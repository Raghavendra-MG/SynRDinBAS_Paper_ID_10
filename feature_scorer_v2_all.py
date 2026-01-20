#!/usr/bin/env python3
import os
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from diffusers import StableDiffusion3Pipeline

# ---------------- CONFIG ----------------
MODEL_ID = "/stable-diffusion-3.5-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAE_BASE_DIR = ""
EXPORT_DIR = ""
os.makedirs(EXPORT_DIR, exist_ok=True)

BLOCKS = [0, 24]              
SAE_K = 32
N_SAMPLES = 120              
INFER_STEPS = 1
GUIDANCE_SCALE = 0.0
TOP_K_EXPORT = 20

TARGET_CONCEPTS = [
    {
        "axis": "expression_smile",
        "pos_prompt": "A close-up professional portrait photo of a person smiling broadly, showing teeth, raised cheeks, bright expression.",
        "neg_prompt": "A close-up professional portrait photo of a person with a neutral expression, relaxed lips, no smile."
    },
    {
        "axis": "gender_male_vs_female",
        "pos_prompt": "A close-up portrait of an adult male professional, masculine features, short hair.",
        "neg_prompt": "A close-up portrait of an adult female professional, feminine features, long hair."
    },
    {
        "axis": "age_young_vs_elderly",
        "pos_prompt": "A portrait of a young adult (about 25 years old), smooth skin, youthful features.",
        "neg_prompt": "A portrait of an elderly adult (about 70 years old), aged skin texture, wrinkles."
    },
    {
        "axis": "race_white_vs_black",
        "pos_prompt": "A professional portrait photo of an adult with fair skin tone, even lighting.",
        "neg_prompt": "A professional portrait photo of an adult with dark skin tone, even lighting."
    },
    {
        "axis": "nationality_american_vs_japanese",
        "pos_prompt": "A portrait of an adult with visual cues associated with American nationality.",
        "neg_prompt": "A portrait of an adult with visual cues associated with Japanese nationality."
    },
    {
        "axis": "hairstyle_short_vs_long",
        "pos_prompt": "A portrait of a person with a short haircut, clean trimmed hair.",
        "neg_prompt": "A portrait of a person with long styled hair."
    }
]

D_MODEL = 1536
IMAGE_TOKENS = 4096
MIN_IMAGE_TOKENS = 1024   

class TopKActivation(nn.Module):
    def __init__(self, k:int):
        super().__init__()
        self.k = k
    def forward(self, x):
        topk_vals, topk_inds = torch.topk(x, self.k, dim=-1)
        hard = torch.zeros_like(x).scatter(-1, topk_inds, topk_vals.relu())
        return hard + (x - x.detach())

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model:int, d_hidden:int, k:int):
        super().__init__()
        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.b_act = nn.Parameter(torch.zeros(d_hidden))
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)
        self.topk = TopKActivation(k)
    def encode(self, x):
        x_cent = x - self.b_pre
        pre = self.encoder(x_cent) + self.b_act
        sparse = self.topk(pre)
        return sparse, pre

_delta_buf = {"h_in": None, "delta": None}

def is_image_like(t: torch.Tensor):
    return (isinstance(t, torch.Tensor)
            and t.dim() == 3
            and t.shape[-1] == D_MODEL
            and t.shape[1] >= MIN_IMAGE_TOKENS)

def pre_hook(module, args, kwargs):
    cand = None
    if kwargs:
        for v in kwargs.values():
            if is_image_like(v):
                cand = v
                break
    if cand is None:
        for a in args:
            if is_image_like(a):
                cand = a
                break
    if cand is None:
        _delta_buf["h_in"] = None
        return None
    _delta_buf["h_in"] = cand.detach().cpu().clone()
    return None  

def post_hook(module, args, output):
    out_t = None
    if isinstance(output, tuple):
        for el in output:
            if is_image_like(el):
                out_t = el
                break
    elif is_image_like(output):
        out_t = output

    if out_t is None or _delta_buf.get("h_in") is None:
        _delta_buf["delta"] = None
        return

    out_cpu = out_t.detach().cpu()
    h_in = _delta_buf.get("h_in")
    if out_cpu.shape[1] != h_in.shape[1]:
        if out_cpu.shape[1] >= MIN_IMAGE_TOKENS and h_in.shape[1] >= MIN_IMAGE_TOKENS:
            pass
        else:
            _delta_buf["delta"] = None
            return

    delta = out_cpu - h_in
    delta_flat = delta.reshape(-1, delta.shape[-1])   
    _delta_buf["delta"] = delta_flat
    return

def load_sae_from_checkpoint(sae_ckpt_path, device):
    state = torch.load(sae_ckpt_path, map_location="cpu")
    if "decoder.weight" not in state:
        dec_key = next((k for k in state.keys() if k.endswith("decoder.weight")), None)
        enc_key = next((k for k in state.keys() if k.endswith("encoder.weight")), None)
        bpre = next((k for k in state.keys() if k.endswith("b_pre")), None)
        bact = next((k for k in state.keys() if k.endswith("b_act")), None)
        new = {}
        new["encoder.weight"] = state[enc_key]
        new["decoder.weight"] = state[dec_key]
        if bpre: new["b_pre"] = state[bpre]
        if bact: new["b_act"] = state[bact]
        state = new
    d_model, d_hidden = state["decoder.weight"].shape[0], state["decoder.weight"].shape[1]
    sae = SparseAutoencoder(d_model, d_hidden, SAE_K)
    sd = sae.state_dict()
    for k in sd.keys():
        if k in state:
            sd[k] = state[k]
        else:
            suff = k.split(".")[-1]
            cand_key = next((v for vk,v in state.items() if vk.endswith(suff)), None)
            if cand_key is not None:
                sd[k] = cand_key
    sae.load_state_dict(sd)
    sae.to(torch.float32).to(device)
    sae.eval()
    return sae, d_model, d_hidden

def score_axis_for_block(pipe, sae, block_idx, axis, pos_prompt, neg_prompt, n_samples=N_SAMPLES):
    d_hidden = sae.decoder.weight.shape[1]
    pos_acc = torch.zeros(d_hidden, dtype=torch.float64, device="cpu")
    neg_acc = torch.zeros(d_hidden, dtype=torch.float64, device="cpu")

    layer = pipe.transformer.transformer_blocks[block_idx]

    sae.to(torch.float32).to(DEVICE)

    def run_and_accumulate(prompt, accumulator, name):
        for i in tqdm(range(n_samples), desc=f"[B{block_idx}] {axis} {name}", leave=False):
            gen = torch.Generator(DEVICE).manual_seed(1000 + i)
            h1 = h2 = None
            try:
                h1, h2 = layer.register_forward_pre_hook(pre_hook, with_kwargs=True), layer.register_forward_hook(post_hook)
                with torch.no_grad():
                    pipe(prompt, num_inference_steps=INFER_STEPS, guidance_scale=GUIDANCE_SCALE, generator=gen, output_type="latent")
                delta = _delta_buf.get("delta", None)
                if delta is None:
                    continue
                delta_gpu = delta.to(DEVICE).to(torch.float32)
                mean = delta_gpu.mean(dim=0, keepdim=True)
                centered = delta_gpu - mean
                scale = centered.norm(p=2, dim=-1, keepdim=True).mean() + 1e-6
                normalized = centered / scale
                with torch.no_grad():
                    sparse, pre = sae.encode(normalized)
                    # use max over tokens (causal signal)
                    feature_strength = pre.max(dim=0)[0]
                    accumulator.add_(feature_strength.detach().cpu().to(torch.float64))
                _delta_buf["delta"] = None
                _delta_buf["h_in"] = None
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"  Warning sample error B{block_idx} {axis} {name}: {e}")
            finally:
                if h1: h1.remove()
                if h2: h2.remove()

    run_and_accumulate(pos_prompt, pos_acc, "POS")
    run_and_accumulate(neg_prompt, neg_acc, "NEG")

    pos_mean = (pos_acc / max(1, n_samples)).numpy()
    neg_mean = (neg_acc / max(1, n_samples)).numpy()
    diff = pos_mean - neg_mean
    idx_sorted = np.argsort(diff)[::-1]

    rows = []
    for rank, feat_idx in enumerate(idx_sorted[:TOP_K_EXPORT]):
        rows.append({
            "axis": axis, "block": block_idx, "feature_idx": int(feat_idx),
            "score_diff": float(diff[feat_idx]), "rank": rank+1,
            "pos_prompt": pos_prompt, "neg_prompt": neg_prompt
        })

    best = int(idx_sorted[0])
    decoder = sae.decoder.weight.detach().cpu().numpy()  
    vec = decoder[:, best]
    out_block_dir = Path(EXPORT_DIR) / f"block{block_idx}"
    out_block_dir.mkdir(parents=True, exist_ok=True)
    vec_path = out_block_dir / f"feat{best}.npy"
    np.save(str(vec_path), vec)
    print(f"  Exported top vector -> {vec_path}")

    return rows

def main():
    print("Loading pipeline:", MODEL_ID)
    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, use_safetensors=True)
    pipe.enable_model_cpu_offload()
    try:
        pipe.transformer.to(DEVICE)
    except Exception:
        pass
    pipe.set_progress_bar_config(disable=True)
    print("Pipeline ready. Transformer moved to device where possible.")

    all_rows = []
    for concept in TARGET_CONCEPTS:
        axis = concept["axis"]
        for block in BLOCKS:
            ck = os.path.join(SAE_BASE_DIR, f"block_{block}", f"sae_block_{block}.pt")
            if not os.path.exists(ck):
                print(f"  SAE checkpoint missing for block {block}: {ck} (skipping)")
                continue
            print(f"  Loading SAE -> {ck}")
            sae, dm, dh = load_sae_from_checkpoint(ck, DEVICE)
            if dm != D_MODEL:
                print(f"  WARNING: SAE d_model={dm} != expected {D_MODEL}")
            rows = score_axis_for_block(pipe, sae, block, axis, concept["pos_prompt"], concept["neg_prompt"], n_samples=N_SAMPLES)
            all_rows.extend(rows)
            del sae
            torch.cuda.empty_cache()
            gc.collect()

    if not all_rows:
        print("No results.")
        return
    df = pd.DataFrame(all_rows)
    out_csv = Path(EXPORT_DIR) / "feature_scoring_results_v2_export_all.csv"
    df.to_csv(out_csv, index=False)
    print("Saved scoring CSV:", out_csv)

if __name__ == "__main__":
    main()
