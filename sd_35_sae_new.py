import os
import sys
import glob
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Optional

try:
    from diffusers import StableDiffusion3Pipeline
except ImportError:
    print("Please install: pip install diffusers transformers accelerate")
    sys.exit(1)


MODEL_ID = "/stable-diffusion-3.5-medium" 
DATASET_CSV = "/laion_coco_aesthetic_clean.csv" 
DATASET_COLUMN = "caption"

MAX_PROMPTS = 80000
PROMPTS_PER_CHUNK = 500  

TARGET_BLOCK_RANGE = range(0,12)

SAE_DIM_EXPANSION = 4     
SAE_K = 32                
SAE_LR = 3e-4
BATCH_SIZE = 4096
EPOCHS = 10                

BASE_OUTPUT_DIR = "/sd3_outputs_all_blocks_201125"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(BASE_OUTPUT_DIR, "training_part_nbl_1.log"))],
    force=True
)
logger = logging.getLogger("ClusterSAE_Part1")

def ensure_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_prompts(csv_path, limit):
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    prompts = df[DATASET_COLUMN].dropna().astype(str).tolist()
    random.shuffle(prompts) 
    
    if len(prompts) > limit:
        prompts = prompts[:limit]
        
    logger.info(f"Loaded {len(prompts)} unique prompts for training.")
    return prompts


class TopKActivation(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x):
        topk_vals, topk_inds = torch.topk(x, self.k, dim=-1)
        hard = torch.zeros_like(x).scatter(-1, topk_inds, topk_vals.relu())
        return hard + (x - x.detach())

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_hidden, k):
        super().__init__()
        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.b_act = nn.Parameter(torch.zeros(d_hidden))
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)
        self.topk = TopKActivation(k)
        
        with torch.no_grad():
            self.decoder.weight.data[:] = self.decoder.weight.data / (self.decoder.weight.data.norm(dim=0, keepdim=True) + 1e-8)

    def encode(self, x):
        x_cent = x - self.b_pre
        pre_acts = self.encoder(x_cent) + self.b_act
        return self.topk(pre_acts), pre_acts

    def decode(self, acts):
        return self.decoder(acts) + self.b_pre

    def forward(self, x):
        sparse_acts, pre_acts = self.encode(x)
        recon = self.decode(sparse_acts)
        return recon, sparse_acts, pre_acts


class ChunkedDataset(IterableDataset):
    def __init__(self, file_list, batch_size):
        self.file_list = file_list
        self.batch_size = batch_size
        
    def __iter__(self):
        random.shuffle(self.file_list)
        for fpath in self.file_list:
            try:
                data = torch.load(fpath, map_location="cpu")
                if data.dtype != torch.float32: data = data.float()
                idx = torch.randperm(data.size(0))
                data = data[idx]
                for i in range(0, len(data), self.batch_size):
                    batch = data[i:i+self.batch_size]
                    if len(batch) == self.batch_size: yield batch
                del data
            except Exception as e:
                logger.error(f"Failed to load {fpath}: {e}")

def collect_deltas(pipe, prompts, block_idx, out_dir, inference_batch_size=32):
    pipe.enable_model_cpu_offload() 
    
    activations = []
    chunk_files = []
    hook_data = {"input": None}

    def pre_hook(module, args, kwargs):
        if 'hidden_states' in kwargs:
            hook_data["input"] = kwargs['hidden_states'].detach().cpu()
        elif len(args) > 0:
            hook_data["input"] = args[0].detach().cpu()

    def post_hook(module, args, output):
        if isinstance(output, tuple): h_out = output[1].detach().cpu()
        else: h_out = output.detach().cpu()
            
        h_in = hook_data["input"]
        if h_in is not None:
            delta = h_out - h_in
            activations.append(delta.float()) 
            hook_data["input"] = None

    layer = pipe.transformer.transformer_blocks[block_idx]
    h1 = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = layer.register_forward_hook(post_hook)

    logger.info(f"Generating activations for Block {block_idx}...")
    
    with torch.no_grad():
        for i in range(0, len(prompts), inference_batch_size):
            batch_prompts = prompts[i : i + inference_batch_size]
            if i % (inference_batch_size * 10) == 0: 
                print(f"  Gen {i}/{len(prompts)}...", end="\r")
            
            pipe(batch_prompts, num_inference_steps=1, output_type="latent", guidance_scale=0.0)
            
            current_batch_count = len(activations) * inference_batch_size
            if current_batch_count >= PROMPTS_PER_CHUNK:
                flat_acts = torch.cat(activations, dim=0)
                flat_acts = flat_acts.reshape(-1, flat_acts.shape[-1])
                fname = os.path.join(out_dir, f"activations_{i}.pt")
                torch.save(flat_acts, fname)
                chunk_files.append(fname)
                activations = [] 

        if activations:
            flat_acts = torch.cat(activations, dim=0)
            flat_acts = flat_acts.reshape(-1, flat_acts.shape[-1])
            fname = os.path.join(out_dir, f"activations_final.pt")
            torch.save(flat_acts, fname)
            chunk_files.append(fname)

    h1.remove()
    h2.remove()
    return chunk_files


def train_sae(chunk_files, output_path, epochs=1):
    device = ensure_device()
    
    sample = torch.load(chunk_files[0], map_location="cpu")
    d_model = sample.shape[-1]
    d_hidden = d_model * SAE_DIM_EXPANSION
    del sample
    
    logger.info(f"Training SAE: d_model={d_model}, d_hidden={d_hidden}, k={SAE_K}")
    
    sae = SparseAutoencoder(d_model, d_hidden, SAE_K).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=SAE_LR)
    
    dataset = ChunkedDataset(chunk_files, BATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=None)

    sae.train()
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        for batch in dataloader:
            batch = batch.to(device)
            
            mean = batch.mean(dim=0, keepdim=True)
            batch_centered = batch - mean
            scale = batch_centered.norm(p=2, dim=-1, keepdim=True).mean() + 1e-6
            batch_norm = batch_centered / scale
            
            optimizer.zero_grad()
            recon, sparse, encoded = sae(batch_norm)
            
            main_loss = nn.MSELoss()(recon, batch_norm)
            
            with torch.no_grad(): pass 
            
            loss = main_loss 
            loss.backward()
            
            with torch.no_grad():
                 sae.decoder.weight.grad -= (sae.decoder.weight.grad * sae.decoder.weight).sum(dim=0, keepdim=True) * sae.decoder.weight

            optimizer.step()
            total_loss += loss.item()
            steps += 1
            
            if steps % 100 == 0: print(f"  Epoch {epoch+1} | Step {steps} | Loss: {loss.item():.6f}", end="\r")
                
        logger.info(f"Epoch {epoch+1} Complete | Avg Loss: {total_loss/steps:.6f}")

    torch.save(sae.state_dict(), output_path)
    logger.info(f"Saved final SAE to {output_path}")


if __name__ == "__main__":
    prompts = load_prompts(DATASET_CSV, MAX_PROMPTS)
    
    logger.info(f"Initializing Pipeline: {MODEL_ID}")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        use_safetensors=True
    )

    for block_idx in TARGET_BLOCK_RANGE:
        logger.info(f"\n{'='*50}\nSTARTING BLOCK {block_idx}\n{'='*50}")
        
        block_dir = os.path.join(BASE_OUTPUT_DIR, f"block_{block_idx}")
        os.makedirs(block_dir, exist_ok=True)
        
        sae_path = os.path.join(block_dir, f"sae_block_{block_idx}.pt")
        
        if os.path.exists(sae_path):
            logger.info(f"SAE for Block {block_idx} already exists. Skipping.")
            continue

        chunks = collect_deltas(pipe, prompts, block_idx, block_dir)
        
        if not chunks:
            logger.error(f"No data collected for Block {block_idx}!")
            continue
            
        train_sae(chunks, sae_path, EPOCHS)
        
        logger.info(f"Cleaning up activation chunks for Block {block_idx}...")
        for f in chunks:
            if os.path.exists(f): os.remove(f)
            
    logger.info("Part 1 (Blocks 0-10) Complete!")