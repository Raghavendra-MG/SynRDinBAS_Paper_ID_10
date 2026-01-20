import os
import torch
import numpy as np
from torch import nn

# ----------------------
# CONFIG
# ----------------------
SAE_BASE_DIR = ""
OUT_DIR = ""

BLOCKS = [0, 24]   # only these


# --- Architecture (must match training) ---
class TopKActivation(nn.Module):
    def __init__(self, k):
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
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)   # (d_hidden, d_model)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)   # (d_model, d_hidden)
        self.topk = TopKActivation(k)

    def encode(self, x):
        x_cent = x - self.b_pre
        pre_acts = self.encoder(x_cent) + self.b_act
        return self.topk(pre_acts), pre_acts


# ----------------------
# LOAD + DETECT SHAPES
# ----------------------
def load_sae(block_idx):
    block_dir = os.path.join(SAE_BASE_DIR, f"block_{block_idx}")
    sae_path = os.path.join(block_dir, f"sae_block_{block_idx}.pt")

    print(f"\nLoading SAE for block {block_idx}: {sae_path}")

    state = torch.load(sae_path, map_location="cpu")

    encoder_w = state["encoder.weight"]  # (d_hidden, d_model)
    decoder_w = state["decoder.weight"]  # (d_model, d_hidden)

    d_model = encoder_w.shape[1]         # 1536
    d_hidden = encoder_w.shape[0]        # 6144

    print(f"Detected architecture: d_model={d_model}, d_hidden={d_hidden}")

    sae = SparseAutoencoder(d_model, d_hidden, k=32)
    sae.load_state_dict(state)
    sae.eval()

    return sae, d_hidden, d_model


# ----------------------
# EXPORT
# ----------------------
def export_block(block_idx):
    sae, d_hidden, d_model = load_sae(block_idx)

    decoder_matrix = sae.decoder.weight.detach().cpu().numpy()  # shape (1536, 6144)

    out_block_dir = os.path.join(OUT_DIR, f"block{block_idx}")
    os.makedirs(out_block_dir, exist_ok=True)

    print(f"Exporting {d_hidden} features → {out_block_dir}")

    for j in range(d_hidden):  # 0 .. 6143
        vec = decoder_matrix[:, j]   # COLUMN extraction (1536,)
        out_path = os.path.join(out_block_dir, f"feat{j}.npy")
        np.save(out_path, vec)

        if j % 500 == 0:
            print(f"  Saved feature {j}/{d_hidden}")

    print(f"✓ Finished exporting block {block_idx}")


# ----------------------
# MAIN
# ----------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    for block in BLOCKS:
        export_block(block)

    print("\nAll SAE decoder features exported successfully!")
