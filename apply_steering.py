import os
import torch
import numpy as np
from diffusers import StableDiffusion3Pipeline



MODEL_ID = "/stable-diffusion-3.5-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VECTOR_DIR = "./vectors"        
OUT_DIR = "./axis_showcase"
os.makedirs(OUT_DIR, exist_ok=True)

BLOCKS = [3, 4]

ALPHA = 8.0

SEED = 12345

PROMPT_BASE = (
    "A high quality professional passport-style portrait photo, "
    "neutral lighting, frontal face, plain background"
)

AXES = {
    "smile": {
        "pos": "expression_smile",
        "neg": None,
    },
    "icao_compliant": {
        "pos": None,
        "neg": "expression_smile",
    },
    "male": {
        "pos": "gender_male_vs_female",
        "neg": None,
    },
    "female": {
        "pos": None,
        "neg": "gender_male_vs_female",
    },
}


def load_vector(axis, block):
    path = os.path.join(VECTOR_DIR, f"{axis}_block{block}_combined.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    v = np.load(path)
    v = v / (np.linalg.norm(v) + 1e-8)
    return torch.tensor(v, dtype=torch.float32, device=DEVICE)


def apply_hook(block, vec, alpha):
    def hook(module, args, kwargs):
        if "hidden_states" in kwargs:
            h = kwargs["hidden_states"]
            kwargs["hidden_states"] = h + alpha * vec
        elif len(args) > 0:
            args = list(args)
            args[0] = args[0] + alpha * vec
            return tuple(args), kwargs
        return None
    return hook


def main():
    print("Loading pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    for name, cfg in AXES.items():
        print(f"\nGenerating: {name}")

        hooks = []

        for block in BLOCKS:
            layer = pipe.transformer.transformer_blocks[block]

            if cfg["pos"] is not None:
                vec = load_vector(cfg["pos"], block)
                h = layer.register_forward_pre_hook(
                    apply_hook(block, vec, +ALPHA),
                    with_kwargs=True,
                )
                hooks.append(h)

            if cfg["neg"] is not None:
                vec = load_vector(cfg["neg"], block)
                h = layer.register_forward_pre_hook(
                    apply_hook(block, vec, -ALPHA),
                    with_kwargs=True,
                )
                hooks.append(h)

        generator = torch.Generator(device=DEVICE).manual_seed(SEED)

        with torch.no_grad():
            image = pipe(
                PROMPT_BASE,
                num_inference_steps=30,
                guidance_scale=5.0,
                generator=generator,
            ).images[0]

        out_path = os.path.join(OUT_DIR, f"{name}.png")
        image.save(out_path)
        print("Saved →", out_path)

        for h in hooks:
            h.remove()

        torch.cuda.empty_cache()

    print("Axis showcase generation complete.")


if __name__ == "__main__":
    main()
