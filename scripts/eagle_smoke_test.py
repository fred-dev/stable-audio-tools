# Quick check that an EAGLE (climate-conditioned) checkpoint still loads and generates
# with this version of stable-audio-tools.
#
#   uv run python scripts/eagle_smoke_test.py --model-config model_config.json --ckpt-path model.ckpt

import argparse
import json
import sys

import torch
import torchaudio
from einops import rearrange

from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.interface.gradio import unwrap_state_dict
from stable_audio_tools.interface.interfaces.climate_cond import create_climate_cond_ui, get_climate_configs, is_climate_model, presets
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main(args):
    with open(args.model_config) as f:
        model_config = json.load(f)

    assert is_climate_model(model_config), "Config has no numeric climate conditioning"

    model = create_model_from_config(model_config)

    try:
        state_dict = load_ckpt_state_dict(args.ckpt_path)
    except Exception as e:
        # Older Lightning checkpoints can hold non-tensor objects that weights_only=True rejects
        print(f"weights_only load failed ({e}), retrying with weights_only=False")
        state_dict = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

    state_dict = unwrap_state_dict(state_dict, model_config["model_type"])

    model_state = model.state_dict()
    loaded = [k for k in state_dict if k in model_state and state_dict[k].shape == model_state[k].shape]
    unused = [k for k in state_dict if k not in loaded]
    missing = [k for k in model_state if k not in state_dict]

    print(f"Checkpoint tensors loaded: {len(loaded)} / model tensors: {len(model_state)}")
    print(f"Checkpoint tensors unused: {len(unused)}")
    for k in unused[:20]:
        print(f"  unused: {k}")
    print(f"Model tensors not in checkpoint (left at init): {len(missing)}")
    for k in missing[:20]:
        print(f"  missing: {k}")

    model.load_state_dict({k: state_dict[k] for k in loaded}, strict=False)

    device = get_device()
    model.to(device).eval().requires_grad_(False)
    print(f"Device: {device}")

    preset_name = next(iter(presets))
    climate_ids = [c["id"] for c in get_climate_configs(model_config)]
    conditioning = {k: v for k, v in presets[preset_name].items() if k in climate_ids}
    conditioning.update({"seconds_start": 0, "seconds_total": args.seconds})
    print(f"Generating {args.seconds}s with preset '{preset_name}', {args.steps} steps")

    audio = generate_diffusion_cond(
        model,
        conditioning=[conditioning],
        steps=args.steps,
        cfg_scale=4.0,
        sample_size=model_config["sample_size"],
        seed=42,
        device=device,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=500,
    )

    audio = audio[:, :, :int(args.seconds * model_config["sample_rate"])]
    audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32)
    peak = audio.abs().max().item()
    audio = audio.div(max(peak, 1e-8)).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    torchaudio.save(args.output, audio, model_config["sample_rate"])
    print(f"Saved {args.output} (raw peak {peak:.4f})")

    # Build (but don't launch) the Gradio UI to check it constructs under this Gradio version
    create_climate_cond_ui(model_config, model)
    print("Gradio climate UI built OK")

    if missing:
        print("WARNING: some model weights were not in the checkpoint - output may be noise")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smoke test an EAGLE checkpoint')
    parser.add_argument('--model-config', type=str, required=True)
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--output', type=str, default='eagle_smoke_test.wav')
    main(parser.parse_args())
