# Batch generation for the Synthetic Ornithology models: reads weather/location JSON
# files, generates audio at several CFG scales for each, then moves each JSON into done/.

import gc
import json
import os
import re

import torch
import torchaudio

from einops import rearrange

from ..inference.generation import generate_diffusion_cond
from .gradio import load_model

# An array of cfg scale values to test
cfg_scales = [2.0, 4.0, 8.0]

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def generate_cond_with_path(
        model,
        conditions,
        sample_size,
        sample_rate,
        seconds_start=0,
        seconds_total=11,
        cfg_scale=6.0,
        steps=250,
        seed=-1,
        sampler_type="dpmpp-2m-sde",
        sigma_min=0.03,
        sigma_max=80,
        cfg_rescale=0.4,
        destination_folder=None,
        file_name=None
    ):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    conditioning = [{**conditions, "seconds_start": seconds_start, "seconds_total": seconds_total}]

    device = next(model.parameters()).device

    audio = generate_diffusion_cond(
        model,
        conditioning=conditioning,
        steps=steps,
        cfg_scale=cfg_scale,
        batch_size=1,
        sample_size=sample_size,
        seed=int(seed),
        device=device,
        sampler_type=sampler_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        scale_phi=cfg_rescale
    )

    # Peak normalise and save
    audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32)
    audio = audio.div(audio.abs().max().clamp(min=1e-8)).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    if destination_folder is not None and file_name is not None:
        torchaudio.save(os.path.join(destination_folder, f"{file_name}.wav"), audio, sample_rate)

def load_and_generate(model, model_config, model_path, json_dir, output_dir):
    """Load JSON files and generate audio for each set of conditions."""
    # Make a subdirectory in the json_dir called done to move the files that have been processed
    done_dir = os.path.join(json_dir, 'done')
    os.makedirs(done_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    json_files = sorted(file for file in os.listdir(json_dir) if file.endswith('.json'))

    if not json_files:
        print(f"No JSON files found in {json_dir}. Please check the directory path and file permissions.")
        return

    step_match = re.search(r'step=(\d+)', model_path or "")
    step_number = step_match.group(1) if step_match else "pretrained"

    for json_filename in json_files:
        json_file_path = os.path.join(json_dir, json_filename)

        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
        except Exception as e:
            print(f"Failed to read or parse {json_file_path}: {e}")
            continue

        print(json_file_path)

        conditions = {
            'latitude': data['coord']['lat'],
            'longitude': data['coord']['lon'],
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data['wind']['speed'],
            'day_of_year': data['dayOfYear'],
            'minutes_of_day': data['minutesOfDay']
        }

        bird_species = data['birdSpecies'].replace(' ', '_')
        base_filename = f"{os.path.splitext(json_filename)[0]}_{step_number}_{bird_species}_cfg_scale_"

        for scale in cfg_scales:
            generate_cond_with_path(
                model,
                conditions,
                sample_size=model_config["sample_size"],
                sample_rate=model_config["sample_rate"],
                seconds_start=0,
                seconds_total=11,
                cfg_scale=scale,
                steps=250,
                seed=-1,
                sampler_type="dpmpp-2m-sde",
                sigma_min=0.03,
                sigma_max=80,
                cfg_rescale=0.4,
                destination_folder=output_dir,
                file_name=base_filename + str(scale))

        # Move the processed JSON file to the done directory
        os.rename(json_file_path, os.path.join(done_dir, json_filename))
        print(f"Moved {json_filename} to {done_dir}")

def runTests(model_config_path=None, ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, model_half=False, json_dir=None, output_dir=None):
    assert (pretrained_name is not None) ^ (model_config_path is not None and ckpt_path is not None), "Must specify either pretrained name or provide a model config and checkpoint, but not both"

    if model_config_path is not None:
        with open(model_config_path) as f:
            model_config = json.load(f)
    else:
        model_config = None

    model, model_config = load_model(model_config, ckpt_path, pretrained_name=pretrained_name, pretransform_ckpt_path=pretransform_ckpt_path, model_half=model_half, device=get_device())

    load_and_generate(model, model_config, ckpt_path, json_dir, output_dir)
