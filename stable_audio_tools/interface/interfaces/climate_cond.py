# Synthetic Ornithology (EAGLE) interface: diffusion models conditioned on numeric
# climate/location values instead of a text prompt. Kept in its own file so upstream
# changes to diffusion_cond.py merge cleanly.

import gc
import tempfile

import numpy as np
import gradio as gr
import torch
import torchaudio

from einops import rearrange
from torchaudio import transforms as T

from ..aeiou import audio_spectrogram_image
from ...inference.generation import generate_diffusion_cond

model = None
sample_size = 524288
sample_rate = 44100
model_half = False
climate_ids = []

TIMING_IDS = ("seconds_start", "seconds_total")

presets = {
    "Pied Currawong": {
        "latitude": -33.6467,
        "longitude": 150.3246,
        "temperature": 12.43,
        "humidity": 86,
        "wind_speed": 0.66,
        "pressure": 1013,
        "minutes_of_day": 369,
        "day_of_year": 297,
    },
    "Yellow-tailed Black Cockatoo": {
        "latitude": -32.8334,
        "longitude": 150.2001,
        "temperature": 23.23,
        "humidity": 45,
        "wind_speed": 1.37,
        "pressure": 1009,
        "minutes_of_day": 986,
        "day_of_year": 78,
    },
    "Australian Magpie": {
        "latitude": -38.522,
        "longitude": 145.3365,
        "temperature": 18.75,
        "humidity": 67,
        "wind_speed": 1.5,
        "pressure": 1023,
        "minutes_of_day": 940,
        "day_of_year": 307,
    },
    "Laughing Kookaburra": {
        "latitude": -27.2685099,
        "longitude": 152.8587437,
        "temperature": 9.02,
        "humidity": 94,
        "wind_speed": 1.5,
        "pressure": 1025,
        "minutes_of_day": 320,
        "day_of_year": 236,
    },
}

default_values = {
    "latitude": -29.8913,
    "longitude": 152.4951,
    "temperature": 22.05,
    "humidity": 88,
    "wind_speed": 0.54,
    "pressure": 1021,
    "minutes_of_day": 1354,
    "day_of_year": 342,
}

slider_labels = {
    "day_of_year": "Day of year",
}

def get_climate_configs(model_config):
    """Number conditioners other than the timing ones, in config order."""
    conditioning = model_config["model"].get("conditioning") or {}
    return [c for c in conditioning.get("configs", []) if c["type"] == "number" and c["id"] not in TIMING_IDS]

def is_climate_model(model_config):
    conditioning = model_config["model"].get("conditioning") or {}
    ids = [c["id"] for c in conditioning.get("configs", [])]
    return "prompt" not in ids and len(get_climate_configs(model_config)) > 0

def audio_to_tensor(audio):
    in_sr, audio = audio

    if audio.dtype == np.float32:
        audio = torch.from_numpy(audio)
    elif audio.dtype == np.int16:
        audio = torch.from_numpy(audio).float().div(32767)
    elif audio.dtype == np.int32:
        audio = torch.from_numpy(audio).float().div(2147483647)
    else:
        raise ValueError(f"Unsupported audio data type: {audio.dtype}")

    if audio.dim() == 1:
        audio = audio.unsqueeze(0) # [1, n]
    elif audio.dim() == 2:
        audio = audio.transpose(0, 1) # [n, 2] -> [2, n]

    if in_sr != sample_rate:
        audio = T.Resample(in_sr, sample_rate)(audio)

    return audio[:, :sample_size]

def generate_cond(
        seconds_start,
        seconds_total,
        cfg_scale,
        steps,
        preview_every,
        seed,
        sampler_type,
        sigma_min,
        sigma_max,
        cfg_rescale,
        init_audio,
        init_noise_level,
        *climate_values
    ):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    preview_images = []
    if preview_every == 0:
        preview_every = None

    conditioning = dict(zip(climate_ids, climate_values))
    conditioning.update({"seconds_start": seconds_start, "seconds_total": seconds_total})

    device = next(model.parameters()).device

    seed = int(seed)
    if seed == -1:
        seed = np.random.randint(0, 2**31 - 1)

    if init_audio is not None:
        init_audio = audio_to_tensor(init_audio)
        if model_half:
            init_audio = init_audio.to(torch.float16)
        init_audio = (sample_rate, init_audio)

    def progress_callback(callback_info):
        current_step = callback_info["i"]
        if (current_step - 1) % preview_every != 0:
            return
        sigma = callback_info["sigma"]
        if isinstance(sigma, torch.Tensor):
            sigma = sigma[0].item() if sigma.dim() > 0 else sigma.item()
        denoised = callback_info["denoised"]
        if model.pretransform is not None:
            denoised = model.pretransform.decode(denoised)
        denoised = rearrange(denoised, "b d n -> d (b n)")
        denoised = denoised.float().clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        preview_images.append((audio_spectrogram_image(denoised, sample_rate=sample_rate), f"Step {current_step} sigma={sigma:.3f}"))

    audio = generate_diffusion_cond(
        model,
        conditioning=[conditioning],
        steps=steps,
        cfg_scale=cfg_scale,
        batch_size=1,
        sample_size=sample_size,
        seed=seed,
        device=device,
        sampler_type=sampler_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        init_audio=init_audio,
        init_noise_level=init_noise_level,
        callback=progress_callback if preview_every is not None else None,
        scale_phi=cfg_rescale,
    )

    # Cut to the requested length, then peak normalise
    audio = audio[:, :, :int(seconds_total * sample_rate)]
    audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32)
    audio = audio.div(audio.abs().max().clamp(min=1e-8)).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    output_path = tempfile.NamedTemporaryFile(suffix=f".seed{seed}.wav", delete=False).name
    torchaudio.save(output_path, audio, sample_rate)

    return (output_path, [audio_spectrogram_image(audio, sample_rate=sample_rate), *preview_images])

def create_sampling_ui(model_config):
    climate_configs = get_climate_configs(model_config)

    diffusion_objective = model.diffusion_objective
    is_rf = diffusion_objective == "rectified_flow"
    is_rf_denoiser = diffusion_objective == "rf_denoiser"

    conditioning_ids = [c["id"] for c in model_config["model"]["conditioning"]["configs"]]
    has_seconds_start = "seconds_start" in conditioning_ids
    has_seconds_total = "seconds_total" in conditioning_ids

    with gr.Row():
        generate_button = gr.Button("Generate", variant='primary', scale=1)

    with gr.Row(equal_height=False):
        with gr.Column():
            with gr.Row():
                seconds_start_slider = gr.Slider(minimum=0, maximum=512, step=1, value=0, label="Seconds start", visible=has_seconds_start)
                max_seconds = max(sample_size // sample_rate, 1)
                seconds_total_slider = gr.Slider(minimum=1, maximum=max_seconds, step=1, value=max_seconds, label="Seconds total", visible=has_seconds_total)

            with gr.Row():
                steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=250, label="Steps")
                preview_every_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Preview Every")
                cfg_scale_slider = gr.Slider(minimum=0.0, maximum=25.0, step=0.1, value=4.0, label="CFG scale")

            with gr.Accordion("Climate and location", open=True):
                preset_dropdown = gr.Dropdown(choices=list(presets.keys()), value=None, label="Select Preset")

                climate_sliders = []
                for config in climate_configs:
                    min_val = config["config"]["min_val"]
                    max_val = config["config"]["max_val"]
                    default = default_values.get(config["id"], (min_val + max_val) / 2)
                    climate_sliders.append(gr.Slider(
                        minimum=min_val,
                        maximum=max_val,
                        step=(max_val - min_val) / 1000,
                        value=min(max(default, min_val), max_val),
                        label=slider_labels.get(config["id"], config["id"])))

            with gr.Accordion("Sampler params", open=False):
                seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")

                with gr.Row():
                    if is_rf:
                        sampler_types, default_sampler, sigma_max_max, sigma_max_default = ["euler", "rk4", "dpmpp"], "dpmpp", 1.0, 1.0
                    elif is_rf_denoiser:
                        sampler_types, default_sampler, sigma_max_max, sigma_max_default = ["pingpong"], "pingpong", 1.0, 1.0
                    else:
                        sampler_types = ["dpmpp-2m-sde", "dpmpp-3m-sde", "dpmpp-2m", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-adaptive", "k-dpm-fast", "v-ddim", "v-ddim-cfgpp"]
                        default_sampler, sigma_max_max, sigma_max_default = "dpmpp-3m-sde", 1000.0, 500.0

                    sampler_type_dropdown = gr.Dropdown(sampler_types, label="Sampler type", value=default_sampler)
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma min")
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=sigma_max_max, step=0.1, value=sigma_max_default, label="Sigma max")
                    cfg_rescale_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG rescale amount")

            with gr.Accordion("Init audio", open=False):
                init_audio_input = gr.Audio(label="Init audio", waveform_options=gr.WaveformOptions(show_recording_waveform=False))
                init_noise_level_slider = gr.Slider(minimum=0.01, maximum=100.0, step=0.01, value=1.0, label="Init noise level")

        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False)
            audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)
            send_to_init_button = gr.Button("Send to init audio", scale=1)
            send_to_init_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[init_audio_input])

    inputs = [
        seconds_start_slider,
        seconds_total_slider,
        cfg_scale_slider,
        steps_slider,
        preview_every_slider,
        seed_textbox,
        sampler_type_dropdown,
        sigma_min_slider,
        sigma_max_slider,
        cfg_rescale_slider,
        init_audio_input,
        init_noise_level_slider,
    ] + climate_sliders

    generate_button.click(fn=generate_cond,
        inputs=inputs,
        outputs=[audio_output, audio_spectrogram_output],
        api_name="generate")

    def update_sliders(preset_name):
        preset = presets.get(preset_name, {})
        return [gr.update(value=preset[cid]) if cid in preset else gr.update() for cid in climate_ids]

    preset_dropdown.change(fn=update_sliders, inputs=[preset_dropdown], outputs=climate_sliders)

def create_climate_cond_ui(model_config, in_model, in_model_half=False, gradio_title=""):
    global model, sample_size, sample_rate, model_half, climate_ids

    model = in_model
    sample_size = model_config["sample_size"]
    sample_rate = model_config["sample_rate"]
    model_half = in_model_half
    climate_ids = [c["id"] for c in get_climate_configs(model_config)]

    with gr.Blocks() as ui:
        if gradio_title:
            gr.Markdown("### %s" % gradio_title)
        with gr.Tab("Generation"):
            create_sampling_ui(model_config)

    return ui
