import gc
import numpy as np
import json 
import torch
import torchaudio
import sys
import os

from aeiou.viz import audio_spectrogram_image
from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T

from ..inference.generation import generate_diffusion_cond, generate_diffusion_uncond
from ..models.factory import create_model_from_config
from ..models.pretrained import get_pretrained_model
from ..models.utils import load_ckpt_state_dict
from ..inference.utils import prepare_audio
from ..training.utils import copy_state_dict


model = None
sample_rate = 44100
sample_size = 524288

def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device="cuda", model_half=False):
    global model, sample_rate, sample_size
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        # Load checkpoint
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))
        #model.load_state_dict(load_ckpt_state_dict(model_ckpt_path))

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path), strict=False)
        print(f"Done loading pretransform")

    model.to(device).eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)
        
    print(f"Done loading model")

    return model, model_config

def generate_cond(
        prompt,
        negative_prompt=None,
        seconds_start=0,
        seconds_total=30,
        latitude = 0.0,
        longitude = 0.0,
        temperature = 0.0,
        humidity = 0.0,
        wind_speed = 0.0,
        pressure = 0.0,
        minutes_of_day = 0.0,
        day_of_year = 0.0,
        cfg_scale=6.0,
        steps=250,
        preview_every=None,
        seed=-1,
        sampler_type="dpmpp-2m-sde",
        sigma_min=0.03,
        sigma_max=50,
        cfg_rescale=0.4,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        mask_cropfrom=None,
        mask_pastefrom=None,
        mask_pasteto=None,
        mask_maskstart=None,
        mask_maskend=None,
        mask_softnessL=None,
        mask_softnessR=None,
        mask_marination=None,
        batch_size=1,
        destination_folder=None,
        file_name=None    
    ):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"Prompt: {prompt}")

    global preview_images
    preview_images = []
    if preview_every == 0:
        preview_every = None

    # Return fake stereo audio
    conditioning = [{"prompt": prompt, "latitude": latitude, "longitude": longitude, "temperature": temperature, "humidity": humidity, "wind_speed": wind_speed, "pressure": pressure, "minutes_of_day": minutes_of_day,"day_of_year": day_of_year, "seconds_start":seconds_start, "seconds_total": seconds_total }] * batch_size

    if negative_prompt:
        negative_conditioning = [{"prompt": negative_prompt, "latitude": latitude, "longitude": longitude, "temperature": temperature, "humidity": humidity, "wind_speed": wind_speed, "pressure": pressure, "minutes_of_day": minutes_of_day,"day_of_year": day_of_year, "seconds_start":seconds_start, "seconds_total": seconds_total}] * batch_size
    else:
        negative_conditioning = None
        
    #Get the device from the model
    device = next(model.parameters()).device

    seed = int(seed)

    if not use_init:
        init_audio = None
    
    input_sample_size = sample_size

    if init_audio is not None:
        in_sr, init_audio = init_audio
        # Turn into torch tensor, converting from int16 to float32
        init_audio = torch.from_numpy(init_audio).float().div(32767)
        
        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0) # [1, n]
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1) # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:

            input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length

        init_audio = (sample_rate, init_audio)

    def progress_callback(callback_info):
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        if (current_step - 1) % preview_every == 0:
            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)
            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})"))

    # If inpainting, send mask args
    # This will definitely change in the future
    if mask_cropfrom is not None: 
        mask_args = {
            "cropfrom": mask_cropfrom,
            "pastefrom": mask_pastefrom,
            "pasteto": mask_pasteto,
            "maskstart": mask_maskstart,
            "maskend": mask_maskend,
            "softnessL": mask_softnessL,
            "softnessR": mask_softnessR,
            "marination": mask_marination,
        }
    else:
        mask_args = None 

    # Do the audio generation
    audio = generate_diffusion_cond(
        model, 
        conditioning=conditioning,
        negative_conditioning=negative_conditioning,
        steps=steps,
        cfg_scale=cfg_scale,
        batch_size=batch_size,
        sample_size=input_sample_size,
        sample_rate=sample_rate,
        seed=seed,
        device=device,
        sampler_type=sampler_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        init_audio=init_audio,
        init_noise_level=init_noise_level,
        mask_args = mask_args,
        callback = progress_callback if preview_every is not None else None,
        scale_phi = cfg_rescale
    )

    # Convert to WAV file
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    #save to the desired folder with the required filename and add the .wav extension
    
    if destination_folder is not None and file_name is not None:
        torchaudio.save(f"{destination_folder}/{file_name}.wav", audio, sample_rate)
        
        

    # Let's look at a nice spectrogram too
    # audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    # return ("output.wav", [audio_spectrogram, *preview_images])



def generate_lm(
        temperature=1.0,
        top_p=0.95,
        top_k=0,    
        batch_size=1,
        ):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    audio = model.generate_audio(
        batch_size=batch_size,
        max_gen_len = sample_size//model.pretransform.downsampling_ratio,
        conditioning=None,
        temp=temperature,
        top_p=top_p,
        top_k=top_k,
        use_cache=True
    )

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    return ("output.wav", [audio_spectrogram])




def autoencoder_process(audio, latent_noise, n_quantizers):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    in_sr, audio = audio

    audio = torch.from_numpy(audio).float().div(32767).to(device)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.transpose(0, 1)

    audio = model.preprocess_audio_for_encoder(audio, in_sr)
    # Note: If you need to do chunked encoding, to reduce VRAM, 
    # then add these arguments to encode_audio and decode_audio: chunked=True, overlap=32, chunk_size=128
    # To turn it off, do chunked=False
    # Optimal overlap and chunk_size values will depend on the model. 
    # See encode_audio & decode_audio in autoencoders.py for more info
    # Get dtype of model
    dtype = next(model.parameters()).dtype

    audio = audio.to(dtype)

    if n_quantizers > 0:
        latents = model.encode_audio(audio, chunked=False, n_quantizers=n_quantizers)
    else:
        latents = model.encode_audio(audio, chunked=False)

    if latent_noise > 0:
        latents = latents + torch.randn_like(latents) * latent_noise

    audio = model.decode_audio(latents, chunked=False)

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav"

def load_and_generate(model_path, json_dir, output_dir):
    """ Load JSON files and generate audio for each set of conditions """
    for json_file in glob(os.path.join(json_dir, '*.json')):
        with open(json_file, 'r') as file:
            data = json.load(file)
        #print the json path
        print(json_file)
        # Extract conditions from JSON
        conditions = {
            'birdSpecies': data['birdSpecies'],
            'latitude': data['coord']['lat'],
            'longitude': data['coord']['lon'],
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data['wind']['speed'],
            'day_of_year': data['dayOfYear'],
            'minutes_of_day': data['minutesOfDay']
        }
        
        # Extract base filename components
        step_number = re.search(r'step=(\d+)', model_path).group(1)  # Adjust regex as needed
        json_filename = os.path.splitext(os.path.basename(json_file))[0]
        bird_species = data['birdSpecies'].replace(' ', '_')
        
        base_filename = f"{bird_species}_{json_filename}_{step_number}_cfg_scale_"
        
        #An array of cfg scale values to test
        cfg_scales = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
        
        # Generate audio we do this 4 times with a loop
        for i in range(5):
            generate_cond(prompt = "",
            negative_prompt="",
            seconds_start=0,
            seconds_total=22,
            latitude = conditions['latitude'],
            longitude = conditions['longitude'],
            temperature = conditions['temperature'],
            humidity = conditions['humidity'],
            wind_speed = conditions['wind_speed'],
            pressure = conditions['pressure'],
            minutes_of_day = conditions['minutes_of_day'],
            day_of_year = conditions['day_of_year'],
            cfg_scale=cfg_scales[i],
            steps=250,
            preview_every=None,
            seed=-1,
            sampler_type="dpmpp-2m-sde",
            sigma_min=0.03,
            sigma_max=50,
            cfg_rescale=0.4,
            use_init=False,
            init_audio=None,
            init_noise_level=1.0,
            mask_cropfrom=None,
            mask_pastefrom=None,
            mask_pasteto=None,
            mask_maskstart=None,
            mask_maskend=None,
            mask_softnessL=None,
            mask_softnessR=None,
            mask_marination=None,
            batch_size=1,
            destination_folder=output_dir,
            file_name=base_filename + str(cfg_scales[i]))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, help='Path to model configuration file')
    parser.add_argument('--model_ckpt_path', type=str, help='Path to model checkpoint file')
    parser.add_argument('--pretrained_name', type=str, help='Name of pretrained model')
    parser.add_argument('--pretransform_ckpt_path', type=str, help='Path to pretransform checkpoint file')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    parser.add_argument('--json_dir', type=str, help='Path to directory containing JSON files')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.pretrained_name:
        model, model_config = load_model(pretrained_name=args.pretrained_name, device=device)
    else:
        with open(args.model_config) as f:
            model_config = json.load(f)
        model, model_config = load_model(model_config=model_config, model_ckpt_path=args.model_ckpt_path, pretransform_ckpt_path=args.pretransform_ckpt_path, device=device)
        
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all JSON files and generate audio
    load_and_generate(args.model_ckpt_path, args.json_dir, args.output_dir)    
