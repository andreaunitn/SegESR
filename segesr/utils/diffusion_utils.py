import torch

def _expand_to(values, reference):
    while len(values.shape) < len(reference.shape):
        values = values.unsqueeze(-1)
    return values

def get_diffusion_target(noise_scheduler, latents, noise, timesteps):
    """Returns the regression target of the diffusion loss for the scheduler's prediction type."""

    if noise_scheduler.config.prediction_type == "epsilon":
        return noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        return noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

def predict_original_latents(noise_scheduler, noisy_latents, model_pred, timesteps):
    """Estimates x_0 from x_t and the model prediction (epsilon or v)."""

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=torch.float32)
    alpha_bar_t = _expand_to(alphas_cumprod[timesteps], noisy_latents)

    noisy_latents = noisy_latents.float()
    model_pred = model_pred.float()

    if noise_scheduler.config.prediction_type == "epsilon":
        return (noisy_latents - (1 - alpha_bar_t).sqrt() * model_pred) / alpha_bar_t.sqrt()
    elif noise_scheduler.config.prediction_type == "v_prediction":
        return alpha_bar_t.sqrt() * noisy_latents - (1 - alpha_bar_t).sqrt() * model_pred
    else:
        raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

def decode_latents_to_rgb(vae, latents, dtype):
    """Decodes (scaled) latents with `vae` and maps the output from [-1, 1] to [0, 1]."""

    image = vae.decode((latents / vae.config.scaling_factor).to(dtype)).sample
    return (image.float().clamp(-1.0, 1.0) + 1.0) / 2.0
