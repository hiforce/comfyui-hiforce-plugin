import math
from abc import abstractmethod

import comfy.model_management
import comfy.sample
import comfy.utils
import latent_preview
import nodes
import numpy as np
import torch


class SamplerLoader:
    @abstractmethod
    def sample(self, latent_image):
        pass


def prepare_noise(latent_image, noise_inds=None):
    generator = torch.Generator(device='cuda')
    if noise_inds is None:
        return torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                           generator=generator, device="cuda")

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1] + 1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout,
                            generator=generator, device="cuda")
        if i in unique_inds:
            print(f"add noise: {i}")
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises


class LoopbackSamplerLoader(SamplerLoader):

    def __init__(self, enable, model, add_noise, noise_seed, steps, cfg, sampler_name,
                 scheduler, positive, negative, full_drawing, loopback):
        self.enable = enable
        self.model = model
        self.disable_noise = False
        if add_noise == "disable":
            self.disable_noise = True
        self.noise_seed = noise_seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.positive = positive
        self.negative = negative
        self.full_drawing = full_drawing
        self.loopback = loopback

    @staticmethod
    def get_loop_denoise_list(start: float, end: float, size: int, policy: str) -> list:
        result = list()
        if size == 1:
            result.append(end)
            return result
        if end < start:
            start = end

        for x in range(size):
            if start == end:
                result.append(end)
                continue
            strength = LoopbackSamplerLoader.get_step_by_policy(x / (size - 1), start, end, policy)
            result.append(strength)
        return result

    @staticmethod
    def get_step_by_policy(progress: float, start: float, end: float, policy: str) -> float:
        if policy == "Aggressive":
            num = math.sin(progress * math.pi * 0.5)
        elif policy == "Lazy":
            num = 1 - math.cos(progress * math.pi * 0.5)
        else:
            num = progress
        strength = start + num * (end - start)
        return strength

    def sample(self, latent):
        if not self.enable:
            return latent
        latent_image = latent["samples"]

        if not self.loopback.enable:
            latest_latent = BaseSamplerLoader.do_sample(self.model, not self.disable_noise, self.noise_seed,
                                                        self.steps, self.cfg, self.sampler_name, self.scheduler,
                                                        self.positive, self.negative, latent, 0, 10000,
                                                        self.full_drawing, self.loopback.final_denoise)[0]
            return latest_latent

        if self.disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                                device="cuda")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, self.noise_seed, batch_inds)

        noise_list = LoopbackSamplerLoader.get_loop_denoise_list(self.loopback.start_denoise,
                                                                 self.loopback.final_denoise,
                                                                 self.loopback.loops, self.loopback.policy)
        latest_out = None
        latest_latent = None

        for x in range(self.loopback.loops):
            denoise = noise_list[x]

            if latest_latent is None:
                latest_latent = loopback_ksampler(noise, self.model, self.noise_seed, self.steps, self.cfg,
                                                  self.sampler_name, self.scheduler, self.positive, self.negative,
                                                  latent, denoise, self.disable_noise,
                                                  force_full_denoise=self.full_drawing)
                latest_out = latest_latent
            else:
                latent_image = latest_latent["samples"]
                noise = prepare_noise(latent_image, 0)

                latest_latent = loopback_ksampler(noise, self.model, self.noise_seed, self.steps, self.cfg,
                                                  self.sampler_name, self.scheduler,
                                                  self.positive, self.negative, latent, denoise, False,
                                                  force_full_denoise=self.full_drawing)

                latest_out = latent_batch(latest_out, latest_latent)
        return latest_out


class BaseSamplerLoader(SamplerLoader):

    @staticmethod
    def do_sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
                  latent_image, start_at_step, end_at_step, full_drawing, denoise=1.0):
        force_full_denoise = full_drawing
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        return nodes.common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                     latent_image, denoise=denoise, disable_noise=disable_noise,
                                     start_step=start_at_step, last_step=end_at_step,
                                     force_full_denoise=force_full_denoise)

    def __init__(self, enable, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive,
                 negative, start_at_step, end_at_step, full_drawing, denoise):
        self.enable = enable
        self.model = model
        self.add_noise = add_noise
        self.noise_seed = noise_seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.positive = positive
        self.negative = negative
        self.start_at_step = start_at_step
        self.end_at_step = end_at_step
        self.full_drawing = full_drawing
        self.denoise = denoise

    def sample(self, latent_image):
        if not self.enable:
            return (latent_image,)

        return BaseSamplerLoader.do_sample(self.model, self.add_noise, self.noise_seed, self.steps, self.cfg,
                                           self.sampler_name,
                                           self.scheduler, self.positive, self.negative, latent_image,
                                           self.start_at_step, self.end_at_step, self.full_drawing, self.denoise)[0]


class Loopback:
    def __init__(self):
        self.policy = None
        self.final_denoise = 1.00
        self.start_denoise = 0.35
        self.loops = 1
        self.enable = False

    def set_enable(self, enable: bool):
        self.enable = enable

    def set_loops(self, loops: int):
        self.loops = loops

    def set_start_denoise(self, start_denoise: float):
        self.start_denoise = start_denoise

    def set_final_denoise(self, final_denoise: float):
        self.final_denoise = final_denoise

    def set_policy(self, policy: str):
        self.policy = policy


def loopback_ksampler(noise, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0,
                      disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step,
                                  last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback,
                                  disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return out


def latent_batch(samples1, samples2):
    samples_out = samples1.copy()
    s1 = samples1["samples"]
    s2 = samples2["samples"]

    if s1.shape[1:] != s2.shape[1:]:
        s2 = comfy.utils.common_upscale(s2, s1.shape[3], s1.shape[2], "bilinear", "center")
    s = torch.cat((s1, s2), dim=0)
    samples_out["samples"] = s
    samples_out["batch_index"] = samples1.get("batch_index", [x for x in range(0, s1.shape[0])]) + samples2.get(
        "batch_index", [x for x in range(0, s2.shape[0])])
    return samples_out
