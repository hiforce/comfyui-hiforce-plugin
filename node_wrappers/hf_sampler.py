import comfy.sample
import comfy.samplers
import impact.core as core
import torch

from hiforce.mask import composite
from hiforce.sampler import Loopback, BaseSamplerLoader, LoopbackSamplerLoader


class HfSwitchKSampleStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("INT", {"default": 1, "min": 1, "max": 2}),
            }
        }

    RETURN_TYPES = (["Sample", "Hold"],)
    RETURN_NAMES = ("status",)
    FUNCTION = "process"
    CATEGORY = "HiFORCE/Sampler"

    def process(self, enable):
        out = None
        if enable == 1:
            out = "Hold"
        if enable == 2:
            out = "Sample"
        return (out,)


class HfBoolSwitchKSampleStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (["Sample", "Hold"],)
    RETURN_NAMES = ("status",)
    FUNCTION = "process"
    CATEGORY = "HiFORCE/Sampler"

    def process(self, enable):
        out = None
        if not enable:
            out = "Hold"
        if enable:
            out = "Sample"
        return (out,)


class HfLoopback:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enable": ("BOOLEAN", {"default": True}),
                "loops": ("INT", {"default": 4, "min": 1, "max": 16}),
                "start_denoise": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 1.0, "step": 0.01}),
                "final_denoise": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 1.0, "step": 0.01}),
                "policy": (["Aggressive", "Linear", "Lazy"],),
            }
        }

    RETURN_TYPES = ("LOOPBACK",)
    RETURN_NAMES = ("loopback",)
    FUNCTION = "process"

    CATEGORY = "HiFORCE/Sampler"

    def process(self, enable, loops, start_denoise, final_denoise, policy):
        loopback = Loopback()
        loopback.set_enable(enable)
        loopback.set_loops(loops)
        loopback.set_start_denoise(start_denoise)
        loopback.set_final_denoise(final_denoise)
        loopback.set_policy(policy)

        return (loopback,)


class HfLookbackSamplerLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sampler_state": (["Sample", "Hold"],),
                "add_noise": (["enable", "disable"],),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "full_drawing": ("BOOLEAN", {"default": True}),
                "loopback": ("LOOPBACK",),
            }
        }

    RETURN_TYPES = ("KSAMPLER",)
    FUNCTION = "doit"

    CATEGORY = "HiFORCE/Sampler"

    def doit(self, model, sampler_state, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive,
             negative, full_drawing=True, loopback=None):
        enable = True
        if sampler_state == "Hold":
            enable = False
        sampler_loader = LoopbackSamplerLoader(enable, model, add_noise, noise_seed, steps, cfg, sampler_name,
                                               scheduler,
                                               positive, negative, full_drawing, loopback)
        return (sampler_loader,)


class HfSamplerLoopback:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "sampler_state": (["Sample", "Hold"],),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "full_drawing": ("BOOLEAN", {"default": True}),
                "loopback": ("LOOPBACK",),
            },
            "optional": {
                "add_noise": (["enable", "disable"],),
            },

        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("last_latent", "images")

    FUNCTION = "process"

    CATEGORY = "HiFORCE/Sampler"

    def process(self, model, vae, sampler_state, noise_seed, steps, cfg, sampler_name, scheduler, positive,
                negative, latent, full_drawing=True, loopback=None, add_noise="enable"):
        enable = True
        if sampler_state == "Hold":
            enable = False

        sampler_loader = LoopbackSamplerLoader(enable, model, add_noise, noise_seed, steps, cfg, sampler_name,
                                               scheduler, positive, negative, full_drawing, loopback)

        latent_out = sampler_loader.sample(latent)
        image_out = vae.decode(latent_out["samples"]).cpu()

        return latent_out, image_out


class HfSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "sampler_state": (["Sample", "Hold"],),
                     "add_noise": (["enable", "disable"],),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "latent_image": ("LATENT",),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "full_drawing": ("BOOLEAN", {"default": True}),
                     "denoise": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 1.00, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "process"

    CATEGORY = "HiFORCE/Sampler"

    def process(self, model, sampler_state, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive,
                negative, latent_image, start_at_step, end_at_step, full_drawing=True, denoise=1.0):
        if sampler_state == "Hold":
            return (latent_image,)

        return BaseSamplerLoader.do_sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive,
                                           negative, latent_image, start_at_step, end_at_step, full_drawing, denoise)


class HfIterativeLatentUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enable": ("BOOLEAN", {"default": True}),
                "samples": ("LATENT",),
                "upscale_factor": ("FLOAT", {"default": 2, "min": 1, "max": 10000, "step": 0.1}),
                "steps": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1}),
                "temp_prefix": ("STRING", {"default": ""}),
                "upscaler": ("UPSCALER",)
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "doit"

    CATEGORY = "HiFORCE/Sampler/Upscale"

    def doit(self, enable, samples, upscale_factor, steps, temp_prefix, upscaler, unique_id):
        if not enable:
            return (samples,)

        w = samples['samples'].shape[3] * 8  # image width
        h = samples['samples'].shape[2] * 8  # image height

        if temp_prefix == "":
            temp_prefix = None

        upscale_factor_unit = max(0, (upscale_factor - 1.0) / steps)
        current_latent = samples
        scale = 1

        for i in range(steps - 1):
            scale += upscale_factor_unit
            new_w = w * scale
            new_h = h * scale
            core.update_node_status(unique_id, f"{i + 1}/{steps} steps | x{scale:.2f}", (i + 1) / steps)
            print(f"IterativeLatentUpscale[{i + 1}/{steps}]: {new_w:.1f}x{new_h:.1f} (scale:{scale:.2f}) ")
            step_info = i, steps
            current_latent = upscaler.upscale_shape(step_info, current_latent, new_w, new_h, temp_prefix)

        if scale < upscale_factor:
            new_w = w * upscale_factor
            new_h = h * upscale_factor
            core.update_node_status(unique_id, f"Final step | x{upscale_factor:.2f}", 1.0)
            print(f"IterativeLatentUpscale[Final]: {new_w:.1f}x{new_h:.1f} (scale:{upscale_factor:.2f}) ")
            step_info = steps, steps
            current_latent = upscaler.upscale_shape(step_info, current_latent, new_w, new_h, temp_prefix)

        core.update_node_status(unique_id, "", None)

        return (current_latent,)


class HfSamplerLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "sampler_state": (["Sample", "Hold"],),
                     "add_noise": (["enable", "disable"],),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "full_drawing": ("BOOLEAN", {"default": True}),
                     "denoise": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 1.00, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("KSAMPLER",)
    FUNCTION = "doit"

    CATEGORY = "HiFORCE/Sampler"

    def doit(self, model, sampler_state, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive,
             negative, start_at_step, end_at_step, full_drawing=True, denoise=1.0):
        enable = True
        if sampler_state == "Hold":
            enable = False
        sampler_loader = BaseSamplerLoader(enable, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler,
                                           positive, negative, start_at_step, end_at_step, full_drawing, denoise)
        return (sampler_loader,)


def latent_composite_mask(destination, source, mask=None):
    output = destination.copy()
    destination = destination["samples"].clone()
    source = source["samples"]
    output["samples"] = composite(destination, source, 0, 0, mask, 8, False)
    return output


class HfTwoSamplersForMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enable": ("BOOLEAN", {"default": True}),
                "latent_image": ("LATENT",),
                "base_sampler": ("KSAMPLER",),
                "mask_sampler": ("KSAMPLER",),
                "mask": ("MASK",)
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "doit"

    CATEGORY = "HiFORCE/Sampler"

    def doit(self, enable, latent_image, base_sampler, mask_sampler, mask):
        if not enable:
            return (latent_image,)
        inv_mask = torch.where(mask != 1.0, torch.tensor(1.0), torch.tensor(0.0))

        latent_image['noise_mask'] = inv_mask
        new_latent_image = base_sampler.sample(latent_image)

        new_latent_image['noise_mask'] = mask
        new_latent_image = mask_sampler.sample(new_latent_image)

        del new_latent_image['noise_mask']

        return (new_latent_image,)


class HfTwoStepSamplers:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enable": ("BOOLEAN", {"default": True}),
                "latent_image": ("LATENT",),
                "step1_sampler": ("KSAMPLER",),
                "step2_sampler": ("KSAMPLER",),
            },
            "optional": {
                "step1_mask": ("MASK",),
                "step2_mask": ("MASK",),
                "recover_mask": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "doit"

    CATEGORY = "HiFORCE/Sampler"

    def doit(self, enable, latent_image, step1_sampler, step2_sampler,
             step1_mask=None, step2_mask=None, recover_mask=False):
        if not enable:
            return (latent_image,)

        if step1_mask is not None:
            latent_image['noise_mask'] = step1_mask
        last_latent = latent_image
        if step1_sampler is not None and (step1_sampler.enable is None or step1_sampler.enable):
            last_latent = step1_sampler.sample(latent_image)
            if step1_mask is not None:
                if recover_mask:
                    inv_mask = torch.where(step1_mask != 1.0, torch.tensor(1.0), torch.tensor(0.0))
                    last_latent = latent_composite_mask(last_latent, latent_image, inv_mask)
                del last_latent['noise_mask']

        new_latent_image = last_latent
        if step2_sampler is not None and (step2_sampler.enable is None or step2_sampler.enable):
            if step2_mask is not None:
                last_latent['noise_mask'] = step2_mask
            new_latent_image = step2_sampler.sample(last_latent)
            if step2_mask is not None:
                if recover_mask:
                    inv_mask = torch.where(step2_mask != 1.0, torch.tensor(1.0), torch.tensor(0.0))
                    new_latent_image = latent_composite_mask(new_latent_image, last_latent, inv_mask)
                del new_latent_image['noise_mask']
        return (new_latent_image,)


NODE_CLASS_MAPPINGS = {
    "HfSampler": HfSampler,
    "HfSamplerLoopback": HfSamplerLoopback,
    "HfLoopback": HfLoopback,
    "HfSwitchKSampleStatus": HfSwitchKSampleStatus,
    "HfBoolSwitchKSampleStatus": HfBoolSwitchKSampleStatus,
    "HfIterativeLatentUpscale": HfIterativeLatentUpscale,
    "HfTwoSamplersForMask": HfTwoSamplersForMask,
    "HfSamplerLoader": HfSamplerLoader,
    "HfLookbackSamplerLoader": HfLookbackSamplerLoader,
    "HfTwoStepSamplers": HfTwoStepSamplers
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HfSampler": "Basic Sampler",
    "HfSamplerLoopback": "Loopback Sampler",
    "HfTwoStepSamplers": "Two Step Sampler",
    "HfTwoSamplersForMask": "Two Samplers for Mask",
    "HfLoopback": "Loopback Setting",
    "HfSwitchKSampleStatus": "Sampler Switch-INT",
    "HfBoolSwitchKSampleStatus": "Sampler Switch-BOOL",
    "HfIterativeLatentUpscale": "Sampler Upscale - Iterative Latent",
    "HfSamplerLoader": "Sampler Loader - Basic",
    "HfLookbackSamplerLoader": "Sampler Loader - Loopback",
}
