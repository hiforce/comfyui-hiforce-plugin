import datetime
import os
import uuid

import folder_paths
import numpy as np
from PIL import Image

from hiforce.image import tensor2rgba, tensor2rgb, process_resize_image, adjust_image_with_max_size
from hiforce.notify import ImageSaveNotify


class HfInitImageWithMaxSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mas_size": ("INT", {"default": 512, "min": 256, "max": 2048}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    CATEGORY = "HiFORCE/Image/Create"

    def process(self, image, mas_size):
        return (adjust_image_with_max_size(image, mas_size),)


class HfResizeImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_resize"

    CATEGORY = "HiFORCE/Image/Zoom"

    def process_resize(self, images, width, height):
        return (process_resize_image(images, width, height),)


class HfImageToRGBA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    CATEGORY = "HiFORCE/Image/Convert"

    def process(self, images):
        out = tensor2rgba(images)
        return (out,)


class HfImageToRGB:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    CATEGORY = "HiFORCE/Image/Convert"

    def process(self, images):
        out = tensor2rgb(images)
        return (out,)


class HfSaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enable": ("BOOLEAN", {"default": True}),
                "images": ("IMAGE",)
            },
            "optional": {
                "notify_hook": ("NOTIFY",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "HiFORCE/Image/Save"

    def save_images(self, enable, images, notify: ImageSaveNotify = None):
        results = list()
        if not enable:
            return {"ui": {"images": results}}

        b = uuid.uuid4()

        date_str = datetime.datetime.now().strftime("%Y%m%d")
        sub_folder = f"{date_str}/"
        full_output_folder = f"{self.output_dir}/{sub_folder}"
        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder)

        filename = f"hf_{b}"

        counter = 1
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            file = f"{filename}_{counter:05}.png"
            full_file_path = os.path.join(full_output_folder, file)
            img.save(full_file_path, pnginfo=metadata, compress_level=0)
            results.append({
                "filename": file,
                "subfolder": sub_folder,
                "type": self.type,
            })
            counter += 1

        if notify is not None:
            notify.notify(images)

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "HfResizeImage": HfResizeImage,
    "HfInitImageWithMaxSize": HfInitImageWithMaxSize,
    "HfSaveImage": HfSaveImage,
    "HfImageToRGB": HfImageToRGB,
    "HfImageToRGBA": HfImageToRGBA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HfResizeImage": "Image Resize",
    "HfInitImageWithMaxSize": "Init Image to limited Size",
    "HfSaveImage": "Save Image",
    "HfImageToRGB": "Convert Image to RGB",
    "HfImageToRGBA": "Convert Image to RGBA"
}
