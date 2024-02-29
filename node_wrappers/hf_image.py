import datetime
import os
import uuid

import folder_paths
import numpy as np
import torch
from PIL import Image, ImageOps

from hiforce.image import tensor2rgba, tensor2rgb, process_resize_image, adjust_image_with_max_size, get_image_from_url, \
    convert_ptl_image_array_to_np_array, ImageExpansionSquareCropper, ImageCropper
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


class LoadImageFromURL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "url")
    FUNCTION = "process"

    CATEGORY = "HiFORCE/Image/Save"

    def process(self, url):
        spec_url_list = url.split(",")
        temp_path = folder_paths.get_temp_directory()
        os.makedirs(temp_path, exist_ok=True)

        image_list = []
        for url in spec_url_list:
            i = get_image_from_url(url)
            if i is not None:
                image_list.append(i)
        out = convert_ptl_image_array_to_np_array(image_list)
        return out, url


class HfImageAutoExpansionSquare:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "expansion_multiple": ("FLOAT", {"default": 1.2, "min": 0.01, "max": 100, "step": 0.01}),
                "size": ("INT", {"default": 1024, "min": 512, "max": 1024}),
                "align": (["bottom", "center", "top"],)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "process"

    CATEGORY = "HiFORCE/Image/Zoom"

    def process(self, images, expansion_multiple: float, size: int, align="bottom"):
        copper = ImageExpansionSquareCropper(expansion_multiple, size, align)
        out = copper.crop_tensor_images(images)
        return (out,)


class HfLoadImageWithCropper:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True})
            },
            "optional": {
                "image_copper": ("IMAGE_COPPER",)
            }
        }

    CATEGORY = "HiFORCE/Image/Create"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "load_image"

    def load_image(self, image, image_copper: ImageCropper = None):

        image_list = []

        image_path = folder_paths.get_annotated_filepath(image)
        if not os.path.exists(image_path):
            i = Image.new("RGB", (512, 512))
        else:
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)

        image = i.convert("RGB")
        image_list.append(image)

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if image_copper is not None:
            out = image_copper.crop_tensor_images((image,))
            return (out,)

        return (image,)


NODE_CLASS_MAPPINGS = {
    "HfResizeImage": HfResizeImage,
    "HfInitImageWithMaxSize": HfInitImageWithMaxSize,
    "HfSaveImage": HfSaveImage,
    "HfImageToRGB": HfImageToRGB,
    "HfImageToRGBA": HfImageToRGBA,
    "LoadImageFromURL": LoadImageFromURL,
    "HfImageAutoExpansionSquare": HfImageAutoExpansionSquare,
    "HfLoadImageWithCropper": HfLoadImageWithCropper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HfResizeImage": "Image Resize",
    "HfInitImageWithMaxSize": "Init Image to limited Size",
    "HfSaveImage": "Save Image",
    "HfImageToRGB": "Convert Image to RGB",
    "HfImageToRGBA": "Convert Image to RGBA",
    "LoadImageFromURL": "Load Images from URL",
    "HfImageAutoExpansionSquare": "Image Enlargement - Zoom and Crop into a Square",
    "HfLoadImageWithCropper": "Load and Crop Image",
}
