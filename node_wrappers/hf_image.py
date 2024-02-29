from hiforce.image import tensor2rgba, tensor2rgb, process_resize_image, adjust_image_with_max_size


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


NODE_CLASS_MAPPINGS = {
    "HfResizeImage": HfResizeImage,
    "HfInitImageWithMaxSize": HfInitImageWithMaxSize,
    "HfImageToRGB": HfImageToRGB,
    "HfImageToRGBA": HfImageToRGBA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HfResizeImage": "Image Resize",
    "HfInitImageWithMaxSize": "Init Image to limited Size",
    "HfImageToRGB": "Convert Image to RGB",
    "HfImageToRGBA": "Convert Image to RGBA"
}
