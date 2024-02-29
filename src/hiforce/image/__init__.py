import os
import random
from abc import abstractmethod

import folder_paths
import numpy as np
import requests
import torch
from PIL import Image
from torch import Tensor


class ImageCropper:
    def __init__(self):
        pass

    @abstractmethod
    def crop(self, image: Image) -> Image:
        pass

    def crop_tensor_images(self, images):
        out = None
        for image in images:
            ptl_image = tensor_image_to_ptl_image(image)
            img = self.crop(ptl_image)
            np_image_array = convert_image_to_tensor_array(img)

            if out is None:
                out = np_image_array
            else:
                out = torch.cat((out, np_image_array), dim=0)
        return out


class ImageExpansionSquareCropper(ImageCropper):
    def __init__(self, expansion_multiple=1.5, size=1024, align="bottom"):
        super().__init__()
        self.enable = True
        self.expansion_multiple = expansion_multiple
        self.size = size
        self.align = align

    def crop(self, ptl_image) -> Image:
        if not self.enable:
            return ptl_image

        width, height = ptl_image.size
        new_base_image = Image.new(
            'RGB',
            size=(int(width * self.expansion_multiple), int(height * self.expansion_multiple)),
            color=(255, 255, 255))

        ptl_image = ptl_image.convert("RGBA")

        adjust_x = int((new_base_image.width - ptl_image.width) / 2)
        if self.align == "center":
            adjust_y = int((new_base_image.height - ptl_image.height) / 2)
        elif self.align == "bottom":
            adjust_y = int(new_base_image.height - ptl_image.height)
        else:
            adjust_y = 0
        new_base_image.paste(ptl_image, (adjust_x, adjust_y), mask=ptl_image)

        new_height = self.size
        new_width = int((new_height / height) * width)
        new_base_image = new_base_image.resize((new_width, new_height), Image.ANTIALIAS)
        new_base_image = new_base_image.convert("RGBA")

        adjust_x = int((self.size - new_base_image.width) / 2)
        img = Image.new('RGB', size=(self.size, self.size), color=(255, 255, 255))
        img.paste(new_base_image, (adjust_x, 0), mask=new_base_image)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img


def tensor_image_to_ptl_image(image: Tensor):
    i = 255. * image.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    return img


def resize_the_image(image: Image, width: int, height: int):
    resize = (width, height)
    image = image.resize(resize, Image.ANTIALIAS)
    return image


def process_resize_image(images, width, height):
    image_size = images.size()
    image_width = int(image_size[2])
    image_height = int(image_size[1])

    out = None

    need_cop = width % 8 != 0 or height % 8 != 0
    if image_width == width and image_height == height and not need_cop:
        return images

    new_width = int(width - width % 8)
    new_height = int(height - height % 8)

    for image in images:
        origin = tensor_image_to_ptl_image(image)
        ptl_img = resize_the_image(origin, int(new_width), int(new_height))
        if need_cop:
            if ptl_img.mode != 'RGBA':
                ptl_img = ptl_img.convert('RGBA')
            img = Image.new("RGB", (new_width, new_height))
            img.paste(ptl_img, (0, 0), mask=ptl_img)

        new_img_out = convert_image_to_tensor_array(ptl_img)
        if out is None:
            out = new_img_out
        else:
            out = torch.cat((out, new_img_out), dim=0)
    return out


def convert_image_to_tensor_array(image: Image):
    the_image = np.array(image).astype(np.float32) / 255.0
    the_image = torch.from_numpy(the_image)[None,]
    return the_image


def tensor2rgb(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if len(size) < 4:
        return t.unsqueeze(3).repeat(1, 1, 1, 3)
    if size[3] == 1:
        return t.repeat(1, 1, 1, 3)
    elif size[3] == 4:
        return t[:, :, :, :3]
    else:
        return t


def tensor2rgba(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if len(size) < 4:
        return t.unsqueeze(3).repeat(1, 1, 1, 4)
    elif size[3] == 1:
        return t.repeat(1, 1, 1, 4)
    elif size[3] == 3:
        alpha_tensor = torch.ones((size[0], size[1], size[2], 1))
        return torch.cat((t, alpha_tensor), dim=3)
    else:
        return t


def adjust_image_with_max_size(image: Tensor, max_size: int):
    image_size = image.size()
    image_width = int(image_size[2])
    image_height = int(image_size[1])

    if max_size % 8 != 0:
        max_size = max_size - (max_size % 8)

    if image_width > image_height:
        if image_width > max_size:
            width = max_size
            height = image_height * max_size / image_width
            return process_resize_image(image, width, height)
        return image
    elif image_height > image_width:
        if image_height > max_size:
            height = max_size
            width = image_width * max_size / image_height
            return process_resize_image(image, width, height)
        return image
    if image_width > max_size:
        return process_resize_image(image, max_size, max_size)
    return process_resize_image(image, image_width, image_height)


def get_image_from_url(url: str) -> Image:
    temp_path = folder_paths.get_temp_directory()
    os.makedirs(temp_path, exist_ok=True)
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        temp_str = ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        temp_file = f"hf_temp_{temp_str}.png"
        local_file = f"{temp_path}/{temp_file}"
        open(local_file, "wb").write(r.content)
        i = Image.open(local_file)
        del r
        return i
    del r
    return None


def convert_ptl_image_array_to_np_array(images):
    torch_images = None
    for image in images:
        if torch_images is None:
            torch_images = convert_image_to_tensor_array(image)
        else:
            the_images = convert_image_to_tensor_array(image)
            torch_images = torch.cat((torch_images, the_images), dim=0)
    return torch_images
