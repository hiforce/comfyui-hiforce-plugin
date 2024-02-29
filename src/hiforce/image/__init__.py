import numpy as np
import torch
from PIL import Image
from torch import Tensor


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
    print(f"width:{new_width}, height: {new_height}")
    print(f"width mod:{width % 8}, height: {new_height}")

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
