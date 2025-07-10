import torch
import numpy as np
import blobfile as bf
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

def normalize(image, shape=(256, 256)):
    """
    Given an PIL image, resize it and normalize each pixel into [-1, 1].
    Args:
        image: image to be normalized, PIL.Image
        shape: the desired shape of the image

    Returns: the normalized image

    """
    image = np.array(image.convert("RGB").resize(shape)) # 转变为numpy数组,shape:(H,W,C)
    image = image.astype(np.float32) / 255.0 # 重置数据类型到float32，归一化到[0.0,1.0]
    image = image[None].transpose(0, 3, 1, 2) # image[None]在数组的最前面添加一个新的维度（即“批量维度”）
    image = torch.from_numpy(image) # 从numpy数组转换为torch张量
    image = image * 2.0 - 1.0 # 归一化到[-1.0,1.0]
    return image


# Copied from Repaint code
def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def save_grid(tensor_img, path, nrow=5):
    """
    tensor_img: [B, 3, H, W] or [tensor(3, H, W)]
    """
    if isinstance(tensor_img, list):
        tensor_img = torch.stack(tensor_img)
    assert len(tensor_img.shape) == 4
    tensor_img = tensor_img.clamp(min=0.0, max=1.0)
    grid = make_grid(tensor_img, nrow=nrow)
    pil = ToPILImage()(grid)
    pil.save(path)
