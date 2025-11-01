from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from torchvision.transforms import RandAugment
from torchvision.io import read_image, ImageReadMode

SOURCE_DIR = "CellData/OCT"
TARGET_DIR_RESIZED = "CellData_resized"
TARGET_DIR_PREPROCESSED = "CellData_preprocessed"

class OCTRandAugment(RandAugment):
    def _augmentation_space(self, num_bins, image_size):
        return {
            # We remove some of the transformations for OCT
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            #"Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            #"Color": (torch.linspace(0.0, 0.9, num_bins), True),
            #"Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            #"Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            #"Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            #"Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            #"AutoContrast": (torch.tensor(0.0), False),
            #"Equalize": (torch.tensor(0.0), False),
            #"Invert": (torch.tensor(0.0), False),
        }

class MendeleyOCTDataset(Dataset):
    @classmethod
    def get_classes(cls):
        if hasattr(cls, "CLASSES"):
            return cls.CLASSES
        cls_tmp = list((TARGET_DIR_RESIZED / Path("train")).iterdir())
        cls.CLASSES = [clss.name for clss in cls_tmp]
        return cls.CLASSES
    
    def __init__(self, _type: str):
        self.type = _type
        self.path = TARGET_DIR_PREPROCESSED / Path(_type)
        self.items = list(self.path.iterdir())

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        path: Path = self.items[index]
        instance = read_image(path, mode=ImageReadMode.GRAY)
        label_str = path.name.split("-")[0]
        label = self.get_classes().index(label_str)
        return instance, label

class TransferLearning:

    @dataclass
    class TransferLearningHead:
        label: str
        input_size: int
        
    BASE_OUTPUT_SINGLE_LAYER: dict[str, TransferLearningHead] = {
        "resnet152": TransferLearningHead("fc", 2048),
        "densenet169": TransferLearningHead("classifier", 1664),
    }

    @staticmethod
    def replace_head(model_name, model, num_classes):
        if model_name in TransferLearning.BASE_OUTPUT_SINGLE_LAYER:
            tlh = TransferLearning.BASE_OUTPUT_SINGLE_LAYER[model_name]
            setattr(model, tlh.label, nn.Linear(tlh.input_size, num_classes))
        else: 
            raise ValueError(f"Invalid model name {model_name}")

def plotimg(img: Tensor, label: str):
    img = img.permute(1,2,0)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze()

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(label)
    plt.show()