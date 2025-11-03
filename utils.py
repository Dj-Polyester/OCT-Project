from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

from matplotlib import pyplot as plt

import torch
from torch import nn, Tensor
from torch import optim
from torch.utils.data import Dataset
from torchvision.transforms import RandAugment
from torchvision.io import read_image, ImageReadMode

import lightning as L

SOURCE_DIR = "CellData/OCT"
TARGET_DIR_RESIZED = "CellData_resized"
TARGET_DIR_PREPROCESSED = "CellData_preprocessed"
CLASSES_TXT_FILE = "classes.txt"

class OCTRandAugment(RandAugment):
	def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
		return {
			# op_name: (magnitudes, signed)
			"Identity": (torch.tensor(0.0), False),
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
		}
	
class CustomDataset(Dataset):
	@classmethod
	def classes(cls):
		if hasattr(cls, "CLASSES"):
			return cls.CLASSES
		root = cls.root
		classes_txt_file = root / Path("classes.txt")
		cls.CLASSES = None
		if classes_txt_file.exists():
			with open(classes_txt_file, "r") as classes_f:
				cls.CLASSES = [line.rstrip("\n") for line in classes_f.readlines()]
		else:
			cls.CLASSES = [clss.name for clss in (root / Path("train")).iterdir()]
		return cls.CLASSES
	
	def __init__(self, root:str, _type: str):
		CustomDataset.root = root
		self.type = _type
		self.path = root / Path(_type)
		self.items = list(self.path.iterdir())

	def __len__(self):
		return len(self.items)

class CustomImageDataset(CustomDataset):
	def __init__(self, root: str, _type: str, mode):
		super().__init__(root, _type)
		self.mode = mode

	def path_instance_pair(self, index: int):
		path: Path = self.items[index]
		instance = read_image(path, mode=self.mode) 
		return path, instance

class OCTMendeleyDataset(CustomImageDataset):
	def __init__(self, _type):
		super().__init__(TARGET_DIR_PREPROCESSED, _type, ImageReadMode.RGB)

	def __getitem__(self, index: int):
		path, instance = self.path_instance_pair(index)
		label_str = path.name.split("-")[0]
		label = self.classes().index(label_str)
		return instance, label

class TransferLearning:
	@dataclass
	class TransferLearningHead:
		label: str
		input_size: int
		
	BASE_OUTPUT_SINGLE_LAYER: dict[str, TransferLearningHead] = {
		"resnet": TransferLearningHead("fc", 2048),
		"densenet": TransferLearningHead("classifier", 1664),
	}

	@staticmethod
	def model_index(model_name):
		if model_name[:6] in TransferLearning.BASE_OUTPUT_SINGLE_LAYER:
			return model_name[:6]
		elif model_name[:8] in TransferLearning.BASE_OUTPUT_SINGLE_LAYER:
			return model_name[:8]
		raise ValueError(f"Invalid model name {model_name}")
	
	@classmethod
	def _get_model_and_process(cls, model_name, callback):
		_model_index = TransferLearning.model_index(model_name)
		model = cls.models[model_name]
		if _model_index != None:
			callback(model, _model_index)
		else: 
			raise ValueError(f"Invalid model name {model_name}")

	@staticmethod
	def get_tlhead(model_name):
		_model_index = TransferLearning.model_index(model_name)
		return TransferLearning.BASE_OUTPUT_SINGLE_LAYER[_model_index]
	
	@classmethod
	def replace_head(cls, model_name, num_classes):
		tlhead = TransferLearning.get_tlhead(model_name)
		model = cls.models[model_name]
		setattr(model, tlhead.label, nn.Linear(tlhead.input_size, num_classes))
	
	@classmethod
	def set_grads(cls, model_name, fill = None):
		tlhead = TransferLearning.get_tlhead(model_name)
		model: nn.Module = cls.models[model_name]
		for name, param in model.named_parameters():
			param.requires_grad = fill if isinstance(fill, bool) else name.startswith(tlhead.label)
			print(name, param.grad, param.requires_grad)



def plotimg(img: Tensor, label: str):
	img = img.permute(1,2,0)
	if img.ndim == 3 and img.shape[-1] == 1:
		img = img.squeeze()

	plt.imshow(img, cmap='gray')
	plt.axis('off')
	plt.title(label)
	plt.show()

# LightningModule for Training
class LitOCTTL(L.LightningModule):
	def __init__(self, model):
		super().__init__()
		self.model = model

	def training_step(self, batch, batch_idx):
		# training_step defines the train loop.
		# it is independent of forward
		x, _ = batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)
		loss = nn.functional.mse_loss(x_hat, x)
		# Logging to TensorBoard (if installed) by default
		self.log("train_loss", loss)
		return loss

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=1e-3)
		return optimizer