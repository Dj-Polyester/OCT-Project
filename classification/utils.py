import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Mapping, Callable, Optional

from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import RandAugment, v2
from torchvision.io import read_image, ImageReadMode
from torchvision.models import WeightsEnum

import lightning as L
import torchmetrics.functional as MF

# Config from paper
GLOBAL_CONFIG = {
	# train data loader
	"batch_size": 80,
	# optimizer
	"optimizer": ["Adam"],
}

LOGDIR = "lightning_logs"
CLASSES_TXT_FILE = "classes.txt"
MAGNITUDE_MAX_RANGE = 10

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
	def __init__(
			self, 
			_type,
			target_dir="OCTData", 
			preprocessed_suffix = "preprocessed",
		):
		target_dir_preprocessed = f'{target_dir}_{preprocessed_suffix}'
		super().__init__(target_dir_preprocessed, _type, ImageReadMode.RGB)

	def lbl2cls1(self, path: Path):
		return path.name.split("-")[0]

	def lbl2cls2(self, path: Path):
		return path.name.split("_")[0].upper()

	def __getitem__(self, index: int):
		path, instance = self.path_instance_pair(index)
		label_str = self.lbl2cls2(path)
		label = self.classes().index(label_str)
		return instance, label

@dataclass
class TransferLearningHead:
	label: str
	input_size: int
	
@dataclass
class TransferLearningModel:
	model_callable: Callable
	weights: Optional[WeightsEnum]
	_model: Optional[nn.Module] = None
	@property
	def model(self):
		if self._model == None:
			self._model = self.model_callable(weights=self.weights)
		return self._model

class TransferLearning:
	models: dict[str, TransferLearningModel]
	BASE_OUTPUT_SINGLE_LAYER: dict[str, TransferLearningHead] = {
		"resnet18": TransferLearningHead("fc", 512), 
		"resnet34": TransferLearningHead("fc", 512), 
		"resnet50": TransferLearningHead("fc", 2048),
		"resnet101": TransferLearningHead("fc", 2048),
		"resnet152": TransferLearningHead("fc", 2048),
		
		"densenet121": TransferLearningHead("classifier", 1024), 
		"densenet161": TransferLearningHead("classifier", 2208), 
		"densenet169": TransferLearningHead("classifier", 1664),
		"densenet201": TransferLearningHead("classifier", 1920),
	}

	@staticmethod
	def model_index(model_name):
		if model_name in TransferLearning.BASE_OUTPUT_SINGLE_LAYER:
			return model_name
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
		model = cls.models[model_name].model
		setattr(model, tlhead.label, nn.Linear(tlhead.input_size, num_classes))
	
	@classmethod
	def set_grads(cls, model_name, fill = None, verbose = False):
		tlhead = TransferLearning.get_tlhead(model_name)
		model = cls.models[model_name].model
		for name, param in model.named_parameters():
			param.requires_grad = fill if isinstance(fill, bool) else name.startswith(tlhead.label)
			if verbose:
				print(name, param.grad, param.requires_grad)



def plotimg(img: Tensor, label: str):
	img = img.permute(1,2,0)
	if img.ndim == 3 and img.shape[-1] == 1:
		img = img.squeeze()

	plt.imshow(img, cmap='gray')
	plt.axis('off')
	plt.title(label)
	plt.show()

def product_dict(
	valid_mapping: Mapping,
	condition: Callable = lambda **_: True,

):
	"""
	Given a `valid_mapping` of key value pairs as (name, iterable_of_valid_values),
	generates a mapping that satisfies `condition` for each iteration as
	(name, possible_value) using cartesian product.

	Usage:
```
for combination in product_dict(valid_mapping, condition):
	# Do sth with the combination
	...
```
	"""
	keys = valid_mapping.keys()
	iter_of_valid_vals = valid_mapping.values()
	for val_comb in itertools.product(*iter_of_valid_vals):
		comb = dict(zip(keys, val_comb))
		if condition(**comb):
			yield comb

def stringify_map(_map: Mapping, delim="_"):
	return delim.join([f"{k}:{v}" for k, v in _map.items()])

# LightningModule for Training
class LitSupervised(L.LightningModule):
	TRAIN_LOSS = "Train Loss"
	VALIDATION_LOSS = "Validation Loss"
	TRAIN_ACC = "Train Accuracy"
	VALIDATION_ACC = "Validation Accuracy"

	TRANSFORMS = v2.Compose([
		v2.ToDtype(torch.float32),
		#v2.Normalize(mean=[49.4128, 49.4128, 49.4128], std=[57.3348, 57.3348, 57.3348]), #Mendeley 1000
		#v2.Normalize(mean=[49.0308, 49.0308, 49.0308], std=[55.3943, 55.3943, 55.3943]), #Mendeley
		#v2.Normalize(mean=[54.0711, 54.0711, 54.0711], std=[45.5358, 45.5358, 45.5357]), #obulisainaren
		v2.Normalize(mean=[82.0378, 82.0378, 82.0378], std=[79.9393, 79.9393, 79.9393]), #obulisainaren mv to middle
		v2.ToDtype(torch.float32, scale=True),
	])
	def __init__(self, config):
		super().__init__()
		runname = config["run"]
		model_name = config["model"]

		tlmodel = TransferLearning.models[model_name]
		self.model = tlmodel.model
		# Replace head with a single layer MLP
		TransferLearning.replace_head(model_name, len(OCTMendeleyDataset.classes()))
		if runname == "step1":
			print("Freezing weights")
			# Step 1: Freeze weights
			TransferLearning.set_grads(model_name)
		elif runname == "step2":
			print("Unfreezing weights")
			# Step 2: Unfreeze weights
			TransferLearning.set_grads(model_name, True)
		self.config = config

	def reset_parameters(self, verbose=False):
		for name, layer in self.model.named_modules():
			if verbose:
				print(name, layer)
			if hasattr(layer, 'reset_parameters'):
				layer.reset_parameters()

	def get_metrics(self, batch, transforms, labels: Mapping[str, str]):
		instance, target = batch
		instance = transforms(instance)
		preds: Tensor = self.model(instance)

		# Logging to TensorBoard (if installed) by default
		values = {}
		if "loss" in labels:
			values[labels["loss"]] = F.cross_entropy(preds, target) 
		if "accuracy" in labels:
			values[labels["accuracy"]] = MF.accuracy(preds, target, task="multiclass", num_classes=preds.size(1))
			
		self.log_dict(
			values, 
			prog_bar=True,
			on_step=False,
			on_epoch=True,
			reduce_fx="mean",
		)
		return {k: values[v] for k,v in labels.items()}

	def training_step(self, batch, batch_idx):
		TRAIN_TRANSFORMS = v2.Compose([
		   	LitSupervised.TRANSFORMS,
			OCTRandAugment(
				magnitude=self.config["magnitude"], 
				num_magnitude_bins=MAGNITUDE_MAX_RANGE
			),
		])
		return self.get_metrics(
			batch, 
			TRAIN_TRANSFORMS,
			{"loss": LitSupervised.TRAIN_LOSS, "accuracy": LitSupervised.TRAIN_ACC},
		)
	
	def validation_step(self, batch, batch_idx):
		return self.get_metrics(
			batch, 
			LitSupervised.TRANSFORMS,
			{"loss": LitSupervised.VALIDATION_LOSS, "accuracy": LitSupervised.VALIDATION_ACC},
		)

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.config["lr"])
		return optimizer