import itertools, random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Mapping, Callable, Optional, Iterable
from collections import defaultdict

from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray
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

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from core_utils import IMG_EXTS

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
	
class CustomImageDataset(Dataset):
	@classmethod
	def from_path(cls, root: str, mode: ImageReadMode = ImageReadMode.RGB):
		CustomImageDataset.root = Path(root)
		paths = [
			p for p in CustomImageDataset.root.rglob("*") if p.suffix.lower() in IMG_EXTS
		]
		return cls(paths, mode)
	
	@classmethod
	def classes(cls):
		if hasattr(cls, "CLASSES"):
			return cls.CLASSES
		cls.CLASSES = [p.name for p in list(cls.root.iterdir())[0].iterdir()]
		return cls.CLASSES
	
	def __init__(self, paths: list[Path], mode: ImageReadMode = ImageReadMode.RGB):
		self.mode = mode
		self.paths = paths
		self.targets = [self.classes().index(p.parts[-2]) for p in self.paths]
		
		self.indices_by_cls = defaultdict(list)
		for i, p in enumerate(self.paths):
			cls_name = p.parent.name
			self.indices_by_cls[cls_name].append(i)

	def path_instance_pair(self, index: int):
		path: Path = self.paths[index]
		instance = read_image(path, mode=self.mode) 
		return path, instance
	
	def __len__(self):
		return len(self.paths)
	
	def __getitem__(self, index: int):
		_, instance = self.path_instance_pair(index)
		label = self.targets[index]
		return instance, label

	def sample_from_class(self, cls_name: str, n: int):
		return random.sample(self.indices_by_cls[cls_name], n)

class OCTDataset(CustomImageDataset):
	pass

class IterableOCTDataset(Iterable):
	def __init__(
			self, 
			dataset: OCTDataset,
			until = None, 
			transforms = v2.ToDtype(torch.float32),
		):
		self.dataset = dataset
		self.until = until
		self.transforms = transforms
	def __iter__(self):
		for instance_count, (instance, _) in enumerate(self.dataset,1):
			if isinstance(self.until, int) and instance_count == self.until:
				break
			instance = self.transforms(instance)
			yield instance
	def __len__(self):
		max_length = len(self.dataset)
		if isinstance(self.until, int):
			return min(self.until, max_length)
		return max_length
	
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

		"mobilenet_v2":TransferLearningHead(
			"classifier", 
			lambda c: nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, c))
		),
		"mobilenet_v3_small":TransferLearningHead(
			"classifier", 
			lambda c: nn.Sequential(
  			  nn.Linear(576, 1024),
  			  nn.Hardswish(),
  			  nn.Dropout(p=0.2, inplace=True),
  			  nn.Linear(1024, c),
  			)
		),
		"mobilenet_v3_large":TransferLearningHead(
			"classifier", 
			lambda c: nn.Sequential(
  			  nn.Linear(960, 1280, bias=True),
  			  nn.Hardswish(),
  			  nn.Dropout(p=0.2, inplace=True),
  			  nn.Linear(1280, c, bias=True),
  			)
		),
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
		new_head = None
		if isinstance(tlhead.input_size, int):
			new_head = nn.Linear(tlhead.input_size, num_classes)
		elif isinstance(tlhead.input_size, Callable):
			new_head = tlhead.input_size(num_classes)
		else:
			raise TypeError(f"tlhead has invalid type {type(tlhead.input_size)}")

		setattr(model, tlhead.label, new_head)
	
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

# LightningModule for Training
class LitSupervised(L.LightningModule):
	METRICS = (
		"accuracy",
		"precision",
		"recall",
		"f1_score",
	)
	SET_NAME_TRAIN = "train" 
	SET_NAME_VAL = "val" 
	TRANSFORMS = v2.Compose([
		v2.ToDtype(torch.float32),
		#v2.Normalize(mean=[49.4128, 49.4128, 49.4128], std=[57.3348, 57.3348, 57.3348]), #OCTDataMendeley 1000
		#v2.Normalize(mean=[49.0308, 49.0308, 49.0308], std=[55.3943, 55.3943, 55.3943]), #OCTDataMendeley (Undersampled)
		#v2.Normalize(mean=[54.0711, 54.0711, 54.0711], std=[45.5358, 45.5358, 45.5357]), #OCTData8C
		#v2.Normalize(mean=[82.0378, 82.0378, 82.0378], std=[79.9393, 79.9393, 79.9393]), #obulisainaren mv to middle

		#v2.Normalize(mean=[48.6115, 48.6115, 48.6115], std=[56.3848, 56.3848, 56.3848], #OCTDataMendeley
		#v2.Normalize(mean=[53.4781, 53.4781, 53.4781], std=[47.4589, 47.4589, 47.4589], #OCTData8C
		v2.ToDtype(torch.float32, scale=True),
	])
	def __init__(self, config, classes=None):
		super().__init__()
		runname = config["run"]
		model_name = config["model"]

		tlmodel = TransferLearning.models[model_name]
		self.model = tlmodel.model
		# Replace head with a single layer MLP
		TransferLearning.replace_head(model_name, len(OCTDataset.classes()))
		if runname == "step1":
			print("Freezing weights")
			# Step 1: Freeze weights
			TransferLearning.set_grads(model_name)
		elif runname == "step2":
			print("Unfreezing weights")
			# Step 2: Unfreeze weights
			TransferLearning.set_grads(model_name, True)
		self.config = config
		self.classes = classes
	def reset_parameters(self, verbose=False):
		for name, layer in self.model.named_modules():
			if verbose:
				print(name, layer)
			if hasattr(layer, 'reset_parameters'):
				layer.reset_parameters()

	def get_metrics(self, batch, transforms, set_name: str, metric_labels: Iterable[str]):
		instance, target = batch
		instance = transforms(instance)
		preds: Tensor = self.model(instance)

		# Logging to TensorBoard (if installed) by default
		values = {}
		def log_agg(label, avg="micro"):
			new_label = f"{set_name}_{label}_{avg}"
			values[new_label] = getattr(MF, label)(
				preds, 
				target, 
				task="multiclass",
				average=avg, 
				num_classes=preds.size(1)
			)

		def log_per_cls(label):
			scores = getattr(MF, label)(
				preds, 
				target, 
				task="multiclass",
				average="none", 
				num_classes=preds.size(1)
			)
			for cls, score in zip(self.classes, scores):
				new_label = f"{set_name}_{label}_{cls}"
				values[new_label] = score 

		for label in metric_labels:
			log_agg(label)
			log_agg(label, "macro")
			log_per_cls(label)

		new_label = f"{set_name}_loss"
		values[new_label] = F.cross_entropy(preds, target) 
		self.log_dict(
			values, 
			prog_bar=True,
			on_step=False,
			on_epoch=True,
			reduce_fx="mean",
		)
		values["loss"] = values[new_label]
		return values

	@staticmethod
	def metric_names_avg(set_names, avgs):
		return [
			f"{set_name}_{metric}_{avg}" for set_name in set_names for metric in LitSupervised.METRICS for avg in avgs
		]
	
	@staticmethod
	def metric_names_classes(set_names, classes):
		return [
			f"{set_name}_{metric}_{cls}" for set_name in set_names for metric in LitSupervised.METRICS for cls in classes
		]
	
	def metric_names(self):
		return LitSupervised.metric_names_avg(
			[LitSupervised.SET_NAME_TRAIN, LitSupervised.SET_NAME_VAL], 
			["micro", "macro"],
		) + LitSupervised.metric_names_classes(
			[LitSupervised.SET_NAME_TRAIN, LitSupervised.SET_NAME_VAL], 
			self.classes,
		)

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
			LitSupervised.SET_NAME_TRAIN,
			LitSupervised.METRICS,
		)
	
	def validation_step(self, batch, batch_idx):
		return self.get_metrics(
			batch, 
			LitSupervised.TRANSFORMS,
			LitSupervised.SET_NAME_VAL,
			LitSupervised.METRICS,
		)

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.config["lr"])
		return optimizer