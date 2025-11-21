#!/usr/bin/env python3
import time
from collections import defaultdict
import shutil
from pathlib import Path
from typing import Mapping, Union, Iterable
import random

import numpy as np
import cv2 as cv

import torch
from torch import Tensor
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
import kagglehub
from utils import CLASSES_TXT_FILE

class Timer:
	def __init__(self):
		self.start = time.perf_counter()
	def print(self):
		self.duration = time.perf_counter() - self.start
		print(f'\n✓ Done in {self.duration} secs!')

def print_progress(count, total, end = ""):
	_end = f" {end}" if isinstance(end, str) else ""
	pct = (count / total * 100) if total else 0
	bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
	print(f'\r[{bar}] {pct:.1f}% ({count}/{total})', end=_end, flush=True)

def download_dataset(dataset_name: str):
	print(f"Downloading dataset {dataset_name} from Kaggle...")
	path = kagglehub.dataset_download(dataset_name)
	print("Path to dataset files:", path)
	return path

def copy_folder(src, dst):
	src_path = Path(src)
	dst_path = Path(dst)
	# Get total image count
	total = sum(1 for _ in src_path.rglob('*') if not _.is_dir())
	# Copy folder structure and process images
	count = 0
	timer = Timer()
	for item in src_path.rglob('*'):
		rel_path = item.relative_to(src_path)
		dest_item = dst_path / rel_path
		if item.is_dir():
			dest_item.mkdir(parents=True, exist_ok=True)
		else:
			count += 1
			dest_item.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(item, dest_item)
			print_progress(count, total)
	timer.print()

def cvreadimg(path: Path):
	img = cv.imread(str(path))
	assert img is not None, "file could not be read, check with os.path.exists()"
	return img

def resize_image(path: Path, width=128, height=128):
	img = cvreadimg(path)
	resized_img = cv.resize(img, (width, height), interpolation=cv.INTER_LANCZOS4)
	return resized_img

def copy_folder_process(src, dst, callback, **kwargs):
	src_path = Path(src)
	dst_path = Path(dst)
	# Get total image count
	img_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
	total = sum(1 for _ in src_path.rglob('*') if _.suffix.lower() in img_exts)
	# Copy folder structure and process images
	img_count = 0
	timer = Timer()
	for item in src_path.rglob('*'):
		rel_path = item.relative_to(src_path)
		dest_item = dst_path / rel_path

		if item.is_dir():
			dest_item.mkdir(parents=True, exist_ok=True)
		elif item.suffix.lower() in img_exts:
			img_count += 1
			dest_item.parent.mkdir(parents=True, exist_ok=True)
			try:
				processed_img = callback(item, **kwargs)
				cv.imwrite(str(dest_item), processed_img, [int(cv.IMWRITE_JPEG_QUALITY), 95])
			except Exception as e:
				print(f"\nError processing {item}: {e} (falling back to plain copy)")
				shutil.copy2(item, dest_item)
			print_progress(img_count, total)
		else:
			dest_item.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(item, dest_item)
	timer.print()

def check_equal(src, dst, width=128, height=128):
	src_path = Path(src)
	dst_path = Path(dst)
	# Get total image count
	img_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
	total = sum(1 for _ in src_path.rglob('*') if _.suffix.lower() in img_exts)
	# Copy folder structure and process images
	equal_count = 0
	img_count = 0
	timer = Timer()
	for src_item, dst_item in zip(src_path.rglob('*'), dst_path.rglob('*')):
		rel_src_path = src_item.relative_to(src_path)
		rel_dst_path = dst_item.relative_to(dst_path)

		if src_item.suffix.lower() in img_exts:
			img_count += 1
			if rel_src_path == rel_dst_path:
				equal_count += 1
			print_progress(img_count, total, end=f"({equal_count}/{total} equal)")
	timer.print()

class ClassPopulation(dict):
	def __init__(self, mapping: Mapping[Path, int]):
		super().__init__(mapping) 
	def __repr__(self):
		return str({k.name: v for k, v in self.items()})
	
def get_class_dist(dir_path: Path):
	class_paths = list(dir_path.iterdir())
	class_dist = ClassPopulation(
		{
			class_path: len(list(class_path.iterdir())) 
			for class_path in class_paths
		}
	)
	print(f"Number of files in {dir_path.name} set per class:")
	return class_dist

def get_statistics(dir_path: Path):
	class_dist = get_class_dist(dir_path)
	return class_dist, sum(class_dist.values())

def get_undersample_statistics(dir_path: Path):
	class_dist = get_class_dist(dir_path)
	num_files2del_per_class = ClassPopulation({k: v - min(class_dist.values()) for k, v in class_dist.items()})
	print(f"Number of files to delete in {dir_path.name} set per class:")
	print(num_files2del_per_class)
	return num_files2del_per_class, sum(num_files2del_per_class.values())

def undersample(directory):
	dir_path = Path(directory)
	# Get statistics
	num_files2del_per_class, total = get_undersample_statistics(dir_path)
	# Copy folder structure and process images
	instance_count = 0
	timer = Timer()
	for class_path, num_files2del in num_files2del_per_class.items():
		files2del = random.sample(list(class_path.iterdir()), k=num_files2del)
		#print(num_files2del, files2del)
		for file in files2del:
			instance_count += 1
			file.unlink()
			print_progress(instance_count, total)
	timer.print()
		
def flatten_directory(directory):
	dir_path = Path(directory)
	# Get statistics
	_, total = get_statistics(dir_path)
	# Copy folder structure and process images
	instance_count = 0
	timer = Timer()
	for file in dir_path.rglob('*'):
		if file.is_file():
			dst = dir_path / file.name
			if file != dst:
				instance_count += 1
				file.rename(dst)
				print_progress(instance_count, total)
	
	# Remove empty subdirectories
	for directory in sorted(dir_path.rglob('*'), reverse=True):
		if directory.is_dir() and not list(directory.iterdir()):
			directory.rmdir()
	timer.print()

def save_classes(directory):
	dir_path = Path(directory)
	parent_path = dir_path.parent
	with open(parent_path / CLASSES_TXT_FILE, "w") as classes_f:
		classes_f.writelines("\n".join([class_path.name for class_path in dir_path.iterdir()]))
	
def divide_into_subsets(directory, names: Iterable[str], ps: Iterable[float]):
	dir_path = Path(directory)
	parent_path = dir_path.parent
	for name in names:
		if (parent_path / name).exists():
			raise FileExistsError(f"File {name} exists in directory {parent_path}")
	# Copy folder structure and process images
	instance_count = 0
	instances = list(dir_path.iterdir())
	total = len(instances)
	timer = Timer()
	for name, p in zip(names, ps):
		ptotal = int(p*total)
		files2mv = random.sample(instances, k=ptotal)
		for file in files2mv:
			if file.is_file():
				dst = parent_path / name / file.name
				if file != dst:
					instance_count += 1
					dst.parent.mkdir(parents=True, exist_ok=True)
					file.rename(dst)
					print_progress(instance_count, ptotal)
	if dir_path.is_dir() and not list(dir_path.iterdir()):
		dir_path.rmdir()
	timer.print()

def combined_mean(x1,n1,x2,n2):
	return (n1*x1 + n2*x2) / (n1+n2)

def combined_var(v1,x1,n1,v2,x2,n2,x, c=1):
	"""c is the correction term in the variance"""
	return ((n1-c)*v1 + (n2-c)*v2 + n1*(x1-x)**2 + n2*(x2-x)**2) / (n1+n2-c)

def get_normal_statistics_whole(
		directory, 
		dim = (0,2,3),
		mode = ImageReadMode.RGB,
		until = None, 
		transforms = v2.ToDtype(torch.float32),
	):
	dir_path = Path(directory)
	total = len(list(dir_path.iterdir()))
	all_instances = None
	timer = Timer()
	for instance_count, path in enumerate(dir_path.iterdir(),1):
		if isinstance(until, int) and instance_count == until:
			break
		instance: Tensor = transforms(read_image(path, mode=mode))
		all_instances = instance.unsqueeze(0) if all_instances == None else torch.cat((all_instances, instance.unsqueeze(0)))
		print_progress(instance_count, total)
	timer.print()
	return all_instances.mean(dim=dim), all_instances.std(dim=dim)

def get_normal_statistics(
		directory, 
		dim = (1,2),
		mode = ImageReadMode.RGB,
		until = None, 
		transforms = v2.ToDtype(torch.float32),
	):
	dir_path = Path(directory)
	mean = 0
	var = 0
	numelems = 0
	total = len(list(dir_path.iterdir()))
	timer = Timer()
	for instance_count, path in enumerate(dir_path.iterdir(),1):
		if isinstance(until, int) and instance_count == until:
			break
		instance: Tensor = transforms(read_image(path, mode=mode))
		instance_numelems = instance.numel()
		# Mean
		instance_mean = instance.mean(dim=dim)
		_combined_mean = combined_mean(
			mean, numelems, 
			instance_mean, instance_numelems
		)
		# var
		instance_var = instance.var(dim=dim)
		_combined_var = combined_var(
			var, mean, numelems,
			instance_var, instance_mean, instance_numelems,
			_combined_mean,
		)
		numelems += instance_numelems
		mean = _combined_mean
		var = _combined_var
		print_progress(instance_count, total)
	timer.print()
	return mean, var.sqrt()

def get_class_index(literal: str, classes):
	for c in classes:
		if literal.startswith(c):
			return c
	return None

def get_class_dist_flattened(dir_path: Path):
	classes = None
	parent_path = dir_path.parent
	with open(parent_path / CLASSES_TXT_FILE, "r") as fclasses:
		classes = [c.rstrip("\n") for c in fclasses.readlines()]
	classdict = {c:0 for c in classes}
	total = len(list(dir_path.iterdir()))
	timer = Timer()
	for count, path in enumerate(dir_path.iterdir(), 1):
		class_index = get_class_index(path.name, classes)
		if class_index != None:
			classdict[class_index] += 1
		print_progress(count, total)
	timer.print()
	return classdict

# MOVE TO MIDDLE

def fill_corners(gray):
	h, w = gray.shape
	mask = np.zeros((h+2, w+2), dtype=np.uint8)
	corners = [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)]
	for corner_seed in corners:
		# Low and high brightness tolerance (e.g., 5-unit tolerance in each channel)
		lo_diff = (0, 0, 0)
		up_diff = (0, 0, 0)

		# Perform the flood fill
		# The function modifies the 'img' array in-place
		black_color = (0,0,0)
		cv.floodFill(
			gray, 
			mask, 
			corner_seed, 
			black_color, 
			loDiff=lo_diff, 
			upDiff=up_diff
		)
	return gray

def binary_cluster_calc_mean(img):
	y, x = np.where(img == 255)
	mean_coo = (int(np.mean(x)), int(np.mean(y)))

	h, w = img.shape
	middle_coo = (w/2, h/2)
	return mean_coo, middle_coo

def simple_segmentation(img, cls):
	corner_filled = fill_corners(img)

	# start segmentation
	threshold_type = cv.THRESH_BINARY
	threshold_type += cv.THRESH_OTSU

	_, thresh = cv.threshold(corner_filled,60,255,threshold_type)

	denoised = cv.medianBlur(thresh, 5)

	return thresh, denoised


def move2middle(img, segment): 
	mean_coo, middle_coo = binary_cluster_calc_mean(segment)
	translation_vector = (int(middle_coo[0] - mean_coo[0]), int(middle_coo[1] - mean_coo[1]))

	translation_matrix = np.float32([
		[1, 0, translation_vector[0]],
		[0, 1, translation_vector[1]]
	])

	h, w, _ = img.shape
	white_color = (255, 255, 255)
	translated_img = cv.warpAffine(
		img, 
		translation_matrix, 
		(w, h),
		borderMode=cv.BORDER_CONSTANT, # Tells OpenCV to use a constant color
		borderValue=white_color          # Specifies the constant color (White)
	)
	return translated_img, mean_coo, middle_coo

def mvimg2middle(path: Path):
	img = cvreadimg(path)
	cls = path.parent.name.split("_")[0]
	gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

	gray_contrasted = cv.convertScaleAbs(gray, alpha=1.5, beta=0)

	thresh, denoised = simple_segmentation(gray_contrasted, cls)
	translated_img, mean_coo, middle_coo = move2middle(img, denoised)
	return translated_img

def preprocess(
	src_dir, 
	target_dir=None, 
	preprocessed_suffix = "preprocessed",
	resize_flag=False, 
	undersample_flag=False,
	mv2middle_flag = False,
):

	src_path = Path(src_dir).resolve()

	if target_dir is None:
		target_dir = src_dir
	
	target_dir_resized = f'{target_dir}_resized' 
	target_path_resized = Path(target_dir_resized).resolve() 

	if resize_flag:
		print("Copy-resizing...")
		copy_folder_process(src_path, target_path_resized, resize_image)
		print("Checking for equality...")
		check_equal(src_path, target_path_resized)

	target_dir_preprocessed = f'{target_dir}_{preprocessed_suffix}'
	target_path_preprocessed = Path(target_dir_preprocessed).resolve() 

	if mv2middle_flag:
		print("Copy-moving to middle...")
		copy_folder_process(
			target_path_resized, 
			target_path_preprocessed, 
			mvimg2middle
		)
	else:
		print("Backing up resized files...")
		copy_folder(target_path_resized, target_path_preprocessed)
	print("Checking for equality...")
	check_equal(target_path_resized, target_path_preprocessed)

	train_dir = Path(target_path_preprocessed, "train")
	validation_dir = Path(target_path_preprocessed, "val")
	test_dir = Path(target_path_preprocessed, "test")

	if undersample_flag:
		print("Undersampling...")
		undersample(train_dir)

	print(f"Saving classes to {str(Path(target_path_preprocessed, CLASSES_TXT_FILE))}...")
	save_classes(test_dir)

	print("Flattening...")
	flatten_directory(train_dir)
	flatten_directory(validation_dir)
	flatten_directory(test_dir)
	
	mean, std = get_normal_statistics(train_dir)
	print(mean, std)
if __name__ == '__main__':
	preprocess("OCTData", preprocessed_suffix="mv2middle", mv2middle_flag=True)