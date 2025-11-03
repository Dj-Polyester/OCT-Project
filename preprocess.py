#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
from typing import Mapping, Union, Iterable
import random
from PIL import Image
import torch
from torchvision.io import read_image, ImageReadMode
from utils import SOURCE_DIR, TARGET_DIR_RESIZED, TARGET_DIR_PREPROCESSED, CLASSES_TXT_FILE

def print_progress(count, total, end = ""):
	_end = f" {end}" if isinstance(end, str) else ""
	pct = (count / total * 100) if total else 0
	bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
	print(f'\r[{bar}] {pct:.1f}% ({count}/{total})', end=_end, flush=True)

def copy_folder(src, dst):
	src_path = Path(src)
	dst_path = Path(dst)
	# Get total image count
	total = sum(1 for _ in src_path.rglob('*') if not _.is_dir())
	# Copy folder structure and process images
	count = 0
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
	print('\n✓ Done!')

def copy_folder_resize(src, dst, width=128, height=128):
	src_path = Path(src)
	dst_path = Path(dst)
	# Get total image count
	img_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
	total = sum(1 for _ in src_path.rglob('*') if _.suffix.lower() in img_exts)
	# Copy folder structure and process images
	img_count = 0
	for item in src_path.rglob('*'):
		rel_path = item.relative_to(src_path)
		dest_item = dst_path / rel_path

		if item.is_dir():
			dest_item.mkdir(parents=True, exist_ok=True)
		elif item.suffix.lower() in img_exts:
			img_count += 1
			dest_item.parent.mkdir(parents=True, exist_ok=True)
			try:
				img = Image.open(item)
				resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
				resized_img.save(dest_item, quality=95)
			except Exception as e:
				print(f"\nError processing {item}: {e} (falling back to plain copy)")
				shutil.copy2(item, dest_item)
			print_progress(img_count, total)
		else:
			dest_item.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(item, dest_item)

	print('\n✓ Done!')

def check_equal(src, dst, width=128, height=128):
	src_path = Path(src)
	dst_path = Path(dst)
	# Get total image count
	img_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
	total = sum(1 for _ in src_path.rglob('*') if _.suffix.lower() in img_exts)
	# Copy folder structure and process images
	equal_count = 0
	img_count = 0
	for src_item, dst_item in zip(src_path.rglob('*'), dst_path.rglob('*')):
		rel_src_path = src_item.relative_to(src_path)
		rel_dst_path = dst_item.relative_to(dst_path)

		if src_item.suffix.lower() in img_exts:
			img_count += 1
			if rel_src_path == rel_dst_path:
				equal_count += 1
			print_progress(img_count, total, end=f"({equal_count}/{total} equal)")
	print('\n✓ Done!')

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
	for class_path, num_files2del in num_files2del_per_class.items():
		files2del = random.sample(list(class_path.iterdir()), k=num_files2del)
		#print(num_files2del, files2del)
		for file in files2del:
			instance_count += 1
			file.unlink()
			print_progress(instance_count, total)
	print('\n✓ Done!')
		
def flatten_directory(directory):
	dir_path = Path(directory)
	# Get statistics
	_, total = get_statistics(dir_path)
	# Copy folder structure and process images
	instance_count = 0
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
	print('\n✓ Done!')

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
	print('\n✓ Done!')

def combined_mean(x1,n1,x2,n2):
	return (n1*x1 + n2*x2) / (n1+n2)

def combined_var(v1,x1,n1,v2,x2,n2,x, c=1):
	"""c is the correction term in the variance"""
	return ((n1-c)*v1 + (n2-c)*v2 + n1*(x1-x)**2 + n2*(x2-x)**2) / (n1+n2-c)


def get_normal_statistics_whole(directory, until = None):
	dir_path = Path(directory)
	total = len(list(dir_path.iterdir()))
	all_instances = None
	for instance_count, path in enumerate(dir_path.iterdir(),1):
		if isinstance(until, int) and instance_count == until:
			break
		instance = read_image(path, mode=ImageReadMode.GRAY).type(torch.float32)
		all_instances = instance if all_instances == None else torch.cat((all_instances, instance))
		print_progress(instance_count, total)
	return all_instances.mean(), all_instances.std()

def get_normal_statistics(directory, until = None):
	dir_path = Path(directory)
	mean = 0
	var = 0
	numelems = 0
	total = len(list(dir_path.iterdir()))
	for instance_count, path in enumerate(dir_path.iterdir(),1):
		if isinstance(until, int) and instance_count == until:
			break
		instance = read_image(path, mode=ImageReadMode.GRAY).type(torch.float32)
		instance_numelems = instance.numel()
		# Mean
		instance_mean = instance.mean()
		_combined_mean = combined_mean(
			mean, numelems, 
			instance_mean, instance_numelems
		)
		# var
		instance_var = instance.var()
		_combined_var = combined_var(
			var, mean, numelems,
			instance_var, instance_mean, instance_numelems,
			_combined_mean,
		)
		numelems += instance_numelems
		mean = _combined_mean
		var = _combined_var
		print_progress(instance_count, total)
	return mean, var.sqrt()

if __name__ == '__main__':
	# print("Copying resized files...")
	# copy_folder_resize(SOURCE_DIR, TARGET_DIR_RESIZED)
	# print("Checking for equality...")
	# check_equal(SOURCE_DIR, TARGET_DIR_RESIZED)
	
	# print("Backing up resized files...")
	# copy_folder(TARGET_DIR_RESIZED, TARGET_DIR_PREPROCESSED)
	# print("Checking for equality...")
	# check_equal(TARGET_DIR_RESIZED, TARGET_DIR_PREPROCESSED)
	train_dir = TARGET_DIR_PREPROCESSED / Path("train")
	test_dir = TARGET_DIR_PREPROCESSED / Path("test")
	# print(f"Saving classes to {str(TARGET_DIR_PREPROCESSED / Path(CLASSES_TXT_FILE))}...")
	# save_classes(test_dir)
	# print("Undersampling...")
	# undersample(train_dir)
	# print("Flattening...")
	# flatten_directory(train_dir)
	# flatten_directory(test_dir)
	# print("Dividing...")
	# divide_into_subsets(train_dir, ["validation"], [.2])
	print("Obtaining normal statistics...")
	mean, std = get_normal_statistics(train_dir)
	print(mean, std)