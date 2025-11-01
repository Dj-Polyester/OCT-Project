#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
from PIL import Image
import random
from typing import Mapping, Union, Iterable
from utils import SOURCE_DIR, TARGET_DIR_RESIZED, TARGET_DIR_PREPROCESSED

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

if __name__ == '__main__':
	# print("Copying resized files")
	# copy_folder_resize(SOURCE_DIR, TARGET_DIR_RESIZED)
	# print("Checking for equality")
	# check_equal(SOURCE_DIR, TARGET_DIR_RESIZED)
	
	print("Backing up resized files")
	copy_folder(TARGET_DIR_RESIZED, TARGET_DIR_PREPROCESSED)
	print("Checking for equality")
	check_equal(TARGET_DIR_RESIZED, TARGET_DIR_PREPROCESSED)
	print("Undersampling")
	undersample(TARGET_DIR_PREPROCESSED / Path("train"))
	print("Flattening")
	flatten_directory(TARGET_DIR_PREPROCESSED / Path("train"))
	flatten_directory(TARGET_DIR_PREPROCESSED / Path("test"))
	print("Dividing")
	divide_into_subsets(TARGET_DIR_PREPROCESSED / Path("train"), ["validation"], [.2])