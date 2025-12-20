#!/usr/bin/env python3
from pathlib import Path
from typing import Mapping, Union, Iterable
import random
from tqdm import tqdm

from utils import (
	copy_folder, 
	resize_image, 
	copy_folder_process, 
	check_equal, 
	get_normal_statistics
)

from classification.utils import CLASSES_TXT_FILE

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
	with tqdm(
		total=total,
		desc="Undersampling...",
	) as pbar:
		for class_path, num_files2del in num_files2del_per_class.items():
			files2del = random.sample(list(class_path.iterdir()), k=num_files2del)
			#print(num_files2del, files2del)
			for file in tqdm(files2del):
				file.unlink()
				pbar.update()
		
def flatten_directory(directory):
	dir_path = Path(directory)
	# Get statistics
	_, total = get_statistics(dir_path)
	# Copy folder structure and process images
	with tqdm(
		total=total,
		desc="Flattening...",
	) as pbar:
		for file in dir_path.rglob('*'):
			if file.is_file():
				dst = dir_path / file.name
				if file != dst:
					file.rename(dst)
					pbar.update()
	
	# Remove empty subdirectories
	for directory in sorted(dir_path.rglob('*'), reverse=True):
		if directory.is_dir() and not list(directory.iterdir()):
			directory.rmdir()

def save_classes(directory):
	dir_path = Path(directory)
	parent_path = dir_path.parent
	print(f"Saving classes to {str(Path(parent_path, CLASSES_TXT_FILE))}...")
	with open(parent_path / CLASSES_TXT_FILE, "w") as classes_f:
		classes_f.writelines("\n".join([class_path.name for class_path in dir_path.iterdir()]))

def get_class_index(literal: str, classes):
	for c in classes:
		if literal.startswith(c):
			return c
	return None

def preprocess(
	src_dir, 
	target_dir=None, 
	preprocessed_suffix = "preprocessed",
	resize_flag=False, 
	undersample_flag=False,
):

	src_path = Path(src_dir).resolve()

	if target_dir is None:
		target_dir = src_dir
	
	target_dir_resized = f'{target_dir}_resized' 
	target_path_resized = Path(target_dir_resized).resolve() 

	if resize_flag:
		copy_folder_process(src_path, target_path_resized, "Copy-resizing...", resize_image)
		check_equal(src_path, target_path_resized)

	target_dir_preprocessed = f'{target_dir}_{preprocessed_suffix}'
	target_path_preprocessed = Path(target_dir_preprocessed).resolve() 

	copy_folder(target_path_resized, target_path_preprocessed, "Backing up resized files...")

	check_equal(target_path_resized, target_path_preprocessed)

	train_dir = Path(target_path_preprocessed, "train")
	validation_dir = Path(target_path_preprocessed, "val")
	test_dir = Path(target_path_preprocessed, "test")

	if undersample_flag:
		undersample(train_dir)

	save_classes(test_dir)

	flatten_directory(train_dir)
	flatten_directory(validation_dir)
	flatten_directory(test_dir)
	
	mean, std = get_normal_statistics(train_dir)
	print(mean, std)
if __name__ == '__main__':
	preprocess("classification/OCTData", undersample_flag=True)