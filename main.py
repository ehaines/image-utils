# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import pathlib
from image_utils import *
from config.paths import *


def alpha_composite_directory_images(dir_path: pathlib.Path):
    with os.scandir(dir_path) as directory:
        for entry in directory:
            if entry.is_file() and "-protected-" in entry.name:
                original_image_name: str = entry.name.split("-protected-")[0]
                original_path = pathlib.Path.joinpath(dir_path, original_image_name + ".png")
                if pathlib.Path.exists(original_path):
                    # clip the image with the original
                    image: Image = apply_file_image_mask(pathlib.Path(entry.path), original_path)
                    image = image.reduce(4)
                    save_path = pathlib.Path.joinpath(dir_path, original_image_name + "-processed.png")
                    image.save(save_path)


alpha_composite_directory_images(pathlib.Path(glazed_files))
