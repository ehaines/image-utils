import os
import argparse
import pathlib
from image_utils import *


def alpha_composite_directory_images(dir_path: pathlib.Path):
    with os.scandir(dir_path) as directory:
        for entry in directory:
            if entry.is_file() and "-protected-" in entry.name:
                print(f"processing {entry}")
                original_image_name: str = entry.name.split("-protected-")[0]
                original_path = pathlib.Path.joinpath(dir_path, original_image_name + ".png")
                if pathlib.Path.exists(original_path):
                    # clip the image with the original
                    image: Image = apply_file_image_mask(pathlib.Path(entry.path), original_path)
                    image = image.reduce(4)
                    save_path = pathlib.Path.joinpath(dir_path, original_image_name + "-processed.png")
                    image.save(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="remove extraneous background from glazed images.")
    parser.add_argument("directory", help="path to directory of images to process")
    args = parser.parse_args()
    alpha_composite_directory_images(pathlib.Path(args.directory))
