import pathlib, config, os, argparse
from image_utils import *

images = [config.test_image]


def stroke_test():
    for image_path in images:
        path = pathlib.Path(image_path)
        output = add_smooth_stroke(100, path, alias_size=10)
        output.show()


def tear_edges_test():
    for image_path in images:
        path = pathlib.Path(image_path)
        # output = tear_paper_edge(image_path, 15, color=(255, 174, 196))
        output = tear_paper_edge(image_path, 15, color=(255, 245, 250))
        output.show()

def tear_edges(dir_path, name_change):
    with os.scandir(dir_path) as directory:
        for entry in directory:
            original_image_name = pathlib.Path(entry.name).stem
            original_suffix = pathlib.Path(entry).suffix
            if entry.is_file() and original_suffix.endswith(".png"):
                output = tear_paper_edge(pathlib.Path(entry), 15, color=(255, 245, 250))
                print(type(dir_path))
                save_path = pathlib.Path.joinpath(pathlib.Path(dir_path), "papered", original_image_name + f"-{name_change}.png")
                output.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("directory", help="path to directory of images to process")
    parser.add_argument("--namechange", default="torn", help="string addition to filename for final torn paper image")

    args = parser.parse_args()
    tear_edges(args.directory, args.namechange)

