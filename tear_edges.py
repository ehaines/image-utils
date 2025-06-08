import pathlib, config
from PIL import Image
from image_utils import *

images = [config.test_image,
          config.test_image2,
          config.test_image3,
          config.test_image4,
          config.test_image5,
          config.test_image6]


def stroke_test():
    for image_path in images:
        path = pathlib.Path(image_path)
        output = add_smooth_stroke(100, path, alias_size=10)
        output.show()


def tear_edges_test():
    tear_paper_edge(images[0], 15)

if __name__ == "__main__":
    tear_edges_test()

