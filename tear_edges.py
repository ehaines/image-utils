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
    for image_path in images:
        path = pathlib.Path(image_path)
        # output = tear_paper_edge(image_path, 15, color=(255, 174, 196))
        output = tear_paper_edge(image_path, 15, color=(255, 245, 250))
        output.show()

if __name__ == "__main__":
    tear_edges_test()

