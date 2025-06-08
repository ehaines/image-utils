import pathlib

from PIL import Image
import numpy as np


def find_centroid(image: Image.Image | pathlib.Path):
    # note - consider reducing the image size
    #  (and then adjusting coordinates of answer) to make this process faster
    img: Image
    if isinstance(image, pathlib.Path) or isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise TypeError("image is not correct type (Path or Image)")
    immat = img.load()
    (X, Y) = img.size
    m = np.zeros((X, Y))

    for x in range(X):
        for y in range(Y):
            m[x, y] = immat[(x, y)] != (255, 255, 255)
    m = m / np.sum(np.sum(m))

    # marginal distributions
    dx = np.sum(m, 1)
    dy = np.sum(m, 0)

    # expected values
    cx = np.sum(dx * np.arange(X))
    cy = np.sum(dy * np.arange(Y))

    return cx, cy
