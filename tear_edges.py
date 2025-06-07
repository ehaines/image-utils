import pathlib, config
from PIL import Image
from image_utils import *

path = pathlib.Path(config.test_image)
# image: Image = Image.open(path, formats=["PNG", "JPEG"])
# image = image.reduce(4)
# solid_image = make_solid_overlay_mask(image, (255, 255, 255))
output = add_smooth_stroke(125, path)
# output = dilate_image(path)
output.show()

