import pathlib
import time
from random import random, seed, uniform, randint

from PIL import Image, ImageFilter, ImagePath
from PIL.ImageDraw import Draw, ImageDraw

import probability_utils


def make_solid_overlay_mask(img: Image, new_rgb: tuple[int, int, int] | str):
    """uses alpha layer replacement technique with a solid color layer to create a color overlay layer
    with no ghosting of unchanged pixels around the edges. All colors are changed to the mask layer and only the
    alpha values are preserved"""
    # Open original image and extract the alpha channel
    alpha = img.getchannel('A')

    # Create red image the same size and copy alpha channel across
    new_img = Image.new('RGBA', img.size, color=new_rgb)
    new_img.putalpha(alpha)
    return new_img


def alpha_composite(source_image: Image, mask_image: Image) -> Image:
    alpha = mask_image.getchannel("A")
    source_image_rgba = source_image.convert("RGBA")
    source_image_rgba.putalpha(alpha)
    return source_image_rgba


def apply_file_image_mask(source_image_path: pathlib.Path, mask_image_path: pathlib.Path) -> Image:
    source_image = Image.open(source_image_path, formats=["PNG", "JPEG"])
    mask_image = Image.open(mask_image_path, formats=["PNG", "JPEG"])
    alpha_composited_image = alpha_composite(source_image, mask_image)
    return alpha_composited_image


def smoother(noise):
    output = []
    for i in range(len(noise) - 1):
        output.append(0.5 * (noise[i] + noise[i + 1]))
    return output


def generate_random_noise(noise_level=1):
    """Generate random noise for the torn effect."""
    noise = uniform(-noise_level, noise_level)
    return noise


def red_noise(noise_range):
    for i in range(noise_range):
        seed(i)
        # noise = [uniform(-1, +1) for i in range(mapsize)]
        # print_chart(i, smoother(noise))
        pass


def add_stroke():
    pass


def add_smooth_stroke(stroke_size, image_src: pathlib.Path | str | Image.Image, alias_size=10):
    if isinstance(image_src, pathlib.Path) or isinstance(image_src, str):
        img = Image.open(image_src).convert("RGBA")
    elif isinstance(image_src, Image.Image):
        img = image_src.convert("RGBA")
    else:
        raise TypeError("image is not correct type (Path/str or Image)")

    process_start = time.time()
    print(process_start)
    stroke_radius = stroke_size or 25
    stroke_image = Image.new("RGBA", img.size, (255, 255, 255, 1))
    img_alpha = (img.getchannel(3).point(lambda x: 255 if x > 240 else 0))
    alpha_time = time.time()
    alpha_diff = alpha_time - process_start
    print("time to get alpha: " + str(alpha_diff))
    start = time.time()
    stroke_alpha = circle_stroke_buffer(img_alpha, stroke_radius)
    # stroke_alpha.show()
    stroke_alpha.paste(circle_stroke_buffer(stroke_alpha, alias_size))
    stroke_alpha.paste(circle_stroke_buffer(stroke_alpha, alias_size // 2))
    stroke_alpha.paste(circle_stroke_buffer(stroke_alpha, alias_size // 5))
    # stroke_alpha.show()
    stroke_diff = time.time() - start
    print(fr"max filter stroke time for size {stroke_size}: {str(stroke_diff)}")
    start = time.time()
    stroke_alpha.filter(ImageFilter.MedianFilter(size=9))
    for i in range(4):
        stroke_alpha = stroke_alpha.filter(ImageFilter.SMOOTH_MORE)
    # stroke_alpha.paste(stroke_alpha
    #                    .filter(ImageFilter.MedianFilter(size=9)))
    smooth_time = time.time() - start
    print(f"smooth time: {str(smooth_time)} seconds")
    stroke_image.putalpha(stroke_alpha.convert("RGBA").getchannel(3))
    output = Image.alpha_composite(stroke_image, img)
    final_time = time.time() - process_start
    print(f"total time: {str(final_time)}")
    return output


def tear_paper_edge(image: pathlib.Path | Image.Image, size: int, color: str = 'pink'):
    img: Image
    # Open the image and convert to grayscale
    if isinstance(image, pathlib.Path) or isinstance(image, str):
        img = Image.open(image).convert("RGBA")
    elif isinstance(image, Image.Image):
        img = image.convert("RGBA")
    else:
        raise TypeError("image is not correct type (Path or Image)")
    stroke = add_smooth_stroke(60, img, 15)  # todo just get stroke buffer piece out
    # stroke.show()
    paper = stroke.copy()
    paper_mask = paper.getchannel(3).point(lambda x: 255 if x > 250 else 0)
    paper_mask.show()
    X, Y = paper.size
    edge = paper_mask.convert("L").filter(ImageFilter.FIND_EDGES)
    # edge.show()
    edge_mask = edge.point(lambda x: 255 if x > 240 else 0)
    edge_mask.show()
    edge_pixels = edge_mask.getdata()
    edge_path = ImagePath.Path(edge_pixels)
    print(f"edge path type: {str(type(edge_path))}")
    edge_path.compact()
    cx, cy = probability_utils.find_centroid(image)
    draw_centroid_point(image=img, centroid=(cx, cy), size=10)
    edge_path.map(torn_paper_effect_by_path)

    # Create a new image
    new_image = Image.new('RGB', (400, 400), (128, 128, 128))
    draw = Draw(new_image)
    draw.line(edge_path)
    new_image.show()
    # options:
    #  try with LUT and point(): https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.point
    #  or try varying path by function


def torn_paper_effect_by_path():
    #may need to find the centroid, the angle to the centroid, then vary point on that line

    #optionally, compress the path for greater speed

    def smoother(noise):
        output = []
        for i in range(len(noise) - 1):
            output.append(0.5 * (noise[i] + noise[i + 1]))
        return output

    # Function to apply random offsets to the points
    def apply_random_offsets(points, max_offset=10):
        noise = [(x + random.randint(-max_offset, max_offset), y + random.randint(-max_offset, max_offset)) for x, y in
                points]
        return smoother(noise)

    # Apply random offsets to the points
    random.seed(time.time())

    modified_points = apply_random_offsets(path.tolist(True))

    # Create a Path object with the modified points
    path = ImagePath.Path(modified_points)

    # noise = [uniform(-1, +1) for i in range(len(path))]
    # smooth_red = smoother(noise)
    # path_points = path.tolist(True)
    # assert len(smooth_red) == len(path_points)

    # line_pts = []
    # for i in range(len(path)):
    #     ptx, pty = path_points[i]
    #     ptx = smooth_red[i] + ptx
    #     pty = smooth_red[i] + pty
    #     line_pts[i] = []
    #     line_pts[i][0] = ptx
    #     line_pts[i][1] = pty
    # path = ImagePath.Path(line_pts)

def random_ift(amplitude, frequencies):
        random.seed(i)
        amplitudes = [amplitude(f) for f in frequencies]
        noises = [noise(f) for f in frequencies]
        sum_of_noises = weighted_sum(amplitudes, noises)

def draw_centroid_point(image: Image, centroid: tuple[int, int], size: int, color="red"):
    """For debugging: draws and displays
     the centroid point of an image (or any other point passed to this function)
     as a circle overlay on the image with radius [size] in [color(default = red)].
     Color may be in any form recognized by PIL"""
    img = image.copy()
    centroid_img = Image.new(img.mode, img.size, (0,0,0,0))
    x, y = centroid
    draw = Draw(centroid_img)
    draw.ellipse((x - size, y - size, x + size, y + size), fill=color)
    img.paste(centroid_img, (0, 0)) #pastes the point on top of the original image
    img.show()

def torn_paper_by_point():
    pass

def dilate_image(image: pathlib.Path | Image.Image):
    from PIL import Image
    import numpy as np
    from scipy.ndimage import binary_dilation

    img: Image
    # Open the image and convert to grayscale
    if isinstance(image, pathlib.Path) or isinstance(image, str):
        img = Image.open(image).convert('L')
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise TypeError("image wrong type")
    # Convert the image to a binary array (thresholding)
    threshold = 128
    binary_img = np.array(img) > threshold

    # Define a 3x3 structuring element for dilation
    structure = np.ones((125, 125), dtype=bool)

    # Apply the dilation operation
    dilated_img = binary_dilation(binary_img, structure=structure)

    # Convert the result back to an image
    dilated_img_pil = Image.fromarray(dilated_img.astype(np.uint8) * 255)
    return dilated_img_pil
    # Save the result
    # dilated_img_pil.save("dilated_image.jpg")

    # Display the result
    # dilated_img_pil.show()


# def mystroke(filename: pathlib.Path, size: int, color: str = 'white'):
#     outf = filename.parent/'mystroke'
#     if not outf.exists():
#         outf.mkdir()
#     img = Image.open(filename)
#     X, Y = img.size
#     edge = img.filter(ImageFilter.FIND_EDGES).load()
#     stroke = Image.new(img.mode, img.size, (0,0,0,0))
#     draw = Draw(stroke)
#     for x in range(X):
#         for y in range(Y):
#             if edge[x,y][3] > 0:
#                 draw.ellipse((x-size,y-size,x+size,y+size), fill=color)
#     stroke.paste(img, (0, 0), img )
#     # stroke.show()
#     # stroke.save(outf/filename.name)

def circle_stroke_buffer(image: pathlib.Path | Image.Image, size: int, color: str = 'white'):
    img: Image
    # Open the image and convert to grayscale
    if isinstance(image, pathlib.Path) or isinstance(image, str):
        img = Image.open(image).convert("L")
    elif isinstance(image, Image.Image):
        img = image.convert("L")
    else:
        raise TypeError("image is not correct type (Path or Image)")
    X, Y = img.size
    edge = img.filter(ImageFilter.FIND_EDGES).load()
    # stroke = Image.new(img.mode, img.size, (0, 0, 0, 0))
    stroke = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = Draw(stroke)
    for x in range(X):
        for y in range(Y):
            if edge[x, y] > 0:
                draw.ellipse((x - size, y - size, x + size, y + size), fill=color)

    stroke.paste(img, (0, 0), img)
    return stroke


def add_area_noise(image: Image.Image | pathlib.Path):
    im: Image
    # Open the image and convert to RGBA
    if isinstance(image, pathlib.Path) or isinstance(image, str):
        im = Image.open(image).convert("RGBA")
    elif isinstance(image, Image.Image):
        im = image.copy().convert("RGBA")
    else:
        raise TypeError("image is not correct type (Path or Image)")

    for i in range(round(im.size[0] * im.size[1] / 5)):
        im.putpixel(
            (randint(0, im.size[0] - 1), randint(0, im.size[1] - 1)),
            (randint(0, 255), randint(0, 255), randint(0, 255))
        )
    return im
