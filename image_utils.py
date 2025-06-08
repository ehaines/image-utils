import pathlib
import time
from random import random, seed, uniform

from PIL import Image, ImageFilter
from PIL.ImageDraw import Draw


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
        #noise = [uniform(-1, +1) for i in range(mapsize)]
        #print_chart(i, smoother(noise))
        pass

def add_stroke():
    pass


def add_smooth_stroke(stroke_size, image_path: pathlib.Path | str, alias_size=10):
    process_start = time.time()
    print (process_start)
    stroke_radius = stroke_size or 25
    img = Image.open(image_path)  # RGBA image
    stroke_image = Image.new("RGBA", img.size, (255, 255, 255, 1))
    img_alpha = (img.getchannel(3).point(lambda x: 255 if x > 200 else 0))
    # img_alpha.filter(ImageFilter.FIND_EDGES).load()
    # img_alpha = img.getchannel(3)
    alpha_time = time.time()
    alpha_diff = alpha_time - process_start
    print("time to get alpha: " + str(alpha_diff))
    start = time.time()
    stroke_alpha = circle_stroke(img_alpha, stroke_radius)
    # stroke_alpha.show()
    stroke_alpha.paste(circle_stroke(stroke_alpha, alias_size))
    stroke_alpha.paste(circle_stroke(stroke_alpha, alias_size//2))
    stroke_alpha.paste(circle_stroke(stroke_alpha, alias_size //5))
    # stroke_alpha.show()
    stroke_diff = time.time() - start
    print(fr"max filter stroke time for size {stroke_size}: {str(stroke_diff)}")
    start = time.time()
    for i in range(8):
        stroke_alpha = stroke_alpha.filter(ImageFilter.SMOOTH_MORE)
    smooth_time = time.time() - start
    print(f"smooth time: {str(smooth_time)} seconds")
    stroke_image.putalpha(stroke_alpha.convert("RGBA").getchannel(3))
    output = Image.alpha_composite(stroke_image, img)
    final_time = time.time() - process_start
    print(f"total time: {str(final_time)}")
    return output


def tear_paper_edge():
    pass


def torn_paper_effect():
    pass

def dilate_image(image: pathlib.Path|Image.Image):
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
    #dilated_img_pil.save("dilated_image.jpg")

    # Display the result
    #dilated_img_pil.show()

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

def circle_stroke(image: pathlib.Path | Image.Image, size: int, color: str = 'white'):

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