import pathlib
from random import random, seed, uniform

from PIL import Image, ImageFilter


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


def add_smooth_stroke(stroke_size, image_path: pathlib.Path | str):
    stroke_radius = stroke_size or 5
    img = Image.open(image_path)  # RGBA image
    stroke_image = Image.new("RGBA", img.size, (255, 255, 255, 1))
    img_alpha = img.getchannel(3).point(lambda x: 255 if x > 0 else 0)
    stroke_alpha = img_alpha.filter(ImageFilter.MaxFilter(stroke_radius))
    # optionally, smooth the result
    stroke_alpha = stroke_alpha.filter(ImageFilter.SMOOTH)
    stroke_image.putalpha(stroke_alpha)
    output = Image.alpha_composite(stroke_image, img)
    return output


def tear_paper_edge():
    pass


def torn_paper_effect():
    pass