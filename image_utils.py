import pathlib

from PIL import Image


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