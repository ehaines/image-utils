import pathlib
import time
from random import random, seed, uniform, randint

from PIL import Image, ImageFilter, ImagePath, ImageDraw
from PIL.Image import composite

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


def generate_red_noise(length, amplitude=1.0, correlation=0.7):
    """Generate red noise (temporally correlated) for path points.
    
    Args:
        length: Number of noise values to generate
        amplitude: Maximum amplitude of noise
        correlation: Correlation factor (0.0 = white noise, 1.0 = fully correlated)
    
    Returns:
        List of correlated noise values
    """
    if length <= 0:
        return []

    noise = [uniform(-amplitude, amplitude)]  # First point is random

    for i in range(1, length):
        # Each subsequent point is influenced by the previous point
        previous_noise = noise[i - 1]
        new_random = uniform(-amplitude, amplitude)
        # Blend previous noise with new random value based on correlation
        correlated_noise = correlation * previous_noise + (1 - correlation) * new_random
        noise.append(correlated_noise)

    return noise


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


def trace_contour(edge_coords_set, start_point):
    """Trace a connected contour starting from start_point"""
    contour = [start_point]
    current = start_point
    edge_coords_set.remove(start_point)

    while True:
        # Find next connected neighbor (8-connectivity)
        next_point = None
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in edge_coords_set:
                    next_point = neighbor
                    break
            if next_point:
                break

        if next_point is None:
            break  # End of contour

        contour.append(next_point)
        edge_coords_set.remove(next_point)
        current = next_point

    return contour


def tear_paper_edge(image: pathlib.Path | Image.Image, size: int, color: tuple):
    from PIL import ImageDraw
    img: Image
    if isinstance(image, pathlib.Path) or isinstance(image, str):
        img = Image.open(image).convert("RGBA")
    elif isinstance(image, Image.Image):
        img = image.convert("RGBA")
    else:
        raise TypeError("image is not correct type (Path or Image)")
    stroke = add_smooth_stroke(15, img, 15)  # todo just get stroke buffer piece out
    # stroke.show()
    paper = stroke.copy()
    paper_mask = paper.getchannel(3).point(lambda x: 255 if x > 250 else 0)
    # paper_mask.show()
    X, Y = paper.size
    edge = paper_mask.convert("L").filter(ImageFilter.FIND_EDGES)
    # edge.show()
    edge_mask = edge.point(lambda x: 255 if x > 240 else 0)
    # edge_mask.show()
    # Extract edge pixels and organize them into connected contour segments
    width, height = edge_mask.size
    edge_pixels = list(edge_mask.getdata())

    # Get all edge pixel coordinates
    all_edge_coords = []
    for i, pixel_val in enumerate(edge_pixels):
        if pixel_val == 255:  # White edge pixels
            x = i % width
            y = i // width
            all_edge_coords.append((x, y))

    print(f"DEBUG: Found {len(all_edge_coords)} total edge pixels")

    # Find all connected contour segments
    edge_coords_set = set(all_edge_coords)
    contour_segments = []

    while edge_coords_set:
        start_point = next(iter(edge_coords_set))
        contour = trace_contour(edge_coords_set, start_point)
        if len(contour) > 10:  # Only keep significant contours
            contour_segments.append(contour)

    print(f"DEBUG: Found {len(contour_segments)} contour segments")
    for i, segment in enumerate(contour_segments):
        print(f"DEBUG: Segment {i}: {len(segment)} points")

    # Use the largest contour segment (likely the main outline)
    if contour_segments:
        edge_coordinates = max(contour_segments, key=len)
        print(f"DEBUG: Using largest contour with {len(edge_coordinates)} points")
    else:
        edge_coordinates = []
        print("DEBUG: No contour segments found!")

    if not edge_coordinates:
        print("DEBUG: No edge coordinates found!")
        return

    cx, cy = probability_utils.find_centroid(stroke)
    # draw_centroid_point(image=stroke, centroid=(cx, cy), size=10)

    # Apply red noise to create torn paper effect
    modified_points = apply_torn_paper_effect(edge_coordinates, noise_amplitude=20, correlation=0.6)

    # Create a new image to show the result
    new_image = Image.new('RGBA', (width, height), (128, 128, 128, 0))
    draw = ImageDraw.Draw(new_image)

    # Create ImagePath from the ordered contour points and draw
    if len(modified_points) > 1:
        # Create ImagePath.Path from the modified contour points
        torn_edge_path = ImagePath.Path(modified_points)

        # Draw the path (as a polygon to make sure it's closed)
        draw.polygon(torn_edge_path, fill=color, width=8)

    # ImageDraw.floodfill(new_image, (cx,cy), (255,255,255))
    # new_image.show()
    
    # Flood fill from centroid to create filled torn paper effect
    filled_image = new_image.copy().convert("RGBA")
    noisy_fill = add_area_noise(filled_image)
    noisy_fill = Image.composite(noisy_fill, filled_image, filled_image)
    # noisy_fill.show()
    ImageDraw.floodfill(filled_image, (int(cx), int(cy)), color, thresh=100)
    # filled_image.show()
    filled_image.filter(ImageFilter.GaussianBlur(9))
    # filled_image.show()
    final_image = Image.alpha_composite(noisy_fill, img)
    # final_image.show()
    return final_image

    
    # options:
    #  try with LUT and point(): https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.point
    #  or try varying path by function


def apply_torn_paper_effect(path_points, noise_amplitude=10, correlation=0.7):
    """Apply red noise to path points for torn paper effect.
    
    Args:
        path_points: List of (x, y) coordinate tuples
        noise_amplitude: Maximum amplitude of noise displacement
        correlation: Correlation factor for red noise (0.0 = white noise, 1.0 = fully correlated)
    
    Returns:
        List of modified (x, y) coordinate tuples
    """
    if not path_points:
        return []

    num_points = len(path_points)
    print(f"DEBUG: Applying torn paper effect to {num_points} points")

    # Generate red noise for x and y directions separately
    noise_x = generate_red_noise(num_points, noise_amplitude, correlation)
    noise_y = generate_red_noise(num_points, noise_amplitude, correlation)

    # Apply noise to each point
    modified_points = []
    for i, (x, y) in enumerate(path_points):
        new_x = x + noise_x[i]
        new_y = y + noise_y[i]
        modified_points.append((new_x, new_y))

        if i < 5:  # Debug first few points
            print(f"DEBUG: Point {i}: ({x}, {y}) -> ({new_x}, {new_y}) (offset: {noise_x[i]:.2f}, {noise_y[i]:.2f})")

    return modified_points


def random_ift(amplitude, frequencies):
    """Placeholder for inverse Fourier transform noise generation"""
    # TODO: Implement if needed for advanced noise generation
    pass


def draw_centroid_point(image: Image, centroid: tuple[int, int], size: int, color="red"):
    """For debugging: draws and displays
     the centroid point of an image (or any other point passed to this function)
     as a circle overlay on the image with radius [size] in [color(default = red)].
     Color may be in any form recognized by PIL"""
    img = image.copy()
    centroid_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    x, y = centroid
    draw = ImageDraw.Draw(centroid_img)
    draw.ellipse((x - size, y - size, x + size, y + size), fill=color)
    # Use alpha_composite to properly blend the centroid with the stroke image
    img = Image.alpha_composite(img.convert("RGBA"), centroid_img)
    img.show()


def torn_paper_by_point():
    pass


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
    draw = ImageDraw.Draw(stroke)
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

