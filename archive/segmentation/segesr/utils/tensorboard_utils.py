import numpy as np
from PIL import Image

def get_tensorboard_writer(accelerator):
    """The SummaryWriter of the accelerator's TensorBoard tracker (None without one, or off the main process)."""

    if not accelerator.is_main_process:
        return None
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            return tracker.writer
    return None

def to_tensorboard_image(image, max_size=768):
    """PIL image -> HWC uint8 array, downscaled so that its longest side is at most `max_size` (keeps logs small)."""

    image = image.convert("RGB")
    scale = max_size / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)
    return np.asarray(image)

def side_by_side(images):
    """Concatenates PIL images horizontally, resizing them to the height of the first one."""

    height = images[0].height
    images = [im if im.height == height else im.resize((round(im.width * height / im.height), height), Image.BICUBIC) for im in images]
    canvas = Image.new("RGB", (sum(im.width for im in images), height))
    x = 0
    for im in images:
        canvas.paste(im.convert("RGB"), (x, 0))
        x += im.width
    return canvas
