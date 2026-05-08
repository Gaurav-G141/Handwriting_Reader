"""
First actual file in processing a handwritten letter
Goal: Convert a piece of writing to a dichormatic image (only pure white or pure black)
This will make analysis much easier
"""
import ctypes
from PIL import Image

lib = ctypes.CDLL("./black_and_white.so")
lib.colorize.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int]
lib.colorize.restype = None

"""
Colorizes an image, making it all black/white
Ensure inputted image is converted to RGB
"""
def colorize(img: Image.Image) -> Image.Image:
    pixel_bytes = bytearray(img.tobytes())   # mutable byte buffer
    buf = (ctypes.c_uint8 * len(pixel_bytes)).from_buffer(pixel_bytes)

    lib.colorize(buf, img.width, img.height)

    return Image.frombytes("RGB", img.size, bytes(pixel_bytes))
