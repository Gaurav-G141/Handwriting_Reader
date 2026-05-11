"""
Second file in processing handwritten data
Goal: Split a file into lines
"""
import ctypes
from PIL import Image
import numpy as np
lib = ctypes.CDLL("./text_splitter.so")
lib.find_line_splits.restype = ctypes.POINTER(ctypes.c_bool)
lib.find_line_splits.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int]
lib.make_image.restype = ctypes.POINTER(ctypes.c_uint8)
lib.make_image.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_int]
"""
Splits image. Assume result already colorized
"""
def split(img: Image.Image) -> list[Image.Image]:
    pixel_bytes = bytearray(img.tobytes())   # mutable byte buffer
    print(img.height)
    buf = (ctypes.c_uint8 * len(pixel_bytes)).from_buffer(pixel_bytes)
    result_ptr = lib.find_line_splits(buf, img.width, img.height)
    result = result_ptr[:img.height]
    cuts = [i for i in range(len(result)) if result[i]]
    cuts = [c for c in cuts if (c + 1 not in cuts or c - 1 not in cuts)]
    print(cuts)
    print(len(cuts))
    results = []
    for i in range(1,len(cuts) - 1,2):
        results.append(img.crop((0, cuts[i], img.width, cuts[i+1])))
    return results

im = Image.open("colorized_images/APLIT1_cropped-3.png").convert("RGB")
res = split(im)
for r in res:
    r.show()