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
lib.isline.restype = ctypes.c_bool
lib.isline.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int]
"""
Splits image. Assume result already colorized
"""
def split(img: Image.Image) -> list[Image.Image]:
    #Make pixel buffer
    pixel_bytes = bytearray(img.tobytes())   # mutable byte buffer

    buf = (ctypes.c_uint8 * len(pixel_bytes)).from_buffer(pixel_bytes)
    result_ptr = lib.find_line_splits(buf, img.width, img.height)
    result = result_ptr[:img.height]
    #Makes obvious cuts, get out lines
    cuts = [i for i in range(len(result)) if result[i]]
    cuts = [c for c in cuts if (c + 1 not in cuts or c - 1 not in cuts)]

    results = []
    for i in range(1,len(cuts) - 1,2):
        results.append(img.crop((0, cuts[i], img.width, cuts[i+1])))
  
    filtered_results = [r for r in results if lib.isline((ctypes.c_uint8 * len(bytearray(img.tobytes()))).from_buffer(bytearray(img.tobytes())), r.width, r.height)]
    #Get line splits
    filtered_splits = [r.height for r in filtered_results]

    
    return filtered_results



