"""
Testing harness of colorize
"""
import os
from colorizer import colorize
from PIL import Image
import time

def batch_colorize(input_folder: str, output_folder: str) -> None:
    os.makedirs(output_folder, exist_ok=True)
 
    image_files = [f for f in os.listdir(input_folder)]
 
    for filename in image_files:
        print(f"Colorizing: {filename}")
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
 
        try:
            start_time = time.time()
            img = Image.open(input_path).convert("RGB")
            pixels = colorize(img)
 
            # Reconstruct a saveable image from the pixel grid
            result = Image.new(img.mode if img.mode == "RGB" else "RGB", img.size)
            result_pixels = result.load()
            for x in range(img.width):
                for y in range(img.height):
                    result_pixels[x, y] = pixels[x, y]
 
            result.save(output_path)
            print(f"[OK]    {filename}")
            print(f"Time to colorize: {time.time() - start_time}")
 
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

batch_colorize("test_images","result_images")
