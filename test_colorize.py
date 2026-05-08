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
 
        start_time = time.time()
        img = Image.open(input_path).convert("RGB")
        result = colorize(img)
        result.save(output_path)
        print(f"[OK]    {filename}")
        print(f"Time to colorize: {time.time() - start_time}")


batch_colorize("test_images","result_images")
