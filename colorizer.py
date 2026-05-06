"""
First actual file in processing a handwritten letter
Goal: Convert a piece of writing to a dichormatic image (only pure white or pure black)
This will make analysis much easier
"""
from PIL import Image
import os, sys
import matplotlib.pyplot as plt
import json
whiteness = {} #Standard dictionary of insertion speeds
with open("whiteness.json", "r") as f:
    whiteness = json.load(f) #Pre-calculated to avoid compute time of checking a dictionary
whiteness = {int(w):whiteness[w] for w in whiteness.keys()}

#By Claude, purely for testing and visualizing
def graph_integers(integers: list[int]) -> None:
    x = integers
    y = list(range(len(integers)))

    plt.plot(x, y, marker='o')
    plt.xlabel("Value")
    plt.ylabel("Index")
    plt.title("Index vs Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

"""
Colorizes an image, making it all black/white
Ensure inputted image is converted to RGB
"""
def colorize(im):
    pixels = im.load()
    width, height = im.size
    #Determine how "white" every pixel is, to denote a clear boundary
    for y in range(height):
        for x in range(width):
            pixel_white = pixels[x,y][0] * pixels[x,y][1] * pixels[x,y][2]
            whiteness[pixel_white] += 1
    white_keys = [key for key in whiteness.keys() if whiteness[key] != 0]
    white_keys = sorted(white_keys)
    threshold = white_keys[int(len(white_keys)/2)] #For speed, make a heurstic guess
    #Threshold has been set, convert the image
    for y in range(height):
        for x in range(width):
            pixel_white = pixels[x,y][0] * pixels[x,y][1] * pixels[x,y][2]
            pixels[x,y] = (0,0,0,255)
            if (pixel_white >= threshold):
                pixels[x,y] = (255,255,255,255)
    #Send pixels back
    return pixels   

