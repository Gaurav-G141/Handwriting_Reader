#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
const int pure_black = 10 * 10 * 10;
const int buffer = 100;
bool* find_line_splits(uint8_t* pixels, int width, int height) {
    bool * ans = malloc(height);
    for (int i = 0; i < height; i++) {
        bool found = true;
        for (int j = buffer; (j < width - buffer) & found; j++) {
            int pixel_white = (pixels[3 * (i * width + j)] + 10) * (pixels[3 * (i * width + j) + 1] + 10) * (pixels[3 * (i * width + j) + 2] + 10);
            if (pixel_white == pure_black) {
                found = false;
            }
        }
        ans[i] = found;
    }
    return ans;
}
uint8_t* make_image(uint8_t* pixels, int width, int upper, int lower) {
    uint8_t * image = malloc(3 * width * (lower - upper));
    int x = 0;
    for (int i = upper; i < lower; i++) {
    for (int j = 0; j < buffer; j++) {
        image[3 * (x * width + j)] = 255;
        image[3 * (x * width + j) + 1] = 255;
        image[3 * (x * width + j) + 2] = 255;        
    }
    for (int j = buffer; (j < width - buffer); j++) {
        image[3 * (x * width + j)] = pixels[3 * (i * width + j)];
        image[3 * (x * width + j) + 1] = pixels[3 * (i * width + j) + 1];
        image[3 * (x * width + j) + 2] = pixels[3 * (i * width + j) + 2];
    }
    x++;
    }
    return image;
}