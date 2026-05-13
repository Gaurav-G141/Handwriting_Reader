#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
const int pure_black = 10 * 10 * 10; //whiteness of a purely black pixel
int buffer = 100; //Amount to forgive at corners (ex: lines at left and right)
const float percent_black = 0.4; //Maximum amount of black a line can have to be still text
/*
Check where text splits
*/
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
/*
Check if a "line" is actually a line of text or garbage
*/
bool isline(uint8_t* pixels, int width, int height) {
    int amount = 0;
    int max = (int)(width * height * percent_black);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int pixel_white = (pixels[3 * (i * width + j)] + 10) * (pixels[3 * (i * width + j) + 1] + 10) * (pixels[3 * (i * width + j) + 2] + 10);
            if (pixel_white == pure_black) {
                amount++;
                if (amount > max) {
                    return false;
                }
            }            
        }
    }
    return amount > 100; //Shouldn't be a pure white screen
}