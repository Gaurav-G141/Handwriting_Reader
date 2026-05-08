#include "khash.h"
#include "ksort.h"
#include <stdlib.h>
#include <stdio.h>
KHASH_MAP_INIT_INT(int_map, int)   // declares a int→int map type
KSORT_INIT_GENERIC(int);
const int threshold = 138 * 138 * 138;
/*
Colorizes an image, making non-written parts white
Ensure inputted image is converted to RGB
void colorize(uint8_t* pixels, int width, int height) {

    khash_t(int_map) * table = kh_init(int_map); 
    //Get pixelwhite values
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int pixel_white = pixels[3 * (i * width + j)] * pixels[3 * (i * width + j) + 1] * pixels[3 * (i * width + j) + 2];
            int absent;
            khiter_t k = kh_put(int_map, table, pixel_white, &absent);
            if (absent == 1) {
                kh_value(table, k) = 1;
            } else {
                kh_value(table, k)++;
            }
        }
    }

    //Make array of white keys
    int size = kh_size(table);
    int * white_keys = malloc(size << 2);
    for (int i = 0; i < size; i++) {
        white_keys[i] = kh_key(table, i);
    }
    ks_introsort(int, size, white_keys);
    int threshold = white_keys[size << 1];
    //Modifies pixels
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int pixel_white = pixels[3 * (i * width + j)] * pixels[3 * (i * width + j) + 1] * pixels[3 * (i * width + j) + 2];
            pixels[3 * (i * width + j)] = 0;
            pixels[3 * (i * width + j) + 1] = 0;
            pixels[3 * (i * width + j) + 2] = 0;
            if (pixel_white > threshold) {
                pixels[3 * (i * width + j)] = 255;
                pixels[3 * (i * width + j) + 1] = 255;
                pixels[3 * (i * width + j) + 2] = 255;
            }
        }
    }
    kh_destroy(int_map, table);
}
*/

//Version without hashtable, better but has flaws
void colorize(uint8_t* pixels, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int pixel_white = (pixels[3 * (i * width + j)] + 10) * (pixels[3 * (i * width + j) + 1] + 10) * (pixels[3 * (i * width + j) + 2] + 10);
            pixels[3 * (i * width + j)] = 0;
            pixels[3 * (i * width + j) + 1] = 0;
            pixels[3 * (i * width + j) + 2] = 0;
            if (pixel_white > threshold) {
                pixels[3 * (i * width + j)] = 255;
                pixels[3 * (i * width + j) + 1] = 255;
                pixels[3 * (i * width + j) + 2] = 255;
            }
        }
    }    
}
