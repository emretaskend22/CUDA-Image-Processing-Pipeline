#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Data Structures ---

// Simple RGB Pixel structure
typedef struct {
    unsigned char r, g, b;
} RgbPixel;

// Simple Grayscale Pixel structure
typedef struct {
    unsigned char gray;
} GrayPixel;

// --- Gaussian 5x5 Kernel ---
// The assignment uses a 5x5 blur, meaning a radius of 2.
// This kernel is normalized such that the sum of all elements is 1.0.
#define KERNEL_RADIUS 2
#define KERNEL_WIDTH (2 * KERNEL_RADIUS + 1) // 5

static const float GAUSSIAN_KERNEL[KERNEL_WIDTH * KERNEL_WIDTH] = {
    1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f,
    4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
    7.0f/273.0f, 26.0f/273.0f, 41.0f/273.0f, 26.0f/273.0f, 7.0f/273.0f,
    4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
    1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f
};

// --- Image I/O Functions ---

// Read a PPM image (P3 or P6 format)
static inline int read_ppm(const char* filename, RgbPixel** image, int* width, int* height) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return 0;
    }
    
    char magic[3];
    if (fscanf(fp, "%2s", magic) != 1) {
        fprintf(stderr, "Error: Invalid PPM header\n");
        fclose(fp);
        return 0;
    }
    
    int is_p3 = (strcmp(magic, "P3") == 0);
    int is_p6 = (strcmp(magic, "P6") == 0);
    
    if (!is_p3 && !is_p6) {
        fprintf(stderr, "Error: Invalid PPM format (expected P3 or P6, got %s)\n", magic);
        fclose(fp);
        return 0;
    }
    
    // Skip comments
    int c;
    while ((c = fgetc(fp)) == '#') {
        while (fgetc(fp) != '\n');
    }
    ungetc(c, fp);
    
    int max_val;
    if (fscanf(fp, "%d %d %d", width, height, &max_val) != 3) {
        fprintf(stderr, "Error: Invalid PPM header\n");
        fclose(fp);
        return 0;
    }
    fgetc(fp); // consume newline
    
    if (max_val != 255) {
        fprintf(stderr, "Error: Only 8-bit PPM files supported\n");
        fclose(fp);
        return 0;
    }
    
    size_t pixel_count = (*width) * (*height);
    *image = (RgbPixel*)malloc(pixel_count * sizeof(RgbPixel));
    if (!*image) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(fp);
        return 0;
    }
    
    if (is_p6) {
        // Binary format - read directly
        if (fread(*image, sizeof(RgbPixel), pixel_count, fp) != pixel_count) {
            fprintf(stderr, "Error: Failed to read image data\n");
            free(*image);
            fclose(fp);
            return 0;
        }
    } else {
        // ASCII format (P3) - read values one by one
        for (size_t i = 0; i < pixel_count; i++) {
            int r, g, b;
            if (fscanf(fp, "%d %d %d", &r, &g, &b) != 3) {
                fprintf(stderr, "Error: Failed to read pixel data at index %zu\n", i);
                free(*image);
                fclose(fp);
                return 0;
            }
            (*image)[i].r = (unsigned char)r;
            (*image)[i].g = (unsigned char)g;
            (*image)[i].b = (unsigned char)b;
        }
    }
    
    fclose(fp);
    printf("Loaded %dx%d image from %s (%s format)\n", *width, *height, filename, is_p6 ? "P6" : "P3");
    return 1;
}

// Write a grayscale PPM image (P5 format)
static inline int write_ppm(const char* filename, const GrayPixel* image, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        return 0;
    }
    
    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    
    size_t pixel_count = width * height;
    if (fwrite(image, sizeof(GrayPixel), pixel_count, fp) != pixel_count) {
        fprintf(stderr, "Error: Failed to write image data\n");
        fclose(fp);
        return 0;
    }
    
    fclose(fp);
    printf("Wrote %dx%d result to %s\n", width, height, filename);
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif // IMAGE_UTILS_H
