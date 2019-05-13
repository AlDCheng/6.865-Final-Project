#ifndef A10_H_PHUDVTKB
#define A10_H_PHUDVTKB

#include <cmath>
#include <iostream>
#include <vector>

#include "Image.h"
#include "basicImageManipulation.h"
#include "filtering.h"
#include "panorama.h"

#include "matrix.h"

// Write your declarations here, or extend the Makefile if you add source
// files

float Access_Subpixel(const Image &im, float x, float y, int z = 0, bool clamp = true);

Image Create_Pyramid(const Image &im, bool clamp = true);
Image Compute_Tensor_Window(const Image &im_dx, const Image &im_dy, Vec2f point, 
						int block_size = 31, float sigmaG = 1.0, float factorSigma = 4.0);
std::vector<Vec2f> Generate_Sparse_Points(unsigned width, unsigned height, unsigned stride);

Vec2f Lucas_Kanade(const Image &im1, const Image &im2, const Image &gx, const Image &gy, Vec2f point, Vec2f displacement, 
						int block_size = 31, float error = 0.001, int max_iters = 100, bool fast = false);
vector<Vec2f> LK_Pyr(const Image &im1, const Image &im2, vector<Vec2f> points, unsigned pyramid_levels = 4,
						int block_size = 31, float error = 0.001, int max_iters = 100, bool fast = false);

Image Align_LK_Tile(Image im1, Image im2, unsigned pyramid_levels = 4, unsigned stride = 10,
						int block_size = 31, float error = 0.001, int max_iters = 100, bool fast = false);
Image Align_LK_Harris(Image im1, Image im2, unsigned pyramid_levels = 4, string mode = "tile",
						int block_size = 31, float error = 0.001, int max_iters = 100, bool fast = false);


#endif /* end of include guard: A10_H_PHUDVTKB */

