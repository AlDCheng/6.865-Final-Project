#include "a10.h"

using namespace std;

// Helper Functions
inline float Access_Subpixel(const Image &im, float x, float y, int z, bool clamp) {
	return interpolateLin(im, x, y, z, clamp);
}

Image Compute_Tensor_Window(const Image &im_dx, const Image &im_dy, Vec2f point, int block_size, float sigmaG, float factorSigma) {
    Image output = Image(block_size, block_size, 3);

    int w = (int)((block_size - 1.0) / 2.0);
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
        	float x = point.x() - w + j;
        	float y = point.y() - w + i;
            output(j, i, 0) = pow(Access_Subpixel(im_dx, x, y), 2);
            output(j, i, 1) = Access_Subpixel(im_dx, x, y) * Access_Subpixel(im_dy, x, y);
            output(j, i, 2) = pow(Access_Subpixel(im_dy, x, y), 2);
        }
    }

    return gaussianBlur_separable(output, sigmaG*factorSigma);
}

Image Create_Pyramid(const Image &im, bool clamp) {
	Filter guassian(vector<float>{
		1,  4,  6,  4, 1,
    	4, 16, 24, 16, 4,
    	6, 24, 36, 24, 6,
    	4, 16, 24, 16, 4,
    	1,  4,  6,  4, 1}, 5, 5);

	Image convolved = guassian.convolve(im, clamp) / 16.0;
	int width  = (int)((im.width()) / 2.0);
	int height = (int)((im.height()) / 2.0);

	Image out(width, height, im.channels());
	for (int i = 1; i < im.height(); i+=2) {
		for (int j = 1; j < im.width(); j+=2) {
			for (int k = 0; k < im.channels(); k++) {
				out((j - 1) / 2, (i - 1) / 2, k) = im(j, i, k);
			}
		}
	}

	return out;
}

vector<Vec2f> Generate_Sparse_Points(unsigned width, unsigned height, unsigned stride)
{
	vector<Vec2f> sparse_points;

	unsigned margin = stride / 2;
	
	for (unsigned i = margin; i <= height - margin; i += stride) {
		for (unsigned j = margin; j <= width - margin; j += stride) {
			Vec2f point;
			point << j, i;
			sparse_points.push_back(point);
		}
	}
	return sparse_points;
}

// Lucas-Kanade Implementation
Vec2f Lucas_Kanade(const Image &im1, const Image &im2, const Image &gx, const Image &gy, Vec2f point, Vec2f displacement, 
					int block_size, float error, int max_iters, bool fast) {
	Matrix G(2,2);
	G << 0, 0, 0, 0;
	int w = (int)((block_size - 1.0) / 2.0);

	if (fast) {
		for (float i = point.y() - w; i <= point.y() + w; i++) {
			for (float j = point.x() - w; j <= point.x() + w; j++) {
				float dx = (Access_Subpixel(im1, j + 1, i) - Access_Subpixel(im1, j - 1, i)) / 2.0;
				float dy = (Access_Subpixel(im1, j, i + 1) - Access_Subpixel(im1, j, i - 1)) / 2.0;

				G(0,0) += pow(dx, 2);
				G(0,1) += dx * dy;
				G(1,0) += dx * dy;
				G(1,1) += pow(dy, 2);
			}
		}
	} else {
		Image tensor = Compute_Tensor_Window(gx, gy, point, block_size);
	    for (int i = 0; i < block_size; i++) {
	    	for (int j = 0; j < block_size; j++) {
	    		G(0, 0) += tensor(j, i, 0);
	    		G(0, 1) += tensor(j, i, 1);
	    		G(1, 0) += tensor(j, i, 1);
	    		G(1, 1) += tensor(j, i, 2);
	    	}
	    }
	}

	Vec2f b, v;
	v << 0, 0;
	for (int n = 0; n < max_iters; n++) {
		b << 0, 0;
		Vec2f gv = displacement + v;
		for (float i = point.y() - w; i <= point.y() + w; i++) {
			for (float j = point.x() - w; j <= point.x() + w; j++) {
				float im1_I = Access_Subpixel(im1, j, i);
				float im2_I = Access_Subpixel(im2, j + gv.x(), i + gv.y());
				float dt = im1_I - im2_I;

				if (fast) {
					float dx = (Access_Subpixel(im1, j + 1, i) - Access_Subpixel(im1, j - 1, i)) / 2.0;
					float dy = (Access_Subpixel(im1, j, i + 1) - Access_Subpixel(im1, j, i - 1)) / 2.0;
					b.x() += dx * dt;
					b.y() += dy * dt;
				} else {
					float dx = Access_Subpixel(gx, j, i);
					float dy = Access_Subpixel(gy, j, i);
					b.x() -= dx * dt;
					b.y() -= dy * dt;
				}
			}
		}
		Vec2f off = G.inverse() * b;
		v += off;
		if (off.squaredNorm() < error) break;
	}

	return v;
}

vector<Vec2f> LK_Pyr(const Image &im1, const Image &im2, vector<Vec2f> points, 
						unsigned pyramid_levels, int block_size, float error, int max_iters, bool fast)
{
	vector<Vec2f> of_points;

	Image im1_gray(im1.width(), im1.height(), 1);
	Image im2_gray(im2.width(), im2.height(), 1);
	if (im1.channels() > 1) im1_gray = lumiChromi(im1)[0];
	else im1_gray = im1;
	if (im2.channels() > 1) im2_gray = lumiChromi(im2)[0];
	else im2_gray = im2;

	vector<Image> im1_L{im1_gray};
	vector<Image> im2_L{im2_gray};
	vector<Image> gx_L, gy_L;

	for (unsigned l = 1; l < pyramid_levels; l++) {
		im1_L.push_back(Create_Pyramid(im1_L[l-1]));
		im2_L.push_back(Create_Pyramid(im2_L[l-1]));
	}

	for (unsigned l = 0; l < pyramid_levels; l++) {
		gx_L.push_back(gradientX(gaussianBlur_separable(im1_L[l], 1.0)));
		gy_L.push_back(gradientY(gaussianBlur_separable(im1_L[l], 1.0)));
	}

	for (unsigned n = 0; n < points.size(); n++) {
		// cout << n+1 << " of " << points.size() << endl;
		Vec2f u = points[n];
		Vec2f g;
		g << 0, 0;

		for (int l = (int)pyramid_levels-1; l >= 0; l--) {
			Vec2f u_l = u / pow(2, l);
			Vec2f d = Lucas_Kanade(im1_L[l], im2_L[l], gx_L[l], gy_L[l], u_l, g,
				block_size, error, max_iters, fast);

			if (l != 0) g = 2 * (g + d);
			else g += d;
		}

		of_points.push_back(g);
	}

	return of_points;
}

// Alignment Menthods

Image Align_LK_Harris(Image im1, Image im2, unsigned pyramid_levels, string mode,
						int block_size, float error, int max_iters, bool fast) {
	Image out(im1.width(), im1.height(), im1.channels());
	vector<Point> h1   = HarrisCorners(im1);
	vector<Vec2f> harris_points;
	for(unsigned n = 0; n < h1.size(); n++) {
		harris_points.push_back(Vec2f(h1[n].x, h1[n].y));
	}

	vector<Vec2f> of_points = LK_Pyr(im1, im2, harris_points,
										pyramid_levels, block_size, error, max_iters, fast);

	Vec2f mean_of, median_of;
	int size = of_points.size();
	vector<int> of_x, of_y;
	for (unsigned n = 0; n < of_points.size(); n++) {
		mean_of(0) += of_points[n].x();
		mean_of(1) += of_points[n].y();
		of_x.push_back(of_points[n].x());
		of_y.push_back(of_points[n].y());
	}
	mean_of /= size;

	sort(of_x.begin(), of_x.end());
    if (size % 2 == 0) {
    	median_of(0) = (of_x[size / 2 - 1] + of_x[size / 2]) / 2;
    	median_of(1) = (of_y[size / 2 - 1] + of_y[size / 2]) / 2;
    }
    else {
    	median_of(0) = of_x[size / 2];
    	median_of(1) = of_y[size / 2];
    }

    cout << "Mean: " << mean_of << endl;
    cout << "Median: " << median_of << endl;

    if (mode == "mean") {
		for(int i = 0; i <= out.height(); i++) {
			for(int j = 0; j <= out.width(); j++) {
				if ((i < 0) || (i >= im1.height()) || (j < 0) || (j >= im1.width())) continue;
				for (int k = 0; k < im1.channels(); k++) {
					out(j, i, k) = im2.smartAccessor(j + mean_of.x(), i + mean_of.y(), k);
				}
			}
		}
	} else if (mode == "median") {
		for(int i = 0; i <= out.height(); i++) {
			for(int j = 0; j <= out.width(); j++) {
				if ((i < 0) || (i >= im1.height()) || (j < 0) || (j >= im1.width())) continue;
				for (int k = 0; k < im1.channels(); k++) {
					out(j, i, k) = im2.smartAccessor(j + median_of.x(), i + median_of.y(), k);
				}
			}
		}
	} else {
		if (mode == "overlay") {	// overlay grayscale
			Image im1_gray = lumiChromi(im1)[0];
			for(int i = 0; i < im1.height(); i++) {
				for(int j = 0; j < im1.width(); j++) {
					out(j, i, 0) = im1_gray(j, i);
					out(j, i, 1) = im1_gray(j, i);
					out(j, i, 2) = im1_gray(j, i);
				}
			}
		}

		int stride = 15;	// set to size of window
		unsigned offset = stride / 2;
		for (unsigned n = 0; n < of_points.size(); n++) {
			Vec2f og_point = harris_points[n];
			Vec2f of_point = of_points[n];
			for(int i = og_point.y() - offset; i <= og_point.y() + offset; i++) {
				for(int j = og_point.x() - offset; j <= og_point.x() + offset; j++) {
					if ((i < 0) || (i >= im1.height()) || (j < 0) || (j >= im1.width())) continue;
					for (int k = 0; k < im1.channels(); k++) {
						out(j, i, k) = im2.smartAccessor(j + of_point.x(), i + of_point.y(), k);
					}
				}
			}
		}
	}

	return out;
}

Image Align_LK_Tile(Image im1, Image im2, unsigned pyramid_levels, unsigned stride,
					int block_size, float error, int max_iters, bool fast) {
	Image out(im1.width(), im1.height(), im1.channels());
	vector<Vec2f> sparse_points = Generate_Sparse_Points(im1.width(), im2.height(), stride);
	vector<Vec2f> of_points = LK_Pyr(im1, im2, sparse_points,
										pyramid_levels, block_size, error, max_iters, fast);

	unsigned offset = stride / 2;
	for (unsigned n = 0; n < of_points.size(); n++) {
		Vec2f og_point = sparse_points[n];
		Vec2f of_point = of_points[n];
		for(int i = og_point.y() - offset; i <= og_point.y() + offset; i++) {
			for(int j = og_point.x() - offset; j <= og_point.x() + offset; j++) {
				if ((i < 0) || (i >= im1.height()) || (j < 0) || (j >= im1.width())) continue;
				for (int k = 0; k < im1.channels(); k++) {
					out(j, i, k) = im2.smartAccessor(j + of_point.x(), i + of_point.y(), k);
				}
			}
		}
	}

	return out;
}