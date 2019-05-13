#include <iostream>

#include "a10.h"
#include "basicImageManipulation.h"

using namespace std;

// Function tests
void Tensor_Test() {
    Image im("./Input/green/noise-small-1.png");

    Vec2f point = Vec2f(281, 408);

    int block_size = 101;
    int w = (int)((block_size - 1.0) / 2.0);
    Image window = Image(block_size, block_size, 3);
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            float x = point.x() - w + j;
            float y = point.y() - w + i;

            for (int k = 0; k < im.channels(); k++) {
                window(j, i, k) = im(x, y, k);
            }
        }
    }

    window.write("./Output/green_window.png");

    Image gx = gradientX(gaussianBlur_separable(lumiChromi(im)[0], 1.0));
    Image gy = gradientY(gaussianBlur_separable(lumiChromi(im)[0], 1.0));

    gx.write("./Output/green_dx.png");
    gy.write("./Output/green_dy.png");

    Image tensor = Compute_Tensor_Window(gx, gy, point, block_size);
    float maxi = tensor.max();
    if(maxi != 0) {
        tensor = tensor / maxi ;
        tensor.write("./Output/green_tensor.png");
    }
}

void Sparse_Points_Test() {
    unsigned width = 80;
    unsigned height = 100;
    unsigned stride = 40;
    vector<Vec2f> ground_truth{Vec2f(20, 20), 
                               Vec2f(60, 20),
                               Vec2f(20, 60),
                               Vec2f(60, 60)};
    vector<Vec2f> points = Generate_Sparse_Points(width, height, stride);

    assert(ground_truth.size() == points.size());

    for (unsigned n = 0; n < points.size(); n++) {
        assert(points[n] == ground_truth[n]);
    }
}

void Pyramid_Test() {
    Image im1("./Input/tum/xyz-1.png");

    unsigned pyramid_levels = 4;
    Image im1_gray(im1.width(), im1.height(), 1);
    if (im1.channels() > 1) im1_gray = lumiChromi(im1)[0];
    else im1_gray = im1;

    vector<Image> im1_L{im1_gray};

    for (unsigned l = 1; l < pyramid_levels; l++) {
        im1_L.push_back(Create_Pyramid(im1_L[l-1]));
    }

    for (unsigned l = 0; l < pyramid_levels; l++) {
        im1_L[l].write("./Output/tum_pyramid_" + std::to_string(l) + ".png");
        (gradientX(gaussianBlur_separable(im1_L[l], 1.0))+0.5).write(
                "./Output/tum_pyramid_gradient_x_" + std::to_string(l) + ".png");
        (gradientY(gaussianBlur_separable(im1_L[l], 1.0))+0.5).write(
                "./Output/tum_pyramid_gradient_y_" + std::to_string(l) + ".png");
    }
}

// NO Pyramid in this test
void LK_Basic_Shift_Test() {
    Image im1("./Input/alignment/align_rect.png");
    Image im2("./Input/alignment/align_rect_close.png");

    Image gx = gradientX(gaussianBlur_separable(lumiChromi(im1)[0], 1.0));
    Image gy = gradientY(gaussianBlur_separable(lumiChromi(im1)[0], 1.0));

    // Bottom left corner of rectangle
    Vec2f point(40, 50);

    Vec2f optical_flow = Lucas_Kanade(im1, im2, gx, gy, point, Vec2f(0, 0));
    assert(round(optical_flow.x()) == 2);
    assert(round(optical_flow.y()) == 2);
}

void LK_Pyr_Shift_Close_X() {
    Image im1("./Input/alignment/align_square.png");
    Image im2("./Input/alignment/align_square_close_x.png");

    Image gx = gradientX(gaussianBlur_separable(lumiChromi(im1)[0], 1.0));
    Image gy = gradientY(gaussianBlur_separable(lumiChromi(im1)[0], 1.0));

    // Bottom left corner of rectangle
    vector<Vec2f> point{Vec2f(40, 50)};

    vector<Vec2f> flow = LK_Pyr(im1, im2, point);
    assert(round(flow[0].x()) == 3);
    assert(round(flow[0].y()) == 0);
}

void LK_Pyr_Shift_Far_X() {
    Image im1("./Input/alignment/align_square.png");
    Image im2("./Input/alignment/align_square_far_x.png");

    Image gx = gradientX(gaussianBlur_separable(lumiChromi(im1)[0], 1.0));
    Image gy = gradientY(gaussianBlur_separable(lumiChromi(im1)[0], 1.0));

    // Bottom left corner of rectangle
    vector<Vec2f> point{Vec2f(40, 50)};

    vector<Vec2f> flow = LK_Pyr(im1, im2, point);
    assert(round(flow[0].x()) == 16);
    assert(round(flow[0].y()) == 0);
}

void LK_Pyr_Shift_Close_Y() {
    Image im1("./Input/alignment/align_square.png");
    Image im2("./Input/alignment/align_square_close_y.png");

    Image gx = gradientX(gaussianBlur_separable(lumiChromi(im1)[0], 1.0));
    Image gy = gradientY(gaussianBlur_separable(lumiChromi(im1)[0], 1.0));

    // Bottom left corner of rectangle
    vector<Vec2f> point{Vec2f(40, 50)};

    vector<Vec2f> flow = LK_Pyr(im1, im2, point);
    assert(round(flow[0].x()) == 0);
    assert(round(flow[0].y()) == 2);
}

void LK_Pyr_Shift_Far_Y() {
    Image im1("./Input/alignment/align_square.png");
    Image im2("./Input/alignment/align_square_far_y.png");

    Image gx = gradientX(gaussianBlur_separable(lumiChromi(im1)[0], 1.0));
    Image gy = gradientY(gaussianBlur_separable(lumiChromi(im1)[0], 1.0));

    // Bottom left corner of rectangle
    vector<Vec2f> point{Vec2f(40, 50)};

    vector<Vec2f> flow = LK_Pyr(im1, im2, point);
    assert(round(flow[0].x()) == 0);
    assert(round(flow[0].y()) == 15);
}

void Align_Green_Tile() {
	Image im1("./Input/green/noise-small-1.png");
    Image im2("./Input/green/noise-small-2.png");

    Image aligned = Align_LK_Tile(im1, im2);
    aligned.write("./Output/align_green_tile.png");
}

void Align_Green_Harris() {
    Image im1("./Input/green/noise-small-1.png");
    Image im2("./Input/green/noise-small-2.png");

    Image aligned = Align_LK_Harris(im1, im2, 4, "median");
    aligned.write("./Output/align_green_harris.png");
}

void Align_Green_Harris_Tiles() {
    Image im1("./Input/green/noise-small-1.png");
    Image im2("./Input/green/noise-small-2.png");

    Image aligned = Align_LK_Harris(im1, im2, 4);
    aligned.write("./Output/align_green_harris_tiles.png");
}

void Align_Tum_Tile() {
    Image im1("./Input/tum/xyz-shift-1.png");
    Image im2("./Input/tum/xyz-shift-2.png");

    Image aligned = Align_LK_Tile(im1, im2);
    aligned.write("./Output/align_tum_tile.png");
}

void Align_Tum_Harris() {
    Image im1("./Input/tum/xyz-shift-1.png");
    Image im2("./Input/tum/xyz-shift-2.png");

    Image aligned = Align_LK_Harris(im1, im2, 4, "median");
    aligned.write("./Output/align_tum_harris.png");
}

void Align_Tum_Harris_Tiles() {
    Image im1("./Input/tum/xyz-shift-1.png");
    Image im2("./Input/tum/xyz-shift-2.png");

    Image aligned = Align_LK_Harris(im1, im2, 4);
    aligned.write("./Output/align_tum_harris_tiles.png");
}



void Align_Harris() {
	// Image im1 = Image("./Input/green/noise-small-1.png");
	// Image im2 = Image("./Input/green/noise-small-2.png");

	// Image im1 = Image("./Input/tum/xyz_shift_1.png");
	// Image im2 = Image("./Input/tum/xyz_shift_2.png");

    // Image im1 = Image("./Input/test/source_1.png");
    // Image im2 = Image("./Input/test/source_2.png");

    Image im1 = Image("./Input/test/pru.png");
    Image im2 = Image("./Input/test/pru3.png");

    // Image im1 = Image("./Input/alignment/align1.png");
    // Image im2 = Image("./Input/alignment/align4.png");

    Image aligned = Align_LK_Harris(im1, im2, 4, "median");
    aligned.debug_write();
}

int main()
{
    // Intermediate functions test
    // Tensor_Test();
    // Sparse_Points_Test();
    // Pyramid_Test();
    // LK_Basic_Shift_Test();
    // LK_Pyr_Shift_Close_X();
    // LK_Pyr_Shift_Far_X();
    // LK_Pyr_Shift_Close_Y();
    // LK_Pyr_Shift_Far_Y();

    // Examples
    // Align_Green_Tile();
    // Align_Green_Harris();
    // Align_Green_Harris_Tiles();

    Align_Tum_Tile();
    Align_Tum_Harris();
    Align_Tum_Harris_Tiles();

    // Align_Test();
    // Align_Harris();
    return EXIT_SUCCESS;
}
