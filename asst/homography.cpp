#include "homography.h"
#include "matrix.h"

using namespace std;


void applyHomography(const Image &source, const Matrix &H, Image &out, bool bilinear) {
    // // --------- HANDOUT  PS06 ------------------------------
    // Transform image source using the homography H, and composite in onto out.
    // if bilinear == true, using bilinear interpolation. Use nearest neighbor
    // otherwise.
    for (int i = 0; i < out.height(); i++) {
        for (int j = 0; j < out.width(); j++) {
            Vec3f out_point(j, i, 1);
            Vec3f transformed_point = H.inverse() * out_point;
            float w = transformed_point.z();
            float source_x = transformed_point.x() / w;
            float source_y = transformed_point.y() / w;

            if ((source_x >= 0) && (source_x < source.width()) &&
                (source_y >= 0) && (source_y < source.height())) {
                for (int k = 0; k < out.channels(); k++) {
                    if (bilinear) {
                        out(j, i, k) = interpolateLin(source, source_x, source_y, k, true);
                    } else {
                        out(j, i, k) = source.smartAccessor(round(source_x), round(source_y), k, true);
                    }
                }
            }
        }
    }
}

Matrix computeHomographyEquations(CorrespondencePair points) {
    Matrix eq(2, 8);
    float x = points.point1.x();
    float y = points.point1.y();
    float xp = points.point2.x();
    float yp = points.point2.y();

    eq(0, 0) = x;
    eq(0, 1) = y;
    eq(0, 2) = 1;
    eq(0, 3) = 0;
    eq(0, 4) = 0;
    eq(0, 5) = 0;
    eq(0, 6) = -xp*x;
    eq(0, 7) = -xp*y;

    eq(1, 0) = 0;
    eq(1, 1) = 0;
    eq(1, 2) = 0;
    eq(1, 3) = x;
    eq(1, 4) = y;
    eq(1, 5) = 1;
    eq(1, 6) = -yp*x;
    eq(1, 7) = -yp*y;

    return eq;
}

Matrix computeHomography(const CorrespondencePair correspondences[4]) {
    // --------- HANDOUT  PS06 ------------------------------
    // Compute a homography from 4 point correspondences.

    Matrix A(8, 8);
    Matrix b(8, 1);


    vector<Matrix> eqs_vec;
    for (unsigned i = 0; i < 4; i++) {
        CorrespondencePair correspondence = correspondences[i];
        Matrix eqs = computeHomographyEquations(correspondence);
        b(2 * i, 0) = correspondence.point2.x();
        b(2 * i + 1, 0) = correspondence.point2.y();
        eqs_vec.push_back(eqs);
    }

    A << eqs_vec[0], eqs_vec[1], eqs_vec[2], eqs_vec[3];
    Matrix values = A.inverse() * b;

    Matrix H(3, 3);
    H(0, 0) = values(0, 0);
    H(0, 1) = values(1, 0);
    H(0, 2) = values(2, 0);
    H(1, 0) = values(3, 0);
    H(1, 1) = values(4, 0);
    H(1, 2) = values(5, 0);
    H(2, 0) = values(6, 0);
    H(2, 1) = values(7, 0);
    H(2, 2) = 1;

    return H;
}


BoundingBox computeTransformedBBox(int imwidth, int imheight, Matrix H) {
    // --------- HANDOUT  PS06 ------------------------------
    // Predict the bounding boxes that encompasses all the transformed
    // coordinates for pixels frow and Image with size (imwidth, imheight)
    Vec3f ul(0, 0, 1);
    Vec3f ur(imwidth, 0, 1);
    Vec3f bl(0, imheight, 1);
    Vec3f br(imwidth, imheight, 1);

    Vec3f transformed_point_ul = H * ul;
    float w_ul = transformed_point_ul.z();
    float x_ul = transformed_point_ul.x() / w_ul;
    float y_ul = transformed_point_ul.y() / w_ul;

    Vec3f transformed_point_ur = H * ur;
    float w_ur = transformed_point_ur.z();
    float x_ur = transformed_point_ur.x() / w_ur;
    float y_ur = transformed_point_ur.y() / w_ur;

    Vec3f transformed_point_bl = H * bl;
    float w_bl = transformed_point_bl.z();
    float x_bl = transformed_point_bl.x() / w_bl;
    float y_bl = transformed_point_bl.y() / w_bl;

    Vec3f transformed_point_br = H * br;
    float w_br = transformed_point_br.z();
    float x_br = transformed_point_br.x() / w_br;
    float y_br = transformed_point_br.y() / w_br;

    return BoundingBox(min(x_ul,x_bl), max(x_ur,x_br), min(y_ul,y_ur), max(y_bl,y_br));
}


BoundingBox bboxUnion(BoundingBox B1, BoundingBox B2) {
    // --------- HANDOUT  PS06 ------------------------------
    // Compute the bounding box that tightly bounds the union of B1 an B2.
    return BoundingBox(min(B1.x1,B2.x1), max(B1.x2,B2.x2), min(B1.y1,B2.y1), max(B1.y2,B2.y2));

}


Matrix makeTranslation(BoundingBox B) {
    // --------- HANDOUT  PS06 ------------------------------
    // Compute a translation matrix (as a homography matrix) that translates the
    // top-left corner of B to (0,0).
    Matrix T = Matrix::Identity(3,3);
    T(0,2) = -B.x1;
    T(1,2) = -B.y1;
    return T;
}


Image stitch(const Image &im1, const Image &im2, const CorrespondencePair correspondences[4]) {
    // --------- HANDOUT  PS06 ------------------------------
    // Transform im1 to align with im2 according to the set of correspondences.
    // make sure the union of the bounding boxes for im2 and transformed_im1 is
    // translated properly (use makeTranslation)
    Matrix H = computeHomography(correspondences);
    BoundingBox BB1 = computeTransformedBBox(im1.width(), im1.height(), H);
    BoundingBox BB2 = BoundingBox(0,im2.width()-1,0,im2.height()-1);
    BoundingBox BBU = bboxUnion(BB1, BB2);
    Matrix T = makeTranslation(BBU);

    Image out(BBU.x2-BBU.x1, BBU.y2-BBU.y1, im1.channels());
    applyHomography(im1, T*H, out, true);
    applyHomography(im2, T, out, true);

    return out;

}

// debug-useful
Image drawBoundingBox(const Image &im, BoundingBox bbox) {
    // // --------- HANDOUT  PS06 ------------------------------
    /*
      ________________________________________
     / Draw me a bounding box!                \
     |                                        |
     | "I jumped to my                        |
     | feet, completely thunderstruck. I      |
     | blinked my eyes hard. I looked         |
     | carefully all around me. And I saw a   |
     | most extraordinary small person, who   |
     | stood there examining me with great    |
     | seriousness."                          |
     \              Antoine de Saint-Exupery  /
      ----------------------------------------
             \   ^__^
              \  (oo)\_______
                 (__)\       )\/\
                     ||----w |
                     ||     ||
    */
    Image out = im;
    out.create_line(bbox.x1, bbox.y1, bbox.x2, bbox.y1, 0, 1, 0);
    out.create_line(bbox.x1, bbox.y1, bbox.x1, bbox.y2, 0, 1, 0);
    out.create_line(bbox.x2, bbox.y1, bbox.x2, bbox.y2, 0, 1, 0);
    out.create_line(bbox.x1, bbox.y2, bbox.x2, bbox.y2, 0, 1, 0);
    return out;
}

void applyHomographyFast(const Image &source, const Matrix &H, Image &out, bool bilinear) {
    // // --------- HANDOUT  PS06 ------------------------------
    // Same as apply but change only the pixels of out that are within the
    // predicted bounding box (when H maps source to its new position).
    BoundingBox BB = computeTransformedBBox(source.width(), source.height(), H);
    for (int i = BB.y1; i <= BB.y2; i++) {
        for (int j = BB.x1; j <= BB.x2; j++) {
            Vec3f out_point(j, i, 1);
            Vec3f transformed_point = H.inverse() * out_point;
            float w = transformed_point.z();
            float source_x = transformed_point.x() / w;
            float source_y = transformed_point.y() / w;

            if ((source_x >= 0) && (source_x < source.width()) &&
                (source_y >= 0) && (source_y < source.height())) {
                for (int k = 0; k < out.channels(); k++) {
                    if (bilinear) {
                        out(j, i, k) = interpolateLin(source, source_x, source_y, k, true);
                    } else {
                        out(j, i, k) = source.smartAccessor(source_x, source_y, k, true);
                    }
                }
            }
        }
    }
}
