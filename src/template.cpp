#include "opencv2/imgproc.hpp"

#include "chilitrack.h"

using namespace std;
using namespace cv;
using namespace chilitrack;

void erosion_mask(Mat src, OutputArray out, int erosion_size )
{
    Mat bw;

    if (src.channels() == 1) {// greyscale
        // ** assumes the background is white! **
        threshold(src, bw, 250, 255, THRESH_BINARY );
        bitwise_not(bw,bw);
    }
    else if (src.channels() == 3) { // BGR
        // ** assumes the background is white! **
        cvtColor(src, bw, COLOR_BGR2GRAY);
        threshold(bw, bw, 250, 255, THRESH_BINARY);
        bitwise_not(bw,bw);
    }
    else { // BGRA
        Mat channels[4];
        split(src, channels);
        bw = channels[3];
    }


    Mat element = getStructuringElement( MORPH_ELLIPSE,
            Size( 2*erosion_size + 1, 2*erosion_size+1 ),
            Point( erosion_size, erosion_size ) );

    /// Apply the erosion operation
    erode( bw, out, element );

}

Template::Template(Mat tpl, 
                   Size size, 
                   Ptr<Feature2D> detector) :
        tracked(false)
    {

    _tpl = tpl.clone();

    if (!detector) {
        detector = Feature2D::create("ORB");
        detector->set("nFeatures", 700);
    }

    // Extract features and descriptors from the tpl image
    // -------------------------------
    (*detector)(tpl, noArray(), kpts, desc);

    cout << kpts.size() << " keypoints on the template" << endl;

    // Store the template bounding box
    // -------------------------------

    bb = Rect(-size.width / 2, -size.height / 2, size.width, size.height);
 
    // Find good features for tracking
    // -------------------------------

    // Compute an eroded mask to obtain features only *inside*
    // the template
    Mat mask;
    erosion_mask(_tpl, mask, 4);

    // cf http://docs.opencv.org/trunk/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    double quality = 0.01;
    // TODO: adapt this euclidian distance to the actual tpl size
    double min_distance = 10;
    tracking_kpts.reserve(NB_FEATURES);
    goodFeaturesToTrack(tpl, tracking_kpts, NB_FEATURES, quality, min_distance, mask);
    
    if (tracking_kpts.size() < NB_FEATURES) {
        cerr << "Attention: found " << tracking_kpts.size() << " good features to track" 
        << " while " << NB_FEATURES << " were desired... Check template.cpp" << endl;
    }

    vector<KeyPoint> kpts_viz;
    KeyPoint::convert(tracking_kpts, kpts_viz);
    drawKeypoints(_tpl, kpts_viz, _tpl_debug);

    // Store the physical location of the tracking features 
    // ----------------------------------------------------

    // assume (0,0) at the center of the template
    tpl_points.reserve(NB_FEATURES);

    int physical_width = size.width; int physical_height = size.height;
    int width = tpl.cols; int height = tpl.rows;

    float xratio = (float) physical_width/width;
    float yratio = (float) physical_height/height;

    for (auto pt : tracking_kpts) {
        tpl_points.push_back(
                    Point3f(pt.x * xratio - physical_width / 2,
                            pt.y * yratio - physical_height / 2,
                            0.f)
                    );
    }
}

