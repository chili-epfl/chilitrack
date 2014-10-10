#include <opencv2/features2d.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#include <iostream>
#include <iomanip>

#include "stats.h" // Stats structure definition
#include "utils.h" // Drawing and printing functions

using namespace std;
using namespace cv;

const double akaze_thresh = 3e-4; // AKAZE detection threshold set to locate about 1000 keypoints
const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
const int bb_min_inliers = 50; // Minimal number of inliers to draw bounding box
const int stats_update_period = 10; // On-screen statistics are updated every 10 frames

Point2f mean(const vector<Point2f>& vals)
{
    size_t nbvals = vals.size();

    auto sum = vals[0];
    for(uint i = 1 ; i < nbvals ; ++i) sum += vals[i];
    return sum * (1.f/nbvals);
}

double variance(const vector<Point2f>& vals)
{
    size_t nbvals = vals.size();

    auto current_mean = mean(vals);

    auto temp = norm(current_mean-vals[0])*norm(current_mean-vals[0]);
    for(uint i = 1 ; i < vals.size() ; ++i)
        temp += norm(current_mean-vals[i])*norm(current_mean-vals[i]);
    return temp/nbvals;
}


class Template
{

public:
    // Nb of features used for tracking the template
    static const int NB_FEATURES = 50;

    Template(cv::Mat tpl, 
             cv::Size size, 
             cv::Ptr<cv::Feature2D> detector);

    cv::Mat image() const {return _tpl;}
    cv::Mat debug() const {return _tpl_debug;}

    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    std::vector<cv::Point2f> tracking_kpts;

    std::vector<cv::Point3f> tpl_points;
    cv::Rect bb;

protected:
    cv::Mat _tpl;
    cv::Mat _tpl_debug;
};

Template::Template(cv::Mat tpl, 
             cv::Size size, 
             cv::Ptr<cv::Feature2D> detector) {


    _tpl = tpl.clone();

    // Extract features and descriptors from the tpl image
    // -------------------------------

    (*detector)(tpl, noArray(), kpts, desc);

    cout << kpts.size() << " keypoints on the template" << endl;

    // Store the template bounding box
    // -------------------------------

    bb = Rect(-size.width / 2, -size.height / 2, size.width, size.height);
 
    // Find good features for tracking
    // -------------------------------

    // cf http://docs.opencv.org/trunk/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    double quality = 0.1;
    // TODO: adapt this euclidian distance to the actual tpl size
    double min_distance = 15;
    tracking_kpts.reserve(NB_FEATURES);
    goodFeaturesToTrack(tpl, tracking_kpts, NB_FEATURES, quality, min_distance);

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

class Tracker
{
public:
    Tracker(Ptr<Feature2D> _detector,
            Ptr<DescriptorMatcher> _matcher,
            Size sceneResolution);

    Mat process(const Mat frame, Ptr<Template> tpl, Stats& stats);
    Mat match(const Mat frame, Ptr<Template> tpl, Stats& stats);
    Mat track(const Mat frame, Ptr<Template> tpl, Stats& stats);
    Ptr<Feature2D> getDetector() {
        return detector;
    }
protected:

    Matx44d computeTransformation(Ptr<Template> tpl) const;

    Ptr<Feature2D> detector;
    Ptr<DescriptorMatcher> matcher;

    std::vector<cv::Point3f> tpl_points;
    std::vector<cv::Point2f> features;

    // when tracking, store the previous frame for optical flow computation
    cv::Mat prev_frame;
    bool tracking_enabled;

    cv::Size cameraResolution;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

private:
    void pruneFeatures(std::vector<cv::Point2f>& features, 
                       std::vector<cv::Point3f>& tpl_points );
    cv::Point2f _centroid;
    double _variance;


};

Tracker::Tracker(Ptr<Feature2D> _detector, 
                 Ptr<DescriptorMatcher> _matcher,
                 Size sceneResolution) :
        detector(_detector),
        matcher(_matcher),
        tracking_enabled(false),
        cameraResolution(sceneResolution),
        cameraMatrix(),
        distCoeffs()
{
    double focalLength = 700.;
    cameraMatrix = (cv::Mat_<double>(3,3) <<
        focalLength ,            0 , cameraResolution.width /2,
                  0 ,  focalLength , cameraResolution.height/2,
                  0,             0 , 1
    );

}

Mat Tracker::process(const Mat frame, Ptr<Template> tpl, Stats& stats)
{
    if (!tracking_enabled)
        return match(frame, tpl, stats);
    else
        return track(frame, tpl, stats);
}

Mat Tracker::match(const Mat frame, Ptr<Template> tpl, Stats& stats)
{
    auto t1 = getTickCount();

    vector<KeyPoint> kp;
    Mat desc;
    (*detector)(frame, noArray(), kp, desc);
    stats.keypoints = (int)kp.size();

    vector< vector<DMatch> > matches;
    vector<KeyPoint> matched1, matched2;
    matcher->knnMatch(tpl->desc, desc, matches, 2);
    for(unsigned i = 0; i < matches.size(); i++) {
        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
            matched1.push_back(tpl->kpts[matches[i][0].queryIdx]);
            matched2.push_back(      kp[matches[i][0].trainIdx]);
        }
    }
    stats.matches = (int)matched1.size();

    Mat inlier_mask, homography;
    vector<KeyPoint> inliers1, inliers2;
    vector<DMatch> inlier_matches;
    if(matched1.size() >= 4) {
        homography = findHomography(Points(matched1), Points(matched2),
                                    RANSAC, ransac_thresh, inlier_mask);
    }

    if(matched1.size() < 4 || homography.empty()) {
        Mat res;

        Mat outImg;
        Size img1size = tpl->image().size(), img2size = frame.size();
        Size size( img1size.width + img2size.width, MAX(img1size.height, img2size.height)   );
        res.create( size, CV_MAKETYPE(tpl->image().depth(), 3) );
        res = Scalar::all(0);
        Mat outImg1 = res( Rect(0, 0, img1size.width, img1size.height) );
        Mat outImg2 = res( Rect(img1size.width, 0, img2size.width, img2size.height) );
        tpl->debug().copyTo(outImg1);
        frame.copyTo(outImg2);




        stats.inliers = 0;
        stats.ratio = 0;
        return res;
    }
    for(unsigned i = 0; i < matched1.size(); i++) {
        if(inlier_mask.at<uchar>(i)) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            inlier_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    stats.inliers = (int)inliers1.size();
    stats.ratio = stats.inliers * 1.0 / stats.matches;

    stats.duration = (int)((double)(getTickCount() - t1)/getTickFrequency() * 1000);

    // Project the tpl tracking features onto the scene and enable tracking
    if(stats.inliers >= bb_min_inliers) {
        perspectiveTransform(tpl->tracking_kpts, features, homography);
        prev_frame = frame.clone();
        tracking_enabled = true;
    
        // initially, all tpl features are present
        tpl_points = tpl->tpl_points;
   
        // compute the initial centroid and variance of the feature cloud.
        _centroid = mean(features);
        _variance = variance(features);
    }

    // Debugging
    Mat res;
    drawMatches(tpl->debug(), inliers1, frame, inliers2,
                inlier_matches, res,
                Scalar(255, 0, 0), Scalar(255, 0, 0));
    return res;
}


Mat Tracker::track(const Mat frame, Ptr<Template> tpl, Stats& stats)
{
    auto t1 = getTickCount();

    vector<Point2f> next_features;
    next_features.reserve(Template::NB_FEATURES);
    vector<unsigned char> status;
    vector<float> err;

    calcOpticalFlowPyrLK(prev_frame, frame, 
                         features, next_features, 
                         status, err, 
                         Size(10,10), 3, 
                         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 20, 0.01));

    prev_frame = frame.clone();

    vector<Point3f> remaining_tpl_features;
    vector<Point2f> found;
    found.reserve(Template::NB_FEATURES);

    for ( size_t i = 0 ; i < features.size() ; i++ ) {
        if (status[i] == 1) {
            remaining_tpl_features.push_back(tpl_points[i]);
            found.push_back(next_features[i]);
        }
        else {
        }
    }

    pruneFeatures(found, remaining_tpl_features);

    // update the centroid and variance.
    // The centroid may have changed if the shape is rotated for instance
    // The variance will change when scaling the shape
    _centroid = mean(found);
    _variance = variance(found);

    tpl_points = remaining_tpl_features;
    features = found;
    
    // If too many features are lost, leave tracking and go back to match
    if (found.size() < Template::NB_FEATURES / 2) {
        tracking_enabled = false;

        // Debug
        // -----

        Mat outImg;
        vector<KeyPoint> kpts_viz;
        KeyPoint::convert(found, kpts_viz);
        drawKeypoints(frame, kpts_viz, outImg, Scalar::all(255));

        return combineImages(tpl->debug(), outImg);
    }

    auto transformation = computeTransformation(tpl);

    stats.duration = (int)((double)(getTickCount() - t1)/getTickFrequency() * 1000);

    // Debug
    // -----

    Mat outImg;
    vector<KeyPoint> kpts_viz;
    KeyPoint::convert(found, kpts_viz);
    drawKeypoints(frame, kpts_viz, outImg, Scalar::all(255));

    draw3DAxis(outImg, transformation, cameraMatrix);
    draw3DRect(outImg, tpl->bb, transformation, cameraMatrix);


    return combineImages(tpl->debug(), outImg);
}

Matx44d Tracker::computeTransformation(Ptr<Template> tpl) const
{
    // Rotation & translation vectors, computed by cv::solvePnP
    cv::Mat rotation, translation;

    // Find the 3D pose of our template
    cv::solvePnP(tpl_points,
                 features,
                 cameraMatrix, distCoeffs,
                 rotation, translation, false,
                 SOLVEPNP_ITERATIVE);

    cv::Matx33d rotMat;
    cv::Rodrigues(rotation, rotMat);

    return {
        rotMat(0,0) , rotMat(0,1) , rotMat(0,2) , translation.at<double>(0) ,
        rotMat(1,0) , rotMat(1,1) , rotMat(1,2) , translation.at<double>(1) ,
        rotMat(2,0) , rotMat(2,1) , rotMat(2,2) , translation.at<double>(2) ,
                  0 ,           0 ,           0 ,                         1 ,
    };
}

/**
 * Only keep features 'close enough' to the feature cloud centroid.
 * 'close' is dependent on the initial variance of the cloud.
 */
void Tracker::pruneFeatures(vector<Point2f>& features, vector<Point3f>& tpl_points ) {

    vector<Point2f> prunedFeatures;
    vector<Point3f> prunedTplPoints;

    for ( size_t i = 0 ; i < features.size() ; i++ ) {
        auto p = features[i];
        if (pow(norm(p - _centroid),2) < 3 * _variance) {
            prunedFeatures.push_back(p);
            prunedTplPoints.push_back(tpl_points[i]);
        }
    }

    features = prunedFeatures;
    tpl_points = prunedTplPoints;

}
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////



int main(int argc, char **argv)
{
    namedWindow("tracking");

    if(argc < 2) {
        cerr << "Usage: " << endl <<
                "tracker template_path" << endl;
        return 1;
    }
    VideoCapture video_in(1);

    if(!video_in.isOpened()) {
        cerr << "Couldn't open camera" << endl;
        return 1;
    }

    double inputWidth  = video_in.get(cv::CAP_PROP_FRAME_WIDTH);
    double inputHeight = video_in.get(cv::CAP_PROP_FRAME_HEIGHT);

    Mat image_object = imread(argv[1], IMREAD_GRAYSCALE);
    if( !image_object.data )
          { cerr << " --(!) Error reading object image " << std::endl; return -1; }


    Stats stats, global_stats;
    Mat frame;

    //Ptr<Feature2D> detector = Feature2D::create("AKAZE");
    //detector->set("threshold", akaze_thresh);
    Ptr<Feature2D> detector = Feature2D::create("ORB");
    detector->set("nFeatures", 500); // default: 500

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    Tracker tracker(detector, matcher, Size(inputWidth, inputHeight));

    auto tpl = makePtr<Template>(Template(image_object, Size(210, 297), detector));


    Stats draw_stats;
    Mat res_frame;
    int frame_count = 0;

    for(; ; frame_count++) {
        bool update_stats = (frame_count % stats_update_period == 0);
        video_in >> frame;

        res_frame = tracker.process(frame, tpl, stats);
        global_stats += stats;
        if(update_stats) {
            draw_stats = stats;
        }

        drawStatistics(res_frame, draw_stats);

        imshow("tracking", res_frame);
        if (waitKey(10) >= 0) break;
    }
    global_stats /= frame_count;
    printStatistics("Tracker", stats);
    return 0;
}
