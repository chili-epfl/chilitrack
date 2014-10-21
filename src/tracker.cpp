#include <opencv2/features2d.hpp>
#include <opencv2/video.hpp>
#include <opencv2/opencv.hpp>

#include "utils.h" // Drawing and printing functions

#include "chilitrack.h"

using namespace std;
using namespace cv;
using namespace chilitrack;

const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio

const int min_inliers = 10; // Minimal number of inliers to compute transform
const int min_inliers_ratio = 0.5; // Minimal ratio of inliers vs matches to compute transform

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


Tracker::Tracker(Ptr<Feature2D> _detector, 
                 Ptr<DescriptorMatcher> _matcher) :
        detector(_detector),
        matcher(_matcher),
        cameraMatrix(),
        distCoeffs(),
        rotation(3,1),
        translation(3,1)
{
    if (!detector) {
        detector = Feature2D::create("ORB");
        detector->set("nFeatures", 700);
    }
    if(!matcher) {
        matcher = DescriptorMatcher::create("BruteForce-Hamming");
    }

}

map<string, Matx44d> Tracker::estimate(const Mat frame, Ptr<Stats> stats)
{

    map<string, Matx44d> res;

    for (auto& kv : _templates) {
        if (!kv.second->tracked) {
            match(frame, kv.second, stats);
        }
        else {
            auto mat = track(frame, kv.second, stats);
            if (!Mat(mat).empty()) res[kv.first] = mat;
        }
    }

    return res;
}

void Tracker::match(const Mat frame, Ptr<Template> tpl, Ptr<Stats> stats)
{

    if (cameraMatrix.empty()) {
        frameSize = frame.size();
        double focalLength = 700.;
        cameraMatrix = (cv::Mat_<double>(3,3) <<
            focalLength ,            0 , frameSize.width /2,
                    0 ,  focalLength , frameSize.height/2,
                    0,             0 , 1
        );
    }

    auto t1 = getTickCount();

    vector<KeyPoint> kp;
    Mat desc;
    (*detector)(frame, noArray(), kp, desc);

    if (stats) stats->keypoints = (int)kp.size();

    vector< vector<DMatch> > matches;
    vector<KeyPoint> matched1, matched2;
    matcher->knnMatch(tpl->desc, desc, matches, 2);
    for(unsigned i = 0; i < matches.size(); i++) {
        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
            matched1.push_back(tpl->kpts[matches[i][0].queryIdx]);
            matched2.push_back(      kp[matches[i][0].trainIdx]);
        }
    }
    if(stats) stats->matches = (int)matched1.size();

    Mat inlier_mask, homography;
    vector<KeyPoint> inliers1, inliers2;
    vector<DMatch> inlier_matches;
    if(matched1.size() >= 4) {
        homography = findHomography(Points(matched1), Points(matched2),
                                    RANSAC, ransac_thresh, inlier_mask);
    }

    if(matched1.size() < 4 || homography.empty()) {
#ifdef CHILITRACK_DEBUG
        Mat outImg;
        Size img1size = tpl->image().size(), img2size = frame.size();
        Size size( img1size.width + img2size.width, MAX(img1size.height, img2size.height)   );
        _debug.create( size, CV_MAKETYPE(tpl->image().depth(), 3) );
        _debug = Scalar::all(0);
        Mat outImg1 = _debug( Rect(0, 0, img1size.width, img1size.height) );
        Mat outImg2 = _debug( Rect(img1size.width, 0, img2size.width, img2size.height) );
        tpl->debug().copyTo(outImg1);
        frame.copyTo(outImg2);
#endif

        if(stats) {
            stats->inliers = 0;
            stats->ratio = 0;
        }
        return;
    }
    for(unsigned i = 0; i < matched1.size(); i++) {
        if(inlier_mask.at<uchar>(i)) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            inlier_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    if(stats) {
        stats->inliers = (int)inliers1.size();
        stats->ratio = stats->inliers * 1.0 / stats->matches;

        stats->duration = (int)((double)(getTickCount() - t1)/getTickFrequency() * 1000);
    }

    // Project the tpl tracking features onto the scene and enable tracking
    if((float)(inliers1.size()) / matched1.size() >= min_inliers_ratio
        && inliers1.size() >= min_inliers) 
    {
        perspectiveTransform(tpl->tracking_kpts, features, homography);
        prev_frame = frame.clone();
        tpl->tracked = true;
    
        // initially, all tpl features are present
        tpl_points = tpl->tpl_points;
   
        // compute the initial centroid and variance of the feature cloud.
        _centroid = mean(features);
        _variance = variance(features);
    }

#ifdef CHILITRACK_DEBUG
    drawMatches(tpl->debug(), inliers1, frame, inliers2,
                inlier_matches, _debug,
                Scalar(255, 0, 0), Scalar(255, 0, 0));
#endif
}


Matx44d Tracker::track(const Mat frame, Ptr<Template> tpl, Ptr<Stats> stats)
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
        tpl->tracked = false;

#ifdef CHILITRACK_DEBUG
        Mat outImg;
        vector<KeyPoint> kpts_viz;
        KeyPoint::convert(found, kpts_viz);
        drawKeypoints(frame, kpts_viz, outImg, Scalar::all(255));

        _debug = combineImages(tpl->debug(), outImg);
#endif
        return Matx44d();
    }

    auto transformation = computeTransformation(tpl);

    if (stats) stats->duration = (int)((double)(getTickCount() - t1)/getTickFrequency() * 1000);

#ifdef CHILITRACK_DEBUG
    Mat outImg;
    vector<KeyPoint> kpts_viz;
    KeyPoint::convert(found, kpts_viz);
    drawKeypoints(frame, kpts_viz, outImg, Scalar::all(255));

    draw3DAxis(outImg, transformation, cameraMatrix);
    draw3DRect(outImg, tpl->bb, transformation, cameraMatrix);


    _debug = combineImages(tpl->debug(), outImg);
#endif

    return transformation;
}

Matx44d Tracker::computeTransformation(Ptr<Template> tpl) const
{
    // Find the 3D pose of our template
    cv::solvePnPRansac(tpl_points,
                 features,
                 cameraMatrix, distCoeffs,
                 rotation, translation, false,
                 SOLVEPNP_ITERATIVE);

    cv::Matx33d rotMat;
    cv::Rodrigues(rotation, rotMat);

    return {
        rotMat(0,0) , rotMat(0,1) , rotMat(0,2) , translation(0) ,
        rotMat(1,0) , rotMat(1,1) , rotMat(1,2) , translation(1) ,
        rotMat(2,0) , rotMat(2,1) , rotMat(2,2) , translation(2) ,
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

/** Sets new camera calibration values.
 */
void Tracker::setCalibration(cv::InputArray newCameraMatrix,
                    cv::InputArray newDistCoeffs){
    cameraMatrix = newCameraMatrix.getMat();
    distCoeffs = newDistCoeffs.getMat();
}

void Tracker::readCalibration(const std::string &filename) {

    cv::Size size;
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Could not read calibration file " << filename << endl;
        cerr << "Using default camera parameter as fallback." << endl;
        return;
    }

    fs["image_width"]             >> size.width;
    fs["image_height"]            >> size.height;

    if (frameSize.width * frameSize.height != 0
        && size != frameSize) {
        cerr << "The calibration was done for a resolution (" << size 
             << ") that does not match the current one (" << frameSize 
             << ")." <<endl;
        cerr << "Using default camera parameter as fallback." << endl;
        return;
    }

    fs["distortion_coefficients"] >> distCoeffs;
    fs["camera_matrix"]           >> cameraMatrix;

    if( distCoeffs.type() != CV_64F )
        distCoeffs = cv::Mat_<double>(distCoeffs);
    if( cameraMatrix.type() != CV_64F )
        cameraMatrix = cv::Mat_<double>(cameraMatrix);

    cout << "Using camera calibration from file " << filename << "." << endl;
}

