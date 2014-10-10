#ifndef _CHILITRACK_H
#define _CHILITRACK_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "stats.h" // Stats structure definition

namespace chilitrack {

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

class Tracker
{
public:
    Tracker(cv::Ptr<cv::Feature2D> _detector,
            cv::Ptr<cv::DescriptorMatcher> _matcher,
            cv::Size sceneResolution);

    cv::Mat process(const cv::Mat frame, cv::Ptr<Template> tpl, Stats& stats);
    cv::Ptr<cv::Feature2D> getDetector() {return detector;}

protected:
    cv::Mat match(const cv::Mat frame, cv::Ptr<Template> tpl, Stats& stats);
    cv::Mat track(const cv::Mat frame, cv::Ptr<Template> tpl, Stats& stats);


    cv::Matx44d computeTransformation(cv::Ptr<Template> tpl) const;

    cv::Ptr<cv::Feature2D> detector;
    cv::Ptr<cv::DescriptorMatcher> matcher;

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

};

#endif // _CHILITRACK_H
