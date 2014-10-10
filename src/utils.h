#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core.hpp>
#include <vector>
#include "stats.h"

using namespace std;
using namespace cv;

void drawBoundingBox(Mat image, vector<Point2f> bb);
void drawStatistics(Mat image, const Stats& stats);
void printStatistics(string name, Stats stats);
vector<Point2f> Points(vector<KeyPoint> keypoints);

void drawBoundingBox(Mat image, vector<Point2f> bb)
{
    for(unsigned i = 0; i < bb.size() - 1; i++) {
        line(image, bb[i], bb[i + 1], Scalar(0, 0, 255), 2);
    }
    line(image, bb[bb.size() - 1], bb[0], Scalar(0, 0, 255), 2);
}

// by Quentin Bonnard
void draw3DAxis(Mat image, Matx44d transformation, Mat cameraMatrix)
{

    static const float DEFAULT_SIZE = 20.f;
    static const cv::Vec4d UNITS[4] {
        {0.f, 0.f, 0.f, 1.f},
            {DEFAULT_SIZE, 0.f, 0.f, 1.f},
            {0.f, DEFAULT_SIZE, 0.f, 1.f},
            {0.f, 0.f, DEFAULT_SIZE, 1.f},
    };

    Mat projectionMat = Mat::zeros(4,4,CV_64F);
    cameraMatrix.copyTo(projectionMat(cv::Rect(0,0,3,3)));
    Matx44d projection = projectionMat;
    projection(3,2) = 1;

    cv::Vec4f referential[4] = {
        projection*transformation*UNITS[0],
        projection*transformation*UNITS[1],
        projection*transformation*UNITS[2],
        projection*transformation*UNITS[3],
    };

    std::vector<cv::Point2f> t2DPoints;
    for (auto homogenousPoint : referential)
        t2DPoints.push_back(cv::Point2f(
                    homogenousPoint[0]/homogenousPoint[3],
                    homogenousPoint[1]/homogenousPoint[3]));

    static const int SHIFT = 16;
    static const float PRECISION = 1<<SHIFT;
    static const std::string AXIS_NAMES[3] = { "x", "y", "z" };
    static const cv::Scalar AXIS_COLORS[3] = {
        {0,0,255},{0,255,0},{255,0,0},
    };
    for (int i : {1,2,3}) {
        cv::line(
                image,
                PRECISION*t2DPoints[0],
                PRECISION*t2DPoints[i],
                AXIS_COLORS[i-1],
                1, LINE_AA, SHIFT);
        cv::putText(image, AXIS_NAMES[i-1], t2DPoints[i],
                cv::FONT_HERSHEY_SIMPLEX, 0.5, AXIS_COLORS[i-1]);
    }
}

void draw3DRect(Mat image, Rect rect, Matx44d transformation, Mat cameraMatrix)
{

    static const cv::Vec4d RECT[4] {
            {(float) rect.tl().x, (float) rect.tl().y, 0.f, 1.f},
            {(float) rect.tl().x, (float) rect.br().y, 0.f, 1.f},
            {(float) rect.br().x, (float) rect.br().y, 0.f, 1.f},
            {(float) rect.br().x, (float) rect.tl().y, 0.f, 1.f},
    };

    Mat projectionMat = Mat::zeros(4,4,CV_64F);
    cameraMatrix.copyTo(projectionMat(cv::Rect(0,0,3,3)));
    Matx44d projection = projectionMat;
    projection(3,2) = 1;

    cv::Vec4f referential[4] = {
        projection*transformation*RECT[0],
        projection*transformation*RECT[1],
        projection*transformation*RECT[2],
        projection*transformation*RECT[3],
    };

    std::vector<cv::Point2f> t2DPoints;
    for (auto homogenousPoint : referential)
        t2DPoints.push_back(cv::Point2f(
                    homogenousPoint[0]/homogenousPoint[3],
                    homogenousPoint[1]/homogenousPoint[3]));

    static const int SHIFT = 16;
    static const float PRECISION = 1<<SHIFT;
    for (int i : {0,1,2,3}) {
        cv::line(
                image,
                PRECISION*t2DPoints[i],
                PRECISION*t2DPoints[(i + 1) % 4],
                {255, 128, 0},
                1, LINE_AA, SHIFT);
    }
}

Mat combineImages(Mat tpl, Mat frame) {

    Mat res;

    Size img1size = tpl.size(), img2size = frame.size();
    Size size( img1size.width + img2size.width, MAX(img1size.height, img2size.height)   );
    res.create( size, CV_MAKETYPE(tpl.depth(), 3) );
    res = Scalar::all(0);
    Mat outImg1 = res( Rect(0, 0, img1size.width, img1size.height) );
    Mat outImg2 = res( Rect(img1size.width, 0, img2size.width, img2size.height) );
    tpl.copyTo(outImg1);
    frame.copyTo(outImg2);

    return res;
}

void drawStatistics(Mat image, const Stats& stats)
{
    static const int font = FONT_HERSHEY_PLAIN;
    stringstream str0, str1, str2, str3, str4;

    str0 << "Keypoints: " << stats.keypoints;
    str1 << "Matches: " << stats.matches;
    str2 << "Inliers: " << stats.inliers;
    str3 << "Inlier ratio: " << setprecision(2) << stats.ratio;
    str4 << "Process. time: " << stats.duration;

    putText(image, str0.str(), Point(0, image.rows - 120), font, 2, Scalar::all(255), 3);
    putText(image, str1.str(), Point(0, image.rows - 90), font, 2, Scalar::all(255), 3);
    putText(image, str2.str(), Point(0, image.rows - 60), font, 2, Scalar::all(255), 3);
    putText(image, str3.str(), Point(0, image.rows - 30), font, 2, Scalar::all(255), 3);
    putText(image, str4.str(), Point(0, image.rows - 0), font, 2, Scalar::all(255), 3);
}

void printStatistics(string name, Stats stats)
{
    cout << name << endl;
    cout << "----------" << endl;

    cout << "Matches " << stats.matches << endl;
    cout << "Inliers " << stats.inliers << endl;
    cout << "Inlier ratio " << setprecision(2) << stats.ratio << endl;
    cout << "Keypoints " << stats.keypoints << endl;
    cout << endl;
}

vector<Point2f> Points(vector<KeyPoint> keypoints)
{
    vector<Point2f> res;
    for(unsigned i = 0; i < keypoints.size(); i++) {
        res.push_back(keypoints[i].pt);
    }
    return res;
}


#endif // UTILS_H
