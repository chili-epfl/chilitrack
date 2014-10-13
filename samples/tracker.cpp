#include <opencv2/videoio.hpp>
#include <iostream>
#include <iomanip>

#include "chilitrack.h"

using namespace std;
using namespace cv;
using namespace chilitrack;

const int stats_update_period = 10; // On-screen statistics are updated every 10 frames

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


int main(int argc, char **argv)
{
    namedWindow("tracking");

    if(argc < 2) {
        cerr << "Usage: " << endl <<
                "tracker template_path [calibration file]" << endl;
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

    if(argc == 3) {
        tracker.readCalibration(argv[2]);
    }

    auto tpl = makePtr<Template>(Template(image_object, Size(200, 297), detector));


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
    
    cout << "----------" << endl;

    cout << "Matches " << stats.matches << endl;
    cout << "Inliers " << stats.inliers << endl;
    cout << "Inlier ratio " << setprecision(2) << stats.ratio << endl;
    cout << "Keypoints " << stats.keypoints << endl;
    cout << endl;

    return 0;
}
