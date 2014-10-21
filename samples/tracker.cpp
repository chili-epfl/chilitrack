#include <opencv2/videoio.hpp>
#include <iostream>
#include <iomanip>

#include <chilitrack.h>

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
    Mat frame;

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

    // Read one frame to initialize the capture
    video_in >> frame;

    Mat image_object = imread(argv[1], IMREAD_GRAYSCALE);
    if( !image_object.data )
          { cerr << " --(!) Error reading object image " << std::endl; return -1; }


    //Ptr<Feature2D> detector = Feature2D::create("AKAZE");
    Ptr<Feature2D> detector = Feature2D::create("ORB");
    detector->set("nFeatures", 700); // default: 500

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    Tracker tracker(detector, matcher);

    if(argc == 3) {
        tracker.readCalibration(argv[2]);
    }

    auto tpl = makePtr<Template>(Template(image_object,
                                          Size(200, 297), 
                                          detector));


    auto stats = makePtr<Stats>();
    Stats draw_stats, global_stats;
    int frame_count = 0;

    for(; ; frame_count++) {
        bool update_stats = (frame_count % stats_update_period == 0);
        video_in >> frame;

        tracker.process(frame, tpl, stats);
        global_stats += *stats;
        if(update_stats) {
            draw_stats = *stats;
        }

        drawStatistics(tracker._debug, draw_stats);

        imshow("tracking", tracker._debug);
        if (waitKey(10) >= 0) break;
    }
    global_stats /= frame_count;
    
    cout << "----------" << endl;

    cout << "Matches " << stats->matches << endl;
    cout << "Inliers " << stats->inliers << endl;
    cout << "Inlier ratio " << setprecision(2) << stats->ratio << endl;
    cout << "Keypoints " << stats->keypoints << endl;
    cout << endl;

    return 0;
}
