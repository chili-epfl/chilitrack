#include <opencv2/videoio.hpp>
#include <iostream>

#include <chilitrack.h>

using namespace std;
using namespace cv;
using namespace chilitrack;

int main(int argc, char **argv)
{
    Mat frame;
    namedWindow("tracking");

    if(argc < 2) {
        cerr << "Usage: " << endl <<
                "tracker template_path" << endl;
        return 1;
    }

    // Configure the video capture
    // ===========================

    VideoCapture video_in(1);

    if(!video_in.isOpened()) {
        cerr << "Couldn't open camera" << endl;
        return 1;
    }

    // Read the template
    // =================

    Mat image_object = imread(argv[1], IMREAD_GRAYSCALE);

    // the size parameter is the *physical* size of the template bounding box,
    // typically in millimeters.
    auto tpl = makePtr<Template>(Template(image_object,
                                          Size(200, 297)));

    // Create the tracker
    // ==================

    Tracker tracker;

    // Main loop
    // =========

    while(true)
    {
        video_in >> frame;

        cout << tracker.process(frame, tpl);
        imshow("tracking", tracker._debug);
        if (waitKey(10) >= 0) break;
    }

    return 0;
}
