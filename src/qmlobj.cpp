#include "qmlobj.h"

ChilitrackDetection::ChilitrackDetection(QQuickItem *parent) :
    QQuickItem(parent),
    im(480,640,CV_8UC1)
{
    QFile file(":/robot-mid.jpg");
    if (!file.open(QIODevice::ReadOnly)){
        qDebug() << "Could not open...";
        return;
    }
    QByteArray blob = file.readAll();
    std::vector<char> data(blob.begin(), blob.end());

    cv::Mat image_object = imdecode(data, cv::IMREAD_GRAYSCALE);
    //cv::Mat image_object = imread("/sdcard/robot-mid.jpg", cv::IMREAD_GRAYSCALE);
    if( !image_object.data ){
        qDebug() << " --(!) Error reading object image ";
    }
    else{
        qDebug() << "Read " << image_object.size().width << "x" << image_object.size().height << " template image";
    }

    //Ptr<Feature2D> detector = Feature2D::create("AKAZE");
    cv::Ptr<cv::Feature2D> detector = cv::Feature2D::create("ORB");
    detector->set("nFeatures", 700); // default: 500

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    tracker = new chilitrack::Tracker(detector, matcher);

    auto tpl = cv::makePtr<chilitrack::Template>(chilitrack::Template(image_object,
                cv::Size(125, 266),
                detector));

    tracker->add_template("tpl1", tpl);
}

ChilitrackDetection::~ChilitrackDetection()
{
    delete tracker;
}

void ChilitrackDetection::setSourceImage(QVariant sourceImage)
{
    auto stats = cv::makePtr<Stats>();
#ifdef ANDROID
    cv::Mat yuvim = sourceImage.value<cv::Mat>();
    if(yuvim.size().width != 640 || yuvim.size().height != 720)
        return;
    cv::cvtColor(yuvim,im,cv::COLOR_YUV2GRAY_NV21);
    qDebug() << "Processing " << im.size().width << "x" << im.size().height << " " << im.channels() << " channels";
    auto results = tracker->estimate(im, stats);
#else
    cv::Mat rgbim = sourceImage.value<cv::Mat>();
    cv::cvtColor(rgbim,im,cv::COLOR_RGB2GRAY);
    qDebug() << "Processing " << im.size().width << "x" << im.size().height << " " << im.channels() << " channels";
    auto results = tracker->estimate(im, stats);
#endif
    bool wasTagVisible = tagVisible;
    if(results.size() > 0){
        for (int i = 0; i<4; ++i)
            for (int j = 0; j<4; ++j)
                tagPose(i,j) = results.begin()->second(i,j);
        emit tagPoseChanged();
        tagVisible = true;
    }
    else
        tagVisible = false;

    if(tagVisible != wasTagVisible)
        emit tagVisibilityChanged();
}

QMatrix4x4 ChilitrackDetection::getTagPose()
{
    return tagPose;
}

bool ChilitrackDetection::isTagVisible()
{
    return tagVisible;
}
