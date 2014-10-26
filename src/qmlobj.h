#ifndef CHILITRACKDETECTION_H
#define CHILITRACKDETECTION_H

#include<QQuickItem>
#include<QMatrix4x4>
#include<QMetaType>

#include"../include/chilitrack.h"

#include<opencv2/imgcodecs.hpp>

Q_DECLARE_METATYPE(cv::Mat)

class ChilitrackDetection : public QQuickItem {
Q_OBJECT
    Q_PROPERTY(QVariant sourceImage WRITE setSourceImage)
    Q_PROPERTY(QMatrix4x4 tagPose READ getTagPose NOTIFY tagPoseChanged)
    Q_PROPERTY(bool tagVisible READ isTagVisible NOTIFY tagVisibilityChanged)

public:

    explicit ChilitrackDetection(QQuickItem *parent = 0);

    virtual ~ChilitrackDetection();

    void setSourceImage(QVariant sourceImage);

    QMatrix4x4 getTagPose();

    bool isTagVisible();

signals:

    void tagPoseChanged();

    void tagVisibilityChanged();

private:

    chilitrack::Tracker* tracker;

    QMatrix4x4 tagPose;

    bool tagVisible = false;

    cv::Mat im;
};

#endif // CHILITRACKDETECTION_H
