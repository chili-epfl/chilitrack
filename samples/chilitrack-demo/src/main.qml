import QtQuick 2.2
import QtQuick.Window 2.1
import QtMultimedia 5.0
import CVCamera 1.0
import Chilitrack 1.0

Window {
    visible: true
    width: camera.size.width
    height: camera.size.height

    //Set up physical camera
    CVCamera{
        id: camera
        device: 0
        size: "640x480"
    }

    //Set up detection
    Chilitrack{
        id: chilitrack
        sourceImage: camera.cvImage
    }

    //Set up visual output
    VideoOutput{
        source: camera
        anchors.fill: parent

        Item{
            id: cameraFrame
            anchors.fill: parent
            transform: MatrixTransform{ matrix: Qt.matrix4x4(
                                        700*width/640, 0,   width/2, 0,
                                        0,   700*height/480, height/2, 0,
                                        0,   0,     1, 0,
                                        0,   0,     1, 1) }

            Rectangle{
                color: "blue"
                width: 210
                height: 280
                transform: [Translate{ x: -105; y:-140 }, MatrixTransform{ matrix: chilitrack.tagPose }]
                visible: chilitrack.tagVisible
            }
        }
    }
}

