#include "plugin.h"

void ChilitrackPlugin::registerTypes(const char *uri)
{
    qmlRegisterType<ChilitrackDetection>(uri, 1, 0, "Chilitrack");
    qmlRegisterType<MatrixTransform>(uri, 1, 0, "MatrixTransform");
}

