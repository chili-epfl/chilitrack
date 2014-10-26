/*
 * Copyright (C) 2014 EPFL
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 */

/**
 * @file MatrixTransform.cpp
 * @brief Transforms QML items by QMatrix4x4 transform matrix
 * @author Ayberk Özgür
 * @author Quentin Bonnard
 * @version 1.0
 * @date 2014-10-10
 */

#include "MatrixTransform.h"

MatrixTransform::MatrixTransform(QQuickItem* parent) :
    QQuickTransform(parent)
{}

QMatrix4x4 MatrixTransform::getMatrix() const
{
    return matrix;
}

void MatrixTransform::setMatrix(QMatrix4x4 matrix)
{
    if (this->matrix == matrix)
        return;
    this->matrix = matrix;
    update();
    emit transformChanged();
}

void MatrixTransform::applyTo(QMatrix4x4* matrix) const
{
    matrix->operator*=(this->matrix);
}

