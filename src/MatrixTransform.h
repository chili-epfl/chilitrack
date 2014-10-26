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
 * @file MatrixTransform.h
 * @brief Transforms QML items by QMatrix4x4 transform matrix
 * @author Ayberk Özgür
 * @author Quentin Bonnard
 * @version 1.0
 * @date 2014-10-10
 */

#ifndef MATRIXTRANSFORM_H
#define MATRIXTRANSFORM_H

#include <QQuickTransform>
#include <QMatrix4x4>

/**
 * @brief Transforms QML items by a QMatrix4x4
 */
class MatrixTransform : public QQuickTransform
{
Q_OBJECT
    Q_PROPERTY(QMatrix4x4 matrix READ getMatrix WRITE setMatrix NOTIFY transformChanged)
    Q_CLASSINFO("DefaultProperty", "matrix")

public:

    /**
     * @brief Creates a new QMatrix4x4 transform
     *
     * @param parent Parent of the new transform
     */
    explicit MatrixTransform(QQuickItem* parent = 0);

    /**
     * @brief Gets the matrix that describes the transform
     *
     * @return Matrix that describes the transform
     */
    QMatrix4x4 getMatrix() const;

    /**
     * @brief Sets the matrix that describes the transform
     *
     * @param matrix Matrix that describes the transform
     */
    void setMatrix(QMatrix4x4 matrix);

    /**
     * @brief Applies this transform to the given transform from the left
     *
     * @param matrix Transform to apply this transform to
     */
    virtual void applyTo(QMatrix4x4* matrix) const;

signals:

    /**
     * @brief Emitted every time this transform is changed
     */
    void transformChanged();

private:

    QMatrix4x4 matrix;  ///< Matrix that describes this transform
};

#endif // MATRIXTRANSFORM_H
