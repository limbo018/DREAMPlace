/**
 * @file   Box.h
 * @author Yibo Lin
 * @date   Jan 2019
 */

#ifndef _DREAMPLACE_UTILITY_BOX_H
#define _DREAMPLACE_UTILITY_BOX_H

#include <limits>
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct Box 
{
    T xl;
    T yl; 
    T xh; 
    T yh; 

    /// @brief default constructor 
    Box()
    {
        invalidate();
    }
    /// @brief constructor 
    /// @param xxl xl 
    /// @param yyl yl 
    /// @param xxh xh 
    /// @param yyh yh 
    Box(T xxl, T yyl, T xxh, T yyh)
        : xl(xxl)
        , yl(yyl)
        , xh(xxh)
        , yh(yyh)
    {
    }
    /// @brief invalidate the box 
    void invalidate()
    {
        xl = std::numeric_limits<T>::max(); 
        yl = std::numeric_limits<T>::max(); 
        xh = std::numeric_limits<T>::lowest();
        yh = std::numeric_limits<T>::lowest();
    }
    /// @brief encompass a point 
    /// @param x 
    /// @param y
    void encompass(T x, T y)
    {
        xl = std::min(xl, x); 
        xh = std::max(xh, x); 
        yl = std::min(yl, y); 
        yh = std::min(yh, y);
    }
    /// @brief encompass a box 
    /// @param xxl xl 
    /// @param yyl yl 
    /// @param xxh xh 
    /// @param yyh yh 
    void encompass(T xxl, T yyl, T xxh, T yyh)
    {
        encompass(xxl, yyl);
        encompass(xxh, yyh);
    }
    /// @brief bloat x direction by 2*dx, and y direction by 2*dy 
    /// @param dx 
    /// @param dy 
    void bloat(T dx, T dy)
    {
        xl -= dx; 
        xh += dx; 
        yl -= dy; 
        yh += dy; 
    }
    /// @brief check if a point is contained by the box 
    /// @param x 
    /// @param y 
    /// @return true if contains 
    bool contains(T x, T y) const 
    {
        return xl <= x && x <= xh && yl <= y && y <= yh; 
    }
    /// @brief check if a box is contained by the box 
    /// @param xxl xl 
    /// @param yyl yl 
    /// @param xxh xh 
    /// @param yyh yh 
    /// @return true if contains 
    bool contains(T xxl, T yyl, T xxh, T yyh) const 
    {
        return contains(xxl, yyl) && contains(xxh, yyh); 
    }
    /// @return width of the box 
    T width() const {return xh-xl;}
    /// @return height of the box 
    T height() const {return yh-yl;}
    /// @return x coordinate of the center of the box 
    T center_x() const {return (xl+xh)/2;}
    /// @return y coordinate of the center of the box 
    T center_y() const {return (yl+yh)/2;}
    /// @return center manhattan distance to another box 
    T center_distance(const Box& rhs) const 
    {
        return fabs(rhs.center_x()-center_x()) + fabs(rhs.center_y()-center_y());
    }
    /// @brief print the box 
    void print() const 
    {
        printf("(%g, %g, %g, %g)\n", (double)xl, (double)yl, (double)xh, (double)yh);
    }
};

DREAMPLACE_END_NAMESPACE

#endif
