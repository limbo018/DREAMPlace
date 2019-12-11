/**
 * @file   ComplexNumber.h
 * @author Zixuan Jiang, Jiaqi Gu
 * @date   Aug 2019
 * @brief  Complex number for CPU
 */

#ifndef DREAMPLACE_UTILITY_COMPLEXNUMBER_H
#define DREAMPLACE_UTILITY_COMPLEXNUMBER_H

#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct ComplexType
{
    T x;
    T y;
    ComplexType()
    {
        x = 0;
        y = 0;
    }

    ComplexType(T real, T imag)
    {
        x = real;
        y = imag;
    }

    ~ComplexType() {}
};

template <typename T>
inline ComplexType<T> complexMul(const ComplexType<T> &x, const ComplexType<T> &y)
{
    ComplexType<T> res;
    res.x = x.x * y.x - x.y * y.y;
    res.y = x.x * y.y + x.y * y.x;
    return res;
}

template <typename T>
inline T RealPartOfMul(const ComplexType<T> &x, const ComplexType<T> &y)
{
    return x.x * y.x - x.y * y.y;
}

template <typename T>
inline T ImaginaryPartOfMul(const ComplexType<T> &x, const ComplexType<T> &y)
{
    return x.x * y.y + x.y * y.x;
}

template <typename T>
inline ComplexType<T> complexAdd(const ComplexType<T> &x, const ComplexType<T> &y)
{
    ComplexType<T> res;
    res.x = x.x + y.x;
    res.y = x.y + y.y;
    return res;
}

template <typename T>
inline ComplexType<T> complexSubtract(const ComplexType<T> &x, const ComplexType<T> &y)
{
    ComplexType<T> res;
    res.x = x.x - y.x;
    res.y = x.y - y.y;
    return res;
}

template <typename T>
inline ComplexType<T> complexConj(const ComplexType<T> &x)
{
    ComplexType<T> res;
    res.x = x.x;
    res.y = -x.y;
    return res;
}

template <typename T>
inline ComplexType<T> complexMulConj(const ComplexType<T> &x, const ComplexType<T> &y)
{
    ComplexType<T> res;
    res.x = x.x * y.x - x.y * y.y;
    res.y = -(x.x * y.y + x.y * y.x);
    return res;
}

DREAMPLACE_END_NAMESPACE

#endif
