#ifndef DREAMPLACE_COMPLEXNUMBER_H
#define DREAMPLACE_COMPLEXNUMBER_H

template <typename T>
struct ComplexType
{
    T x;
    T y;
    __host__ __device__ ComplexType()
    {
        x = 0;
        y = 0;
    }

    __host__ __device__ ComplexType(T real, T imag)
    {
        x = real;
        y = imag;
    }

    __host__ __device__ ~ComplexType() {}
};

template <typename T>
inline __device__ ComplexType<T> complexMul(const ComplexType<T> &x, const ComplexType<T> &y)
{
    ComplexType<T> res;
    res.x = x.x * y.x - x.y * y.y;
    res.y = x.x * y.y + x.y * y.x;
    return res;
}

template <typename T>
inline __device__ T RealPartOfMul(const ComplexType<T> &x, const ComplexType<T> &y)
{
    return x.x * y.x - x.y * y.y;
}

template <typename T>
inline __device__ T ImaginaryPartOfMul(const ComplexType<T> &x, const ComplexType<T> &y)
{
    return x.x * y.y + x.y * y.x;
}

template <typename T>
inline __device__ ComplexType<T> complexAdd(const ComplexType<T> &x, const ComplexType<T> &y)
{
    ComplexType<T> res;
    res.x = x.x + y.x;
    res.y = x.y + y.y;
    return res;
}

template <typename T>
inline __device__ ComplexType<T> complexSubtract(const ComplexType<T> &x, const ComplexType<T> &y)
{
    ComplexType<T> res;
    res.x = x.x - y.x;
    res.y = x.y - y.y;
    return res;
}

template <typename T>
inline __device__ ComplexType<T> complexConj(const ComplexType<T> &x)
{
    ComplexType<T> res;
    res.x = x.x;
    res.y = -1 * x.y;
    return res;
}

template <typename T>
inline __device__ ComplexType<T> complexMulConj(const ComplexType<T> &x, const ComplexType<T> &y)
{
    ComplexType<T> res;
    res.x = x.x * y.x - x.y * y.y;
    res.y = -1 * (x.x * y.y + x.y * y.x);
    return res;
}

#endif
