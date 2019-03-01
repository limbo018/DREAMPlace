/**
 * @file   cvector.h
 * @author Yibo Lin
 * @date   Sep 2018
 */

#ifndef _GPUPLACE_CVECTOR_H
#define _GPUPLACE_CVECTOR_H

#include <stdio.h>

namespace CVector 
{

template <typename T>
struct CVector 
{
    T* data; ///< data pointer 
    int size; 
    int capacity; 

    /// accessor 
    T& operator[](int i)
    {
        return data[i]; 
    }

    /// accessor 
    const T& operator[](int i) const
    {
        return data[i]; 
    }
};

template <typename T>
void initialize(CVector<T>& array, int capacity)
{
    array.data = nullptr; 
    if (capacity) 
    {
        array.data = new T [capacity]; 
        if (!array.data) 
        {
            printf("CVector allocation failed, capacity %d\n", capacity);
            //fflush(stdout);
            return;
        } 
    }
    array.capacity = capacity; 
    array.size = 0; 
}

template <typename T>
void destroy(CVector<T>& array)
{
    if (array.capacity)
    {
        delete [] array.data; 
    }
    array.data = nullptr; 
    array.size = 0; 
    array.capacity = 0; 
}

/// swap two vectors 
template <typename T>
void swap(CVector<T>& x, CVector<T>& y)
{
    std::swap(x.data, y.data); 
    std::swap(x.size, y.size); 
    std::swap(x.capacity, y.capacity); 
}

/// insert a value to an array 
/// return the pointer to the new element 
template <typename T>
void insert(CVector<T>& array, T value, int pos)
{
    // assume capacity is enough 
    if (array.size+1 >= array.capacity || pos > array.size)
    {
        printf("not enough capacity or invalid position, start %p, size %d, capacity %d, pos %d\n", array.data, array.size, array.capacity, pos);
        //fflush(stdout);
        return; 
    }

    if (pos == array.size) // append
    {
        array.data[pos] = value; 
    }
    else 
    {
        for (int i = array.size; i > pos; --i)
        {
            array.data[i] = array.data[i-1];
        }
        array.data[pos] = value; 
    }
    array.size += 1; 
}

/// append an element to an array 
/// return the pointer to the new element
template <typename T>
void push_back(CVector<T>& array, T value)
{
    // assume capacity is enough 
    if (array.size+1 >= array.capacity)
    {
        printf("not enough capacity, start %p, size %d, capacity %d\n", array.data, array.size, array.capacity);
        //fflush(stdout);
        return; 
    }

    int old_idx = array.size;
    array.size += 1; 
    array.data[old_idx] = value; 
}

/// append an array of values to an array 
/// return the pointer to the beginning of new elements
template <typename T>
void push_back(CVector<T>& array, const T* bgn, const T* end)
{
    // assume capacity is enough 
    if (array.size+end-bgn >= array.capacity)
    {
        printf("not enough capacity, start %p, size %d, capacity %d, insert %ld\n", array.data, array.size, array.capacity, end-bgn);
        //fflush(stdout);
        return; 
    }

    int i = array.size;
    for (const T* ptr = bgn; ptr != end; ++ptr, ++i)
    {
        array.data[i] = *ptr; 
    }
    array.size += end-bgn; 
}

/// erase a value from an array 
/// return the pointer to the element of the erased position 
template <typename T>
void erase(CVector<T>& array, int pos)
{
    if (array.size == 0 || pos >= array.size)
    {
        printf("cannot erase empty array or invalid position, size %d, capacity %d, pos %d\n", array.size, array.capacity, pos);
        //fflush(stdout);
        return; 
    }

    for (int i = pos; i < array.size-1; ++i)
    {
        array.data[i] = array.data[i+1];
    }
    array.size -= 1; 
}

template <typename T>
void print(const CVector<T>& array)
{
    printf("data %p, size %d, capacity %d\n", array.data, array.size, array.capacity);
    for (int i = 0; i < array.size; ++i)
    {
        printf("[%d] = %g\n", i, array[i]);
    }
}

template <typename T>
void print(const CVector<CVector<T> >& array)
{
    printf("data %p, size %d, capacity %d\n", array.data, array.size, array.capacity);
    for (int i = 0; i < array.size; ++i)
    {
        printf("\n[%d] ", i);
        print(array[i]);
    }
    printf("\n");
}

}

#endif
