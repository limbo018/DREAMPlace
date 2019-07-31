/*************************************************************************
    > File Name: util.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sun 14 Jun 2015 04:08:18 PM CDT
 ************************************************************************/

#ifndef DREAMPLACE_UTIL_H
#define DREAMPLACE_UTIL_H

#include <string>
#include "utility/src/Namespace.h"
#include "utility/src/Msg.h"

/// headers for hash tables 
#include <tr1/unordered_map>
#include <tr1/unordered_set>
DREAMPLACE_BEGIN_NAMESPACE
namespace hashspace = std::tr1;
DREAMPLACE_END_NAMESPACE

DREAMPLACE_BEGIN_NAMESPACE

enum Direction1DType {
	kLOW = 0,
	kHIGH = 1, 
	kX = 0, 
	kY = 1, 
    kLEFT = 0, 
    kRIGHT = 1,
    kBOTTOM = 0, 
    kTOP = 1
};

enum Direction2DType {
	kXLOW = 0, 
	kXHIGH = 1,
	kYLOW = 2,
	kYHIGH = 3
};

/// extract x/y information from Direction2DType
inline Direction1DType getXY(Direction2DType d)
{
    return Direction1DType(d>1);
}
/// extract low/high information from Direction2DType
inline Direction1DType getLH(Direction2DType d)
{
    return Direction1DType(d&1);
}
/// construct Direction2DType from x/y and low/high information 
inline Direction2DType to2D(Direction1DType xy, Direction1DType lh)
{
    return Direction2DType(((int)xy<<1)+(int)lh);
}

/// data traits 
/// define a template class of data traits
/// which will make it easier for generic change of data type 
template <typename T>
struct coordinate_traits;

/// specialization for int
template <>
struct coordinate_traits<int>
{
	typedef int coordinate_type;
	typedef double euclidean_distance_type;
    typedef long manhattan_distance_type;
	typedef long area_type;
	typedef unsigned int site_index_type; ///< site index in a row structure 
	typedef unsigned long site_area_type; ///< number of sites for a region 
    typedef unsigned int index_type; ///< index (id) 
    typedef double weight_type; ///< type for net or node weights 
};
/// specialization for unsigned int
template <>
struct coordinate_traits<unsigned int>
{
	typedef unsigned int coordinate_type;
	typedef double euclidean_distance_type;
    typedef long manhattan_distance_type;
	typedef long area_type;
	typedef unsigned int site_index_type; ///< site index in a row structure 
	typedef unsigned long site_area_type; ///< number of sites for a region 
    typedef unsigned int index_type; ///< index (id) 
    typedef double weight_type; ///< type for net or node weights 
};

/// type helper for non-const/const  
template <typename T, int B> struct ConstTypeHelper;
template <typename T> struct ConstTypeHelper<T, 0>
{
    typedef T value_type;
    typedef T& reference_type;
    typedef T* pointer_type;
};
template <typename T> struct ConstTypeHelper<T, 1>
{
    typedef T value_type;
    typedef T const& reference_type;
    typedef T const* pointer_type;
};

DREAMPLACE_END_NAMESPACE

#endif
