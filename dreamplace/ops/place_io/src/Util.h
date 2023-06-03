/*************************************************************************
    > File Name: util.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sun 14 Jun 2015 04:08:18 PM CDT
 ************************************************************************/

#ifndef DREAMPLACE_UTIL_H
#define DREAMPLACE_UTIL_H

#include <limbo/string/String.h>
#include <cstring>
#include <string>
#include <vector>
#include "utility/src/msg.h"

/// headers for hash tables
#include <unordered_map>
#include <unordered_set>
DREAMPLACE_BEGIN_NAMESPACE
namespace hashspace = std;
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

enum Direction2DType { kXLOW = 0, kXHIGH = 1, kYLOW = 2, kYHIGH = 3 };

/// extract x/y information from Direction2DType
inline Direction1DType getXY(Direction2DType d) {
  return Direction1DType(d > 1);
}
/// extract low/high information from Direction2DType
inline Direction1DType getLH(Direction2DType d) {
  return Direction1DType(d & 1);
}
/// construct Direction2DType from x/y and low/high information
inline Direction2DType to2D(Direction1DType xy, Direction1DType lh) {
  return Direction2DType(((int)xy << 1) + (int)lh);
}

/// data traits
/// define a template class of data traits
/// which will make it easier for generic change of data type
template <typename T>
struct coordinate_traits;

/// specialization for int
template <>
struct coordinate_traits<int> {
  typedef int coordinate_type;
  typedef double euclidean_distance_type;
  typedef long manhattan_distance_type;
  typedef long area_type;
  typedef unsigned int site_index_type;  ///< site index in a row structure
  typedef unsigned long site_area_type;  ///< number of sites for a region
  typedef unsigned int index_type;       ///< index (id)
  typedef double weight_type;            ///< type for net or node weights
};
/// specialization for unsigned int
template <>
struct coordinate_traits<unsigned int> {
  typedef unsigned int coordinate_type;
  typedef double euclidean_distance_type;
  typedef long manhattan_distance_type;
  typedef long area_type;
  typedef unsigned int site_index_type;  ///< site index in a row structure
  typedef unsigned long site_area_type;  ///< number of sites for a region
  typedef unsigned int index_type;       ///< index (id)
  typedef double weight_type;            ///< type for net or node weights
};
/// specialization for float
template <>
struct coordinate_traits<float> {
  typedef float coordinate_type;
  typedef double euclidean_distance_type;
  typedef double manhattan_distance_type;
  typedef double area_type;
  typedef unsigned int site_index_type;  ///< site index in a row structure
  typedef double site_area_type;         ///< number of sites for a region
  typedef unsigned int index_type;       ///< index (id)
  typedef float weight_type;             ///< type for net or node weights
};
/// specialization for double
template <>
struct coordinate_traits<double> {
  typedef double coordinate_type;
  typedef long double euclidean_distance_type;
  typedef long double manhattan_distance_type;
  typedef long double area_type;
  typedef unsigned long site_index_type;  ///< site index in a row structure
  typedef long double site_area_type;     ///< number of sites for a region
  typedef unsigned long index_type;       ///< index (id)
  typedef double weight_type;             ///< type for net or node weights
};

/// type helper for non-const/const
template <typename T, int B>
struct ConstTypeHelper;
template <typename T>
struct ConstTypeHelper<T, 0> {
  typedef T value_type;
  typedef T& reference_type;
  typedef T* pointer_type;
};
template <typename T>
struct ConstTypeHelper<T, 1> {
  typedef T value_type;
  typedef T const& reference_type;
  typedef T const* pointer_type;
};

/// @brief Match a string with a wildcard pattern.
/// Copied from geeksforgeeks
/// https://www.geeksforgeeks.org/wildcard-pattern-matching/
class WildcardMatch {
 public:
  /// @param str target string
  /// @param pattern target pattern
  /// @param n length of string
  /// @param m length of pattern
  inline bool operator()(const char* str, const char* pattern, std::size_t n,
                         std::size_t m) {
    // empty pattern can only match with
    // empty string
    if (m == 0) return (n == 0);

    // lookup table for storing results of
    // subproblems
    m_n = n;
    m_m = m;
    m_lookup.resize((n + 1) * (m + 1));

    // initailze lookup table to false
    memset(m_lookup.data(), false, sizeof(unsigned char) * m_lookup.size());

    // empty pattern can match with empty string
    lookup(0, 0) = true;

    // Only '*' can match with empty string
    for (std::size_t j = 1; j <= m; j++)
      if (pattern[j - 1] == '*') lookup(0, j) = lookup(0, j - 1);

    // fill the table in bottom-up fashion
    for (std::size_t i = 1; i <= n; i++) {
      for (std::size_t j = 1; j <= m; j++) {
        // Two cases if we see a '*'
        // a) We ignore ‘*’ character and move
        //    to next  character in the pattern,
        //     i.e., ‘*’ indicates an empty sequence.
        // b) '*' character matches with ith
        //     character in input
        if (pattern[j - 1] == '*')
          lookup(i, j) = lookup(i, j - 1) || lookup(i - 1, j);

        // Current characters are considered as
        // matching in two cases
        // (a) current character of pattern is '?'
        // (b) characters actually match
        else if (pattern[j - 1] == '?' || str[i - 1] == pattern[j - 1])
          lookup(i, j) = lookup(i - 1, j - 1);

        // If characters don't match
        else
          lookup(i, j) = false;
      }
    }

    return lookup(n, m);
  }

 protected:
  inline unsigned char lookup(std::size_t i, std::size_t j) const {
    return m_lookup.at(i * (m_m + 1) + j);
  }
  inline unsigned char& lookup(std::size_t i, std::size_t j) {
    return m_lookup.at(i * (m_m + 1) + j);
  }

  std::vector<unsigned char> m_lookup;  ///< lookup table
  std::size_t m_n;                      ///< length of string
  std::size_t m_m;                      /// < length of pattern
};

DREAMPLACE_END_NAMESPACE

#endif
