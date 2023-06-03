#ifndef FLUTE_HPP_
#define FLUTE_HPP_

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace flute {

#define DEGREE 9                // LUT will be used when d <= DEGREE, DEGREE <= 9
#define FLUTEROUTING 1          // 1 to construct routing, 0 to estimate WL only
#define REMOVE_DUPLICATE_PIN 0  // Remove dup. pin for flute_wl() & flute()
#define ACCURACY 8              // Default accuracy is 3
// #define MAXD 2008840  // max. degree of a net that can be handled
#define MAXD 1000  // max. degree of a net that can be handled

using DType = int;

// TODO:
// 1. replace macros with constexpr variables
// 2. construct a param struct
// 3. replace all malloc-free contents
// 4. clear inconsistent comments

struct Branch {
  DType x, y;  // starting point of the branch
  int n;       // index of neighbor
};

struct Tree {
  int deg;         // degree
  DType length;    // total wirelength
  Branch* branch;  // array of tree branches
};

// Major functions
void readLUT(const char* powv, const char* post);
DType flute_wl(int d, DType* x, DType* y, int acc);
Tree flute(int d, DType* x, DType* y, int acc);
DType wirelength(Tree t);
void printtree(Tree t);

// Other useful functions
DType flutes_wl_LD(int d, DType* xs, DType* ys, int* s);
DType flutes_wl_MD(int d, DType* xs, DType* ys, int* s, int acc);
DType flutes_wl_RDP(int d, DType* xs, DType* ys, int* s, int acc);
Tree flutes_LD(int d, DType* xs, DType* ys, int* s);
Tree flutes_MD(int d, DType* xs, DType* ys, int* s, int acc);
Tree flutes_RDP(int d, DType* xs, DType* ys, int* s, int acc);

#if REMOVE_DUPLICATE_PIN == 1
#define flutes_wl(d, xs, ys, s, acc) flutes_wl_RDP(d, xs, ys, s, acc)
#define flutes(d, xs, ys, s, acc) flutes_RDP(d, xs, ys, s, acc)
#else
#define flutes_wl(d, xs, ys, s, acc) flutes_wl_ALLD(d, xs, ys, s, acc)
#define flutes(d, xs, ys, s, acc) flutes_ALLD(d, xs, ys, s, acc)
#endif

#define flutes_wl_ALLD(d, xs, ys, s, acc) flutes_wl_LMD(d, xs, ys, s, acc)
#define flutes_ALLD(d, xs, ys, s, acc) flutes_LMD(d, xs, ys, s, acc)

#define flutes_wl_LMD(d, xs, ys, s, acc) \
  (d <= DEGREE ? flutes_wl_LD(d, xs, ys, s) : flutes_wl_MD(d, xs, ys, s, acc))
#define flutes_LMD(d, xs, ys, s, acc) \
  (d <= DEGREE ? flutes_LD(d, xs, ys, s) : flutes_MD(d, xs, ys, s, acc))

}  // namespace flute

#endif  // FLUTE_HPP_
