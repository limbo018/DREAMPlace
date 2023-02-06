#ifndef FLUTE_HPP_
#define FLUTE_HPP_

namespace flute {

#define DEGREE 9        // LUT will be used when d <= DEGREE, DEGREE <= 9
#define FLUTEROUTING 1   // 1 to construct routing, 0 to estimate WL only
#define REMOVE_DUPLICATE_PIN 0  // Remove dup. pin for flute_wl() & flute()
#define ACCURACY 8  // Default accuracy is 3
//#define MAXD 2008840  // max. degree of a net that can be handled
#define MAXD 1000  // max. degree of a net that can be handled

#ifndef DTYPE   // Data type for distance
#define DTYPE int
#endif

typedef struct {
  DTYPE x, y;   // starting point of the branch
  int n;   // index of neighbor
} Branch;

typedef struct{
  int deg;   // degree
  DTYPE length;   // total wirelength
  Branch *branch;   // array of tree branches
} Tree;

// Major functions
extern void readLUT(const char* powv, const char* post);
extern DTYPE flute_wl(int d, DTYPE x[], DTYPE y[], int acc);
extern Tree flute(int d, DTYPE x[], DTYPE y[], int acc);
extern DTYPE wirelength(Tree t);
extern void printtree(Tree t);

// Other useful functions
extern DTYPE flutes_wl_LD(int d, DTYPE xs[], DTYPE ys[], int s[]);
extern DTYPE flutes_wl_MD(int d, DTYPE xs[], DTYPE ys[], int s[], int acc);
extern DTYPE flutes_wl_RDP(int d, DTYPE xs[], DTYPE ys[], int s[], int acc);
extern Tree flutes_LD(int d, DTYPE xs[], DTYPE ys[], int s[]);
extern Tree flutes_MD(int d, DTYPE xs[], DTYPE ys[], int s[], int acc);
extern Tree flutes_RDP(int d, DTYPE xs[], DTYPE ys[], int s[], int acc);

#if REMOVE_DUPLICATE_PIN==1
  #define flutes_wl(d, xs, ys, s, acc) flutes_wl_RDP(d, xs, ys, s, acc) 
  #define flutes(d, xs, ys, s, acc) flutes_RDP(d, xs, ys, s, acc) 
#else
  #define flutes_wl(d, xs, ys, s, acc) flutes_wl_ALLD(d, xs, ys, s, acc) 
  #define flutes(d, xs, ys, s, acc) flutes_ALLD(d, xs, ys, s, acc) 
#endif

#define flutes_wl_ALLD(d, xs, ys, s, acc) flutes_wl_LMD(d, xs, ys, s, acc)
#define flutes_ALLD(d, xs, ys, s, acc) flutes_LMD(d, xs, ys, s, acc)

#define flutes_wl_LMD(d, xs, ys, s, acc) \
    (d<=DEGREE ? flutes_wl_LD(d, xs, ys, s) : flutes_wl_MD(d, xs, ys, s, acc))
#define flutes_LMD(d, xs, ys, s, acc) \
    (d<=DEGREE ? flutes_LD(d, xs, ys, s) : flutes_MD(d, xs, ys, s, acc))

#define ADIFF(x,y) ((x)>(y)?(x-y):(y-x))  // Absolute difference

} // namespace flute

#endif // FLUTE_HPP_
