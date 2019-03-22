/*****************************/
/*  User-Defined Parameters  */
/*****************************/
#define MAXD 150    // max. degree that can be handled
#define ACCURACY 3  // Default accuracy
#define ROUTING 1   // 1 to construct routing, 0 to estimate WL only
#define LOCAL_REFINEMENT 0      // Suggestion: Set to 1 if ACCURACY >= 5
#define REMOVE_DUPLICATE_PIN 0  // Remove dup. pin for flute_wl() & flute()

#ifndef DTYPE   // Data type for distance
#define DTYPE int
#endif


/*****************************/
/*  User-Callable Functions  */
/*****************************/
// void readLUT();
// DTYPE flute_wl(int d, DTYPE x[], DTYPE y[], int acc);
// DTYPE flutes_wl(int d, DTYPE xs[], DTYPE ys[], int s[], int acc);
// Tree flute(int d, DTYPE x[], DTYPE y[], int acc);
// Tree flutes(int d, DTYPE xs[], DTYPE ys[], int s[], int acc);
// DTYPE wirelength(Tree t);
// void printtree(Tree t);
// void plottree(Tree t);


/*************************************/
/* Internal Parameters and Functions */
/*************************************/
//#define POWVFILE "POWV9.dat"        // LUT for POWV (Wirelength Vector)
//#define POSTFILE "POST9.dat"        // LUT for POST (Steiner Tree)
#define D 9                         // LUT is used for d <= D, D <= 9
#define TAU(A) (8+1.3*(A))
#define D1(A) (25+120/((A)*(A)))     // flute_mr is used for D1 < d <= D2
#define D2(A) ((A)<=6 ? 500 : 75+5*(A))

typedef struct
{
    DTYPE x, y;   // starting point of the branch
    int n;   // index of neighbor
} Branch;

typedef struct
{
    int deg;   // degree
    DTYPE length;   // total wirelength
    Branch *branch;   // array of tree branches
} Tree;

// User-Callable Functions
extern void readLUT(const char* POWVFILE, const char* POSTFILE);
extern DTYPE flute_wl(int d, DTYPE x[], DTYPE y[], int acc);
//Macro: DTYPE flutes_wl(int d, DTYPE xs[], DTYPE ys[], int s[], int acc);
extern Tree flute(int d, DTYPE x[], DTYPE y[], int acc);
//Macro: Tree flutes(int d, DTYPE xs[], DTYPE ys[], int s[], int acc);
extern DTYPE wirelength(Tree t);
extern void printtree(Tree t);
extern void plottree(Tree t);

// Other useful functions
extern void init_param();
extern DTYPE flutes_wl_LD(int d, DTYPE xs[], DTYPE ys[], int s[]);
extern DTYPE flutes_wl_MD(int d, DTYPE xs[], DTYPE ys[], int s[], int acc);
extern DTYPE flutes_wl_RDP(int d, DTYPE xs[], DTYPE ys[], int s[], int acc);
extern Tree flutes_LD(int d, DTYPE xs[], DTYPE ys[], int s[]);
extern Tree flutes_MD(int d, DTYPE xs[], DTYPE ys[], int s[], int acc);
extern Tree flutes_HD(int d, DTYPE xs[], DTYPE ys[], int s[], int acc);
extern Tree flutes_RDP(int d, DTYPE xs[], DTYPE ys[], int s[], int acc);

#if REMOVE_DUPLICATE_PIN==1
  #define flutes_wl(d, xs, ys, s, acc) flutes_wl_RDP(d, xs, ys, s, acc) 
  #define flutes(d, xs, ys, s, acc) flutes_RDP(d, xs, ys, s, acc) 
#else
  #define flutes_wl(d, xs, ys, s, acc) flutes_wl_ALLD(d, xs, ys, s, acc) 
  #define flutes(d, xs, ys, s, acc) flutes_ALLD(d, xs, ys, s, acc) 
#endif

#define flutes_wl_ALLD(d, xs, ys, s, acc) flutes_wl_LMD(d, xs, ys, s, acc)
#define flutes_ALLD(d, xs, ys, s, acc) \
    (d<=D ? flutes_LD(d, xs, ys, s) \
          : (d<=D1(acc) ? flutes_MD(d, xs, ys, s, acc) \
                        : flutes_HD(d, xs, ys, s, acc)))

#define flutes_wl_LMD(d, xs, ys, s, acc) \
    (d<=D ? flutes_wl_LD(d, xs, ys, s) : flutes_wl_MD(d, xs, ys, s, acc))
#define flutes_LMD(d, xs, ys, s, acc) \
    (d<=D ? flutes_LD(d, xs, ys, s) : flutes_MD(d, xs, ys, s, acc))

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))
#define abs(x) ((x)<0?(-x):(x))
#define ADIFF(x,y) ((x)>(y)?(x-y):(y-x))  // Absolute difference
