/* SPDX-License-Identifier: GPL-2.0-or-later
 * Template for genuine extended80 and research binary64 scalar types.
 * Operations do not convert intermediate values to binary64.
 */
static inline ATTR REAL NAME(from)(double x) { return (REAL)x; }
static inline ATTR REAL NAME(neg)(REAL x) { return -x; }
static inline ATTR REAL NAME(add)(REAL a, REAL b) { return a+b; }
static inline ATTR REAL NAME(sub)(REAL a, REAL b) { return a-b; }
static inline ATTR REAL NAME(mul)(REAL a, REAL b) { return a*b; }
static inline ATTR REAL NAME(half)(REAL x) { return x*(REAL)0.5; }
static inline ATTR REAL NAME(twice)(REAL x) { return x*(REAL)2.0; }
static inline ATTR REAL NAME(scale)(REAL x, int e) { return SCALBN(x,e); }
static inline ATTR REAL NAME(input)(double x, int e) { return SCALBN((REAL)x,e); }
static inline ATTR REAL NAME(div)(REAL a, double b) { return a/(REAL)b; }
static inline ATTR REAL NAME(pi_half)(void) { return PI_HALF; }
static inline ATTR REAL NAME(sin)(REAL x) { return SIN(x); }
static ATTR int NAME(export)(REAL x, int global, struct ac_spectral_bin *out) {
    if (!isfinite(x) || x < 0) return 0;
    if (x == 0) { *out = (struct ac_spectral_bin){0,0,0}; return 1; }
    int e;
    REAL y = FREXP(x,&e);
    double hi = (double)y, lo = (double)(y-(REAL)hi);
    *out = (struct ac_spectral_bin){hi,lo,global+e};
    return 1;
}

static ATTR int NAME(export_signed)(REAL x,int global,struct ac_spectral_bin *out) {
    int negative=x < 0;
    if (!NAME(export)(negative?NAME(neg)(x):x,global,out)) return 0;
    if (negative) { out->hi=-out->hi; out->lo=-out->lo; }
    return 1;
}

static inline ATTR int NAME(abs_ge)(REAL a,REAL b) { return (a<0?-a:a)>=(b<0?-b:b); }

static inline ATTR REAL NAME(div_real)(REAL a,REAL b) { return a/b; }

static inline ATTR int NAME(gt)(REAL a,REAL b) { return a>b; }
static inline ATTR int NAME(finite)(REAL a) { return isfinite(a); }
static inline ATTR REAL NAME(sqrt_real)(REAL a) { return SQRT(a); }
