/* SPDX-License-Identifier: GPL-2.0-or-later
 * Template: CP7's accurate two-limb arithmetic, separately instantiated
 * for the baseline ISA and (on x86) the runtime-selected FMA target.
 * The enclosing file defines NAME, ATTR and REAL.
 */
static inline ATTR REAL NAME(from)(double x) { return (REAL){x, 0.0}; }
static inline ATTR REAL NAME(neg)(REAL x) { return (REAL){-x.hi, -x.lo}; }
static inline ATTR REAL NAME(sum)(double a, double b) {
    double hi = a + b, v = hi - a;
    return (REAL){hi, (a - (hi-v)) + (b-v)};
}
static inline ATTR REAL NAME(add)(REAL a, REAL b) {
    REAL s = NAME(sum)(a.hi, b.hi), t = NAME(sum)(a.lo, b.lo);
    REAL u = NAME(sum)(s.hi, s.lo + t.hi);
    return NAME(sum)(u.hi, u.lo + t.lo);
}
static inline ATTR REAL NAME(sub)(REAL a, REAL b) { return NAME(add)(a, NAME(neg)(b)); }
static inline ATTR REAL NAME(mul)(REAL a, REAL b) {
    double p = a.hi * b.hi, e = fma(a.hi, b.hi, -p);
    double cross = fma(a.hi, b.lo, a.lo * b.hi);
    return NAME(sum)(p, e + (cross + a.lo * b.lo));
}
static inline ATTR REAL NAME(half)(REAL x) { return (REAL){x.hi * 0.5, x.lo * 0.5}; }
static inline ATTR REAL NAME(twice)(REAL x) { return (REAL){x.hi * 2.0, x.lo * 2.0}; }
static inline ATTR REAL NAME(scale)(REAL x, int e) {
    return NAME(sum)(scalbn(x.hi, e), scalbn(x.lo, e));
}
static inline ATTR REAL NAME(input)(double x, int e) { return NAME(from)(scalbn(x, e)); }
static inline ATTR REAL NAME(div)(REAL a, double b) {
    REAL divisor = NAME(from)(b), q = NAME(from)(a.hi / b);
    REAL r = NAME(sub)(a, NAME(mul)(divisor, q));
    q = NAME(add)(q, NAME(from)(r.hi / b));
    r = NAME(sub)(a, NAME(mul)(divisor, q));
    return NAME(add)(q, NAME(from)(r.hi / b));
}
static inline ATTR REAL NAME(pi_half)(void) { return dd_pi_half; }
static ATTR REAL NAME(sin)(REAL x) {
    REAL x2 = NAME(mul)(x, x), p = dd_sine_coefficients[19];
    for (int k = 18; k >= 0; --k) p = NAME(add)(dd_sine_coefficients[k], NAME(mul)(x2, p));
    return NAME(mul)(x, p);
}
static ATTR int NAME(export)(REAL x, int global, struct ac_spectral_bin *out) {
    if (!isfinite(x.hi) || !isfinite(x.lo) || x.hi < 0 || (x.hi == 0 && x.lo < 0)) return 0;
    if (x.hi == 0 && x.lo == 0) { *out = (struct ac_spectral_bin){0,0,0}; return 1; }
    if (x.hi == 0) x = NAME(from)(x.lo);
    int e;
    (void)frexp(x.hi, &e);
    REAL y = NAME(scale)(x, -e);
    *out = (struct ac_spectral_bin){y.hi, y.lo, global + e};
    return 1;
}

static ATTR int NAME(export_signed)(REAL x,int global,struct ac_spectral_bin *out) {
    int negative=x.hi < 0 || (x.hi == 0 && x.lo < 0);
    if (!NAME(export)(negative?NAME(neg)(x):x,global,out)) return 0;
    if (negative) { out->hi=-out->hi; out->lo=-out->lo; }
    return 1;
}

static inline ATTR int NAME(abs_ge)(REAL a,REAL b) { return fabs(a.hi)>=fabs(b.hi); }

static inline ATTR REAL NAME(div_real)(REAL a,REAL b) {
    REAL q=NAME(from)(a.hi/b.hi);
    REAL r=NAME(sub)(a,NAME(mul)(b,q));
    q=NAME(add)(q,NAME(from)(r.hi/b.hi));
    r=NAME(sub)(a,NAME(mul)(b,q));
    return NAME(add)(q,NAME(from)(r.hi/b.hi));
}

static inline ATTR int NAME(gt)(REAL a,REAL b) { return a.hi>b.hi || (a.hi==b.hi && a.lo>b.lo); }
static inline ATTR int NAME(finite)(REAL a) { return isfinite(a.hi) && isfinite(a.lo); }
static inline ATTR REAL NAME(sqrt_real)(REAL a) {
    REAL q=NAME(from)(sqrt(a.hi));
    return NAME(add)(q,NAME(div_real)(NAME(sub)(a,NAME(mul)(q,q)),NAME(twice)(q)));
}
