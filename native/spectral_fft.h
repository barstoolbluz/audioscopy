/* SPDX-License-Identifier: GPL-2.0-or-later
 * Same real FFT and one-sided spectrum in all precision instantiations.
 * N real samples -> N/2 packed complex FFT -> N/2+1 real-signal bins.
 * All window, twiddle, butterfly, power and compensated sum operations
 * use REAL. Input/output double conversion occurs only at ABI boundaries.
 */
typedef struct { REAL r, i; } NAME(complex);
#ifdef AC_VECTOR_LANES
#include "spectral_vector.h"
#endif
typedef struct {
    size_t n, m;
    REAL *window, *sum, *correction;
    NAME(complex) *root, *work;
    uint32_t *reverse;
    int exponent, has_scale;
} NAME(state);

static inline ATTR NAME(complex) NAME(cadd)(NAME(complex) a, NAME(complex) b) {
    return (NAME(complex)){NAME(add)(a.r,b.r),NAME(add)(a.i,b.i)};
}
static inline ATTR NAME(complex) NAME(csub)(NAME(complex) a, NAME(complex) b) {
    return (NAME(complex)){NAME(sub)(a.r,b.r),NAME(sub)(a.i,b.i)};
}
static inline ATTR NAME(complex) NAME(cmul)(NAME(complex) a, NAME(complex) b) {
    return (NAME(complex)){
        NAME(sub)(NAME(mul)(a.r,b.r),NAME(mul)(a.i,b.i)),
        NAME(add)(NAME(mul)(a.r,b.i),NAME(mul)(a.i,b.r))};
}
static inline ATTR REAL NAME(norm)(NAME(complex) x) {
    return NAME(add)(NAME(mul)(x.r,x.r),NAME(mul)(x.i,x.i));
}
static ATTR void NAME(destroy)(void *v) {
    NAME(state) *s = v;
    if (!s) return;
    free(s->window); free(s->sum); free(s->correction); free(s->root);
    free(s->work); free(s->reverse); free(s);
}
static ATTR void *NAME(create)(size_t n) {
    NAME(state) *s = calloc(1,sizeof(*s));
    if (!s) return NULL;
    size_t m = n/2, q = n/4;
    s->n=n; s->m=m;
    s->window=malloc(n*sizeof(*s->window));
    s->root=malloc((m+1)*sizeof(*s->root));
    s->work=malloc(m*sizeof(*s->work));
    s->reverse=malloc(m*sizeof(*s->reverse));
    s->sum=calloc(m+1,sizeof(*s->sum));
    s->correction=calloc(m+1,sizeof(*s->correction));
    if (!s->window || !s->root || !s->work || !s->reverse || !s->sum || !s->correction) {
        NAME(destroy)(s); return NULL;
    }
    REAL ph = NAME(pi_half)(), zero=NAME(from)(0.0), one=NAME(from)(1.0);
    /* Exact quadrant endpoints. Each root is independently evaluated:
     * no O(N) accumulated phase drift, no rounded binary64 pi or window. */
    s->root[0]=(NAME(complex)){one,zero};
    s->root[q]=(NAME(complex)){zero,NAME(neg)(one)};
    s->root[m]=(NAME(complex)){NAME(neg)(one),zero};
    for (size_t k=1;k<q;++k) {
        double fraction=(double)k/(double)q; /* exact dyadic */
        REAL sn=NAME(sin)(NAME(mul)(ph,NAME(from)(fraction)));
        REAL cs=NAME(sin)(NAME(mul)(ph,NAME(from)(1.0-fraction)));
        s->root[k]=(NAME(complex)){cs,NAME(neg)(sn)};
        s->root[m-k]=(NAME(complex)){NAME(neg)(cs),NAME(neg)(sn)};
    }
    /* Same sine-of-Hann / Vorbis window as CP7, half-sample offset,
     * symmetry is exact in the mathematical window. */
    for (size_t k=0;k<m;++k) {
        REAL a=NAME(sin)(NAME(mul)(ph,NAME(from)((double)(2*k+1)/(double)n)));
        REAL w=NAME(sin)(NAME(mul)(ph,NAME(mul)(a,a)));
        s->window[k]=w; s->window[n-1-k]=w;
    }
    size_t j=0;
    for (size_t i=0;i<m;++i) {
        s->reverse[i]=(uint32_t)j;
        size_t bit=m/2;
        while (bit && (j & bit)) { j^=bit; bit>>=1; }
        j^=bit;
    }
    return s;
}
static inline ATTR void NAME(accumulate)(NAME(state) *s, size_t k, REAL x) {
    /* Positive power, compensated in the selected precision. */
    REAL y=NAME(sub)(x,s->correction[k]);
    REAL t=NAME(add)(s->sum[k],y);
    s->correction[k]=NAME(sub)(NAME(sub)(t,s->sum[k]),y);
    s->sum[k]=t;
}
static ATTR void NAME(transform)(NAME(state) *s) {
    size_t m=s->m,n=s->n;
    /* Packed complex data already in bit-reversed order. */
    for (size_t len=2;len<=m;len*=2) {
        size_t half=len/2,step=n/len;
        for (size_t i=0;i<m;i+=len) {
            /* Exactly unit root at j=0; avoids redundant products. */
            NAME(complex) a=s->work[i], b=s->work[i+half];
            s->work[i]=NAME(cadd)(a,b); s->work[i+half]=NAME(csub)(a,b);
            size_t j=1;
#ifdef AC_VECTOR_LANES
            for (;j+AC_VECTOR_LANES<=half;j+=AC_VECTOR_LANES)
                NAME(vector_butterflies)(s->work+i+j,s->work+i+j+half,s->root+j*step,step);
#endif
            for (;j<half;++j) {
                a=s->work[i+j]; b=NAME(cmul)(s->work[i+j+half],s->root[j*step]);
                s->work[i+j]=NAME(cadd)(a,b); s->work[i+j+half]=NAME(csub)(a,b);
            }
        }
    }
}
static ATTR void NAME(collect)(NAME(state) *s) {
    size_t m=s->m;
    REAL dc=NAME(add)(s->work[0].r,s->work[0].i);
    REAL ny=NAME(sub)(s->work[0].r,s->work[0].i);
    NAME(accumulate)(s,0,NAME(mul)(dc,dc));
    NAME(accumulate)(s,m,NAME(mul)(ny,ny));
    for (size_t k=1;k<m/2;++k) {
        NAME(complex) z=s->work[k], v=s->work[m-k];
        NAME(complex) a={NAME(half)(NAME(add)(z.r,v.r)),NAME(half)(NAME(sub)(z.i,v.i))};
        NAME(complex) b={NAME(half)(NAME(add)(z.i,v.i)),NAME(half)(NAME(sub)(v.r,z.r))};
        NAME(complex) t=NAME(cmul)(s->root[k],b);
        /* X[k]=a+t, X[m-k]=conj(a-t); both have interior weight 2. */
        NAME(accumulate)(s,k,NAME(twice)(NAME(norm)(NAME(cadd)(a,t))));
        NAME(accumulate)(s,m-k,NAME(twice)(NAME(norm)(NAME(csub)(a,t))));
    }
    /* k=m/2 is self-paired; X[k]=conj(Z[k]). */
    NAME(accumulate)(s,m/2,NAME(twice)(NAME(norm)(s->work[m/2])));
}
static ATTR void NAME(window)(void *v, const double *pcm, size_t frames, size_t channels, size_t offset, int exponent) {
    NAME(state) *s=v;
    if (!s->has_scale || exponent>s->exponent) {
        if (s->has_scale) {
            int shift=2*(s->exponent-exponent);
            for (size_t k=0;k<=s->m;++k) {
                s->sum[k]=NAME(scale)(s->sum[k],shift);
                s->correction[k]=NAME(scale)(s->correction[k],shift);
            }
        }
        s->exponent=exponent; s->has_scale=1;
    }
    size_t end=offset+frames;
    REAL zero=NAME(from)(0.0);
    for (size_t ch=0;ch<channels;++ch) {
        for (size_t k=0;k<s->m;++k) {
            size_t i=2*k, j=i+1;
            REAL r=zero, im=zero;
            if (i>=offset && i<end) r=NAME(mul)(NAME(input)(pcm[(i-offset)*channels+ch],-s->exponent),s->window[i]);
            if (j>=offset && j<end) im=NAME(mul)(NAME(input)(pcm[(j-offset)*channels+ch],-s->exponent),s->window[j]);
            s->work[s->reverse[k]]=(NAME(complex)){r,im};
        }
        NAME(transform)(s); NAME(collect)(s);
    }
}
static ATTR int NAME(finish)(void *v, uint64_t frames, size_t channels, struct ac_spectral_bin *out) {
    NAME(state) *s=v;
    for (size_t k=0;k<=s->m;++k) {
        REAL p=NAME(sub)(s->sum[k],s->correction[k]);
        /* All divisors individually exactly representable as binary64.
         * No pre-rounded reciprocal; every division uses REAL precision.
         * One-sided Parseval and channel mean, then real-frame denominator. */
        p=NAME(div)(NAME(div)(NAME(div)(p,(double)s->n),(double)channels),(double)frames);
        if (!NAME(export)(p,2*s->exponent,&out[k])) return 0;
    }
    return 1;
}
