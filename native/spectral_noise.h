/* SPDX-License-Identifier: GPL-2.0-or-later
 * Wide noise-band reduction on caller-selected quiet passage spectra.
 * Retain a bounded six-element positive energy accumulator. Each passage
 * contributes linear power times real frames, never an average of dB.
 */
#ifdef AC_WIDE_DD
#define NC(x) CAT(dd_,x)
#else
#define NC(x) ((REAL)CAT(AC_LD_,x))
#endif

typedef struct { REAL sum,correction; int exponent,has_scale; } NAME(noise_acc);
typedef struct { NAME(noise_acc) acc[6]; } NAME(noise_state);

static inline ATTR REAL NAME(square)(REAL x) { return NAME(mul)(x,x); }
static ATTR REAL NAME(weight)(double f,int weighting) {
    REAL one=NAME(from)(1.0);
    if (!weighting) return one;
    if (f==0.0) return NAME(from)(0.0);
    REAL x=NAME(from)(f),s=NAME(square)(x);
    if (weighting==1) {
        /* A: square the existing analogue response, cancelling sqrt. */
        REAL numerator=NAME(mul)(NC(a_f0_sq),NAME(square)(s));
        REAL den=NAME(mul)(NAME(add)(s,NC(a_f1_sq)),NAME(add)(s,NC(a_f0_sq)));
        REAL ratio=NAME(div_real)(numerator,den);
        REAL other=NAME(mul)(NAME(add)(s,NC(a_f2_sq)),NAME(add)(s,NC(a_f3_sq)));
        return NAME(mul)(NAME(div_real)(NAME(square)(ratio),other),NC(a_norm));
    }
    REAL h1=NAME(add)(NAME(mul)(NAME(add)(NAME(mul)(NAME(add)(NAME(mul)(NC(itu_a),s),NC(itu_b)),s),NC(itu_c)),s),one);
    REAL h2=NAME(mul)(NAME(add)(NAME(mul)(NAME(add)(NAME(mul)(NC(itu_d),s),NC(itu_e)),s),NC(itu_f)),x);
    REAL num=NAME(square)(NAME(mul)(NC(itu_num),x));
    return NAME(mul)(NAME(div_real)(num,NAME(add)(NAME(square)(h1),NAME(square)(h2))),NC(itu_norm));
}
static ATTR void *NAME(noise_create)(void) { return calloc(1,sizeof(NAME(noise_state))); }
static inline ATTR REAL NAME(import_bin)(const struct ac_spectral_bin *b) {
    return NAME(add)(NAME(from)(b->hi),NAME(from)(b->lo));
}
static inline ATTR void NAME(noise_accumulate)(NAME(noise_acc) *a,REAL v,int e) {
    if (!a->has_scale || e>a->exponent) {
        if (a->has_scale) {
            a->sum=NAME(scale)(a->sum,a->exponent-e);
            a->correction=NAME(scale)(a->correction,a->exponent-e);
        }
        a->exponent=e;a->has_scale=1;
    }
    REAL y=NAME(sub)(NAME(scale)(v,e-a->exponent),a->correction);
    REAL t=NAME(add)(a->sum,y);
    a->correction=NAME(sub)(NAME(sub)(t,a->sum),y);a->sum=t;
}
static ATTR int NAME(noise_add)(void *state,const struct ac_spectral_bin *bins,size_t count,uint32_t rate,uint64_t frames,int view,int weighting) {
    NAME(noise_state) *s=state;
    double ny=(double)rate/2.0;
    double lo=view==2?0.0:20.0,hi=view==0?fmin(20000.0,ny):ny;
    static const double edges[6]={20.0,200.0,2000.0,10000.0,20000.0,INFINITY};
    double df=(double)rate/(double)(2*(count-1));
    for (size_t i=0;i<count;++i) {
        double f=(double)i*df;
        if (f<lo || f>hi || (f==hi && hi<ny) || bins[i].hi==0.0) continue;
        REAL w=NAME(weight)(f,weighting);
        REAL p=NAME(mul)(NAME(import_bin)(&bins[i]),w);
        struct ac_spectral_bin normalized;
        if (!NAME(export)(p,bins[i].exponent,&normalized)) return 0;
        if (!normalized.hi) continue;
        REAL v=NAME(mul)(NAME(import_bin)(&normalized),NAME(from)((double)frames));
        NAME(noise_accumulate)(&s->acc[0],v,normalized.exponent);
        for (size_t b=0;b<5;++b) {
            double l=fmax(lo,edges[b]),h=fmin(hi,edges[b+1]);
            if (h>l && f>=l && (f<h || (f==h && h==ny))) {
                NAME(noise_accumulate)(&s->acc[b+1],v,normalized.exponent);break;
            }
        }
    }
    return 1;
}
static ATTR int NAME(noise_finish)(void *state,uint64_t frames,struct ac_spectral_bin out[6]) {
    NAME(noise_state) *s=state;
    for (size_t i=0;i<6;++i) {
        NAME(noise_acc) *a=&s->acc[i];
        REAL p=NAME(div)(NAME(sub)(a->sum,a->correction),(double)frames);
        if (!NAME(export)(p,a->exponent,&out[i])) return 0;
    }
    return 1;
}
#undef NC
