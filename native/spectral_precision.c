/* SPDX-License-Identifier: GPL-2.0-or-later
 * Optional high-precision average spectrum, with a narrow C ABI.
 * No global mutable state, no long double crosses the ABI, no allocations
 * during ingestion. Source instantiated for baseline DD, optional x86 FMA,
 * and genuine SysV extended80 only when the existing x87 feature is enabled.
 */
#include <float.h>
#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

#if defined(__aarch64__) && defined(__ARM_NEON) && (defined(__linux__) || defined(__APPLE__))
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#define AC_HAVE_NEON_TARGET 1
#endif

_Static_assert(FLT_RADIX==2 && DBL_MANT_DIG==53 && DBL_MAX_EXP==1024,"binary64 required");
#include "spectral_precision.h"
typedef struct { double hi, lo; } dd_real;
#include "spectral_constants.h"
#include "noise_constants.h"
#include "loudness_constants.h"

struct spectral_ops {
    void *(*create)(size_t);
    void (*destroy)(void *);
    void (*window)(void *,const double *,size_t,size_t,size_t,int);
    int (*finish)(void *,uint64_t,size_t,struct ac_spectral_bin *);
    void *(*cross_create)(size_t);
    void (*cross_destroy)(void *);
    void (*cross_window)(void *,const double *,size_t,size_t,size_t,int);
    int (*cross_finish)(void *,uint64_t,size_t,struct ac_cross_bin *);
    void *(*noise_create)(void);
    int (*noise_add)(void *,const struct ac_spectral_bin *,size_t,uint32_t,uint64_t,int,int);
    int (*noise_finish)(void *,uint64_t,struct ac_spectral_bin *);
    int (*kw_create)(uint32_t,size_t,const unsigned char *,size_t,void **);
    void (*kw_destroy)(void *);
    int (*kw_push)(void *,const double *,size_t);
    int (*kw_finish)(void *,struct ac_loudness_result *);
#ifdef AUDIOSCOPY_SPECTRAL_RESEARCH
    int (*kw_probe)(uint32_t,const double *,size_t,struct ac_spectral_bin *);
#endif
};
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define NAME(x) CAT(PREFIX,x)
#ifdef AUDIOSCOPY_SPECTRAL_RESEARCH
#define PROBE(p) ,CAT(p,_kw_probe)
#else
#define PROBE(p)
#endif
#define REAL dd_real
#define AC_WIDE_DD
#define PREFIX dd_base_
#define ATTR
#include "spectral_dd.h"
#include "spectral_fft.h"
#include "spectral_cross.h"
#include "spectral_noise.h"
#include "loudness_wide.h"
static const struct spectral_ops dd_base_ops={dd_base_create,dd_base_destroy,dd_base_window,dd_base_finish,dd_base_cross_create,dd_base_cross_destroy,dd_base_cross_window,dd_base_cross_finish,dd_base_noise_create,dd_base_noise_add,dd_base_noise_finish,dd_base_kw_create,dd_base_kw_destroy,dd_base_kw_push,dd_base_kw_finish PROBE(dd_base)};
#undef PREFIX
#undef ATTR
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define AC_HAVE_FMA_TARGET 1
#define PREFIX dd_fma_
#define ATTR __attribute__((target("fma")))
#include "spectral_dd.h"
#include "spectral_fft.h"
#include "spectral_cross.h"
#include "spectral_noise.h"
#include "loudness_wide.h"
static const struct spectral_ops dd_fma_ops={dd_fma_create,dd_fma_destroy,dd_fma_window,dd_fma_finish,dd_fma_cross_create,dd_fma_cross_destroy,dd_fma_cross_window,dd_fma_cross_finish,dd_fma_noise_create,dd_fma_noise_add,dd_fma_noise_finish,dd_fma_kw_create,dd_fma_kw_destroy,dd_fma_kw_push,dd_fma_kw_finish PROBE(dd_fma)};
#undef PREFIX
#undef ATTR
#define PREFIX dd_avx2_
#define ATTR __attribute__((target("avx2,fma")))
#define AC_VECTOR_LANES 4
#include "spectral_dd.h"
#include "spectral_fft.h"
#include "spectral_cross.h"
#include "spectral_noise.h"
#include "loudness_wide.h"
static const struct spectral_ops dd_avx2_ops={dd_avx2_create,dd_avx2_destroy,dd_avx2_window,dd_avx2_finish,dd_avx2_cross_create,dd_avx2_cross_destroy,dd_avx2_cross_window,dd_avx2_cross_finish,dd_avx2_noise_create,dd_avx2_noise_add,dd_avx2_noise_finish,dd_avx2_kw_create,dd_avx2_kw_destroy,dd_avx2_kw_push,dd_avx2_kw_finish PROBE(dd_avx2)};
#undef PREFIX
#undef ATTR
#undef AC_VECTOR_LANES
#define PREFIX dd_avx512_
#define ATTR __attribute__((target("avx512f,avx2,fma")))
#define AC_VECTOR_LANES 8
#include "spectral_dd.h"
#include "spectral_fft.h"
#include "spectral_cross.h"
#include "spectral_noise.h"
#include "loudness_wide.h"
static const struct spectral_ops dd_avx512_ops={dd_avx512_create,dd_avx512_destroy,dd_avx512_window,dd_avx512_finish,dd_avx512_cross_create,dd_avx512_cross_destroy,dd_avx512_cross_window,dd_avx512_cross_finish,dd_avx512_noise_create,dd_avx512_noise_add,dd_avx512_noise_finish,dd_avx512_kw_create,dd_avx512_kw_destroy,dd_avx512_kw_push,dd_avx512_kw_finish PROBE(dd_avx512)};
#undef PREFIX
#undef ATTR
#undef AC_VECTOR_LANES
static int fma_available(void) { return __builtin_cpu_supports("avx") && __builtin_cpu_supports("fma"); }
static int avx2_available(void) { return fma_available() && __builtin_cpu_supports("avx2"); }
static int avx512_available(void) { return avx2_available() && __builtin_cpu_supports("avx512f"); }
#endif
#ifdef AC_HAVE_NEON_TARGET
#define PREFIX dd_neon_
#define ATTR
#define AC_VECTOR_LANES 2
#include "spectral_dd.h"
#include "spectral_fft.h"
#include "spectral_cross.h"
#include "spectral_noise.h"
#include "loudness_wide.h"
static const struct spectral_ops dd_neon_ops={dd_neon_create,dd_neon_destroy,dd_neon_window,dd_neon_finish,dd_neon_cross_create,dd_neon_cross_destroy,dd_neon_cross_window,dd_neon_cross_finish,dd_neon_noise_create,dd_neon_noise_add,dd_neon_noise_finish,dd_neon_kw_create,dd_neon_kw_destroy,dd_neon_kw_push,dd_neon_kw_finish PROBE(dd_neon)};
#undef PREFIX
#undef ATTR
#undef AC_VECTOR_LANES
static int neon_available(void) {
#if defined(__linux__)
    return (getauxval(AT_HWCAP)&HWCAP_ASIMD)!=0;
#else
    return 1; /* native Apple arm64 */
#endif
}
#endif
#undef REAL
#undef AC_WIDE_DD
#if defined(AUDIOSCOPY_ENABLE_X87)
#if !defined(__x86_64__) || !defined(__linux__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "x87 spectrum requires native little-endian GNU/Linux x86-64"
#endif
_Static_assert(LDBL_MANT_DIG==64 && LDBL_MAX_EXP==16384 && sizeof(long double)==16,"extended80 required");
#define REAL long double
#define PREFIX x87_
#define ATTR
#define SCALBN scalbnl
#define FREXP frexpl
#define SIN sinl
#define SQRT sqrtl
#define PI_HALF 1.570796326794896619231321691639751442098584699687552910L
#include "spectral_scalar.h"
#include "spectral_fft.h"
#include "spectral_cross.h"
#include "spectral_noise.h"
#include "loudness_wide.h"
static const struct spectral_ops x87_ops={x87_create,x87_destroy,x87_window,x87_finish,x87_cross_create,x87_cross_destroy,x87_cross_window,x87_cross_finish,x87_noise_create,x87_noise_add,x87_noise_finish,x87_kw_create,x87_kw_destroy,x87_kw_push,x87_kw_finish PROBE(x87)};
#undef REAL
#undef PREFIX
#undef ATTR
#undef SCALBN
#undef FREXP
#undef SIN
#undef SQRT
#undef PI_HALF
#endif
#ifdef AUDIOSCOPY_SPECTRAL_RESEARCH
#define REAL double
#define PREFIX f64_
#define ATTR
#define SCALBN scalbn
#define FREXP frexp
#define SIN sin
#define SQRT sqrt
#define PI_HALF 1.570796326794896619231321691639751442098584699687552910
#include "spectral_scalar.h"
#include "spectral_fft.h"
#include "spectral_cross.h"
#include "spectral_noise.h"
#include "loudness_wide.h"
static const struct spectral_ops f64_ops={f64_create,f64_destroy,f64_window,f64_finish,f64_cross_create,f64_cross_destroy,f64_cross_window,f64_cross_finish,f64_noise_create,f64_noise_add,f64_noise_finish,f64_kw_create,f64_kw_destroy,f64_kw_push,f64_kw_finish PROBE(f64)};
#undef REAL
#undef PREFIX
#undef ATTR
#undef SCALBN
#undef FREXP
#undef SIN
#undef SQRT
#undef PI_HALF
#endif
#undef NAME
#undef CAT
#undef CAT2

/* Same environment discipline as the CP7 correlation kernel. On non-x86,
 * C fenv provides nearest/masked exceptions; if gradual underflow is not
 * available, reject the call rather than silently lose precision. */
struct ac_environment {
    fenv_t portable;
#if defined(__x86_64__)
    /* The hardware 28-byte x87 environment. libc fesetenv is permitted to
     * leave condition codes unchanged. Preserve them as well as TOP, tag,
     * control and exception state. The C ABI requires balanced x87 stack. */
    unsigned char x87[28];
#endif
};
static int leave(const struct ac_environment *saved) {
    int ok=fesetenv(&saved->portable)==0;
#if defined(__x86_64__)
    __asm__ volatile("fldenv %0" :: "m"(saved->x87) : "memory");
#endif
    return ok;
}
static int enter(struct ac_environment *saved) {
#if defined(__x86_64__)
    /* fnstenv masks exceptions as a side effect; immediately undo that
     * while recording the exact environment, before portable feholdexcept. */
    __asm__ volatile("fnstenv %0\n\tfldenv %0" : "=m"(saved->x87) :: "memory");
#endif
    if (feholdexcept(&saved->portable)!=0) return 0;
    if (fesetround(FE_TONEAREST)!=0) { (void)leave(saved); return 0; }
#if defined(__x86_64__)
    unsigned short cw;
    unsigned int mxcsr;
    __asm__ volatile("fnstcw %0" : "=m"(cw) :: "memory");
    cw=(unsigned short)((cw & ~0x0f00u)|0x0300u);
    __asm__ volatile("fldcw %0" :: "m"(cw) : "memory");
    __asm__ volatile("stmxcsr %0" : "=m"(mxcsr) :: "memory");
    mxcsr &= ~0x8040u;
    __asm__ volatile("ldmxcsr %0" :: "m"(mxcsr) : "memory");
#endif
    volatile double normal=DBL_MIN, half=0.5, tiny=DBL_TRUE_MIN, two=2.0;
    if (normal*half==0 || tiny*two==0) { (void)leave(saved); return 0; }
    return 1;
}
/* Requests and actual execution identifiers (private ABI):
 * 1 DD auto, 2 x87, 3 baseline DD, 4 x86 target FMA DD, 5 research f64, 6 AVX2 DD, 7 AVX512F DD, 8 AArch64 NEON DD. */
static const struct spectral_ops *select_ops(int mode,int *actual) {
    if (mode==1 || mode==4) {
#ifdef AC_HAVE_FMA_TARGET
        if (fma_available()) { *actual=4; return &dd_fma_ops; }
#endif
        if (mode==4) return NULL;
        mode=3;
    }
    if (mode==3) { *actual=3; return &dd_base_ops; }
#ifdef AC_HAVE_FMA_TARGET
    if (mode==6 && avx2_available()) { *actual=6; return &dd_avx2_ops; }
    if (mode==7 && avx512_available()) { *actual=7; return &dd_avx512_ops; }
#endif
#ifdef AC_HAVE_NEON_TARGET
    if (mode==8 && neon_available()) { *actual=8; return &dd_neon_ops; }
#endif
#ifdef AUDIOSCOPY_ENABLE_X87
    if (mode==2) { *actual=2; return &x87_ops; }
#endif
#ifdef AUDIOSCOPY_SPECTRAL_RESEARCH
    if (mode==5) { *actual=5; return &f64_ops; }
#endif
    return NULL;
}
int audioscopy_spectral_available(int mode) { int actual=0; return select_ops(mode,&actual)!=NULL; }

struct spectrum_plan {
    const struct spectral_ops *ops;
    void *state;
    double *block;
    size_t n,channels,used,pending;
    uint64_t frames;
    int leading,terminal,cross;
};
void audioscopy_spectral_free(void *v) {
    struct spectrum_plan *p=v;
    if (p) { if (p->cross) p->ops->cross_destroy(p->state); else p->ops->destroy(p->state); free(p->block); free(p); }
}
/* 1 success, 0 invalid or environment failure, -1 allocation, -2 backend. */
static int new_plan(uint32_t rate,size_t channels,int mode,int cross,void **out,size_t *fft_len,int *actual) {
    if (!out || !fft_len || !actual) return 0;
    *out=NULL;
    if (rate<8000 || rate>768000 || channels<1 || channels>64) return 0;
    const struct spectral_ops *ops=select_ops(mode,actual);
    if (!ops) return -2;
    struct ac_environment saved;
    if (!enter(&saved)) return 0;
    size_t n=1; while (n<(rate+5u)/6u) n*=2;
    struct spectrum_plan *p=calloc(1,sizeof(*p));
    if (p) {
        p->ops=ops;p->n=n;p->channels=channels;p->leading=1;p->cross=cross;
        p->block=malloc(n*channels*sizeof(*p->block));
        p->state=cross?ops->cross_create(n):ops->create(n);
    }
    int ok=p && p->block && p->state;
    int restored=leave(&saved);
    if (!ok || !restored) { audioscopy_spectral_free(p); return restored?-1:0; }
    *out=p; *fft_len=n; return 1;
}
int audioscopy_spectral_new(uint32_t rate,size_t channels,int mode,void **out,size_t *fft_len,int *actual) {
    return new_plan(rate,channels,mode,0,out,fft_len,actual);
}
int audioscopy_cross_new(uint32_t rate,int mode,void **out,size_t *fft_len,int *actual) {
    return new_plan(rate,2,mode,1,out,fft_len,actual);
}
static void process(struct spectrum_plan *p) {
    if (p->cross) {
        p->ops->cross_window(p->state,p->block,p->used,2,p->leading?p->n/2:0,0);
        return;
    }
    double max=0.0;
    for (size_t i=0;i<p->used*p->channels;++i) { double x=fabs(p->block[i]); if (x>max) max=x; }
    if (max!=0) p->ops->window(p->state,p->block,p->used,p->channels,p->leading?p->n/2:0,ilogb(max));
}
static void advance(struct spectrum_plan *p) {
    if (!p->leading) {
        size_t hop=p->n/2;
        if (hop>p->used) hop=p->used;
        p->used-=hop;
        memmove(p->block,p->block+hop*p->channels,p->used*p->channels*sizeof(double));
    }
    p->leading=0; p->pending=0;
}
int audioscopy_spectral_push(void *v,const double *pcm,size_t frames) {
    struct spectrum_plan *p=v;
    if (!p || p->terminal || frames>(UINT64_C(1)<<48)-p->frames || frames>SIZE_MAX/p->channels) return 0;
    if (!frames) return 1;
    if (!pcm) return 0;
    /* Rust validates the entire push transactionally. This internal ABI
     * accepts only those finite admitted samples, with complete frames. */
    struct ac_environment saved;
    if (!enter(&saved)) { p->terminal=1; return 0; }
    p->frames+=frames;
    while (frames) {
        size_t target=p->leading?p->n/2:p->n, take=target-p->used;
        if (take>frames) take=frames;
        memcpy(p->block+p->used*p->channels,pcm,take*p->channels*sizeof(double));
        p->used+=take;p->pending+=take;pcm+=take*p->channels;frames-=take;
        if (p->used==target) { process(p); advance(p); }
    }
    if (!leave(&saved)) { p->terminal=1; return 0; }
    return 1;
}
static int finish_plan(void *v,struct ac_spectral_bin *out,struct ac_cross_bin *cross_out,size_t bins) {
    struct spectrum_plan *p=v;
    if (!p || p->terminal || !p->frames || (p->cross?!cross_out:!out) || bins!=p->n/2+1) return 0;
    p->terminal=1;
    struct ac_environment saved;
    if (!enter(&saved)) return 0;
    if (p->pending) { process(p); advance(p); }
    if (!p->leading && p->used) process(p);
    int ok=p->cross?p->ops->cross_finish(p->state,p->frames,p->channels,cross_out):p->ops->finish(p->state,p->frames,p->channels,out);
    int restored=leave(&saved);
    return ok && restored;
}

int audioscopy_spectral_finish(void *v,struct ac_spectral_bin *out,size_t bins) {
    return finish_plan(v,out,NULL,bins);
}
int audioscopy_cross_finish(void *v,struct ac_cross_bin *out,size_t bins) {
    return finish_plan(v,NULL,out,bins);
}

/* Private bounded noise reduction plan. Public Rust supplies matching
 * spectrum backend and clock. Validate even manually constructed bins. */
struct noise_plan {
    const struct spectral_ops *ops;
    void *state;
    size_t count;
    uint32_t rate;
    uint64_t frames;
    int view,weighting,terminal;
};
int audioscopy_noise_new(uint32_t rate,int mode,int view,int weighting,void **out,int *actual) {
    if (!out || !actual) return 0;
    *out=NULL;
    if (rate<8000 || rate>768000 || view<0 || view>2 || weighting<0 || weighting>2) return 0;
    const struct spectral_ops *ops=select_ops(mode,actual);
    if (!ops) return -2;
    struct noise_plan *p=calloc(1,sizeof(*p));
    if (!p) return -1;
    p->state=ops->noise_create();
    if (!p->state) { free(p);return -1; }
    size_t n=1;while (n<(rate+5u)/6u) n*=2;
    p->ops=ops;p->rate=rate;p->count=n/2+1;p->view=view;p->weighting=weighting;
    *out=p;return 1;
}
void audioscopy_noise_free(void *v) {
    struct noise_plan *p=v;
    if (p) { free(p->state);free(p); }
}
int audioscopy_noise_add(void *v,const struct ac_spectral_bin *bins,size_t count,uint64_t frames) {
    struct noise_plan *p=v;
    if (!p || p->terminal || !bins || count!=p->count || !frames || frames>(UINT64_C(1)<<48)-p->frames) return 0;
    struct ac_environment saved;
    if (!enter(&saved)) { p->terminal=1;return 0; }
    int ok=1;
    for (size_t i=0;i<count;++i) {
        double h=bins[i].hi,l=bins[i].lo;int e=bins[i].exponent;
        if (!isfinite(h) || !isfinite(l) || h<0 || (h==0?(l!=0 || e!=0):(h<0.5 || h>1.0 || fabs(l)>0x1p-52 || e< -32768 || e>64))) { ok=0;break; }
    }
    if (ok) ok=p->ops->noise_add(p->state,bins,count,p->rate,frames,p->view,p->weighting);
    int restored=leave(&saved);
    if (!ok || !restored) { p->terminal=1;return 0; }
    p->frames+=frames;return 1;
}
int audioscopy_noise_finish(void *v,struct ac_spectral_bin *out,size_t count) {
    struct noise_plan *p=v;
    if (!p || p->terminal || !p->frames || !out || count!=6) return 0;
    p->terminal=1;
    struct ac_environment saved;
    if (!enter(&saved)) return 0;
    int ok=p->ops->noise_finish(p->state,p->frames,out);
    int restored=leave(&saved);
    return ok && restored;
}

/* Integrated native-profile plan. Wide precision does not change the
 * ordinary meter or its compatibility profile. */
struct loudness_plan {
    const struct spectral_ops *ops;
    void *state;
    uint64_t frames;
    size_t channels;
    int terminal;
};
int audioscopy_loudness_new(uint32_t rate,size_t channels,const unsigned char *weights,size_t limit,int mode,void **out,int *actual) {
    if (!out || !actual) return 0;
    *out=NULL;
    if (rate<3364 || rate>2822400 || !channels || channels>64 || !weights || !limit || limit>4000000) return 0;
    int audible=0;for (size_t i=0;i<channels;++i) { if (weights[i]>2) return 0;if (weights[i]) audible=1; }
    if (!audible) return 0;
    if (mode<1 || mode>5) return -2;
    const struct spectral_ops *ops=select_ops(mode,actual);if (!ops) return -2;
    struct loudness_plan *p=calloc(1,sizeof(*p));if (!p) return -1;
    p->ops=ops;p->channels=channels;
    struct ac_environment saved;if (!enter(&saved)) { free(p);return 0; }
    int status=ops->kw_create(rate,channels,weights,limit,&p->state);
    int restored=leave(&saved);
    if (status!=1 || !restored) { ops->kw_destroy(p->state);free(p);return restored?status:0; }
    *out=p;return 1;
}
void audioscopy_loudness_free(void *v) {
    struct loudness_plan *p=v;if (p) { p->ops->kw_destroy(p->state);free(p); }
}
int audioscopy_loudness_push(void *v,const double *pcm,size_t frames) {
    struct loudness_plan *p=v;
    if (!p || p->terminal || frames>(UINT64_C(1)<<48)-p->frames || frames>SIZE_MAX/p->channels) return 0;
    if (!frames) return 1;
    if (!pcm) return 0;
    struct ac_environment saved;if (!enter(&saved)) { p->terminal=1;return 0; }
    int status=p->ops->kw_push(p->state,pcm,frames);int restored=leave(&saved);
    if (status!=1 || !restored) { p->terminal=1;return restored?status:0; }
    p->frames+=frames;return 1;
}
int audioscopy_loudness_finish(void *v,struct ac_loudness_result *out) {
    struct loudness_plan *p=v;if (!p || p->terminal || !out) return 0;
    p->terminal=1;struct ac_environment saved;if (!enter(&saved)) return 0;
    int status=p->ops->kw_finish(p->state,out);int restored=leave(&saved);return status && restored;
}
#ifdef AUDIOSCOPY_SPECTRAL_RESEARCH
int audioscopy_loudness_probe(uint32_t rate,int mode,const double *pcm,size_t n,struct ac_spectral_bin *out) {
    if (rate<3364 || rate>2822400 || !pcm || !out) return 0;
    int actual;const struct spectral_ops *ops=select_ops(mode,&actual);if (!ops) return 0;
    struct ac_environment saved;if (!enter(&saved)) return 0;
    int status=ops->kw_probe(rate,pcm,n,out);int restored=leave(&saved);return status && restored;
}
#endif
