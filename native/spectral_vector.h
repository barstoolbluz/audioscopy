/* SPDX-License-Identifier: GPL-2.0-or-later
 * Independent DD butterflies across contiguous complex samples. The vector
 * graph is identical to spectral_dd.h; every lane has its own accumulator.
 * Transpose four AoS complex values into limb vectors; memcpy avoids
 * alignment and alias assumptions. No gather of arbitrary addresses.
 */
_Static_assert(sizeof(NAME(complex))==4*sizeof(double),"DD complex layout");
#if AC_VECTOR_LANES==2
#define VEC float64x2_t
#define VADD vaddq_f64
#define VSUB vsubq_f64
#define VMUL vmulq_f64
#define VFMA(a,b,c) vfmaq_f64((c),(a),(b))
#define VNEG vnegq_f64
#else
typedef struct { __m256d rh,rl,ih,il; } NAME(four);
static inline ATTR NAME(four) NAME(transpose)(__m256d a,__m256d b,__m256d c,__m256d d) {
    __m256d ab0=_mm256_unpacklo_pd(a,b),ab1=_mm256_unpackhi_pd(a,b);
    __m256d cd0=_mm256_unpacklo_pd(c,d),cd1=_mm256_unpackhi_pd(c,d);
    return (NAME(four)){
        _mm256_permute2f128_pd(ab0,cd0,0x20),_mm256_permute2f128_pd(ab1,cd1,0x20),
        _mm256_permute2f128_pd(ab0,cd0,0x31),_mm256_permute2f128_pd(ab1,cd1,0x31)};
}
static inline ATTR NAME(four) NAME(load4)(const NAME(complex) *p,size_t step) {
    __m256d a,b,c,d;
    memcpy(&a,p,sizeof(a)); memcpy(&b,p+step,sizeof(b));
    memcpy(&c,p+2*step,sizeof(c)); memcpy(&d,p+3*step,sizeof(d));
    return NAME(transpose)(a,b,c,d);
}
static inline ATTR void NAME(store4)(NAME(complex) *p,NAME(four) v) {
    NAME(four) t=NAME(transpose)(v.rh,v.rl,v.ih,v.il);
    memcpy(p,&t.rh,sizeof(t.rh)); memcpy(p+1,&t.rl,sizeof(t.rl));
    memcpy(p+2,&t.ih,sizeof(t.ih)); memcpy(p+3,&t.il,sizeof(t.il));
}
#if AC_VECTOR_LANES==4
#define VEC __m256d
#define VADD _mm256_add_pd
#define VSUB _mm256_sub_pd
#define VMUL _mm256_mul_pd
#define VFMA _mm256_fmadd_pd
#define VNEG(x) _mm256_xor_pd((x),_mm256_set1_pd(-0.0))
#else
#define VEC __m512d
#define VADD _mm512_add_pd
#define VSUB _mm512_sub_pd
#define VMUL _mm512_mul_pd
#define VFMA _mm512_fmadd_pd
#define VNEG(x) _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(x),_mm512_set1_epi64(INT64_MIN)))
static inline ATTR VEC NAME(combine)(__m256d a,__m256d b) {
    return _mm512_insertf64x4(_mm512_castpd256_pd512(a),b,1);
}
#endif
#endif
typedef struct { VEC hi,lo; } NAME(vreal);
typedef struct { NAME(vreal) r,i; } NAME(vcomplex);
static inline ATTR NAME(vreal) NAME(vsum)(VEC a,VEC b) {
    VEC hi=VADD(a,b),v=VSUB(hi,a);
    return (NAME(vreal)){hi,VADD(VSUB(a,VSUB(hi,v)),VSUB(b,v))};
}
static inline ATTR NAME(vreal) NAME(vadd)(NAME(vreal) a,NAME(vreal) b) {
    NAME(vreal) s=NAME(vsum)(a.hi,b.hi),t=NAME(vsum)(a.lo,b.lo);
    NAME(vreal) u=NAME(vsum)(s.hi,VADD(s.lo,t.hi));
    return NAME(vsum)(u.hi,VADD(u.lo,t.lo));
}
static inline ATTR NAME(vreal) NAME(vsub)(NAME(vreal) a,NAME(vreal) b) {
    return NAME(vadd)(a,(NAME(vreal)){VNEG(b.hi),VNEG(b.lo)});
}
static inline ATTR NAME(vreal) NAME(vmul)(NAME(vreal) a,NAME(vreal) b) {
    VEC p=VMUL(a.hi,b.hi),e=VFMA(a.hi,b.hi,VNEG(p));
    VEC cross=VFMA(a.hi,b.lo,VMUL(a.lo,b.hi));
    return NAME(vsum)(p,VADD(e,VADD(cross,VMUL(a.lo,b.lo))));
}
#if AC_VECTOR_LANES==2
static inline ATTR NAME(vcomplex) NAME(vload)(const NAME(complex) *p,size_t step) {
    double limbs[8];
    memcpy(limbs,p,4*sizeof(double));memcpy(limbs+4,p+step,4*sizeof(double));
    float64x2x4_t v=vld4q_f64(limbs);
    return (NAME(vcomplex)){{v.val[0],v.val[1]},{v.val[2],v.val[3]}};
}
static inline ATTR void NAME(vstore)(NAME(complex) *p,NAME(vcomplex) v) {
    double limbs[8];float64x2x4_t a={{v.r.hi,v.r.lo,v.i.hi,v.i.lo}};
    vst4q_f64(limbs,a);memcpy(p,limbs,sizeof(limbs));
}
#else
static inline ATTR NAME(vcomplex) NAME(vload)(const NAME(complex) *p,size_t step) {
    NAME(four) a=NAME(load4)(p,step);
#if AC_VECTOR_LANES==4
    return (NAME(vcomplex)){{a.rh,a.rl},{a.ih,a.il}};
#else
    NAME(four) b=NAME(load4)(p+4*step,step);
    return (NAME(vcomplex)){{NAME(combine)(a.rh,b.rh),NAME(combine)(a.rl,b.rl)},
        {NAME(combine)(a.ih,b.ih),NAME(combine)(a.il,b.il)}};
#endif
}
static inline ATTR void NAME(vstore)(NAME(complex) *p,NAME(vcomplex) v) {
#if AC_VECTOR_LANES==4
    NAME(store4)(p,(NAME(four)){v.r.hi,v.r.lo,v.i.hi,v.i.lo});
#else
    NAME(store4)(p,(NAME(four)){_mm512_castpd512_pd256(v.r.hi),_mm512_castpd512_pd256(v.r.lo),
        _mm512_castpd512_pd256(v.i.hi),_mm512_castpd512_pd256(v.i.lo)});
    NAME(store4)(p+4,(NAME(four)){_mm512_extractf64x4_pd(v.r.hi,1),_mm512_extractf64x4_pd(v.r.lo,1),
        _mm512_extractf64x4_pd(v.i.hi,1),_mm512_extractf64x4_pd(v.i.lo,1)});
#endif
}
#endif
static inline ATTR void NAME(vector_butterflies)(NAME(complex) *a,NAME(complex) *b,const NAME(complex) *roots,size_t step) {
    NAME(vcomplex) u=NAME(vload)(a,1),v=NAME(vload)(b,1),w=NAME(vload)(roots,step);
    NAME(vcomplex) t={NAME(vsub)(NAME(vmul)(v.r,w.r),NAME(vmul)(v.i,w.i)),
        NAME(vadd)(NAME(vmul)(v.r,w.i),NAME(vmul)(v.i,w.r))};
    NAME(vstore)(a,(NAME(vcomplex)){NAME(vadd)(u.r,t.r),NAME(vadd)(u.i,t.i)});
    NAME(vstore)(b,(NAME(vcomplex)){NAME(vsub)(u.r,t.r),NAME(vsub)(u.i,t.i)});
}
#undef VEC
#undef VADD
#undef VSUB
#undef VMUL
#undef VFMA
#undef VNEG
