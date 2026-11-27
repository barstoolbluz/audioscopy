/* SPDX-License-Identifier: GPL-2.0-or-later
 * Stereo auto and complex cross power. Reuse the CP8 packed-real FFT;
 * independent channel scales preserve cross products at extreme amplitude
 * ratios. No extra state is allocated for the ordinary power analyzer.
 */
typedef struct {
    NAME(state) *fft;
    NAME(complex) *left;
    REAL *sum, *correction; /* four planes: L, R, Re(conj(L)*R), Im */
    int exponent[2], has_scale[2];
} NAME(cross_state);

static ATTR void NAME(cross_destroy)(void *v) {
    NAME(cross_state) *s=v;
    if (!s) return;
    NAME(destroy)(s->fft); free(s->left); free(s->sum); free(s->correction); free(s);
}
static ATTR void *NAME(cross_create)(size_t n) {
    NAME(cross_state) *s=calloc(1,sizeof(*s));
    if (!s) return NULL;
    size_t bins=n/2+1;
    s->fft=NAME(create)(n);
    s->left=malloc(bins*sizeof(*s->left));
    s->sum=calloc(4*bins,sizeof(*s->sum));
    s->correction=calloc(4*bins,sizeof(*s->correction));
    if (!s->fft || !s->left || !s->sum || !s->correction) {
        NAME(cross_destroy)(s); return NULL;
    }
    return s;
}
static inline ATTR void NAME(cross_add)(NAME(cross_state) *s,size_t plane,size_t k,REAL x) {
    size_t i=plane*(s->fft->m+1)+k;
    if (plane>=2) {
        /* Neumaier compensation for signed terms, including large/small/
         * opposite-large cancellation. Positive planes retain CP8 Kahan. */
        REAL a=s->sum[i],t=NAME(add)(a,x);
        REAL residual=NAME(abs_ge)(a,x)?NAME(add)(NAME(sub)(a,t),x):NAME(add)(NAME(sub)(x,t),a);
        s->correction[i]=NAME(add)(s->correction[i],residual);
        s->sum[i]=t;return;
    }
    REAL y=NAME(sub)(x,s->correction[i]);
    REAL t=NAME(add)(s->sum[i],y);
    s->correction[i]=NAME(sub)(NAME(sub)(t,s->sum[i]),y);
    s->sum[i]=t;
}
#ifdef AC_WIDE_DD
/* Reconstruct conjugate partners once. Each bin has independent state;
 * visiting partners consecutively cannot change the per-bin sum order. */
static inline ATTR void NAME(cross_collect)(NAME(cross_state) *s,size_t ch,size_t k,NAME(complex) x) {
    int interior=k!=0 && k!=s->fft->m;
    REAL p=NAME(norm)(x);
    NAME(cross_add)(s,ch,k,interior?NAME(twice)(p):p);
    if (ch==0) s->left[k]=x;
    else {
        NAME(complex) l=s->left[k];l.i=NAME(neg)(l.i);
        NAME(complex) z=NAME(cmul)(l,x);
        NAME(cross_add)(s,2,k,interior?NAME(twice)(z.r):z.r);
        NAME(cross_add)(s,3,k,interior?NAME(twice)(z.i):z.i);
    }
}
static inline ATTR void NAME(cross_collect_all)(NAME(cross_state) *s,size_t ch) {
    NAME(state) *f=s->fft;size_t m=f->m;
    REAL zero=NAME(from)(0.0);
    NAME(cross_collect)(s,ch,0,(NAME(complex)){NAME(add)(f->work[0].r,f->work[0].i),zero});
    NAME(cross_collect)(s,ch,m,(NAME(complex)){NAME(sub)(f->work[0].r,f->work[0].i),zero});
    NAME(cross_collect)(s,ch,m/2,(NAME(complex)){f->work[m/2].r,NAME(neg)(f->work[m/2].i)});
    for (size_t k=1;k<m/2;++k) {
        NAME(complex) z=f->work[k],v=f->work[m-k];
        NAME(complex) a={NAME(half)(NAME(add)(z.r,v.r)),NAME(half)(NAME(sub)(z.i,v.i))};
        NAME(complex) b={NAME(half)(NAME(add)(z.i,v.i)),NAME(half)(NAME(sub)(v.r,z.r))};
        NAME(complex) t=NAME(cmul)(f->root[k],b),q=NAME(csub)(a,t);
        NAME(cross_collect)(s,ch,k,NAME(cadd)(a,t));
        NAME(cross_collect)(s,ch,m-k,(NAME(complex)){q.r,NAME(neg)(q.i)});
    }
}
#else
/* Use the same pair reconstruction as CP8. DC/Nyquist are real, and the
 * middle bin is conj(Z[m/2]). Unlike power, cross products keep the sign. */
static inline ATTR NAME(complex) NAME(real_bin)(NAME(state) *s,size_t k) {
    REAL zero=NAME(from)(0.0);
    if (k==0) return (NAME(complex)){NAME(add)(s->work[0].r,s->work[0].i),zero};
    if (k==s->m) return (NAME(complex)){NAME(sub)(s->work[0].r,s->work[0].i),zero};
    if (k==s->m/2) return (NAME(complex)){s->work[k].r,NAME(neg)(s->work[k].i)};
    size_t i=k<s->m/2?k:s->m-k;
    NAME(complex) z=s->work[i],v=s->work[s->m-i];
    NAME(complex) a={NAME(half)(NAME(add)(z.r,v.r)),NAME(half)(NAME(sub)(z.i,v.i))};
    NAME(complex) b={NAME(half)(NAME(add)(z.i,v.i)),NAME(half)(NAME(sub)(v.r,z.r))};
    NAME(complex) t=NAME(cmul)(s->root[i],b);
    if (k==i) return NAME(cadd)(a,t);
    NAME(complex) q=NAME(csub)(a,t);
    return (NAME(complex)){q.r,NAME(neg)(q.i)};
}
#endif
static ATTR void NAME(cross_window)(void *v,const double *pcm,size_t frames,size_t channels,size_t offset,int ignored) {
    NAME(cross_state) *s=v;
    NAME(state) *f=s->fft;
    (void)channels; (void)ignored;
    double max[2]={0,0};
    for (size_t i=0;i<frames;++i) for (size_t ch=0;ch<2;++ch) {
        double x=fabs(pcm[2*i+ch]); if (x>max[ch]) max[ch]=x;
    }
    int shift[2]={0,0};
    for (size_t ch=0;ch<2;++ch) if (max[ch]!=0) {
        int e=ilogb(max[ch]);
        if (!s->has_scale[ch] || e>s->exponent[ch]) {
            if (s->has_scale[ch]) shift[ch]=s->exponent[ch]-e;
            s->exponent[ch]=e;s->has_scale[ch]=1;
        }
    }
    int shifts[4]={2*shift[0],2*shift[1],shift[0]+shift[1],shift[0]+shift[1]};
    for (size_t plane=0;plane<4;++plane) if (shifts[plane]) {
        size_t end=(plane+1)*(f->m+1);
        for (size_t i=plane*(f->m+1);i<end;++i) {
            s->sum[i]=NAME(scale)(s->sum[i],shifts[plane]);
            s->correction[i]=NAME(scale)(s->correction[i],shifts[plane]);
        }
    }
    REAL zero=NAME(from)(0.0);
    size_t end=offset+frames;
    for (size_t ch=0;ch<2;++ch) {
        for (size_t k=0;k<f->m;++k) {
            size_t i=2*k,j=i+1;
            REAL r=zero,im=zero;
            if (i>=offset && i<end) r=NAME(mul)(NAME(input)(pcm[(i-offset)*2+ch],-s->exponent[ch]),f->window[i]);
            if (j>=offset && j<end) im=NAME(mul)(NAME(input)(pcm[(j-offset)*2+ch],-s->exponent[ch]),f->window[j]);
            f->work[f->reverse[k]]=(NAME(complex)){r,im};
        }
        NAME(transform)(f);
#ifdef AC_WIDE_DD
        NAME(cross_collect_all)(s,ch);
#else
        for (size_t k=0;k<=f->m;++k) {
            NAME(complex) x=NAME(real_bin)(f,k);
            int interior=k!=0 && k!=f->m;
            REAL p=NAME(norm)(x);
            NAME(cross_add)(s,ch,k,interior?NAME(twice)(p):p);
            if (ch==0) s->left[k]=x;
            else {
                NAME(complex) l=s->left[k];l.i=NAME(neg)(l.i);
                NAME(complex) z=NAME(cmul)(l,x);
                NAME(cross_add)(s,2,k,interior?NAME(twice)(z.r):z.r);
                NAME(cross_add)(s,3,k,interior?NAME(twice)(z.i):z.i);
            }
        }
#endif
    }
}
static ATTR int NAME(cross_finish)(void *v,uint64_t frames,size_t channels,struct ac_cross_bin *out) {
    NAME(cross_state) *s=v; (void)channels;
    size_t bins=s->fft->m+1;
    int exponents[4]={2*s->exponent[0],2*s->exponent[1],s->exponent[0]+s->exponent[1],s->exponent[0]+s->exponent[1]};
    for (size_t k=0;k<bins;++k) {
        struct ac_spectral_bin *dest[4]={&out[k].left,&out[k].right,&out[k].real,&out[k].imag};
        for (size_t plane=0;plane<4;++plane) {
            size_t i=plane*bins+k;
            REAL p=plane<2?NAME(sub)(s->sum[i],s->correction[i]):NAME(add)(s->sum[i],s->correction[i]);
            p=NAME(div)(NAME(div)(p,(double)s->fft->n),(double)frames);
            if (plane<2) { if (!NAME(export)(p,exponents[plane],dest[plane])) return 0; }
            else { if (!NAME(export_signed)(p,exponents[plane],dest[plane])) return 0; }
        }
    }
    return 1;
}
