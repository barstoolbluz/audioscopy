/* SPDX-License-Identifier: GPL-2.0-or-later
 * Optional native-profile integrated loudness in extended80 or DD.
 * Conditioned RLB + shelf realization matches the native transfer function.
 * Partition the rational window clock at starts/ends: one filter and one
 * positive energy accumulation per frame, no per-frame energy ring.
 * Complete 400 ms windows only; no LRA/short-term/compatibility substitution.
 */
#ifdef AC_WIDE_DD
#define KC(x) CAT(dd_,x)
#else
#define KC(x) ((REAL)CAT(AC_LD_,x))
#endif

typedef struct { REAL c,s,vh,t,rg,k2,bp_mix,lp_mix; } NAME(kw_coeff);
typedef struct { REAL previous,y,q,a,b; } NAME(kw_channel);
typedef struct { REAL sum,correction; } NAME(kw_acc);
typedef struct {
    NAME(kw_coeff) coeff;
    NAME(kw_channel) channel[64];
    unsigned char weights[64];
    size_t channels,capacity,count,limit;
    uint32_t rate;
    uint64_t frames,window,started,ended;
    NAME(kw_acc) segment,active[5];
    REAL *energies;
} NAME(kw_state);

static inline ATTR REAL NAME(prewarp)(REAL frequency,uint32_t rate) {
    REAL half_pi=NAME(pi_half)();
    REAL angle=NAME(mul)(half_pi,NAME(div)(NAME(twice)(frequency),(double)rate));
    return NAME(div_real)(NAME(sin)(angle),NAME(sin)(NAME(sub)(half_pi,angle)));
}
static ATTR int NAME(kw_coefficients)(uint32_t rate,NAME(kw_coeff) *out) {
    REAL one=NAME(from)(1.0),zero=NAME(from)(0.0),four=NAME(from)(4.0);
    REAL k=NAME(prewarp)(KC(kw_hp_f),rate),r=NAME(div_real)(one,KC(kw_hp_q));
    REAL k2=NAME(square)(k),kr=NAME(mul)(k,r),d=NAME(add)(NAME(add)(one,kr),k2);
    out->c=NAME(div_real)(NAME(add)(NAME(twice)(kr),NAME(mul)(four,k2)),d);
    out->s=NAME(div_real)(NAME(twice)(k),NAME(sqrt_real)(d));
    k=NAME(prewarp)(KC(kw_shelf_f),rate);r=NAME(div_real)(one,KC(kw_shelf_q));
    d=NAME(add)(NAME(add)(one,NAME(mul)(k,r)),NAME(square)(k));
    out->vh=KC(kw_vh);out->t=NAME(div_real)(k,d);out->rg=NAME(add)(r,k);out->k2=NAME(twice)(k);
    out->lp_mix=NAME(sub)(one,out->vh);
    out->bp_mix=NAME(add)(NAME(mul)(NAME(sub)(KC(kw_vb),out->vh),r),NAME(mul)(out->lp_mix,k));
    REAL terms[8]={out->c,out->s,out->vh,out->t,out->rg,out->k2,out->bp_mix,out->lp_mix};
    for (size_t i=0;i<8;++i) if (!NAME(finite)(terms[i])) return 0;
    REAL s2=NAME(square)(out->s);
    return NAME(gt)(s2,zero) && NAME(gt)(out->c,s2) && NAME(gt)(NAME(add)(NAME(sub)(four,NAME(twice)(out->c)),s2),zero)
        && NAME(gt)(NAME(mul)(out->t,out->k2),zero) && NAME(gt)(one,NAME(mul)(out->t,out->rg)) && NAME(gt)(NAME(twice)(out->rg),out->k2);
}
static inline ATTR REAL NAME(kw_filter)(NAME(kw_channel) *state,const NAME(kw_coeff) *c,double input) {
    REAL x=NAME(from)(input),dx=NAME(sub)(x,state->previous);
    REAL y=NAME(add)(state->y,NAME(sub)(NAME(sub)(dx,NAME(mul)(c->c,state->y)),NAME(mul)(c->s,state->q)));
    REAL q=NAME(add)(state->q,NAME(mul)(c->s,state->y));
    REAL t=NAME(sub)(NAME(sub)(y,state->b),NAME(mul)(c->rg,state->a));
    REAL inc=NAME(mul)(c->t,t),bp=NAME(add)(state->a,inc);
    REAL out=NAME(add)(NAME(add)(NAME(mul)(c->vh,y),NAME(mul)(c->bp_mix,bp)),NAME(mul)(c->lp_mix,state->b));
    REAL a=NAME(add)(state->a,NAME(twice)(inc)),b=NAME(add)(state->b,NAME(mul)(c->k2,bp));
    *state=(NAME(kw_channel)){x,y,q,a,b};return out;
}
static inline ATTR void NAME(kw_accumulate)(NAME(kw_acc) *a,REAL x) {
    REAL y=NAME(sub)(x,a->correction),t=NAME(add)(a->sum,y);
    a->correction=NAME(sub)(NAME(sub)(t,a->sum),y);a->sum=t;
}
static inline ATTR REAL NAME(kw_value)(NAME(kw_acc) *a) { return NAME(sub)(a->sum,a->correction); }
static ATTR void NAME(kw_destroy)(void *v) {
    NAME(kw_state) *s=v;if (s) { free(s->energies);free(s); }
}
static ATTR int NAME(kw_create)(uint32_t rate,size_t channels,const unsigned char *weights,size_t limit,void **out) {
    NAME(kw_state) *s=calloc(1,sizeof(*s));if (!s) return -1;
    if (!NAME(kw_coefficients)(rate,&s->coeff)) { free(s);return 0; }
    s->rate=rate;s->channels=channels;s->limit=limit;s->window=(2u*rate+2u)/5u;s->started=1;
    memcpy(s->weights,weights,channels);*out=s;return 1;
}
static ATTR int NAME(kw_record)(NAME(kw_state) *s,REAL value) {
    if (!NAME(finite)(value)) return 0;
    if (!NAME(gt)(value,KC(kw_absolute_gate))) return 1;
    if (s->count==s->limit) return -1;
    if (s->count==s->capacity) {
        size_t next=s->capacity?s->capacity*2:256;
        if (next>s->limit) next=s->limit;
        REAL *p=realloc(s->energies,next*sizeof(*p));if (!p) return -1;
        s->energies=p;s->capacity=next;
    }
    s->energies[s->count++]=value;return 1;
}
static ATTR int NAME(kw_push)(void *v,const double *pcm,size_t frames) {
    NAME(kw_state) *s=v;REAL zero=NAME(from)(0.0);
    while (frames) {
        uint64_t start=s->started*s->rate/10,end=s->window+s->ended*s->rate/10;
        uint64_t boundary=start<end?start:end;
        if (boundary<=s->frames) return 0;
        size_t take=frames;
        if ((uint64_t)take>boundary-s->frames) take=(size_t)(boundary-s->frames);
        for (size_t i=0;i<take;++i) {
            REAL energy=zero;
            for (size_t ch=0;ch<s->channels;++ch) if (s->weights[ch]) {
                REAL y=NAME(kw_filter)(&s->channel[ch],&s->coeff,pcm[i*s->channels+ch]);
                REAL p=NAME(square)(y);
                if (s->weights[ch]==2) p=NAME(mul)(p,KC(kw_surround));
                energy=NAME(add)(energy,p);
            }
            NAME(kw_accumulate)(&s->segment,energy);
        }
        pcm+=take*s->channels;s->frames+=take;frames-=take;
        if (s->frames!=boundary) continue;
        REAL segment=NAME(kw_value)(&s->segment);
        if (!NAME(finite)(segment) || s->started-s->ended>5) return 0;
        for (uint64_t i=s->ended;i<s->started;++i) NAME(kw_accumulate)(&s->active[i%5],segment);
        s->segment=(NAME(kw_acc)){zero,zero};
        if (end==boundary) {
            NAME(kw_acc) *a=&s->active[s->ended%5];
            int status=NAME(kw_record)(s,NAME(div)(NAME(kw_value)(a),(double)s->window));
            if (status!=1) return status;
            ++s->ended;
        }
        if (start==boundary) {
            if (s->started-s->ended>=5) return 0;
            s->active[s->started%5]=(NAME(kw_acc)){zero,zero};++s->started;
        }
    }
    return 1;
}
static ATTR int NAME(kw_finish)(void *v,struct ac_loudness_result *out) {
    NAME(kw_state) *s=v;
    *out=(struct ac_loudness_result){0};out->frames=s->frames;out->required_frames=s->window;
    if (!s->frames) return 1; /* no input */
    if (s->frames<s->window) { out->outcome=1;return 1; }
    if (!s->count) { out->outcome=2;return 1; }
    out->absolute_count=s->count;
    NAME(kw_acc) acc={0};
    for (size_t i=0;i<s->count;++i) NAME(kw_accumulate)(&acc,s->energies[i]);
    REAL threshold=NAME(div)(NAME(div)(NAME(kw_value)(&acc),(double)s->count),10.0);
    if (!NAME(finite)(threshold)) return 0;
    memset(&acc,0,sizeof(acc));
    for (size_t i=0;i<s->count;++i) if (NAME(gt)(s->energies[i],threshold)) {
        NAME(kw_accumulate)(&acc,s->energies[i]);++out->relative_count;
    }
    if (!out->relative_count) { out->outcome=3;return 1; }
    REAL mean=NAME(div)(NAME(kw_value)(&acc),(double)out->relative_count);
    if (!NAME(export)(mean,0,&out->energy)) return 0;
    out->outcome=4;return 1;
}
#ifdef AUDIOSCOPY_SPECTRAL_RESEARCH
/* Test-only direct samples for independent transfer/coefficient checks. */
static ATTR int NAME(kw_probe)(uint32_t rate,const double *pcm,size_t n,struct ac_spectral_bin *out) {
    NAME(kw_coeff) c;NAME(kw_channel) state={0};if (!NAME(kw_coefficients)(rate,&c)) return 0;
    for (size_t i=0;i<n;++i) if (!NAME(export_signed)(NAME(kw_filter)(&state,&c,pcm[i]),0,&out[i])) return 0;
    return 1;
}
#endif
#undef KC
