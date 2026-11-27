/* SPDX-License-Identifier: GPL-2.0-or-later
 * Optional native Linux x86-64 correlation kernel. The ABI exchanges only
 * binary64 samples, integer counts and 7 canonical 10-byte x87 values.
 * No long double crosses the Rust ABI, no dynamic allocation, no callbacks.
 */
#include <float.h>
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if !defined(__x86_64__) || !defined(__linux__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "The x87 storage ABI is limited to little-endian Linux x86-64"
#endif
_Static_assert(FLT_RADIX == 2 && DBL_MANT_DIG == 53, "binary64 required");
_Static_assert(LDBL_MANT_DIG == 64 && LDBL_MAX_EXP == 16384 && sizeof(long double) == 16,
               "genuine SysV x87 extended long double required");

/* Seven persistent values: origins, shifted means, two variances, covariance.
 * Only the meaningful 80 bits are copied; ABI padding is never persisted.
 */
enum { OL, OR, ML, MR, VL, VR, CV, FIELDS };
static long double load80(const uint8_t *p) {
    long double x = 0.0L;
    memcpy(&x, p, 10);
    return x;
}
static void store80(uint8_t *p, long double x) { memcpy(p, &x, 10); }

/* Hold/mask exceptions and clear sticky flags, set round-to-nearest and
 * full 64-bit x87 significand. fesetenv restores both x87 and MXCSR state,
 * including caller flags, masks and rounding. No external callback while
 * the environment is changed. The SysV ABI requires a balanced x87 stack.
 */
static int enter(fenv_t *saved) {
    if (feholdexcept(saved) != 0) return 0;
    if (fesetround(FE_TONEAREST) != 0) {
        (void)fesetenv(saved);
        return 0;
    }
    unsigned short cw;
    unsigned int mxcsr;
    __asm__ volatile("fnstcw %0" : "=m"(cw) :: "memory");
    cw = (unsigned short)((cw & ~0x0f00u) | 0x0300u);
    __asm__ volatile("fldcw %0" :: "m"(cw) : "memory");
    __asm__ volatile("stmxcsr %0" : "=m"(mxcsr) :: "memory");
    mxcsr &= ~(0x8040u); /* FTZ, DAZ */
    __asm__ volatile("ldmxcsr %0" :: "m"(mxcsr) : "memory");
    return 1;
}

/* Rust validates samples and bounds count <= 2^48. Fixed blocks are supplied
 * in stream order. Each block uses a local origin and two passes. Parallel
 * central-moment merging avoids raw-square minus square-of-mean cancellation.
 */
int audioscopy_x87_push(uint8_t *state, uint64_t prior, const double *pcm, size_t n) {
    if (n == 0 || n > 1024 || prior > (UINT64_C(1) << 48) - n) return 0;
    fenv_t saved;
    if (!enter(&saved)) return 0;
    long double s[FIELDS];
    for (size_t i = 0; i < FIELDS; ++i) s[i] = prior ? load80(state + i * 10) : 0.0L;
    const long double al = pcm[0], ar = pcm[1];
    long double ml = 0.0L, mr = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        ml += (long double)pcm[2*i] - al;
        mr += (long double)pcm[2*i+1] - ar;
    }
    ml /= (long double)n;
    mr /= (long double)n;
    long double vl = 0.0L, vr = 0.0L, cv = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        const long double l = ((long double)pcm[2*i] - al) - ml;
        const long double r = ((long double)pcm[2*i+1] - ar) - mr;
        vl += l*l; vr += r*r; cv += l*r;
    }
    if (prior == 0) {
        s[OL] = al; s[OR] = ar; s[ML] = ml; s[MR] = mr;
        s[VL] = vl; s[VR] = vr; s[CV] = cv;
    } else {
        const long double dl = ((al-s[OL])+ml)-s[ML];
        const long double dr = ((ar-s[OR])+mr)-s[MR];
        const long double weight = (long double)n / (long double)(prior+n);
        const long double factor = (long double)prior * weight;
        s[ML] += dl*weight; s[MR] += dr*weight;
        s[VL] += vl + (dl*dl)*factor;
        s[VR] += vr + (dr*dr)*factor;
        s[CV] += cv + (dl*dr)*factor;
    }
    for (size_t i = 0; i < FIELDS; ++i) store80(state + i * 10, s[i]);
    return fesetenv(&saved) == 0;
}

int audioscopy_x87_finish(const uint8_t *state, double *out) {
    fenv_t saved;
    if (!enter(&saved)) return 0;
    const long double vl = load80(state + VL*10), vr = load80(state + VR*10);
    long double r = 0.0L;
    if (vl > 0.0L && vr > 0.0L) {
        r = load80(state + CV*10) / (sqrtl(vl)*sqrtl(vr));
        /* Same zero-variance and physical correlation clamp as CP6. */
        if (r > 1.0L) r = 1.0L;
        if (r < -1.0L) r = -1.0L;
    }
    out[0] = (double)r;
    out[1] = (double)(r - (long double)out[0]);
    return fesetenv(&saved) == 0;
}
