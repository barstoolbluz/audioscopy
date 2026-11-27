/* SPDX-License-Identifier: GPL-2.0-or-later
 * Private C ABI for the optional wide-spectrum implementation and its
 * independent qualification tools. Do not expose native long double here.
 */
#ifndef AUDIOSCOPY_SPECTRAL_PRECISION_H
#define AUDIOSCOPY_SPECTRAL_PRECISION_H
#include <stddef.h>
#include <stdint.h>
struct ac_spectral_bin { double hi, lo; int32_t exponent; };
struct ac_cross_bin { struct ac_spectral_bin left,right,real,imag; };
int audioscopy_cross_new(uint32_t rate,int mode,void **out,size_t *fft_len,int *actual);
int audioscopy_cross_finish(void *plan,struct ac_cross_bin *out,size_t bins);
int audioscopy_spectral_available(int mode);
int audioscopy_spectral_new(uint32_t rate,size_t channels,int mode,void **out,size_t *fft_len,int *actual);
int audioscopy_spectral_push(void *plan,const double *pcm,size_t frames);
int audioscopy_spectral_finish(void *plan,struct ac_spectral_bin *out,size_t bins);
void audioscopy_spectral_free(void *plan);
int audioscopy_noise_new(uint32_t rate,int mode,int view,int weighting,void **out,int *actual);
int audioscopy_noise_add(void *plan,const struct ac_spectral_bin *bins,size_t count,uint64_t frames);
int audioscopy_noise_finish(void *plan,struct ac_spectral_bin *out,size_t count);
void audioscopy_noise_free(void *plan);
struct ac_loudness_result {
    uint64_t frames,required_frames,absolute_count,relative_count;
    int32_t outcome;
    struct ac_spectral_bin energy;
};
int audioscopy_loudness_new(uint32_t rate,size_t channels,const unsigned char *weights,size_t limit,int mode,void **out,int *actual);
int audioscopy_loudness_push(void *plan,const double *pcm,size_t frames);
int audioscopy_loudness_finish(void *plan,struct ac_loudness_result *out);
void audioscopy_loudness_free(void *plan);
#ifdef AUDIOSCOPY_SPECTRAL_RESEARCH
int audioscopy_loudness_probe(uint32_t rate,int mode,const double *pcm,size_t n,struct ac_spectral_bin *out);
#endif
#endif
