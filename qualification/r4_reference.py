#!/usr/bin/env python3
"""R4 bandwidth arithmetic oracle and regression fixtures, not native tests.

The old estimator is the unchanged R2/R3 numerical model. The corrected
estimator uses independently indexed half-hop/endpoint FFTs. Power is
one-sided channel-mean energy (amplitude squared times real frames), with
no per-window energy redistribution or local amplification. A candidate
must pass both its local -80 dB test and the common programme-bin floor.
"""
import math
from time import perf_counter
import numpy as np
import r2_reference as r2
from reference_checks import tone, power_spectrum, db

REL = 1e-8


def endpoint(y, n, rate, floor):
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    length, channels = y.shape
    a = 2*np.pi*(np.arange(length)+.5)/length
    w = .35875 - .48829*np.cos(a) + .14128*np.cos(2*a) - .01168*np.cos(3*a)
    p = (abs(np.fft.rfft(y*w[:, None], n=n, axis=0))**2).sum(axis=1)/(n*channels)
    p[1:-1] *= 2
    if not p.max():
        return 0., rate/length
    mask = ((p >= p.max()*REL) & (p >= floor)
            & np.r_[True, p[1:] >= p[:-1]] & np.r_[p[:-1] > p[1:], True])
    candidates = np.flatnonzero(mask)
    return (candidates[-1]*rate/n if len(candidates) else 0.), rate/length


def extent(x, rate):
    n = 1 << (math.ceil(rate/6)-1).bit_length()
    h = n//2
    length = len(x)
    full = power_spectrum(x, n)*length
    floor = full.max()*REL
    if length < h:
        supports = [x]
        if length > n//8:
            supports += [x[:n//8], x[-n//8:]]
    else:
        supports = [x[:h], x[:n//8], x[-h:], x[-n//8:]]
        if h < length < n:
            supports += [x]
    edges = [endpoint(y, n, rate, floor) for y in supports]
    p = power_spectrum(x, n, True)
    old = np.flatnonzero(p >= p.max()*REL)[-1]*rate/n if p.max() else 0.
    if p.max():
        base = old
    else:
        finest = min(d for _, d in edges)
        base = max(f for f, d in edges if d == finest)
    return max([base] + [f for f, d in edges if f > base+4*d]), old, edges


def fixture():
    rate, n = 96000, 16384
    f = 171*rate/n
    x = np.zeros(8*n)
    active = 4*n
    x[:active] = tone(rate, active, f, .1)
    x[active-n:active] *= .5+.5*np.cos(np.pi*np.arange(n)/(n-1))
    # Exactly representable 24-bit PCM. Triangular +/- one-LSB noise:
    # each of two independent Bernoulli bits has probability 1/2.
    x = np.round(x*2**23)/2**23
    # Match the native regression's deterministic xorshift bit stream.
    state = 0x14a6
    mask = (1 << 64)-1
    def bit():
        nonlocal state
        state ^= (state << 13) & mask
        state ^= state >> 7
        state ^= (state << 17) & mask
        return state >> 63
    noise = np.array([bit()-bit() for _ in range(n//2)])/2**23
    return rate, n, x, noise


def regressions(estimator=extent, *, common_floor=False):
    rate, n, x, noise = fixture()
    base = r2.extent(x, rate)[0]
    assert abs(base-1000) < 100, base
    print(f'Base R3 bandwidth: {base:.9f} Hz; noise RMS {db(np.mean(noise**2)):.6f} dBFS')
    cases = 0
    for start in [5*n, len(x)-len(noise)]:
        y = x.copy()
        y[start:start+len(noise)] = noise
        for z in [y, y[::-1]]:
            before = r2.extent(z, rate)[0]
            after = estimator(z, rate)[0]
            if start == len(x)-len(noise):
                assert before > 45000, before
            else:
                assert abs(before-base) < 20, before
            assert abs(after-base) < 20, (before, after, base)
            print(f'REPRODUCED 24-bit noise at {start}: R3 {before:.9f} Hz, {"R5" if common_floor else "R4"} {after:.9f} Hz')
            cases += 1
            for gain in [2**-15, -.37, 2., 16.]:
                assert abs(estimator(z*gain, rate)[0]-after) < 1e-8
                cases += 1
    # Endpoint tone alone must have programme-relative materiality.
    for amp in [1e-3, 1e-6, 1e-7, 1e-8, 1e-12]:
        y = x.copy()
        length = n//2
        b = tone(rate, length, 30000, amp)*np.sin(np.pi*(np.arange(length)+.5)/length)**2
        y[-len(b):] = b
        for z in [y, y[::-1]]:
            before = r2.extent(z, rate)[0]
            after = estimator(z, rate)[0]
            assert abs(before-30000) < 200, before
            expected = 30000 if amp == 1e-3 else base
            assert abs(after-expected) < (400 if amp == 1e-3 else 20), (amp, before, after)
            print(f'Endpoint peak {amp:g}: R3 {before:.9f} Hz, {"R5" if common_floor else "R4"} {after:.9f} Hz')
            cases += 1
    # A negligible upper peak must not conceal a material lower one in
    # the same local view. The old detector returns only the upper peak.
    y = x.copy()
    w = np.sin(np.pi*(np.arange(960)+.5)/960)**2
    y[-960:] = w*(tone(rate, 960, 6000, .005)+tone(rate, 960, 30000, 3e-6))
    for z in [y, y[::-1]]:
        before = r2.extent(z, rate)[0]
        after = estimator(z, rate)[0]
        assert abs(before-30000)<200 and abs(after-6000)<400, (before, after)
        print(f'Material lower peak with negligible higher peak: R3 {before:.9f} Hz, {"R5" if common_floor else "R4"} {after:.9f} Hz')
        cases += 1
        for gain in [.0001, -.5, 8.]:
            assert estimator(z*gain, rate)[0] == after
            cases += 1
    # Additional unequal-channel and no-interior cases. Quantization is
    # incidental to the inference; channel-mean normalization must agree.
    for channels in [1, 2, 3]:
        for amp in [1e-2, 1e-8]:
            y = np.tile(x[:, None], (1, channels))
            y[-960:, -1] = w*tone(rate, 960, 30000, amp)
            for z in [y, y[::-1]]:
                after = estimator(z, rate)[0]
                assert abs(after-(30000 if amp == 1e-2 else base)) < (400 if amp == 1e-2 else 20), (channels, amp, after)
                cases += 1
    for length in [1000, n//8, n//2-1, n//2, n//2+1, n-1, n, n+1, 4*n+1333]:
        y = np.zeros(length)
        y[-960:] = w*tone(rate, 960, 30000, 1e-9)
        for z in [y, y[::-1]]:
            before = r2.extent(z, rate)[0]
            after = estimator(z, rate)[0]
            if common_floor:
                # R5 deliberately corrects the interior threshold. Test
                # the known 30 kHz content, not equality to R3's estimate.
                # Match the existing native short/endpoint-only tolerance.
                assert abs(after-30000) < 800, (length, before, after)
            else:
                # Historical R4 reproduction: unchanged interior estimate.
                assert after == before and before >= 29200, (length, before, after)
            assert abs(estimator(z*1e5, rate)[0]-estimator(z, rate)[0])<1e-8
            cases += 2
        assert estimator(np.zeros(length), rate)[0] == 0.
        cases += 1
    print(f'PASS {cases} materiality, mirrored, gain, multichannel, short/endpoint-only and silence cases ({"R5" if common_floor else "R4"})')


if __name__ == '__main__':
    print('R4 NUMERICAL MODEL ONLY; NATIVE RUST NOT EXECUTED')
    start = perf_counter()
    regressions()
    old_extent = r2.extent
    try:
        r2.extent = extent
        r2.bandwidth_checks()
    finally:
        r2.extent = old_extent
    print('PASS all 125 existing bandwidth fixture models using R4 criterion')
    print(f'Model elapsed {perf_counter()-start:.3f} s (not a native benchmark)')
