#!/usr/bin/env python3
"""R5 bandwidth oracle. Numerical models only, not compiled Rust evidence.

Reuse the independent direct-index power and endpoint calculations. R5
changes only the complete-window bin floor: the reference is accumulated
whole-record bin energy, including the complementary boundary windows.
"""
import math
from time import perf_counter
import numpy as np
import r2_reference as r2
import r4_reference as r4
from reference_checks import tone, power_spectrum, db


def extent(x, rate, *, details=False):
    n = 1 << (math.ceil(rate/6)-1).bit_length()
    h = n//2
    length = len(x)
    full = power_spectrum(x, n)*length
    floor = full.max()*r4.REL
    if length < h:
        supports = [x]
        if length > n//8:
            supports += [x[:n//8], x[-n//8:]]
    else:
        supports = [x[:h], x[:n//8], x[-h:], x[-n//8:]]
        if h < length < n:
            supports += [x]
    edges = [r4.endpoint(y, n, rate, floor) for y in supports]
    interior = power_spectrum(x, n, True)*length
    local_max = interior.max()
    old = np.flatnonzero(interior >= local_max*r4.REL)[-1]*rate/n if local_max else 0.
    occupied = np.flatnonzero((interior > 0.) & (interior >= floor))
    if len(occupied):
        base = occupied[-1]*rate/n
    else:
        finest = min(d for _, d in edges)
        base = max(f for f, d in edges if d == finest)
    result = max([base] + [f for f, d in edges if f > base+4*d]) if full.max() else 0.
    if details:
        return result, old, edges, len(occupied), local_max, floor
    return result, old, edges


def triangular(length, seed=0x14a6):
    # Same xorshift stream and exact integer codes as the Rust fixture.
    state = seed
    mask = (1 << 64)-1
    def bit():
        nonlocal state
        state ^= (state << 13) & mask
        state ^= state >> 7
        state ^= (state << 17) & mask
        return state >> 63
    return np.array([bit()-bit() for _ in range(length)])/2**23


def fixture(length=960, hz=1000., total=8*16384, offset=0):
    r = 96000
    x = np.zeros(total)
    burst = tone(r,length,hz,.1)*np.sin(np.pi*(np.arange(length)+.5)/length)**2
    x[offset:offset+length] = np.round(burst*2**23)/2**23
    return x


def regressions():
    r = 96000
    x = fixture()
    noise = triangular(len(x))
    print(f'Triangular dither RMS: {db(np.mean(noise**2)):.6f} dBFS')
    baseline = extent(x,r)[0]
    assert 1000. <= baseline <= 1600., baseline
    assert abs(r4.extent(x,r)[0]-2712.890625) < 1e-6
    cases = 0
    for placement in ['silent', 'whole']:
        y = x+noise if placement=='whole' else np.r_[x[:960], noise[960:]]
        for z in [y,y[::-1]]:
            before = r4.extent(z,r)[0]
            after = extent(z,r)[0]
            assert before > 45000., before
            assert abs(after-baseline) < 12., (before,after,baseline)
            print(f'REPRODUCED {placement} dither: R4 {before:.9f} Hz, R5 {after:.9f} Hz')
            cases += 1
            for gain in [2**-15,-.37,2.,8.]:
                assert extent(z*gain,r)[0] == after
                cases += 1
    print(f'Undithered: R4 {r4.extent(x,r)[0]:.9f} Hz, R5 {baseline:.9f} Hz')
    # Nonzero interior noise, but no bin passes the recording-wide floor.
    # A genuine endpoint peak must survive the actual-support fallback.
    x = fixture(32,12000.)
    y = x+noise
    for z in [y,y[::-1]]:
        after,old,edges,count,local_max,floor=extent(z,r,details=True)
        assert count == 0 and 0. < local_max < floor
        assert r4.extent(z,r)[0] > 45000.
        assert abs(after-12000.) < 1000., (after,edges)
        print(f'REPRODUCED nonzero, immaterial interior: R4 {r4.extent(z,r)[0]:.9f} Hz, R5 {after:.9f} Hz; interior max/floor {local_max/floor:.6g}')
        cases+=1
    # Vary realistic attack length, placement, endpoint direction, and
    # channel counts. Compare like-for-like dithered vs undithered content.
    for length in [480,960,1920]:
        for offset in [0,37,16384]:
            for channels in [1,2,3]:
                a=fixture(length,total=4*16384+71,offset=offset)
                # A finite tapered note is not a spectral line. Its own
                # -80 dB extent, independently computed from the actual
                # support, is a useful upper bound for these fixtures.
                support=abs(np.fft.rfft(a[offset:offset+length], n=16384))**2
                support_hz=np.flatnonzero(support>=support.max()*r4.REL)[-1]*r/16384
                b=a+triangular(len(a),seed=0x123456789)
                if channels>1:
                    a=np.stack([a]+[np.zeros_like(a)]*(channels-1),axis=1)
                    b=np.stack([b]+[triangular(len(b),seed=0x123456789+i) for i in range(1,channels)],axis=1)
                for u,v in [(a,b),(a[::-1],b[::-1])]:
                    expected=extent(u,r)[0]
                    actual=extent(v,r)[0]
                    assert 800.<=expected<=support_hz+2*r/16384, (length,offset,channels,expected,support_hz)
                    assert abs(actual-expected)<30., (length,offset,channels,expected,actual)
                    cases+=1
    print(f'PASS {cases} R5 dither, common-floor fallback, gain, placement and multichannel model cases')


if __name__ == '__main__':
    print('R5 NUMERICAL MODEL ONLY; NATIVE RUST NOT EXECUTED')
    start=perf_counter()
    regressions()
    r4.regressions(estimator=extent,common_floor=True)
    original=r2.extent
    try:
        r2.extent=extent
        r2.bandwidth_checks()
    finally:
        r2.extent=original
    print('PASS all 125 preceding bandwidth fixture models using R5 common floor')
    print(f'Model elapsed {perf_counter()-start:.3f} s (not a native benchmark)')
