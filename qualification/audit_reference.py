#!/usr/bin/env python3
"""Corrective-round mathematical/fixture checks. Not execution of Rust code.

Direct finite-record reference and complementary-window reference are separate.
The boundary/lattice checks reproduce the known fixture expectations; they do
not certify the streaming Rust state machine. Run the Rust integration tests.
"""
import math
from functools import reduce
from time import perf_counter
import numpy as np
from reference_checks import power_spectrum, window, weight, tone, db, assert_close


def legacy_power(pcm,n):
    total=np.zeros(n//2+1)
    for start in range(0,len(pcm),n):
        x=pcm[start:start+n]
        w=np.sin(np.pi*(np.arange(len(x))+.5)/len(x))**2
        p=np.abs(np.fft.rfft(x*w,n=n))**2;p[1:-1]*=2
        if p.sum():total+=p*np.dot(x,x)/p.sum()
    return total/len(pcm)


def spectrum_checks():
    n=16384;h=n//2;r=96000;length=8*n;burst_len=2048
    w=window(n);assert_close(np.max(np.abs(w[:h]**2+w[h:]**2-1)),0,1e-14,'squared overlap partition')
    low_hz=171*r/n;low=tone(r,length,low_hz,.1)
    burst=tone(r,burst_len,30000,.079)*np.sin(np.pi*(np.arange(burst_len)+.5)/burst_len)**2
    f=np.fft.rfftfreq(n,1/r);f_ref=np.fft.rfftfreq(length,1/r);above=f>20000
    old=[];new=[]
    for start in [0,1,h-burst_len//2,n-burst_len//2,length-burst_len]:
        x=low.copy();x[start:start+burst_len]+=burst
        ref=np.abs(np.fft.rfft(x))**2/length**2;ref[1:-1]*=2
        a=legacy_power(x,n);b=power_spectrum(x,n)
        raw=np.mean(x*x);expected=db(ref[f_ref>20000].sum()/raw)
        old.append(db(a[above].sum()/a.sum()));new.append(db(b[above].sum()/b.sum()))
        assert_close(new[-1],expected,.5,f'burst start {start} full-record DFT oracle')
        assert_close(b.sum(),raw,1e-12,f'burst start {start} Parseval')
        for mode in ['none','A','468']:
            for lo,hi in [(0,r/2),(20,20000),(20000,r/2)]:
                mask=(f>=lo)&((f<hi)|((hi==r/2)&(f==hi)))
                refmask=(f_ref>=lo)&((f_ref<hi)|((hi==r/2)&(f_ref==hi)))
                assert_close(db(np.dot(b[mask],weight(f[mask],mode))*1e-8),db(np.dot(ref[refmask],weight(f_ref[refmask],mode))*1e-8),.5,f'quiet burst {start} {mode} {lo}-{hi} Hz')
    print(f'LEGACY position spread {max(old)-min(old):.6f} dB; corrected reference spread {max(new)-min(new):.9f} dB')
    assert max(old)-min(old)>5 and max(new)-min(new)<.5
    rng=np.random.default_rng(20261126)
    for frames in [1,2,h-1,h,h+1,n-1,n,n+1,3*n+177]:
        x=rng.normal(0,.01,(frames,3));p=power_spectrum(x,n)
        assert_close(p.sum(),np.mean(x*x),1e-15,f'multichannel partial {frames} frames')
    # A dry-run of the streaming scheduler independently verifies that the
    # start/end offsets produce exactly the same sample-weight contributions.
    # No Rust code is invoked by this check.
    for length in [1,h-1,h,h+1,n-1,n,n+1,3*n+177]:
        for chunk in [1,997,h+3,n+1]:
            leading=True;pending=0;block=[];coverage=np.zeros(length)
            def apply():
                offset=h if leading else 0
                coverage[block]+=w[offset:offset+len(block)]**2
            for start in range(0,length,chunk):
                pcm=list(range(start,min(length,start+chunk)))
                while pcm:
                    target=h if leading else n
                    take=min(len(pcm),target-len(block));block.extend(pcm[:take]);del pcm[:take];pending+=take
                    if len(block)==target:
                        apply()
                        if not leading:block=block[h:]
                        leading=False;pending=0
            if pending:
                apply()
                if not leading:block=block[h:]
                leading=False
            if not leading and block:apply()
            assert np.max(abs(coverage-1))<1e-14,(length,chunk)
    print('PASS streaming-schedule mathematical coverage, 32 length/chunk combinations')


def quiet_checks():
    # Independently identify the exact interval in the explicit step fixture.
    r=48000;B=2400;threshold=1e-6
    def coarse_and_refined(x):
        blocks=[x[i:i+B] for i in range(0,len(x),B)]
        coarse=[np.mean(b*b)<=threshold for b in blocks]
        runs=[];run=[]
        def close():
            nonlocal run
            if len(run)>=r:runs.extend(run)
            run=[]
        for j,b in enumerate(blocks):
            q=coarse[j];below=b*b<=threshold;half=(len(b)+1)//2
            prev_noisy=j==0 or not coarse[j-1];next_noisy=j+1==len(blocks) or not coarse[j+1]
            if q and np.max(b*b)>4*threshold and ((prev_noisy and below[-half:].all()) or (next_noisy and below[:half].all())):q=False
            if q:run.extend(b);continue
            loud=np.flatnonzero(~below)
            if not len(loud):run.extend(b);continue
            run.extend(b[:loud[0]]);close();run.extend(b[loud[-1]+1:])
        close();return np.asarray(runs)
    for loud in [.1,.02]:
        for off in [0,1,97,1200,2399]:
            for length in [r,r+960,r-1]:
                q=tone(r,length,50,1e-5)
                x=np.r_[np.full(B+off,loud),q,np.full(2*B,loud)]
                recovered=coarse_and_refined(x)
                if length>=r:
                    assert np.array_equal(q,recovered),(loud,off,length,len(recovered))
                    assert_close(db(np.mean(recovered**2)),-103.01029995664,1e-9,f'quiet boundary loud {loud} offset {off} length {length}')
                else:assert len(recovered)==0
    for a,b,expected in [(r,r+960,2*r+960),(30000,30000,0),(r,30000,r)]:
        x=np.r_[np.full(a,1e-5),np.full(B+333,.1),np.full(b,1e-5)]
        assert len(coarse_and_refined(x))==expected
    print('PASS transition fixture model, subminimum gaps and separated/boundary runs')


def grids():
    for step,offset,low in [(256,0,16),(192,0,18),(128,0,17),(192,7,24)]:
        codes=np.arange(-32768,32768,dtype=np.int64)*step+offset
        g=reduce(math.gcd,(int(v-codes[0]) for v in codes))
        bit_or=int(np.bitwise_or.reduce(codes))
        tz=(bit_or&-bit_or).bit_length()-1
        observed=((int(codes[-1]-codes[0])//g+1)-1).bit_length()
        assert g==step and 24-tz==low and observed==16
        print(f'PASS lattice step {step} offset {offset}: normalized {low}, observed {observed}')
    for depth,pad in [(16,1),(24,256),(53,1<<37),(24,1),(24,100),(53,100<<29)]:
        # 53-bit signed normalized float grid has 52 fractional bits.
        peak=pad;spacing=pad&-pad
        assert (peak<=8*spacing)==(pad not in [100,100<<29])
    print('PASS binary-aligned dither eligibility and multiples-of-100 negative control')


def benchmark_reference():
    rng=np.random.default_rng(42);x=rng.normal(0,.05,96000*6);n=16384
    for label,fn in [('legacy-nonoverlap',legacy_power),('complementary-half-hop',power_spectrum)]:
        times=[]
        for _ in range(3):
            t=perf_counter();fn(x,n);times.append(perf_counter()-t)
        print(f'NUMPY reference only {label}: median {np.median(times):.6f} s for 6 s PCM')

if __name__=='__main__':
    print('INDEPENDENT NUMERICAL / FIXTURE MODEL ONLY. NOT NATIVE RUST QUALIFICATION.')
    spectrum_checks();quiet_checks();grids();benchmark_reference()
    print('All corrective numerical and fixture-model checks passed.')
