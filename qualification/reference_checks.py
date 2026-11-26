#!/usr/bin/env python3
"""Independent NumPy/SciPy numerical checks of the specified DSP conventions.

This is NOT a Rust execution test. It checks analytic expectations and fixture
geometry used by the Rust tests, and may be run without Cargo. Rust tests remain
mandatory. Requires Python 3, NumPy and SciPy.
"""
import math
import numpy as np
from scipy.signal import lfilter


def assert_close(got, expected, tolerance, label):
    if not math.isfinite(float(got)) or abs(got-expected) > tolerance:
        raise AssertionError(f"{label}: {got} != {expected} within {tolerance}")
    print(f"PASS {label}: {got:.12g} (expected {expected:.12g}, tolerance {tolerance:.3g})")


def db(x):
    return 10 * np.log10(x)


def window(n):
    return np.sin(np.pi/2*np.sin(np.pi*(np.arange(n)+.5)/n)**2)


def windowed_frames(pcm, n, complete_only=False):
    """Vectorized direct indexing of half-hop windows; no streaming buffer."""
    pcm=np.asarray(pcm,dtype=np.float64)
    if pcm.ndim==1:pcm=pcm[:,None]
    h=n//2;w=window(n)
    for start in range(-h,len(pcm),h):
        lo=max(0,start);hi=min(start+n,len(pcm))
        if complete_only and (lo!=start or hi!=start+n):continue
        x=np.zeros((n,pcm.shape[1]));x[lo-start:hi-start]=pcm[lo:hi]
        yield x*w[:,None]


def power_spectrum(pcm,n,complete_only=False):
    pcm=np.asarray(pcm,dtype=np.float64)
    channels=1 if pcm.ndim==1 else pcm.shape[1]
    total=np.zeros(n//2+1)
    for weighted in windowed_frames(pcm,n,complete_only):
        p=np.sum(np.abs(np.fft.rfft(weighted,axis=0))**2,axis=1)/(n*channels)
        p[1:-1]*=2;total+=p
    return total/len(pcm)


def tone(rate, length, freq, amp):
    return amp*np.sin(2*np.pi*freq*np.arange(length)/rate)


def weight(f, mode):
    if mode == 'A':
        s = f*f
        return 10**.2*(12194**2*s*s/((s+20.6**2)*np.sqrt((s+107.7**2)*(s+737.9**2))*(s+12194**2)))**2
    if mode == '468':
        s = f*f
        a = ((-4.737338981378384e-24*s+2.043828333606125e-15)*s-1.363894795463638e-7)*s+1
        b = ((1.306612257412824e-19*s-2.118150887518656e-11)*s+5.559488023498642e-4)*f
        return 10**1.822*(1.246332637532143e-4*f/np.hypot(a,b))**2
    return np.ones_like(f)


def short_term_and_range():
    rate = 48000
    b = np.array([1.5351248595869702,-5.761945908580321,8.116910049252581,-5.08848181111208,1.19839281085285])
    a = np.array([1.0,-3.68070674801639,5.087045247971131,-3.1315463514467305,.7252088884778705])
    x = np.r_[tone(rate,8*rate,1000,.05),tone(rate,8*rate,1000,.5)]
    y = lfilter(b,a,np.r_[x,np.zeros(int(1.5*rate))])
    z = np.exp(-2j*np.pi*1000/rate)
    h = sum(b*z**np.arange(5))/sum(a*z**np.arange(5))
    expected = -.691+db(.125*abs(h)**2)
    windows = np.array([np.mean(y[i-3*rate:i]**2) for i in range(3*rate,len(x)+1,rate//10)])
    assert_close(-.691+db(windows.max()),expected,.015,'short-term analytic K response')
    extended = np.array([np.mean(y[i-3*rate:i]**2) for i in range(3*rate,len(y)+1,rate//10)])
    gated = np.sort(extended[extended>=10**((-70+.691)/10)])
    gated = gated[gated>=gated.mean()*.01]
    low = int(math.floor((len(gated)-1)*.10+.5))
    high = int(math.floor((len(gated)-1)*.95+.5))
    assert_close(db(gated[high]/gated[low]),20,.15,'LRA two-level tone')
    # Rational 100 ms clock: 30 hops always contain exactly three seconds.
    for rate in [44100,44101,48000,96000,192000,768000]:
        ends = np.arange(101,dtype=np.int64)*rate//10
        assert np.all(ends[30:]-ends[:-30]==3*rate)
    print('PASS rational short-term clock')


def spectra_and_noise():
    rate=96000; n=16384; f=np.fft.rfftfreq(n,1/rate); df=f[1]
    pcm=tone(rate,4*n,171*df,.1)+tone(rate,4*n,5120*df,.1)
    p=power_spectrum(pcm,n)
    assert_close(p.sum(),.01,1e-12,'Parseval stereo-independent PCM power')
    assert_close(db(p[f>20000].sum()/p.sum()),-10*math.log10(2),.002,'above 20 kHz ratio')
    interior=power_spectrum(pcm,n,True)
    assert_close(f[np.flatnonzero(interior>=interior.max()*1e-8)[-1]]/1000,30,.04,'bandwidth of bin-aligned high tone')
    rng=np.random.default_rng(20261125)
    arbitrary=rng.normal(0,.01,(12345,2));p=power_spectrum(arbitrary,n)
    assert_close(p.sum(),np.mean(arbitrary**2),1e-15,'partial multichannel block power')
    pcm=sum(tone(rate,12*n,k*df,1e-5) for k in [17,171,853,2560,5120]);p=power_spectrum(pcm,n)
    assert_close(p.sum(),2.5e-10,1e-19,'five-tone quiet full-band power')
    assert_close(db(p[(f>=20)&(f<20000)].sum()),db(2e-10),.002,'five-tone audible quiet power')
    for lo,hi in [(20,200),(200,2000),(2000,10000),(10000,20000),(20000,48001)]:
        assert_close(db(p[(f>=lo)&(f<hi)].sum()),db(5e-11),.005,f'noise band {lo}-{hi} Hz')
    for mode in ['A','468']:
        v=db(np.dot(p,weight(f,mode)))
        assert abs(v-db(p.sum()))>.5
        print(f'PASS {mode} weighted noise differs: {v:.6f} dBFS vs {db(p.sum()):.6f}')
    for f0, a, b in [(100,-19.1,-19.9),(1000,0,0),(4000,1,10.5),(6300,-.1,12.2),(10000,-2.5,8.1)]:
        assert_close(db(weight(np.array(float(f0)),'A')),a,.11,f'A weighting {f0} Hz')
        assert_close(db(weight(np.array(float(f0)),'468')),b,.11,f'468 weighting {f0} Hz')
    assert_close((1*5e-11+2*2e-10)/3,1.5e-10,1e-24,'duration-pooled quiet power')


def cutoff_and_delay():
    rate=96000;n=16384;f=np.fft.rfftfreq(n,1/rate)
    rng=np.random.default_rng(18221)
    phase=rng.uniform(0,2*np.pi,len(f));bins=np.exp(1j*phase);bins[0]=bins[-1]=0;bins[f>22050]=0
    cycle=np.fft.irfft(bins,n);p=power_spectrum(np.tile(cycle,8),n,True)
    def density(lo,hi):return p[(f>=lo)&(f<hi)].sum()/(hi-lo)
    edge=22050.;below=density(.94*edge,.985*edge);above=density(1.015*edge,1.06*edge)
    assert above<below*.001 and density(.88*edge,.94*edge)>below*.01
    assert all(density(edge*(.94+i*.01),edge*(.95+i*.01))>below*.03 for i in range(4))
    print('PASS steep cutoff fixture meets occupied-band and drop criteria')
    rate=48000;n=8192;lag=3;frames=rate
    sequence=rng.normal(0,.01,frames+lag);pcm=np.column_stack((sequence[lag:],sequence[:-lag]));f=np.fft.rfftfreq(n,1/rate)
    cross=np.zeros(n//2+1,complex);le=re=0.;mask=(f>=2000)&(f<=10000)
    for x in windowed_frames(pcm,n,True):
        l,r=np.fft.rfft(x,axis=0).T
        cross[mask]+=np.conj(l[mask])*r[mask];le+=2*np.sum(abs(l[mask])**2);re+=2*np.sum(abs(r[mask])**2)
    c=np.fft.irfft(cross,n)*n/np.sqrt(le*re);indices=np.arange(-48,49);v=abs(c[indices%n]);j=int(np.argmax(v));p=v[j]
    a,b=v[j-1],v[j+1];offset=.5*(a-b)/(a-2*p+b)
    assert p>.5 and v[abs(indices-indices[j])>6].max()<.95*p
    assert_close((indices[j]+offset)*1e6/rate,62.5,2,'upper-band delay sign and fractional interpolation')


def dither_and_tone():
    rate=48000;n=8192;rng=np.random.default_rng(22);x=rng.integers(-1,2,4*rate)/32768.
    f=np.fft.rfftfreq(n,1/rate)
    for signal,shaped in [(x,False),(np.diff(x,prepend=0),True)]:
        p=power_spectrum(signal,n)
        bands=np.array([p[(f>=24000*i/8)&((f<24000*(i+1)/8)|((i==7)&(f==24000)))].sum() for i in range(8)])
        lo=bands[:4].sum();hi=bands[4:].sum()
        assert (hi>2*lo) if shaped else (lo<2*hi and hi<2*lo and bands.min()>.25*bands.max())
        print(f'PASS {"shaped" if shaped else "flat"} dither fixture band ratio {hi/lo:.6f}')
    for freq,amp in [(50,1e-5),(100,2e-5)]:
        p=power_spectrum(tone(rate,2*rate,freq,amp),n);i=np.argmax(p)
        a,b,c=np.log(p[i-1:i+2]);offset=.5*(a-c)/(a-2*b+c)
        assert_close((i+offset)*f[1],freq,2,'hum line frequency')
        assert_close(db(p[i-1:i+2].sum()),db(amp*amp/2),.4,'hum line power')


if __name__=='__main__':
    print('NUMERICAL REFERENCE ONLY — not Rust execution')
    short_term_and_range();spectra_and_noise();cutoff_and_delay();dither_and_tone()
    print('All independent numerical reference checks passed.')
