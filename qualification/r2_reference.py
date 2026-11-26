#!/usr/bin/env python3
"""R2 independently executed fixture models and analytic expectations.

No Rust is loaded. The models reproduce detector and endpoint arithmetic
against known source intervals and frequencies; native tests are mandatory.
The previous-candidate model demonstrates the three targeted failures.
"""
import math
from functools import reduce
from time import perf_counter
import numpy as np
from reference_checks import tone, power_spectrum, db, assert_close

def model(x, rate=48000,threshold=1e-6):
    x=np.asarray(x);x=x[:,None] if x.ndim==1 else x
    energy=x*x
    blocks=[energy[i*rate//20:min((i+1)*rate//20,len(x))] for i in range((len(x)*20+rate-1)//rate)]
    blocks=[b for b in blocks if len(b)]
    q=[np.all(b.sum(axis=0)<=threshold*len(b)) for b in blocks]
    context=[(0.5*min(threshold*len(b)-b.sum(axis=0)),b.max()) if quiet and b.max()<=64*threshold else (0,0) for b,quiet in zip(blocks,q)]
    def edge(b,c):
        if not len(b):return 0
        margin=c[0] if c[1]>threshold and b.max()<=2*c[1] else 0
        if np.all(b.sum(axis=0)<=threshold*len(b)+margin):return len(b)
        bad=np.flatnonzero(np.any(b>threshold,axis=1));return int(bad[0]) if len(bad) else len(b)
    def bounds(b):
        local=(rate+999)//1000
        out=[]
        for j in range(0,len(b),local):
            p=b[j:j+local];s=np.flatnonzero(np.any(p>64*threshold,axis=1))
            if not len(s) and np.any(p.sum(axis=0)>4*threshold*len(p)):s=np.flatnonzero(np.any(p>4*threshold,axis=1))
            if len(s):out.extend([j+s[0],j+s[-1]+1])
        return (out[0],out[-1]) if out else None
    intervals=[]; run=None;start=0
    def append(a,b):
        nonlocal run
        if a==b:return
        if run is None:run=[a,b]
        else:
            assert run[1]==a,(run,a,b)
            run[1]=b
    def end():
        nonlocal run
        if run is not None and run[1]-run[0]>=rate:intervals.append(run)
        run=None
    for i,b in enumerate(blocks):
        prev=i==0 or not q[i-1];nxt=i==len(blocks)-1 or not q[i+1]
        pc=(0,0) if i==0 else context[i-1];nc=(0,0) if i==len(blocks)-1 else context[i+1]
        bb=bounds(b) if b.max()>4*threshold and (not q[i] or prev or nxt) else None
        if bb is not None:
            a,z=bb;prefix=edge(b[:a],pc);suffix=edge(b[z:][::-1],nc)
            split=not q[i] or prev and suffix>=(len(b)+1)//2 or nxt and prefix>=(len(b)+1)//2
        elif q[i] or i==len(blocks)-1 and len(b)<(rate+19)//20 and edge(b,pc)==len(b):split=False
        else:
            bad=np.flatnonzero(np.any(b>threshold,axis=1));prefix=int(bad[0]);suffix=len(b)-1-int(bad[-1]);split=True
        if split:
            append(start,start+prefix);end();append(start+len(b)-suffix,start+len(b))
        else:append(start,start+len(b))
        start+=len(b)
    end();return intervals


def edge_peak(y,n,r):
    y=np.asarray(y);y=y[:,None] if y.ndim==1 else y
    L=len(y);a=2*np.pi*(np.arange(L)+.5)/L
    w=.35875-.48829*np.cos(a)+.14128*np.cos(2*a)-.01168*np.cos(3*a)
    p=(abs(np.fft.rfft(y*w[:,None],n=n,axis=0))**2).sum(axis=1);p[1:-1]*=2
    if not p.max():return 0,r/L
    mask=(p>=p.max()*1e-8)&np.r_[True,p[1:]>=p[:-1]]&np.r_[p[:-1]>p[1:],True]
    return np.flatnonzero(mask)[-1]*r/n,r/L

def extent(x,r):
    n=1<<(math.ceil(r/6)-1).bit_length();h=n//2;L=len(x)
    if L<h:
        edges=[edge_peak(x,n,r)]
        if L>n//8:edges += [edge_peak(x[:n//8],n,r),edge_peak(x[-n//8:],n,r)]
    else:
        edges=[edge_peak(x[:h],n,r),edge_peak(x[:n//8],n,r),edge_peak(x[-h:],n,r),edge_peak(x[-n//8:],n,r)]
        if L<n and L>h:edges += [edge_peak(x,n,r)]
    p=power_spectrum(x,n,True)
    old=0 if not p.max() else np.flatnonzero(p>=p.max()*1e-8)[-1]*r/n
    if p.max():base=old
    else:
        finest=min(d for _,d in edges);base=max(f for f,d in edges if d==finest)
    return max([base]+[f for f,d in edges if f>base+4*d]), old,edges


def previous_quiet_frames(x, rate=48000):
    threshold=1e-6;B=rate//20
    blocks=[x[i:i+B] for i in range(0,len(x),B)]
    quiet=[np.mean(b*b)<=threshold for b in blocks]
    total=run=0
    for j,b in enumerate(blocks):
        q=quiet[j];below=b*b<=threshold;half=(len(b)+1)//2
        prev=j==0 or not quiet[j-1];nxt=j+1==len(blocks) or not quiet[j+1]
        if q and b.max()**2>4*threshold and ((prev and below[-half:].all()) or (nxt and below[:half].all())):q=False
        if q:run+=len(b);continue
        bad=np.flatnonzero(~below)
        if not len(bad):run+=len(b);continue
        run+=int(bad[0]);total+=run if run>=rate else 0;run=len(b)-1-int(bad[-1])
    return total+(run if run>=rate else 0)


def quiet_checks():
    r=48000;Q=tone(r,48960,50,.0012);x=np.r_[np.full(1200,.1),Q,np.full(4800,.1)]
    assert previous_quiet_frames(x)==0
    assert model(x,r)==[[1200,1200+len(Q)]]
    expected=db(np.mean(Q*Q));assert_close(expected,-61.4266750357,1e-8,'R2 supplied quiet-floor oracle')
    print('REPRODUCED previous candidate: quiet passage discarded; R2 model recovers 48,960 frames')
    cases=0
    for r in [48000,96000]:
        B=r//20
        for amp in [1e-5,.0012]:
            for channels,phase,crest in [(1,0,False),(2,.7,False),(1,1.5,False),(2,2.8,False),(1,1.5,True)]:
                if crest and amp<1e-4:continue
                for L in [r-1,r,r+r//50]:
                    q=amp*np.sin(2*np.pi*50*np.arange(L)[:,None]/r+phase+np.arange(channels)[None,:]*np.pi/2)
                    if crest:q[::97]=.003
                    for offset in [0,1,97,333,B//2,B-1]:
                        for lead,tail in [(B+offset,2*B),(0,B+offset),(B+offset,0)]:
                            for loud in [.02,.1]:
                                x=np.concatenate([np.full((lead,channels),loud),q,np.full((tail,channels),loud)])
                                expected=[[lead,lead+L]] if L>=r else []
                                assert model(x,r)==expected,(r,amp,channels,phase,crest,L,offset,lead,tail,loud)
                                cases+=1
    for a,b,expected in [(48000,48960,96960),(30000,30000,0),(48000,30000,48000)]:
        x=np.r_[np.full(a,1e-5),np.full(2733,.1),np.full(b,1e-5)]
        assert sum(z-y for y,z in model(x))==expected
        cases+=1
    # Random high-crest quiet material. The admitted block RMS is below
    # threshold; an individual edge can exceed it because of finite sampling.
    for rms in [.0007,.00085,.00095]:
        for seed in range(25):
            rng=np.random.default_rng(seed);q=rng.normal(0,rms,48000);q*=rms/np.sqrt(np.mean(q*q))
            for off in [1,7,31,97,333,1200,2301,2399]:
                lead=2400+off;x=np.r_[np.full(lead,.02),q,np.full(4800,.02)]
                assert model(x)==[[lead,lead+48000]],(rms,seed,off,model(x))
                cases+=1
    print(f'PASS exact known quiet intervals: {cases} cases (rates, phases, channels, high crests, boundaries, subminimum and separated runs)')


def bandwidth_checks():
    r=96000;n=16384;f=171*r/n;freqs=np.fft.rfftfreq(n,1/r)
    b=tone(r,960,30000,.01)*np.sin(np.pi*(np.arange(960)+.5)/960)**2
    old=[];new=[];ratios=[]
    L=8*n;lo=tone(r,L,f,.1)
    for start in [0,(L-960)//2,L-960]:
        x=lo.copy();x[start:start+960]+=b
        a,c,_=extent(x,r);old.append(c);new.append(a)
        p=power_spectrum(x,n);ratios.append(db(p[freqs>20000].sum()/p.sum()))
        assert_close(a/1000,30,.4,f'R2 burst placement {start} bandwidth kHz')
    assert old[0]<2000 and old[-1]<2000 and old[1]>29000
    print('REPRODUCED previous candidate: bandwidth Hz',old,'; R2 model',new)
    assert max(ratios)-min(ratios)<1e-5
    assert_close(ratios[0],-45.6105,.001,'unchanged complementary power ratio dB')
    cases=0
    for L in [8*n,8*n+333,n-1,12345,9600,4800,2049,1000]:
        for phase in [0,.37,1.5]:
            lo=.1*np.sin(2*np.pi*f*np.arange(L)/r+phase)
            assert abs(extent(lo,r)[0]-f)<40
            assert abs(extent(lo+tone(r,L,30000,.1),r)[0]-30000)<40
            cases+=2
            for start in [0,(L-960)//2,L-960]:
                x=lo.copy();x[start:start+960]+=b
                assert abs(extent(x,r)[0]-30000)<400,(L,phase,start,extent(x,r))
                cases+=1
    for r in [48000,96000,192000]:
        n=1<<(math.ceil(r/6)-1).bit_length();freqs=np.fft.rfftfreq(n,1/r)
        rng=np.random.default_rng(42);bins=np.exp(1j*rng.uniform(0,2*np.pi,len(freqs)));bins[(freqs>22050)|(freqs==0)]=0
        x=np.tile(np.fft.irfft(bins),8)
        assert abs(extent(x,r)[0]-22050)<50
        cases+=1
    r=96000;n=16384;L=4*n+71;lo=tone(r,L,171*r/n,.1);x=np.stack([lo,-lo],axis=1);x[-960:,1]+=b
    assert abs(extent(x,r)[0]-30000)<400
    x=np.zeros(4*n+1333);x[-960:]=b
    assert extent(x,r)[1]==0 and abs(extent(x,r)[0]-30000)<800
    print(f'PASS {cases+2} boundary/short/cutoff/stereo bandwidth fixture models')


def grid_checks():
    cases=0
    for lo,hi in [(-32768,32767),(-30000,30000),(-100,100)]:
        for step,offset in [(192,0),(192,7),(256,7),(256,0),(128,0),(1,0)]:
            codes=np.arange(lo,hi+1,dtype=np.int64)*step+offset
            gcd=reduce(math.gcd,(int(v-codes[0]) for v in codes));v=int(np.bitwise_or.reduce(codes));tz=(v&-v).bit_length()-1
            slots=(int(codes[-1]-codes[0])//gcd)+1
            observed=(slots-1).bit_length();normalized=24-tz
            complete=slots>=4 and slots&(slots-1)==0 and len(codes)>=slots
            affine=gcd>1 and ((gcd&(gcd-1))!=0 or abs(int(codes[0]))%gcd!=0)
            old=min(normalized,observed) if gcd>1 and complete else normalized
            new=min(normalized,observed) if gcd>1 and ((affine and slots>=3) or complete) else normalized
            assert gcd==step and slots==hi-lo+1
            if lo==-30000 and step==192 and offset==0:
                assert (old,new,normalized,observed,gcd)==(18,16,18,16,192)
                print('REPRODUCED previous candidate: used 18; R2 observed width 16, normalized 18, spacing 192')
            if affine:assert new==min(normalized,observed)
            elif gcd==1 or not complete:assert new==normalized
            cases+=1
    print(f'PASS {cases} exact lattice models (complete, headroom, offset, aligned, full grid)')

if __name__=='__main__':
    print('R2 NUMERICAL/FIXTURE MODELS ONLY; NATIVE RUST TESTS NOT EXECUTED')
    start=perf_counter();quiet_checks();bandwidth_checks();grid_checks()
    print(f'R2 reference models completed in {perf_counter()-start:.3f} s (not native throughput)')
