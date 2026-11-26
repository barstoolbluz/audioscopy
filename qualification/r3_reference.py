#!/usr/bin/env python3
"""R3 quiet-edge detector fixture models, not native Rust execution.

The trim=False branch reproduces R2. The enabled branch models the new
bounded local prefix search. Source intervals and raw PCM mean-square
values are independent oracles; they do not depend on the detector.
"""
import numpy as np
from reference_checks import db, assert_close
import r2_reference as r2

def model(x, rate=48000,threshold=1e-6, trim=True, minimum=None):
    minimum=rate if minimum is None else minimum
    x=np.asarray(x);x=x[:,None] if x.ndim==1 else x
    energy=x*x
    blocks=[energy[i*rate//20:min((i+1)*rate//20,len(x))] for i in range((len(x)*20+rate-1)//rate)]
    blocks=[b for b in blocks if len(b)]
    q=[np.all(b.sum(axis=0)<=threshold*len(b)) for b in blocks]
    context=[(0.5*min(threshold*len(b)-b.sum(axis=0)),b.max()) if quiet and b.max()<=64*threshold else (0,0) for b,quiet in zip(blocks,q)]
    def edge(b,c, refine=False):
        if not len(b):return 0
        margin=c[0] if c[1]>threshold and b.max()<=2*c[1] else 0
        if np.all(b.sum(axis=0)<=threshold*len(b)+margin):return len(b)
        bad=np.flatnonzero(np.any(b>threshold,axis=1));exact=int(bad[0]) if len(bad) else len(b)
        if not trim or not refine or c[1]<=threshold:return exact
        count=np.arange(1,len(b)+1)
        good=(count>=max(1,len(b)-2*((rate+999)//1000))) & (count<len(b)) & (np.maximum.accumulate(b.max(axis=1))<=2*c[1]) & (np.cumsum(b,axis=0)<=threshold*count[:,None]+c[0]).all(axis=1)
        return max(exact, int(count[good][-1]) if good.any() else 0)
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
        if run is not None and run[1]-run[0]>=minimum:intervals.append(run)
        run=None
    for i,b in enumerate(blocks):
        prev=i==0 or not q[i-1];nxt=i==len(blocks)-1 or not q[i+1]
        pc=(0,0) if i==0 else context[i-1];nc=(0,0) if i==len(blocks)-1 else context[i+1]
        bb=bounds(b) if b.max()>4*threshold and (not q[i] or prev or nxt) else None
        if bb is not None:
            a,z=bb;prefix=edge(b[:a],pc,True);suffix=edge(b[z:][::-1],nc,True)
            split=not q[i] or prev and suffix>=(len(b)+1)//2 or nxt and prefix>=(len(b)+1)//2
        elif q[i] or i==len(blocks)-1 and len(b)<(rate+19)//20 and edge(b,pc)==len(b):split=False
        else:
            bad=np.flatnonzero(np.any(b>threshold,axis=1));prefix=int(bad[0]);suffix=len(b)-1-int(bad[-1]);split=True
        if split:
            append(start,start+prefix);end();append(start+len(b)-suffix,start+len(b))
        else:append(start,start+len(b))
        start+=len(b)
    end();return intervals


def fixture(rate, channels, offset, loud_phase, quiet_phase, length, loud_hz=440):
    block=rate//20
    lead=block+offset
    channel=np.arange(channels)[None,:]
    t=np.arange(lead+length+2*block)[:,None]/rate
    x=.02*np.sin(2*np.pi*loud_hz*t+loud_phase+channel*.43)
    quiet=.0012*np.sin(2*np.pi*50*np.arange(length)[:,None]/rate+quiet_phase+channel*1.57)
    x[lead:lead+length]=quiet
    return x, quiet, lead


def tonal_checks():
    rate=48000
    x,q,lead=fixture(rate,1,1200,.35,.7,48960)
    prior=model(x,rate,trim=False,minimum=1)
    # Cross-check the unmodified R2 model on the actual eligibility rule.
    assert r2.model(x,rate)==model(x,rate,trim=False)==[]
    assert prior==[[4783,52560]],prior
    assert model(x,rate)==[[3600,52560]]
    expected=db(np.mean(q*q))
    got=db(np.mean(x[3600:52560]**2))
    assert_close(got,-61.4266750357,1e-8,'R3 full-band raw PCM floor')
    print(f'REPRODUCED R2: {prior[0][1]-prior[0][0]} frames ({47777/rate:.9f} s), no qualifying run')
    print(f'PASS R3: 48960 frames (1.02 s), floor {got:.9f} dBFS')
    cases=0; prior_failures=0; worst=0.; worst_duration=0.
    for rate in [48000,96000]:
        for channels in [1,2]:
            for offset in [0,1,97,rate//40,rate//20-1]:
                length=rate+rate//50
                for p in [0,.35,1.5,3.1,4.7,5.9]:
                    for qp in [0,.7,1.4,2.8]:
                        x,q,_=fixture(rate,channels,offset,p,qp,length)
                        expected=db(np.mean(q*q))
                        for y in [x,x[::-1]]:
                            old=r2.model(y,rate)
                            assert model(y,rate,trim=False)==old
                            prior_failures+=not old
                            actual=model(y,rate)
                            assert len(actual)==1,(rate,channels,offset,p,qp,actual)
                            a,z=actual[0]
                            error=abs(db(np.mean(y[a:z]**2))-expected)
                            duration_error=abs(z-a-length)/rate
                            assert error<=.5 and duration_error<=.002,(rate,channels,offset,p,qp,error,duration_error)
                            worst=max(worst,error);worst_duration=max(worst_duration,duration_error)
                            cases+=1
    print(f'PASS {cases} tonal cases (rates, phases, channels, offsets, reverse); {prior_failures} R2 unavailable outcomes; worst floor error {worst:.9f} dB; worst duration error {worst_duration*1000:.6f} ms')
    # Further frequency sweep: the trim must not be specific to a 440 Hz
    # separator or a particular sample phase.
    n=0; worst=0.
    for hz in [110,220,440,880,1760]:
        for p in np.arange(0,6.28,.35):
            for qp in [0,.7,1.4,2.1]:
                x,q,_=fixture(48000,1,1200,p,qp,48960,hz)
                expected=db(np.mean(q*q))
                for y in [x,x[::-1]]:
                    actual=model(y)
                    assert len(actual)==1,(hz,p,qp,actual)
                    a,z=actual[0]
                    err=abs(db(np.mean(y[a:z]**2))-expected)
                    assert err<=.5 and abs(z-a-48960)<=96,(hz,p,qp,err,actual)
                    worst=max(worst,err);n+=1
    print(f'PASS {n} additional loud-frequency/phase/mirror cases; worst floor error {worst:.9f} dB')
    n=0
    for rate in [48000,96000]:
        block=rate//20
        for channels,offset,p,qp in [(1,block//2,.35,.7),(1,1,1.5,0.),(1,block-1,3.1,2.8),(1,97,4.7,1.4),(2,block//2,.35,.7),(2,97,5.9,1.4)]:
            x,_,_=fixture(rate,channels,offset,p,qp,rate-rate//50)
            for y in [x,x[::-1]]:
                assert model(y,rate)==[],(rate,channels,offset,p,qp)
                n+=1
    # Two separated subminimum quiet runs still cannot combine.
    for p in [0,.35,1.5,3.1]:
        x=.02*np.sin(2*np.pi*440*np.arange(60000+7200)/48000+p)
        for a,z in [(1200,31200),(36000,66000)]:
            x[a:z]=.0012*np.sin(2*np.pi*50*np.arange(z-a)/48000+.7)
        for y in [x,x[::-1]]:
            assert model(y)==[]
            n+=1
    print(f'PASS {n} nonqualifying tonal controls (short and separated runs)')


if __name__=='__main__':
    print('R3 NUMERICAL FIXTURE MODELS ONLY; NATIVE RUST NOT EXECUTED')
    tonal_checks()
    # Run the unchanged R2 fixture cases against the R3 detector model as
    # well. Restore the module binding so no historical test is rewritten.
    old_model=r2.model
    try:
        r2.model=model
        r2.quiet_checks()
    finally:
        r2.model=old_model
    print('PASS all 2547 exact R2 quiet-interval cases with R3 refinement')
