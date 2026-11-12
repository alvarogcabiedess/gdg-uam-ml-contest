"""Single-sample preprocessing compiled during Model construction, outside timed inference."""
from __future__ import annotations

import math
import numpy as np
from numba import njit


@njit(cache=True)
def preprocess_into(series,result,work,counts,bounds):
    row=work[0]
    clean=work[1]
    medians=work[2]
    ordered=work[3]
    lows=bounds[0]
    highs=bounds[1]
    for ch in range(4):
        # Linear interpolation, extending the nearest finite value at the edges.
        first=-1
        for i in range(256):
            if math.isfinite(series[ch,i]):
                first=i
                break
        if first<0:
            for i in range(256): result[0,ch,i]=0.
            continue
        for i in range(first+1): row[i]=series[ch,first]
        previous=first
        for i in range(first+1,256):
            if math.isfinite(series[ch,i]):
                value=series[ch,i]
                for j in range(previous+1,i):
                    row[j]=series[ch,previous]+(value-series[ch,previous])*((j-previous)/(i-previous))
                row[i]=value
                previous=i
        for i in range(previous+1,256): row[i]=row[previous]
        # A histogram bounds the two quartiles from actual observed bucket extrema.
        # If every median residual is <= this lower threshold, the full sort
        # cannot replace any value. Otherwise execute the original exact path.
        minimum=float(row[0]); maximum=float(row[0]); max_residual=0.; all_finite=True
        for i in range(256):
            value=float(row[i]); minimum=min(minimum,value); maximum=max(maximum,value)
            all_finite=all_finite and math.isfinite(value)
            a=row[max(0,i-1)];b=row[i];c=row[min(255,i+1)]
            median=max(min(a,b),min(max(a,b),c))
            medians[i]=median
            max_residual=max(max_residual,abs(float(b)-float(median)))
        certified=False
        if all_finite and max_residual<=5.*1e-4:
            certified=True
        elif all_finite and math.isfinite(maximum-minimum) and maximum>minimum:
            for slot in range(16):
                counts[slot]=0;lows[slot]=np.inf;highs[slot]=-np.inf
            factor=15./(maximum-minimum)
            for i in range(256):
                value=float(row[i]);bucket=min(15,max(0,int((value-minimum)*factor)))
                counts[bucket]+=1;lows[bucket]=min(lows[bucket],value);highs[bucket]=max(highs[bucket],value)
            cumulative=0;b63=0;b64=0;b191=0;b192=0
            for bucket in range(16):
                previous_count=cumulative;cumulative+=counts[bucket]
                if previous_count<=63<cumulative:b63=bucket
                if previous_count<=64<cumulative:b64=bucket
                if previous_count<=191<cumulative:b191=bucket
                if previous_count<=192<cumulative:b192=bucket
            upper_q25=highs[b63]*.25+highs[b64]*.75
            lower_q75=lows[b191]*.75+lows[b192]*.25
            lower_threshold=5.*max((lower_q75-upper_q25)/1.349,1e-4)
            certified=max_residual<=lower_threshold
        if certified:
            threshold=max_residual
        else:
            ordered[:]=row
            ordered.sort()
            q25=float(ordered[63])*.25+float(ordered[64])*.75
            q75=float(ordered[191])*.75+float(ordered[192])*.25
            threshold=5.*max((q75-q25)/1.349,1e-4)
        total=0.
        for i in range(256):
            value=row[i]
            if not certified:
                b=row[i];median=medians[i]
                value=median if abs(float(b)-float(median))>threshold else b
            clean[i]=value
            total+=value
        mean=total/256.
        variance=0.
        for i in range(256):
            variance+=(float(clean[i])-mean)**2
        scale=max(math.sqrt(variance/256.),1e-5)
        for i in range(256):
            result[0,ch,i]=min(6.,max(-6.,(float(clean[i])-mean)/scale))
    return result


def warmup():
    pass
