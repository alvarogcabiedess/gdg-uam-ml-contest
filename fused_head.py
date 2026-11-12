"""Original fused CPU head for our frozen network; no training or sample cache.

Sequential float32 reductions can differ from ORT's reduction order. This is a
candidate implementation, not a claim of bitwise equivalence to ORT.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def finish(z, rate, w1, b1, w2, b2, features, hidden, logits):
    channels, length = z.shape
    for c in range(channels):
        total = np.float32(0.)
        largest = np.float32(-np.inf)
        for t in range(length):
            total += z[c, t]
            largest = max(largest, z[c, t])
        mean = total / np.float32(length)
        variance = np.float32(0.)
        for t in range(length):
            delta = z[c, t] - mean
            variance += delta * delta
        features[c] = mean
        features[channels+c] = np.sqrt(variance/np.float32(length)+np.float32(1e-5))
        features[2*channels+c] = largest
        for block in range(4):
            total = np.float32(0.)
            for t in range(block*(length//4), (block+1)*(length//4)):
                total += z[c, t]
            features[3*channels+4*c+block] = total / np.float32(length//4)
    features[-1] = rate
    # SIMD across independent output lanes; no fastmath/reassociation requested.
    for j in range(hidden.size):
        hidden[j] = b1[j]
    for i in range(features.size):
        for j in range(hidden.size):
            hidden[j] += features[i] * w1[i, j]
    for j in range(hidden.size):
        hidden[j] = max(hidden[j], np.float32(0.))
    for j in range(logits.size):
        logits[j] = b2[j]
    for i in range(hidden.size):
        for j in range(logits.size):
            logits[j] += hidden[i] * w2[i, j]
    best = 0
    for j in range(1, logits.size):
        if logits[j] > logits[best]:
            best = j
    return best
