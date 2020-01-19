import operator as op
from functools import reduce

def C(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

t = 10
n = 3 * t + 1
a = 11
CountA = 0.
for pos in range(2*t+1,4*t+2):
    CountA += C(pos,2*t)*C(4*n-t-pos,n+a-3*t-1)*C(3*n+2*t+1-a-pos,n)
CountB = 0.
for pos in range(2*t+1,4*t+2):
    CountB += C(pos,2*t)*C(4*n-t-pos,2*n-a-2*t-1)*C(2*n+t+1+a-pos,n)
print(CountB/(CountA+CountB))

for pos in range(2*t+1,4*t+2):
    up = 0.
    for nn in range(n-t+1,n+1):
        up += C(nn+4*t+1-pos,nn)*C(n-nn+a+n+3*t-1,n-nn)
    print(up/C(2*n+t+1+a-pos,n))
