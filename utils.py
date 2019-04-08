import numpy as np

def SDR(s,sr):
    eps=1e-20
    ml=np.minimum(len(s), len(sr))
    s=s[:ml]
    sr=sr[:ml]
    return ml, 10*np.log10(np.sum(s**2)/(np.sum((s-sr)**2)+eps)+eps)

# WTA portion
def create_permutation(F_len, L, M):
    P = np.zeros([L,M], dtype=np.uint16)
    if F_len > np.iinfo(np.uint16).max:
        print ("WARNING: Need to increase dtype")
        return -1
    for ii in range(L):
        P[ii,:] = np.random.permutation(np.arange(F_len))[:M]
    return P

def WTA(X,P):
    # uint16,uint16 -> uint16
    # Index X with permutations P
    T = X.shape[1]
    L,M = P.shape
    rowSub = np.tile(P.T.reshape(-1), T)
    colSub = np.repeat(np.arange(T, dtype=np.uint16), L*M)
    Y = X[rowSub, colSub]
    Y = np.reshape(Y, [L,M,T], order="F")
    
    # Take the maximum index
    maxY, maxIY = np.expand_dims(np.max(Y,1),1), np.argmax(Y,1)
    
    # Index back to original frequency bins
    maxIX = np.zeros([L,T], dtype=np.uint16)
    for ii in range(L):
        maxIX[ii,:] = P[ii,maxIY[ii,:]]
    return maxIY, maxIX