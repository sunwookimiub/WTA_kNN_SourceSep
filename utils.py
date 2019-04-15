import matplotlib.pyplot as plt
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

def plot_results(snr_true, snr_mean, wta_snr_mean, search_snr_mean, model_nm):
    f, axarr = plt.subplots(1, 2, figsize=(12,4))
    axarr[0].set_title('SNR (Mean)')
    axarr[0].plot(search_snr_mean)
    axarr[1].set_title('SNR vs.')
    axarr[1].bar([0],
                height=[snr_true])
    axarr[1].bar([1],
                height=[snr_mean])
    axarr[1].bar([2],
                height=[wta_snr_mean])
    axarr[1].bar([3],
                height=[search_snr_mean.max()])
    f.tight_layout()
    f.savefig(model_nm)


class model_argparse():
    def __init__(self, model_nm):
        self.n_dr = None
        self.n_spkr = None
        self.errmetric = None
        self.num_L = None
        
        self.noise_idx = None
        self.n_test_spkrs = None
        self.use_only_seen_noises = None
        self.use_mel = None
        self.seed = None
        self.n_rs = None
        
        self.K = None
        self.L = None
        self.M = None
        self.DnC = None
        
        self.print_every = 10
        self.time_th = None
        self.is_save = False
        self.parse(model_nm)
    
    def parse(self, model_nm):
        real_nm = model_nm.split('.npy')[0]
        split_nm = real_nm.split('_')
        first_split = split_nm[0].split('(')[1].split(')')[0].split('|')
        second_split = split_nm[1].split('(')[1].split(')')[0].split('|')
        third_split = split_nm[2].split('(')[1].split(')')[0].split('|')
        fourth_split = split_nm[3].split('(')[1].split(')')[0].split('|')
        
        self.n_dr = int(first_split[0])
        self.n_spkr = int(first_split[1])
        self.n_test_spkrs = int(first_split[2])
        self.n_rs = int(first_split[3])
        self.use_mel = True if first_split[4] == 'True' else False
        self.noise_idx = list(map(int, first_split[5].replace("[", "").replace("]", "").split(',')))
        self.use_only_seen_noises = bool(first_split[6])
        self.seed = int(first_split[7])
        
        self.errmetric = second_split[0]
        self.num_L = int(second_split[1])
        self.time_th = float(second_split[2])
        
        self.L = int(third_split[0])
        self.M = int(third_split[1])
        
        self.DnC = int(fourth_split[0])
        self.K = int(fourth_split[1])


def get_model_nm(args):        
    model_nm = "DSTRMNUS({}|{}|{}|{}|{}|{}|{}|{})_ENT({}|{}|{})_LM({}|{})_DK({}|{})".format(
        args.n_dr, args.n_spkr, args.n_test_spkrs, args.n_rs, 
        args.use_mel, args.noise_idx, args.use_only_seen_noises, args.seed,
        args.errmetric, args.num_L, int(args.time_th),
        args.L, args.M,
        args.DnC, args.K)
    return model_nm