import matplotlib.pyplot as plt
import numpy as np
import _pickle as pickle

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
        self.screen = None
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
        if len(split_nm) > 4:
            fifth_split = split_nm[4].split('(')[1].split(')')[0]
            self.screen = fifth_split
        
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
    model_nm = "DSTRMNUS({}|{}|{}|{}|{}|{}|{}|{})_ENT({}|{}|{})_LM({}|{})_DK({}|{})_S({})".format(
        args.n_dr, args.n_spkr, args.n_test_spkrs, args.n_rs, 
        args.use_mel, args.noise_idx, args.use_only_seen_noises, args.seed,
        args.errmetric, args.num_L, int(args.time_th),
        args.L, args.M,
        args.DnC, args.K,
        args.screen)
    return model_nm

def viz_res(file_dir, title, print_args):
    true_perfs = {k:v for k, v in zip(np.arange(10), np.zeros(10))}
    knn_perfs = {k:v for k, v in zip(np.arange(10), np.zeros(10))}
    wta_perfs = {k:v for k, v in zip(np.arange(10), np.zeros(10))}
    search_perfs = {k:v for k, v in zip(np.arange(10), np.zeros(10))}
    cnts = {k:v for k, v in zip(np.arange(10), np.zeros(10, dtype=np.int))}
    model_names = {k:[] for k in np.arange(10)}

    files = [x for x in os.listdir(file_dir) if '.pkl' in x]
    for model in files:
        f_args = model_argparse(model)
        nidx = f_args.noise_idx
        file_check = True
        for attr, value in f_args.__dict__.items():
            if attr in print_args.__dict__:
                if value != print_args.__dict__[attr]:
                    file_check = False
        
        if file_check:
            n = f_args.noise_idx[0]
            with open(file_dir + model, "rb") as input_file:
                e = pickle.load(input_file)

            true_perfs[n] += e['snr_true']
            knn_perfs[n] += e['snr_mean']
            wta_perfs[n] += e['wta_snr_mean']
            search_perfs[n] += e['search_snr_mean_max']
            cnts[n] += 1
            model_names[n].append(model)

    eps = 1e-10
    avg_true_perfs = [true_perfs[idx]/(cnts[idx]+eps) for idx in range (10)]
    avg_knn_perfs = [knn_perfs[idx]/(cnts[idx]+eps) for idx in range (10)]
    avg_wta_perfs = [wta_perfs[idx]/(cnts[idx]+eps) for idx in range (10)]
    avg_search_perfs = [search_perfs[idx]/(cnts[idx]+eps) for idx in range (10)]

    f, axarr = plt.subplots(1, 3, figsize=(12,4))
    axarr[0].set_title('Counts')
    axarr[0].bar(np.arange(10), height=[cnts[k] for k in range(10)])
    axarr[1].set_title('True vs kNN')
    axarr[1].bar(np.arange(10), height=avg_true_perfs, alpha = 0.9)
    axarr[1].bar(np.arange(10), height=avg_knn_perfs, alpha = 0.7)
    axarr[1].set_ylim(-10,20)
    axarr[2].set_title('WTA vs Search')
    axarr[2].bar(np.arange(10), height=avg_wta_perfs, alpha = 0.9)
    axarr[2].bar(np.arange(10), height=avg_search_perfs, alpha = 0.7)
    axarr[2].set_ylim(-10,20)
    f.suptitle(title)
    
def get_IRM(S, N):
    return np.power(S,2) / (np.power(S,2) + np.power(N,2))