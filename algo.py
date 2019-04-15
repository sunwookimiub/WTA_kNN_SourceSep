import numpy as np
import librosa
import time
import copy
from utils import *
from sklearn.metrics import pairwise_distances as pdist

# Reconstruction helper functions
def get_IBM_med_mean(D, IBM, K, T):
    DsIdx = np.argsort(D)
    kNNIdx = DsIdx[:,:K]
    kNNIdx1D = kNNIdx.T.flatten()
    IBM3D = IBM[:,kNNIdx1D]
    IBM3D = np.reshape(IBM3D, (-1, K, T))
    IBMEstMean = np.reshape(np.mean(IBM3D,1), (-1,T))
    return IBMEstMean

def istft_transform_clean(teX, IBM):
    clean = librosa.istft(teX * IBM, hop_length=512)
    return clean

def get_IBM_from_pairwise_dist(teX_mag, trX_mag, IBM, K, metric, P=None):
    teX_feed = teX_mag
    trX_feed = trX_mag
    if P is not None and metric=='hamming':
        teX_feed, _ = WTA(teX_mag, P)
        trX_feed, _ = WTA(trX_mag, P)
    D = pdist(teX_feed.T, trX_feed.T, metric)
    F, T = teX_mag.shape
    return get_IBM_med_mean(D, IBM, K, T)

def get_DnC_FL_divs(DnC, F_or_L):
    F_or_Ls = np.ones(DnC, dtype=np.int) * (F_or_L//DnC)
    F_or_Ls[0] += F_or_L%DnC
    return F_or_Ls

def DnC_batch(data, args, is_WTA, pmel_Fs, stft_Fs, Ls=None, epochs=1): 
    pdist_metric = 'cosine'
    P=None
    Ps = np.zeros((args.L, args.M), dtype=np.int)
    
    sdr_mean = np.zeros(epochs)
    for j in range(epochs):
        IBMEstMean = np.zeros(data['teX'].shape)
        stft_start_idx = 0
        pmel_start_idx = 0
        if is_WTA:
            P_start_idx = 0
        
        for i in range(args.DnC):
            stft_end_idx = stft_start_idx + stft_Fs[i]
            pmel_end_idx = pmel_start_idx + pmel_Fs[i]
            
            IBM_i = data['IBM'][stft_start_idx:stft_end_idx]
            if args.use_pmel:
                teX_i = data['teX_mag_pmel'][pmel_start_idx:pmel_end_idx]
                trX_i = data['trX_mag_pmel'][pmel_start_idx:pmel_end_idx]
            else:
                teX_i = data['teX_mag'][stft_start_idx:stft_end_idx]
                trX_i = data['trX_mag'][stft_start_idx:stft_end_idx]
            
            if is_WTA:
                if args.use_pmel:
                    P = create_permutation(pmel_Fs[i], Ls[i], args.M)
                else:
                    P = create_permutation(stft_Fs[i], Ls[i], args.M)
                P_end_idx = P_start_idx + Ls[i]
                Ps[P_start_idx:P_end_idx] = P
                P_start_idx += Ls[i]
                pdist_metric = 'hamming'

            IBMEstMean[stft_start_idx:stft_end_idx] = get_IBM_from_pairwise_dist(
                                            teX_i, trX_i, IBM_i, args.K, pdist_metric, P)

            stft_start_idx += stft_Fs[i]
            pmel_start_idx += pmel_Fs[i]
            
        tesReconMean = librosa.istft(data['teX'] * IBMEstMean, hop_length=512)
        sdr_mean[j] = SDR(tesReconMean, data['tes'])[1]
        
    if is_WTA:
        return sdr_mean.mean(), Ps
    return sdr_mean.mean()


def DnC_search_good_Ps(data, args, pmel_Fs, stft_Fs, Ls):
    Ps = np.zeros((args.DnC, Ls[0]*2, args.M), dtype=np.int)
    allerrs = []
    stft_start_idx = 0
    pmel_start_idx = 0
    for i in range(args.DnC):
        stft_end_idx = stft_start_idx + stft_Fs[i]
        pmel_end_idx = pmel_start_idx + pmel_Fs[i]

        IBM_i = data['IBM'][stft_start_idx:stft_end_idx]
        if args.use_pmel:
            teX_i = data['teX_mag_pmel'][pmel_start_idx:pmel_end_idx]
            trX_i = data['trX_mag_pmel'][pmel_start_idx:pmel_end_idx]
        else:
            teX_i = data['teX_mag'][stft_start_idx:stft_end_idx]
            trX_i = data['trX_mag'][stft_start_idx:stft_end_idx]
            
        good_P, errs = search_best_P(
                trX_i, Ls[i], args)
        Ps[i] = good_P
        allerrs.append(errs)
        
        stft_start_idx += stft_Fs[i]
        pmel_start_idx += pmel_Fs[i]
        
    return Ps, allerrs

def DnC_analyze_good_Ps(data, args, pmel_Fs, stft_Fs, Ls, Ps):
    skip_n = Ls[0]
    errs = np.zeros((skip_n, args.DnC))
    snr_mean_all = np.zeros(skip_n)
    model_nm = get_model_nm(args)

    for j in range(skip_n):
        stft_start_idx = 0
        pmel_start_idx = 0
        IBM_Mean_i = []
        for i in range(args.DnC):
            stft_end_idx = stft_start_idx + stft_Fs[i]
            pmel_end_idx = pmel_start_idx + pmel_Fs[i]
            
            IBM_i = data['IBM'][stft_start_idx:stft_end_idx]
            if args.use_pmel:
                teX_i = data['teX_mag_pmel'][pmel_start_idx:pmel_end_idx]
                trX_i = data['trX_mag_pmel'][pmel_start_idx:pmel_end_idx]
            else:
                teX_i = data['teX_mag'][stft_start_idx:stft_end_idx]
                trX_i = data['trX_mag'][stft_start_idx:stft_end_idx]
            
            sim_x = get_sim_matrix(trX_i, 'cosine', args.errmetric)

            P = Ps[i][j:j+skip_n]
            sim_h = get_sim_matrix(trX_i, 'hamming', args.errmetric, P)
            errs[j,i] = get_error(sim_x, sim_h, args.errmetric)
            IBM_Mean = get_IBM_from_pairwise_dist(teX_i, trX_i, IBM_i, args.K, 'hamming', P)
            IBM_Mean_i.append(IBM_Mean)
            
            stft_start_idx += stft_Fs[i]
            pmel_start_idx += pmel_Fs[i]

        IBM_Mean_sk = np.concatenate(IBM_Mean_i)        
        tesReconMean_sk = librosa.istft(data['teX'] * IBM_Mean_sk, hop_length=512)
        snr_mean_sk = SDR(tesReconMean_sk, data['tes'])[1]
        snr_mean_all[j] = snr_mean_sk
        if j % args.print_every == 0:
            print (model_nm, end='| ')
            print ("{} {} {:.2f} {}".format(j,j+skip_n, snr_mean_sk, errs[j]))

    return snr_mean_all, errs

# Adaboost type permutation generation
def get_sim_matrix(trX_mag, metric, errmetric, P=None):
    trX_feed = trX_mag
    if metric == 'hamming' and P is not None:
        trX_Pidx, _ = WTA(trX_mag, P)
        trX_feed = trX_Pidx
    sim = pdist(trX_feed.T, metric=metric)
    if errmetric == 'xent':
        return 1-sim
    return sim

def search_good_M(trX_mag, L, M, errmetric):
    F, _ = trX_mag.shape
    search_space = 15
    means = np.ones(search_space)*0.4
    for m in range(3,search_space,2):
        random_P = create_permutation(F, L, m)
        sim_h = get_sim_matrix(trX_mag, 'hamming', errmetric, random_P)
        sim_x = get_sim_matrix(trX_mag, 'cosine', errmetric)
        means[m] = np.abs(sim_x.mean()-sim_h.mean())
    print (means.argmin())
    return means

def get_error(sim_x, sim_h, errmetric):
    if errmetric == 'sse':
        err = np.sum(np.power(sim_x - sim_h,2))
    else:
        err = np.sum(sim_h * np.log(sim_x+1e-20))
    return err

def search_best_P(X, L, args):
    model_nm = get_model_nm(args)
    F, _ = X.shape
    
    # Inits
    tot_n = L * 2 # for testing purposes run extra
    num_iters = tot_n//args.num_L
    start_th = L//args.num_L # To ensure only num_L are being tested
    
    good_Ps = np.zeros((tot_n, args.M), dtype=np.int)
    errs, times = np.zeros(num_iters), np.zeros(num_iters)

    # Retrieve similarity metric for train set
    sim_x = get_sim_matrix(X, 'cosine', args.errmetric)

    # Get first error approximation
    random_P = create_permutation(F, args.num_L, args.M)
    sim_h = get_sim_matrix(X, 'hamming', args.errmetric, random_P)
    err = get_error(sim_x, sim_h, args.errmetric)

    # Begin iterative search of best permutations
    fix_err = False
    for i in range(num_iters):
        start_idx = i*args.num_L
        toc = time.time()

        # If it takes too long, don't try to get better
        if not fix_err: 
            prev_err = err
        err = np.inf

        # Search for a better permutation than before
        while np.abs(err) > np.abs(prev_err):
            p_start_idx = 0
            if i >= start_th: 
                p_start_idx += (i-start_th+1)*args.num_L
            end_idx = (i+1)*args.num_L
            
            good_Ps[start_idx:end_idx] = create_permutation(F, args.num_L, args.M)
            use_P = good_Ps[p_start_idx:end_idx]
            sim_h = get_sim_matrix(X, 'hamming', args.errmetric, use_P)
            err = get_error(sim_x, sim_h, args.errmetric)

        # Set limiting criterion
        errs[i], times[i] = err, time.time() - toc
        if times[i] > args.time_th and not fix_err and i > 20:
            fix_err = True
            print (model_nm, end='| ')
            print ("Fixed at epoch {} for taking {:.2f}s".format(start_idx, times[i]))
                
        if start_idx % args.print_every == 0:
            print (model_nm, end='| ')
            print("Epoch {} t: {:.2f} err: {:.2f}".format(start_idx, times[i], errs[i]))
                    
    return good_Ps, errs

def random_sampling_search(data, args, pmel_Fs, stft_Fs, Ls, subsample_Ls):
    rs_Ps = None
    _, T = data['trX_mag'].shape
    n_sample_frames = T//args.n_rs
    np.random.seed(args.seed)
    perms = np.random.permutation(T)
    for i in range(args.n_rs):
        newdata = copy.deepcopy(data)
        newdata['trX_mag'] = newdata['trX_mag'][:,perms][:,n_sample_frames*i:n_sample_frames*(i+1)]
        newdata['IBM'] = newdata['IBM'][:,perms][:,n_sample_frames*i:n_sample_frames*(i+1)]
        search_Ps_i, search_errs = DnC_search_good_Ps(newdata, args, pmel_Fs, stft_Fs, subsample_Ls)
        if rs_Ps is not None:
            rs_Ps = np.concatenate((rs_Ps, search_Ps_i),1)
        else:
            rs_Ps = search_Ps_i
        print (rs_Ps.shape)

    return rs_Ps

def DnC_search_good_Ps(data, args, pmel_Fs, stft_Fs, Ls):
    Ps = np.zeros((args.DnC, Ls[0]*2, args.M), dtype=np.int)
    allerrs = []
    stft_start_idx = 0
    pmel_start_idx = 0
    for i in range(args.DnC):
        stft_end_idx = stft_start_idx + stft_Fs[i]
        pmel_end_idx = pmel_start_idx + pmel_Fs[i]

        IBM_i = data['IBM'][stft_start_idx:stft_end_idx]
        if args.use_pmel:
            teX_i = data['teX_mag_pmel'][pmel_start_idx:pmel_end_idx]
            trX_i = data['trX_mag_pmel'][pmel_start_idx:pmel_end_idx]
        else:
            teX_i = data['teX_mag'][stft_start_idx:stft_end_idx]
            trX_i = data['trX_mag'][stft_start_idx:stft_end_idx]
            
        good_P, errs = search_best_P(
                trX_i, Ls[i], args)
        Ps[i] = good_P
        allerrs.append(errs)
        
        stft_start_idx += stft_Fs[i]
        pmel_start_idx += pmel_Fs[i]
        
    return Ps, allerrs


def debug_ind_noise_snr(data, args, pmel_Fs, stft_Fs, model_nm):
    seeded_snr_means = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    tot_seeds = 100
    for seed in range(tot_seeds):
        args.seed = seed
        np.random.seed(args.seed)
        args.noise_idx = [seed % 10]
        data = setup_experiment_data(args)
        _, snr_mean = DnC_batch(data, args, False, pmel_Fs, stft_Fs)
        print (seed, snr_mean)
        seeded_snr_means[str(args.noise_idx[0])] += snr_mean
    snrs = [seeded_snr_means[str(idx)]/(tot_seeds/10) for idx in range (10)]
    plt.bar(np.arange(10), height=snrs)
    plt.savefig("DEBUG_" + model_nm)
    
def load_model_and_get_max(model_nm, data, args, pmel_Fs, stft_Fs, Ls):
    search_Ps = np.load(model_nm)
    search_snr_mean, errs = DnC_analyze_good_Ps(data, args, pmel_Fs, stft_Fs, Ls, search_Ps)
    return search_snr_mean.max()