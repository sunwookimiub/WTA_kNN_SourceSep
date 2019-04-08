import numpy as np
import librosa
import time
from utils import *
from sklearn.metrics import pairwise_distances as pdist

# Reconstruction helper functions
def get_IBM_med_mean(D, IBM, K, teX):
    DsIdx = np.argsort(D)
    kNNIdx = DsIdx[:,:K]
    kNNIdx1D = kNNIdx.T.flatten()
    IBM3D = IBM[:,kNNIdx1D]
    IBM3D = np.reshape(IBM3D, (teX.shape[0], K, teX.shape[1]))
    IBMEstMed = np.reshape(np.median(IBM3D,1), (teX.shape))
    IBMEstMean = np.reshape(np.mean(IBM3D,1), (teX.shape))
    return IBMEstMed, IBMEstMean

def istft_transform_clean(teX, IBMEstMed, IBMEstMean):
    median_clean = librosa.istft(teX * IBMEstMed, hop_length=512)
    mean_clean = librosa.istft(teX * IBMEstMean, hop_length=512)
    return median_clean, mean_clean

def get_IBM_from_pairwise_dist(teX_mag, trX_mag, IBM, K, metric, P=None):
    teX_feed = teX_mag
    trX_feed = trX_mag
    if P is not None and metric=='hamming':
        teX_feed, _ = WTA(teX_mag, P)
        trX_feed, _ = WTA(trX_mag, P)
    D = pdist(teX_feed.T, trX_feed.T, metric)
    return get_IBM_med_mean(D, IBM, K, teX_mag)

def avg_WTA(teX_mag, teX, trX_mag, tes, IBM, K, L, M):
    F, _ = trX_mag.shape
    sdr_med, sdr_mean = np.zeros(50), np.zeros(50)
    for i in range(50):
        P = create_permutation(F, L, M)
        IBMEstMed, IBMEstMean = get_IBM_from_pairwise_dist(teX_mag, trX_mag, IBM, K, 'hamming', P)
        one_tesReconMed_h1, one_tesReconMean_h1 = istft_transform_clean(teX, IBMEstMed, IBMEstMean)
        sdr_med[i], sdr_mean[i] = SDR(one_tesReconMed_h1, tes)[1], SDR(one_tesReconMean_h1, tes)[1]
    return sdr_med.mean(), sdr_mean.mean()

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
        err = np.sum(sim_h * np.log(sim_x))
    return err

def search_best_P(trX_mag, errmetric, num_L, M, num_p, print_every, time_th, extra_p):
    F, _ = trX_mag.shape
    
    # Inits
    tot_n = num_p + extra_p # for testing purposes run extra
    num_iters = tot_n//num_L
    start_th = num_p//num_L # To ensure only num_L are being tested
    
    good_Ps = np.zeros((tot_n, M), dtype=np.int)
    errs, times = np.zeros(num_iters), np.zeros(num_iters)

    # Retrieve similarity metric for train set
    sim_x = get_sim_matrix(trX_mag, 'cosine', errmetric)

    # Get first error approximation
    random_P = create_permutation(F, num_L, M)
    sim_h = get_sim_matrix(trX_mag, 'hamming', errmetric, random_P)
    err = get_error(sim_x, sim_h, errmetric)

    # Begin iterative search of best permutations
    fix_err = False
    for i in range(num_iters):
        start_idx = i*num_L
        toc = time.time()

        # If it takes too long, don't try to get better
        if not fix_err: 
            prev_err = err
        err = np.inf

        # Search for a better permutation than before
        while np.abs(err) > np.abs(prev_err):
            p_start_idx = 0
            if i >= start_th: 
                p_start_idx += (i-start_th+1)*num_L
            end_idx = (i+1)*num_L
            
            good_Ps[start_idx:end_idx] = create_permutation(F, num_L, M)
            use_P = good_Ps[p_start_idx:end_idx]
            sim_h = get_sim_matrix(trX_mag, 'hamming', errmetric, use_P)
            err = get_error(sim_x, sim_h, errmetric)

        # Set limiting criterion
        errs[i], times[i] = err, time.time() - toc
        if times[i] > time_th and not fix_err and i > 20:
                fix_err = True
                print ("Fixed at epoch {} for taking {:.2f}s".format(start_idx, times[i]))
                
        if start_idx % print_every == 0:
            print("Epoch {} t: {:.2f} err: {:.2f}".format(start_idx, times[i], errs[i]))
                    
    return good_Ps, errs

def get_WTA_SNR_and_err(trX_mag, teX_mag, teX, tes, IBM, errmetric, P, sim_x, K):
    sim_h = get_sim_matrix(trX_mag, 'hamming', errmetric, P)
    err = get_error(sim_x, sim_h, errmetric)
    
    IBMEstMed, IBMEstMean = get_IBM_from_pairwise_dist(teX_mag, trX_mag, IBM, K, 'hamming', P)
    one_tesReconMed_h1, one_tesReconMean_h1 = istft_transform_clean(teX, IBMEstMed, IBMEstMean)
    return SDR(one_tesReconMed_h1, tes)[1], SDR(one_tesReconMean_h1, tes)[1], err