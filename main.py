from argparse import ArgumentParser
import matplotlib.pyplot as plt
import _pickle as pickle
import os

from loader import *
from utils import *
from algo import *

def parse_arguments():
    parser = ArgumentParser()
    
    parser.add_argument("n_dr", type=int, 
                        help="Data: Number of dialects")
    parser.add_argument("n_spkr", type=int, 
                        help="Data: Number of speakers per dialect")
    parser.add_argument("errmetric", type=str, 
                        help="Error metric (sse, xent)")
    parser.add_argument("num_L", type=int, 
                        help="Search: Number of permutations per search")

    
    parser.add_argument("--noise_idx", type=int, nargs='*', default=[5],
                        help="Noises (e.g. '5 6' gives 5th and 6th noise [cicada, birds]. Default: 5)")
    
    parser.add_argument("-k", "--K", type=int, default=5,
                        help="kNN: Number of nearest neighbors (Default: 5)")
    parser.add_argument("-l", "--L", type=int, default=100,
                        help="WTA: Number of permutation vectors (Default: 100)")
    parser.add_argument("-m", "--M", type=int, default=3,
                        help="WTA: Number of samples for permutations (Default: 3)")
    parser.add_argument("-d", "--DnC", type=int, default=1,
                        help="Divide and Conquer: Number of partitions (Default: 1)")
    parser.add_argument("-u", "--use_only_seen_noises", action='store_false',
                        help = "Data: Option to select beyond seen noises")
    parser.add_argument("-p", "--print_every", type=int, default=10,
                        help="Search: Number of iterations to print results (Default: 10)")
    parser.add_argument("-t", "--time_th", type=float, default=1.0,
                        help="Search: Number of seconds to limit search (Default: 1.0)")
    parser.add_argument("-n", "--num_p", type=int, default=100, 
                        help="Search: Number of total permutations to search (Default: 100)")
    parser.add_argument("-e", "--extra_p", type=int, default=100,
                        help="Search: Number of extra permutations to search (Default: 100)")
    
    return parser.parse_args()


def main():
    args = parse_arguments()

    model_nm = "DS({}|{})_E({})_NT({}|{})".format(args.n_dr, args.n_spkr, args.errmetric, args.num_L, int(args.time_th))
    print ("Running {}...".format(model_nm))

    eps = 1e-10

    # WTA params
    n_DnC = 513//args.DnC
    F = np.ones(args.DnC) * n_DnC

    data = setup_experiment_data(args.n_dr, args.n_spkr, args.noise_idx, args.use_only_seen_noises)
    tes, teX, trX_mag, teX_mag, IBM = data['tes'], data['teX'], data['trX_mag'], data['teX_mag'], data['IBM']

    IBMEstMed, IBMEstMean = get_IBM_from_pairwise_dist(teX_mag, trX_mag, IBM, args.K, 'cosine')
    tesReconMed, tesReconMean = istft_transform_clean(teX, IBMEstMed, IBMEstMean)
    print("Median SNR: {:.2f} Mean SNR: {:.2f}".format(SDR(tesReconMed, tes)[1], SDR(tesReconMean, tes)[1]))

    wta_med_snr, wta_mean_snr = avg_WTA(teX_mag, teX, trX_mag, tes, IBM, args.K, args.L, args.M)
    print("WTA Median SNR: {:.2f} WTA Mean SNR: {:.2f}".format(wta_med_snr, wta_mean_snr))

    # Generate good perms
    good_Ps, errs = search_best_P(
                        trX_mag, args.errmetric, args.num_L, args.M, args.num_p,
                        args.print_every, args.time_th, args.extra_p)

    sim_x = get_sim_matrix(trX_mag, 'cosine', args.errmetric)

    # Performance
    select_Ps = good_Ps[:100]
    snr_med, snr_mean, err = get_WTA_SNR_and_err(
                                trX_mag, teX_mag, teX, tes, IBM, args.errmetric, select_Ps, sim_x, args.K)
    print ("First 100 Mean SDR: {:.2f} Med SDR: {:.2f} Sim Err: {:.2f}".format(snr_med, snr_mean, err))

    select_Ps = good_Ps[50:150]
    snr_med, snr_mean, err = get_WTA_SNR_and_err(
                                trX_mag, teX_mag, teX, tes, IBM, args.errmetric, select_Ps, sim_x, args.K)
    print ("50 to 150 Mean SDR: {:.2f} Med SDR: {:.2f} Sim Err: {:.2f}".format(snr_med, snr_mean, err))

    select_Ps = good_Ps[-100:]
    snr_med, snr_mean, err = get_WTA_SNR_and_err(
                                trX_mag, teX_mag, teX, tes, IBM, args.errmetric, select_Ps, sim_x, args.K)
    print ("Last 100 Mean SDR: {:.2f} Med SDR: {:.2f} Sim Err: {:.2f}".format(snr_med, snr_mean, err))

    pickle.dump(good_Ps, open("{}.p".format(model_nm),"wb"))
    print ("Dumped")



if __name__ == "__main__":
    main()
