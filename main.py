from argparse import ArgumentParser
import matplotlib.pyplot as plt
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
                        help="Search: Error metric (sse, xent)")
    parser.add_argument("num_L", type=int, 
                        help="Search: Number of permutations per search")

    
    parser.add_argument("--noise_idx", type=int, nargs='*', default=[3],
                        help="Noises (e.g. '5 6' gives 5th and 6th noise [cicada, birds]. Default: 5)")
    parser.add_argument("-q", "--n_test_spkrs", type=int, default=10, 
                        help="Data: Number of test utterances (Default: 10)")
    parser.add_argument("-u", "--use_only_seen_noises", action='store_false',
                        help = "Data: Option to select beyond seen noises")
    parser.add_argument("-e", "--use_pmel", action='store_true',
                        help = "Data: Option to use power mel spectrogram")
    parser.add_argument("-c", "--seed", type=int, default=22,
                        help = "Data: Seed for train and test speaker selection")
    parser.add_argument("-r", "--n_rs", type=int, default=1,
                        help = "Data: Random sampling")
    
    parser.add_argument("-k", "--K", type=int, default=5,
                        help="kNN: Number of nearest neighbors (Default: 5)")
    
    parser.add_argument("-l", "--L", type=int, default=100,
                        help="WTA: Number of permutation vectors (Default: 100)")
    parser.add_argument("-m", "--M", type=int, default=3,
                        help="WTA: Number of samples for permutations (Default: 3)")
    
    parser.add_argument("-d", "--DnC", type=int, default=1,
                        help="Divide and Conquer: Number of partitions (Default: 1)")

    parser.add_argument("-p", "--print_every", type=int, default=10,
                        help="Search: Number of iterations to print results (Default: 10)")
    parser.add_argument("-t", "--time_th", type=float, default=1.0,
                        help="Search: Number of seconds to limit search (Default: 1.0)")
    parser.add_argument("-s", "--is_save", action='store_true',
                        help = "Search: Option to save searched permutations")
    
    parser.add_argument("-b", "--is_debug", action='store_true',
                help = "Debugging: Denoising performance analysis")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    data = setup_experiment_data(args)
    
    pmel_Fs = get_DnC_FL_divs(args.DnC, 128)
    stft_Fs = get_DnC_FL_divs(args.DnC, 513)
    Ls = get_DnC_FL_divs(args.DnC, args.L)
    
    model_nm = "DSTRPNUS({}|{}|{}|{}|{}|{}|{}|{})_ENT({}|{}|{})_LM({}|{})_DK({}|{})".format(
        args.n_dr, args.n_spkr, args.n_test_spkrs, args.n_rs, 
        args.use_pmel, args.noise_idx, args.use_only_seen_noises, args.seed,
        args.errmetric, args.num_L, int(args.time_th),
        args.L, args.M,
        args.DnC, args.K)
    
    if args.is_debug:
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

    else:
        print ("Running {}...".format(model_nm))
        
        recon = librosa.istft(data['teX'] * data['te_IBM'], hop_length=512)
        true_IBM_snr = SDR(recon, data['tes'])[1]
        
        print ("True SNR: {:.2f}".format(true_IBM_snr))
        
        _, snr_mean = DnC_batch(data, args, False, pmel_Fs, stft_Fs)
        print("Mean SNR: {:.2f}".format(snr_mean))

        _, wta_snr_mean, P = DnC_batch(data, args, True, pmel_Fs, stft_Fs, Ls, epochs=1)
        print("WTA Mean SNR: {:.2f}".format(wta_snr_mean))

        # Generate good perms
        search_Ps, search_errs = DnC_search_good_Ps(data, args, pmel_Fs, stft_Fs, Ls)
        search_snr_med, search_snr_mean, errs = DnC_analyze_good_Ps(data, args, pmel_Fs, stft_Fs, Ls, search_Ps)
        plot_results(snr_med, snr_mean, wta_snr_med, wta_snr_mean, search_snr_med, search_snr_mean, model_nm)

        if args.is_save:
            np.save(model_nm, good_Ps)
            print ("Saved")

if __name__ == "__main__":
    main()
