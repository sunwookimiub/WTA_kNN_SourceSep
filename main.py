from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _pickle as pickle
import os

from loader import *
from utils import *
from algo import *

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("screen", type=str,
                        help = "Misc: Screen session being run on")
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
    parser.add_argument("-e", "--use_mel", action='store_true',
                        help = "Data: Option to use mel spectrogram")
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
    
    mel_Fs = get_DnC_FL_divs(args.DnC, 128)
    stft_Fs = get_DnC_FL_divs(args.DnC, 513)
    Ls = get_DnC_FL_divs(args.DnC, args.L)

    model_nm = get_model_nm(args)
    
    if args.is_debug:
        # debug_ind_noise_snr(data, args, mel_Fs, stft_Fs, model_nm) # kNN
        # debug_wta_snr(args, mel_Fs, stft_Fs, Ls) # WTA
        # debug_SDR_reconstruction('DSTRMNUS(8|2|10|5|True|[8]|True|1)_ENT(xent|1|180)_LM(100|2)_DK(1|5)_S(c4).pkl')
        file_dir = "Results_get_argmax1"
        debug_get_argmax(file_dir)
        # print ("Nothing here")

    else:
        print ("Running {}...".format(model_nm))
        
        recon = librosa.istft(data['teX'] * data['te_IRM'], hop_length=512)
        snr_true = SDR(recon, data['tes'])[1]
        
        print ("Oracle SNR: {:.2f}".format(snr_true))
        
        snr_mean = DnC_batch(data, args, False, mel_Fs, stft_Fs)
        print("Mean SNR: {:.2f}".format(snr_mean))

        wta_snr_mean, P = DnC_batch(data, args, True, mel_Fs, stft_Fs, Ls, epochs=1)
        print("WTA Mean SNR: {:.2f}".format(wta_snr_mean))

        # Generate good perms
        search_Ps = random_sampling_search(data, args, mel_Fs, stft_Fs, Ls)
        search_snr_mean, errs = DnC_analyze_good_Ps(data, args, mel_Fs, stft_Fs, Ls, search_Ps)
        plot_results(snr_true, snr_mean, wta_snr_mean, search_snr_mean, model_nm)

        if args.is_save:
            save_data = {'search_Ps': search_Ps, 'snr_true': snr_true, 'snr_mean': snr_mean,
                         'wta_snr_mean': wta_snr_mean, 'search_snr_mean_max': search_snr_mean.max()}
            pickle.dump(save_data, open("{}.pkl".format(model_nm),"wb"))
            print ("Saved")

if __name__ == "__main__":
    main()
