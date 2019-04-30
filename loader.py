import os
import random
import librosa
import numpy as np
from utils import get_IRM

def load_spkr(spkr_dir):
    spkr_files = [x for x in os.listdir(spkr_dir) if 'wav' in x]
    spkr_frqs = [librosa.load('{}/{}'.format(spkr_dir, x), sr=16000)[0] for x in spkr_files]
    spkr_frqs = [frqs/frqs.std() for frqs in spkr_frqs]
    return spkr_frqs

def load_noises(noise_dir):
    noise_files = ['birds.wav', 'casino.wav', 'cicadas.wav', 'computerkeyboard.wav', 'eatingchips.wav', 'frogs.wav', 'jungle.wav', 'machineguns.wav', 'motorcycles.wav', 'ocean.wav']
    noise_frqs = [librosa.load('{}/{}'.format(noise_dir, x), sr=16000)[0] for x in noise_files]
    noise_frqs = [frqs/frqs.std() for frqs in noise_frqs]
    return noise_frqs

def get_and_add_noise(noise, source, seed, test=False):
    np.random.seed(seed)
    if test:
        noise_start = 0
    else:
        noise_start = np.random.randint(0,len(noise) - len(source))
    spkr_noise = noise[noise_start:noise_start + len(source)]
    return spkr_noise, spkr_noise + source

def load_more_spkr_with_noise(spkr_dir, noise, seed):
    spkr_frqs = load_spkr(spkr_dir)
    spkr_s = np.concatenate(spkr_frqs, 0)
    spkr_n, spkr_x = get_and_add_noise(noise, spkr_s, seed)
    return spkr_s, spkr_n, spkr_x

def get_random_dr_f_speakers(dr_idx, num_speakers, seed):
    all_spkrs = ['dr{}/{}'.format(dr_idx, name) for name in 
                    os.listdir('Data/train/dr{}'.format(dr_idx))if 'Store' not in name]
    f_spkrs = [spkr for spkr in all_spkrs if spkr.split('/')[1][0] == 'f']
    np.random.seed(seed)
    perms = np.random.permutation(len(f_spkrs))[:num_speakers]
    return [f_spkrs[i] for i in perms]

def get_random_dr_m_speakers(dr_idx, num_speakers, seed):
    all_spkrs = ['dr{}/{}'.format(dr_idx, name) for name in 
                    os.listdir('Data/train/dr{}'.format(dr_idx))if 'Store' not in name]
    m_spkrs = [spkr for spkr in all_spkrs if spkr.split('/')[1][0] == 'm']
    np.random.seed(seed)
    perms = np.random.permutation(len(m_spkrs))[:num_speakers]
    return [m_spkrs[i] for i in perms]

def get_random_dr_speakers(dr_idx, num_speakers, seed):
    num_f = num_m = num_speakers//2
    if num_speakers % 2 != 0:
        if dr_idx % 2 == 0:
            num_f += 1
        else:
            num_m += 1
    f_spkrs = get_random_dr_f_speakers(dr_idx, num_f, seed)
    m_spkrs = get_random_dr_m_speakers(dr_idx, num_m, seed)
    fm_spkrs = f_spkrs + m_spkrs
    return fm_spkrs

def get_random_tes(speaker_lists, seed):
    dr_idx = np.random.randint(1,9)
    all_spkrs = ['dr{}/{}'.format(dr_idx, name) for name in 
                os.listdir('Data/train/dr{}'.format(dr_idx))
                if 'Store' not in name 
                and 'dr{}/{}'.format(dr_idx,name) not in speaker_lists]
    np.random.seed(seed)
    perms = np.random.permutation(len(all_spkrs))[0]
    return all_spkrs[perms]

def normalize_frqs(max_amp, frqs):
    return [frq/max_amp for frq in frqs]

def stft_transform(frqs):
    return [librosa.stft(frq, n_fft=1024, hop_length=512) for frq in frqs]

def get_magnitudes(stfts):
    return [np.abs(stft) for stft in stfts]

def get_mels(stft_mags):
    return [librosa.feature.melspectrogram(S=stft_mag) for stft_mag in stft_mags]

def load_trainset(trs_spkr_lists, noise_idx_list, noise_frqs, seed):
    trs, trn, trx = [], [], []
    for i, trs_spkr in enumerate(trs_spkr_lists):
        for idx in noise_idx_list:
            s, n, x = load_more_spkr_with_noise('Data/train/{}'.format(trs_spkr), noise_frqs[idx], seed) 
            trs.append(s)
            trn.append(n)
            trx.append(x)
    trs = np.concatenate(trs).ravel()
    trn = np.concatenate(trn).ravel()
    trx = np.concatenate(trx).ravel()
    return trs, trn, trx

def load_testset(tes_spkr_lists, noise_idx_list, use_only_seen_noises, noise_frqs, seed):
    tes, ten, tex = [], [], []
    used_noises = []
    np.random.seed(seed)
    use_perms = np.random.permutation(len(noise_idx_list))
    all_perms = np.random.permutation(len(tes_spkr_lists))%10
    
    for i, tes_spkr in enumerate(tes_spkr_lists):
        tes_list = load_spkr('Data/train/{}'.format(tes_spkr))
        spkr_s = tes_list[np.random.randint(0,len(tes_list))]
        if use_only_seen_noises:            
            noise_idx = noise_idx_list[use_perms[i%len(use_perms)]]
        else:
            noise_idx = all_perms[i%len(all_perms)]
        used_noises.append(noise_idx)
        spkr_n, spkr_x = get_and_add_noise(noise_frqs[noise_idx], spkr_s, seed)
        tes.append(spkr_s)
        ten.append(spkr_n)
        tex.append(spkr_x)
    tes = np.concatenate(tes).ravel()
    ten = np.concatenate(ten).ravel()
    tex = np.concatenate(tex).ravel()
    
    print ("Test Noise Index: {}".format(used_noises))
    return tes, ten, tex

def setup_experiment_data(args):
    noise_frqs = load_noises('Data/Duan')
        
    # Load trainset
    trs_spkr_lists = []
    for i in range(1,args.n_dr+1):
        trs_spkr_lists += get_random_dr_speakers(i, args.n_spkr, args.seed)
    random.seed(args.seed)
    random.shuffle(trs_spkr_lists)
    print ("Train Speakers: {}".format(trs_spkr_lists))
    trs, trn, trx = load_trainset(trs_spkr_lists, args.noise_idx, noise_frqs, args.seed)
    
    # Load testset
    tes_spkr_lists = []
    for i in range(args.n_test_spkrs):
        spkr = get_random_tes(trs_spkr_lists, args.seed)
        tes_spkr_lists.append(spkr)
        trs_spkr_lists.append(spkr)
    random.seed(args.seed)
    random.shuffle(tes_spkr_lists)
    print ("Test Speakers: {}".format(tes_spkr_lists))
    tes, ten, tex = load_testset(tes_spkr_lists, args.noise_idx, args.use_only_seen_noises, noise_frqs, args.seed)

    # Normalize
    max_amp = trx.max()
    trs, trn, trx, tes, ten, tex = normalize_frqs(max_amp, [trs, trn, trx, tes, ten, tex])

    # STFT
    trS, trN, trX, teS, teN, teX = stft_transform([trs, trn, trx, tes, ten, tex])
    trS_mag, trN_mag, trX_mag, teS_mag, teN_mag, teX_mag = get_magnitudes([trS, trN, trX, teS, teN, teX])
    IBM = (trS_mag > trN_mag)*1
    te_IRM = get_IRM(teS_mag, teN_mag)
    
    # Mel spectrogram
    trX_mag_mel, teX_mag_mel = get_mels([trX_mag, teX_mag])

    data = {'tes': tes, 'teX': teX, 'te_IRM': te_IRM, 
            'trX_mag': trX_mag, 'teX_mag': teX_mag, 'IBM': IBM,
            'trX_mag_mel': trX_mag_mel, 'teX_mag_mel': teX_mag_mel}
    
    return data

def setup_debug_data(args):
    noise_frqs = load_noises('Data/Duan')
        
    if args.seed == 0:
        trs_spkr_lists = ['dr6/flag0', 'dr8/fbcg1', 'dr3/mtlb0', 'dr1/mrcg0', 'dr5/mhmg0', 'dr2/fpjf0', 
                          'dr2/mrab0', 'dr6/msds0', 'dr7/mrpc1', 'dr4/mbma0', 'dr5/ftbw0', 'dr3/fntb0', 
                          'dr1/ftbr0', 'dr4/fdkn0', 'dr8/mbsb0', 'dr7/fmkc0']
        tes_spkr_lists = ['dr5/flja0', 'dr2/mmaa0', 'dr4/mlsh0', 'dr5/mwch0', 'dr6/majp0', 'dr4/msrg0', 
                          'dr3/mtpp0', 'dr6/fsbk0', 'dr6/fjdm2', 'dr4/mmbs0']
    elif args.seed == 1:
        trs_spkr_lists = ['dr2/faem0', 'dr6/fklc1', 'dr1/fvfb0', 'dr8/fpls0', 'dr4/fssb0', 'dr3/mrjb1', 
                          'dr2/mjma0', 'dr5/fsag0', 'dr4/mpeb0', 'dr6/mesj0', 'dr8/mmea0', 'dr1/mdac0', 
                          'dr7/fpab1', 'dr7/mdcm0', 'dr5/mrvg0', 'dr3/fdfb0']
        tes_spkr_lists = ['dr4/fcag0', 'dr4/fjwb1', 'dr4/mcss0', 'dr2/fajw0', 'dr2/fdxw0', 'dr2/fskl0', 
                          'dr4/mmbs0', 'dr4/flkm0', 'dr2/marc0', 'dr4/mfwk0']
    elif args.seed == 2:
        trs_spkr_lists = ['dr7/fjen0', 'dr6/mkes0', 'dr4/mjrh0', 'dr7/mbar0', 'dr4/flhd0', 'dr1/fdaw0', 
                          'dr5/fsag0', 'dr2/mrjh0', 'dr5/mewm0', 'dr3/fcmg0', 'dr6/fapb0', 'dr2/faem0', 
                          'dr3/mgaf0', 'dr8/fmbg0', 'dr8/mtcs0', 'dr1/mrws0']
        tes_spkr_lists = ['dr5/mtmt0', 'dr1/ftbr0', 'dr7/mrpc1', 'dr1/mjwt0', 'dr7/fpab1', 'dr1/mtjs0', 
                          'dr5/mjrg0', 'dr5/mjxa0', 'dr6/msds0', 'dr8/mejs0']
    
    # Load trainset    
    print ("Train Speakers: {}".format(trs_spkr_lists))
    trs, trn, trx = load_trainset(trs_spkr_lists, args.noise_idx, noise_frqs, args.seed)
    
    # Load testset
    print ("Test Speakers: {}".format(tes_spkr_lists))
    tes, ten, tex = load_testset(tes_spkr_lists, args.noise_idx, args.use_only_seen_noises, noise_frqs, args.seed)

    # Normalize
    max_amp = trx.max()
    trs, trn, trx, tes, ten, tex = normalize_frqs(max_amp, [trs, trn, trx, tes, ten, tex])

    # STFT
    trS, trN, trX, teS, teN, teX = stft_transform([trs, trn, trx, tes, ten, tex])
    trS_mag, trN_mag, trX_mag, teS_mag, teN_mag, teX_mag = get_magnitudes([trS, trN, trX, teS, teN, teX])
    IBM = (trS_mag > trN_mag)*1
    te_IRM = get_IRM(teS_mag, teN_mag)
    
    # Mel spectrogram
    trX_mag_mel, teX_mag_mel = get_mels([trX_mag, teX_mag])

    data = {'tes': tes, 'teX': teX, 'te_IRM': te_IRM, 
            'trX_mag': trX_mag, 'teX_mag': teX_mag, 'IBM': IBM,
            'trX_mag_mel': trX_mag_mel, 'teX_mag_mel': teX_mag_mel}
    
    return data
