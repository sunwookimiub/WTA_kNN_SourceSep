import os
import random
import librosa
import numpy as np

def load_spkr(spkr_dir):
    spkr_files = [x for x in os.listdir(spkr_dir) if 'wav' in x]
    spkr_frqs = [librosa.load('{}/{}'.format(spkr_dir, x), sr=16000)[0] for x in spkr_files]
    spkr_frqs = [frqs/frqs.std() for frqs in spkr_frqs]
    return spkr_frqs

def load_noises(noise_dir):
    noise_files = os.listdir(noise_dir)
    print (noise_files)
    noise_frqs = [librosa.load('{}/{}'.format(noise_dir, x), sr=16000)[0] for x in noise_files]
    noise_frqs = [frqs/frqs.std() for frqs in noise_frqs]
    return noise_frqs

def get_and_add_noise(noise, source, test=False):
    if test:
        noise_start = 0
    else:
        noise_start = np.random.randint(0,len(noise) - len(source))
    spkr_noise = noise[noise_start:noise_start + len(source)]
    return spkr_noise, spkr_noise + source

def load_more_spkr_with_noise(spkr_dir, noise):
    spkr_frqs = load_spkr(spkr_dir)
    spkr_s = np.concatenate(spkr_frqs, 0)
    spkr_n, spkr_x = get_and_add_noise(noise, spkr_s)
    return spkr_s, spkr_n, spkr_x

def get_random_dr_f_speakers(dr_idx, num_speakers):    
    all_spkrs = ['dr{}/{}'.format(dr_idx, name) for name in 
                    os.listdir('Data/train/dr{}'.format(dr_idx))if 'Store' not in name]
    f_spkrs = [spkr for spkr in all_spkrs if spkr.split('/')[1][0] == 'f']
    return random.sample(f_spkrs, num_speakers)

def get_random_dr_m_speakers(dr_idx, num_speakers):    
    all_spkrs = ['dr{}/{}'.format(dr_idx, name) for name in 
                    os.listdir('Data/train/dr{}'.format(dr_idx))if 'Store' not in name]
    m_spkrs = [spkr for spkr in all_spkrs if spkr.split('/')[1][0] == 'm']
    return random.sample(m_spkrs, num_speakers)

def get_random_dr_speakers(dr_idx, num_speakers):
    num_f = num_m = num_speakers//2
    if num_speakers % 2 != 0:
        num_f += 1
    f_spkrs = get_random_dr_f_speakers(dr_idx, num_f)
    m_spkrs = get_random_dr_m_speakers(dr_idx, num_m)
    fm_spkrs = f_spkrs + m_spkrs
    return fm_spkrs

def get_random_tes(speaker_lists):
    dr_idx = np.random.randint(1,9)
    all_spkrs = ['dr{}/{}'.format(dr_idx, name) for name in 
                os.listdir('Data/train/dr{}'.format(dr_idx))
                if 'Store' not in name 
                and 'dr{}/{}'.format(dr_idx,name) not in speaker_lists]
    return random.choice(all_spkrs)

def normalize_frqs(max_amp, frqs):
    return [frq/max_amp for frq in frqs]

def stft_transform(frqs):
    return [librosa.stft(frq, n_fft=1024, hop_length=512) for frq in frqs]

def get_magnitudes(stfts):
    return [np.abs(stft) for stft in stfts]

def load_trainset(trs_spkr_lists, noise_idx_list, noise_frqs):
    trs, trn, trx = [], [], []
    for i, trs_spkr in enumerate(trs_spkr_lists):
        for noise_idx in noise_idx_list:
            s, n, x = load_more_spkr_with_noise('Data/train/{}'.format(trs_spkr), noise_frqs[noise_idx]) 
            trs.append(s)
            trn.append(n)
            trx.append(x)
    trs = np.concatenate(trs).ravel()
    trn = np.concatenate(trn).ravel()
    trx = np.concatenate(trx).ravel()
    return trs, trn, trx

def load_testset(trs_spkr_lists, noise_idx_list, use_only_seen_noises, noise_frqs):
    tes_spkr = get_random_tes(trs_spkr_lists)
    tes_list = load_spkr('Data/train/{}'.format(tes_spkr))
    tes = tes_list[np.random.randint(0,len(tes_list))]
    if use_only_seen_noises:
        noise_idx = np.random.choice(noise_idx_list)
    else:
        noise_idx = np.random.randint(0,len(noise_frqs)) 
    print ("Test Speaker: {} Noise Index: {}".format(tes_spkr, noise_idx))
    _, tex = get_and_add_noise(noise_frqs[noise_idx], tes)
    return tes, tex

def setup_experiment_data(num_dr, per_dr_spkr_num, noise_idx_list, use_only_seen_noises):
    noise_frqs = load_noises('Data/Duan')
    
    # Load trainset
    trs_spkr_lists = []
    for i in range(1,num_dr+1):
        trs_spkr_lists += get_random_dr_speakers(i, per_dr_spkr_num)
    print ("Train Speakers: {}".format(trs_spkr_lists))
    random.shuffle(trs_spkr_lists)
    trs, trn, trx = load_trainset(trs_spkr_lists, noise_idx_list, noise_frqs)

    # Load testset
    tes, tex = load_testset(trs_spkr_lists, noise_idx_list, use_only_seen_noises, noise_frqs)

    # Normalize
    max_amp = trx.max()
    trs, trn, trx, tes, tex = normalize_frqs(max_amp, [trs, trn, trx, tes, tex])

    # STFT
    trS, trN, trX, teX = stft_transform([trs, trn, trx, tex])
    trS_mag, trN_mag, trX_mag, teX_mag = get_magnitudes([trS, trN, trX, teX])
    IBM = (trS_mag > trN_mag)*1

    data = {'tes': tes, 'teX': teX, 'trX_mag': trX_mag, 'teX_mag': teX_mag, 'IBM': IBM}
    return data