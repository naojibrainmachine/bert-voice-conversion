import tensorflow as tf
import librosa
from scipy import signal
import numpy as np
import random
import math
from scipy.io import wavfile
np.set_printoptions(threshold = 1e6)
#由于win_length=1024,n_fft的窗口为2048，而训练的窗口为401，随意改变win_length和n_fft会造成模型准确率不一致
#import random

np.random.seed(1)
default_config={"sr":16000,"win_length": 1024,"hop_length": 80,"duration": 2,"n_fft": 1024,"n_mels": 80,\
                "n_mfcc": 40,"max_db": 35,"min_db": -55,"preemphasis": 0.97,"n_iter":60,\
                "data_path_phcls":"dataset/English_data/lisa/data/timit/raw/TIMIT/TRAIN/*/*/*.wav",\
                "data_path_phcls_test":"dataset/English_data/lisa/data/timit/raw/TIMIT/TEST/*/*/*.wav",\
                "data_ss_train":"dataset/English_data/arctic/slt/*.wav",\
                "data_ss_test":"dataset/English_data/arctic/slt/*.wav",\
                "data_convert":"dataset/English_data/arctic/bdl/*.wav",\
                "data_convert_save":"datasets/English_data/logdir",\
                "ss_pre_train":"dataset/LibriSpeech/*/*/*/*.wav",\
                "num_banks":8,"hidden_units":256,"num_highway_blocks":4,\
                "emphasis_magnitude": 1.2}
phns = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
        'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
        'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
        'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
        'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
def re_sample(data,sr):
    #y, sr = librosa.load(file_name, sr=None)
    data_16k = librosa.resample(data.astype(np.float32), sr, default_config["sr"])
    #librosa.output.write_wav("./p225_004_2_L_2.wav", y_16k, default_config["sr"],norm=False)#保存之前做32767的归一化
    return data_16k
def save_wave(path,name,data,sr,norm=False):
    path_name=path+'/'+name
    data = np.clip(data, -32767, 32767)
    #librosa.output.write_wav(path_name, data, sr,norm=norm)
    wavfile.write(path_name, sr, data.astype(np.int16))
def get_mfccs_and_spectrogram(wav_file, trim=True, random_crop=False,bReSample=False):
    '''This is applied in `train2`, `test2` or `convert` phase.
    '''


    # Load
    wav, sr_ = librosa.load(wav_file, sr=default_config["sr"])

    #resample
    if bReSample:
        wav=re_sample(wav,sr_)
    
    # Trim
    if trim:
        wav, _ = librosa.effects.trim(wav, frame_length=default_config["win_length"], hop_length=default_config["hop_length"])#去除首部和尾部静音

    if random_crop:
        wav = wav_random_crop(wav, default_config["sr"], default_config["duration"])
    #print(wav.shape)
    # Padding or crop
    length = default_config["sr"] * default_config["duration"]#default_config["sr"]是采样率
    #print(length,"length")
    wav = librosa.util.fix_length(wav, length)#把wav转化为length长度，剪切或者补零

    return _get_mfcc_and_spec(wav, default_config["preemphasis"], default_config["n_fft"], default_config["win_length"], default_config["hop_length"])


# TODO refactoring
def _get_mfcc_and_spec(wav, preemphasis_coeff, n_fft, win_length, hop_length):

    # Pre-emphasis
    y_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Get spectrogram
    D = librosa.stft(y=y_preem, n_fft=n_fft, hop_length=hop_length, win_length=win_length)#短时傅里叶变换，里面包括加窗、分帧、快速傅里叶变换
    mag = np.abs(D)#取绝对值
    #print(mag.shape,"mag")
    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(default_config["sr"], default_config["n_fft"], default_config["n_mels"])  # (n_mels, 1+n_fft//2)# 梅尔滤波器
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram 梅尔频谱图

    # Get mfccs, amp to db
    mag_db = amp2db(mag)#取对数，得到分贝
    mel_db = amp2db(mel)
    #mfccs = np.dot(librosa.filters.dct(default_config["n_mfcc"], mel_db.shape[0]), mel_db)#离散余弦变换，改变数据分布，将冗余数据分开（梅尔频率倒谱系数）
    mfccs=librosa.feature.mfcc(y=np.asarray(list(range(mel_db.shape[1]))),sr=default_config["sr"],n_mfcc=default_config["n_mfcc"],norm=None,S=mel_db)
    
    # Normalization (0 ~ 1)
    mag_db = normalize_0_1(mag_db, default_config["max_db"], default_config["min_db"])#归一化
    mel_db = normalize_0_1(mel_db, default_config["max_db"], default_config["min_db"])
    #print(mfccs.shape,"mfccs")
    #print(mag_db.shape,"mag_db")
    #print(mel_db.shape,"mel_db")
    return mfccs.T, mag_db.T, mel_db.T  # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)


def amp2db(amp):
    return librosa.amplitude_to_db(amp)#相当于power_to_db(S**2)


def _get_mfcc_and_spec_2(wav, preemphasis_coeff, n_fft, win_length, hop_length):#替代

    # Pre-emphasis
    y_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Get spectrogram
    D = librosa.stft(y=y_preem, n_fft=n_fft, hop_length=hop_length, win_length=win_length)#短时傅里叶变换，里面包括加窗、分帧、快速傅里叶变换
    mag = np.abs(D)#取绝对值

    mag=mag**2#开方后得到能量谱图

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(default_config["sr"], default_config["n_fft"], default_config["n_mels"])  # (n_mels, 1+n_fft//2)# 梅尔滤波器
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram

    # Get mfccs, amp to db
    mag_db = librosa.core.power_to_db(mag)#取对数，得到分贝，为了模拟人耳的“对数式”特性（声音响度已经大到一定程度后，即使再有较大的增加，人耳的感觉却无明显变化）
    mel_db = librosa.core.power_to_db(mel)
    #mfccs = np.dot(librosa.filters.dct(default_config["n_mfcc"], mel_db.shape[0]), mel_db)#离散余弦变换，改变数据分布，将冗余数据分开（梅尔频率倒谱系数）
    mfccs=librosa.feature.mfcc(y=np.asarray(list(range(mel_db.shape[1]))),sr=default_config["sr"],n_mfcc=default_config["n_mfcc"],norm=None,S=mel_db)
    
    
    # Normalization (0 ~ 1)
    mag_db = normalize_0_1(mag_db, default_config["max_db"], default_config["min_db"])#归一化
    mel_db = normalize_0_1(mel_db, default_config["max_db"], default_config["min_db"])

    return mfccs.T, mag_db.T, mel_db.T  # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)

def wav_random_crop(wav, sr, duration):#随机取一定长度
    assert (wav.ndim <= 2)

    target_len = sr * duration
    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - target_len)), 1)[0]
    #print(start,"start")
    end = start + target_len
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav


def preemphasis(wav, coeff=0.97):
    """
    Emphasize high frequency range of the waveform by increasing power(squared amplitude).

    Parameters
    ----------
    wav : np.ndarray [shape=(n,)]
        Real-valued the waveform.

    coeff: float <= 1 [scalar]
        Coefficient of pre-emphasis.

    Returns
    -------
    preem_wav : np.ndarray [shape=(n,)]
        The pre-emphasized waveform.
    """
    preem_wav = signal.lfilter([1, -coeff], [1], wav)
    return preem_wav

def normalize_0_1(values, max, min):
    normalized = np.clip((values - min) / (max - min), 0, 1)
    return normalized

def get_mfccs_and_phones(wav_file, trim=False, random_crop=True):

    '''This is applied in `train1` or `test1` phase.
    '''
    #print(wav_file)
    # Load
    wav = read_wav(wav_file, sr=default_config["sr"])

    mfccs, _, _ = _get_mfcc_and_spec(wav, default_config["preemphasis"],default_config["n_fft"],
                                     default_config["win_length"],
                                     default_config["hop_length"])

    # timesteps
    num_timesteps = mfccs.shape[0]
    #print(num_timesteps,"size")
    # phones (targets)
    phn_file = wav_file.replace("WAV", "PHN")#.replace("WAV.wav", "PHN").replace("wav", "PHN")
    phn2idx, idx2phn = load_vocab()
    phns = np.zeros(shape=(num_timesteps,))
    bnd_list = []
    #print(phn_file)#,encoding ='utf-8'
    for line in open(phn_file, mode='r').read().splitlines():
        start_point, _, phn = line.split()
        bnd = int(start_point) // default_config["hop_length"]
        phns[bnd:] = phn2idx[phn]#应该是把开始位置到结束位置都设为开始位置的idx，第二行开始位置会下移动，这样就分开每个音素自己对应idx
        bnd_list.append(bnd)

    # Trim
    if trim:#跳过静音
        start, end = bnd_list[1], bnd_list[-1]
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Random crop
    n_timesteps = (default_config["duration"] * default_config["sr"]) // default_config["hop_length"] + 1 #401
    if random_crop:
        start = np.random.choice(range(np.maximum(1, len(mfccs) - n_timesteps)), 1)[0]#随机截断数据
        end = start + n_timesteps
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Padding or crop
    mfccs = librosa.util.fix_length(mfccs, n_timesteps, axis=0)#对齐
    phns = librosa.util.fix_length(phns, n_timesteps, axis=0)

    return mfccs, phns

def get_mfccs_and_phones_dataAugmentation(wav_file_sample,wav_file_sampled,sample_num=3,trim=False,dataAugmentation=False ,random_crop=True):

    '''This is applied in `train1` or `test1` phase.
    '''
    #print(wav_file)
    # Load
    wav_sample = read_wav(wav_file_sample, sr=default_config["sr"])

    wav_sampled = read_wav(wav_file_sampled, sr=default_config["sr"])

    mfccs_sample, _, _ = _get_mfcc_and_spec(wav_sample, default_config["preemphasis"],default_config["n_fft"],
                                     default_config["win_length"],
                                     default_config["hop_length"])
    
    mfccs_sampled, _, _ = _get_mfcc_and_spec(wav_sampled, default_config["preemphasis"],default_config["n_fft"],
                                     default_config["win_length"],
                                     default_config["hop_length"])

    # timesteps
    num_timesteps_sample = mfccs_sample.shape[0]
    #print(num_timesteps,"size")
    # phones (targets)
    phn_file_sample = wav_file_sample.replace("WAV", "PHN")#.replace("WAV.wav", "PHN").replace("wav", "PHN")
    phn2idx, idx2phn = load_vocab()
    phns_sample = np.zeros(shape=(num_timesteps_sample,))
    bnd_list_sample = []
    #print(phn_file)#,encoding ='utf-8'
    for line in open(phn_file_sample, mode='r').read().splitlines():
        start_point, _, phn = line.split()
        bnd = int(start_point) // default_config["hop_length"]
        phns_sample[bnd:] = phn2idx[phn]#应该是把开始位置到结束位置都设为开始位置的idx，第二行开始位置会下移动，这样就分开每个音素自己对应idx
        bnd_list_sample.append(bnd)


    # timesteps
    num_timesteps_sampled = mfccs_sampled.shape[0]
    #print(num_timesteps,"size")
    # phones (targets)
    phn_file_sampled = wav_file_sampled.replace("WAV", "PHN")#.replace("WAV.wav", "PHN").replace("wav", "PHN")
    phn2idx, idx2phn = load_vocab()
    phns_sampled = np.zeros(shape=(num_timesteps_sampled,))
    bnd_list_sampled = []
    #print(phn_file)#,encoding ='utf-8'
    for line in open(phn_file_sampled, mode='r').read().splitlines():
        start_point, _, phn = line.split()
        bnd = int(start_point) // default_config["hop_length"]
        phns_sampled[bnd:] = phn2idx[phn]#应该是把开始位置到结束位置都设为开始位置的idx，第二行开始位置会下移动，这样就分开每个音素自己对应idx
        bnd_list_sampled.append(bnd)
    

    # Trim
    if trim:#跳过静音
        start, end = bnd_list_sample[1], bnd_list_sample[-1]
        mfccs_sample = mfccs_sample[start:end]
        phns_sample = phns_sample[start:end]
        assert (len(mfccs_sample) == len(phns_sample))

        start, end = bnd_list_sampled[1], bnd_list_sampled[-1]
        mfccs_sampled = mfccs_sampled[start:end]
        phns_sampled = phns_sampled[start:end]
        assert (len(mfccs_sampled) == len(phns_sampled))
    


    count=0
    front=0
    behind=0
    range_phs_sample={}
    
    for i in range(0,phns_sample.shape[0]):
        if(phns_sample[front]!=phns_sample[behind]):
            range_phs_sample[count]=[phns_sample[front],front,behind]
            count=count+1
            front=behind
        
        behind=i

        if (behind+1)==phns_sample.shape[0]:
            range_phs_sample[count]=[phns_sample[front],front,phns_sample.shape[0]]
            count=count+1
            front=phns_sample.shape[0]

    count=0
    front=0
    behind=0
    range_phs_sampled={}
    
    for i in range(0,phns_sampled.shape[0]):
        if(phns_sampled[front]!=phns_sampled[behind]):
            range_phs_sampled[count]=[phns_sampled[front],front,behind]
            count=count+1
            front=behind
        
        behind=i

        if (behind+1)==phns_sampled.shape[0]:
            range_phs_sampled[count]=[phns_sampled[front],front,phns_sampled.shape[0]]
            count=count+1
            front=phns_sampled.shape[0]


    keys_sample=list(range_phs_sample.keys())

    keys_sampled=list(range_phs_sampled.keys())
    #print(range_phs_sample,"range_phs_sample qian")
    #print(range_phs_sampled,"range_phs_sampled qian")
    count=0
    for j in range(len(keys_sample)):
        for i in range(0,len(keys_sampled)):
            if(range_phs_sample[keys_sample[j]][0]==range_phs_sampled[keys_sampled[i]][0]):
                #print(range_phs_sample[keys_sample[j]][0],"keykey")
                front=range_phs_sample[keys_sample[j]][1]
                behind=range_phs_sample[keys_sample[j]][2]
                range_phs_sample[keys_sample[j]][1]=range_phs_sampled[keys_sampled[i]][1]
                range_phs_sample[keys_sample[j]][2]=range_phs_sampled[keys_sampled[i]][2]
                count=count+1
                #range_phs_sampled[keys_sampled[i]][1]=front
                #range_phs_sampled[keys_sampled[i]][2]=behind
    #print(range_phs_sample,"range_phs_sample hou")
    sample_y=[]
    sample_x=[]
    for j in range(len(keys_sample)):
        sample_y.append(phns_sample[range_phs_sample[keys_sample[j]][1]:range_phs_sample[keys_sample[j]][2]])
        sample_x.append(mfccs_sample[range_phs_sample[keys_sample[j]][1]:range_phs_sample[keys_sample[j]][2]][:])

    mfccs=np.concatenate(sample_x,axis=0)
    phns=np.concatenate(sample_y)
    # Random crop
    n_timesteps = (default_config["duration"] * default_config["sr"]) // default_config["hop_length"] + 1 #401
    if random_crop:
        start = np.random.choice(range(np.maximum(1, len(mfccs) - n_timesteps)), 1)[0]#随机截断数据
        end = start + n_timesteps
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Padding or crop
    mfccs = librosa.util.fix_length(mfccs, n_timesteps, axis=0)#对齐
    phns = librosa.util.fix_length(phns, n_timesteps, axis=0)

    return mfccs, phns


def load_vocab():
    phn2idx = {phn: idx for idx, phn in enumerate(phns)}
    idx2phn = {idx: phn for idx, phn in enumerate(phns)}

    return phn2idx, idx2phn

def read_wav(path, sr, duration=None, mono=True):
    wav, _ = librosa.load(path, mono=mono, sr=sr, duration=duration)
    return wav







def denormalize_db(norm_db, max_db, min_db):
    """
    Denormalize the normalized values to be original dB-scaled value.
    :param norm_db: Normalized spectrogram.
    :param max_db: Maximum dB.
    :param min_db: Minimum dB.
    :return: Decibel-scaled spectrogram.
    """
    db = np.clip(norm_db, 0, 1) * (max_db - min_db) + min_db
    return db

def inv_preemphasis(preem_wav, coeff=0.97):
    """
    Invert the pre-emphasized waveform to the original waveform.

    Parameters
    ----------
    preem_wav : np.ndarray [shape=(n,)]
        The pre-emphasized waveform.

    coeff: float <= 1 [scalar]
        Coefficient of pre-emphasis.

    Returns
    -------
    wav : np.ndarray [shape=(n,)]
        Real-valued the waveform.
    """
    wav = signal.lfilter([1], [1, -coeff], preem_wav)
    return wav

def db2amp(db):
    return librosa.db_to_amplitude(db)

def spec2wav(mag, n_fft, win_length, hop_length, num_iters=30, phase=None):
    """
    Get a waveform from the magnitude spectrogram by Griffin-Lim Algorithm.

    Parameters
    ----------
    mag : np.ndarray [shape=(1 + n_fft/2, t)]
        Magnitude spectrogram.

    n_fft : int > 0 [scalar]
        FFT window size.

    win_length  : int <= n_fft [scalar]
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

    hop_length : int > 0 [scalar]
        Number audio of frames between STFT columns.

    num_iters: int > 0 [scalar]
        Number of iterations of Griffin-Lim Algorithm.

    phase : np.ndarray [shape=(1 + n_fft/2, t)]
        Initial phase spectrogram.

    Returns
    -------
    wav : np.ndarray [shape=(n,)]
        The real-valued waveform.

    """
    assert (num_iters > 0)
    if phase is None:
        phase = np.pi * np.random.rand(*mag.shape)
    stft = mag * np.exp(1.j * phase)
    wav = None
    #print(mag.shape,"mag")
    for i in range(num_iters):
        wav = librosa.istft(stft, win_length=win_length, hop_length=hop_length)
        if i != num_iters - 1:
            stft = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            _, phase = librosa.magphase(stft)
            phase = np.angle(phase)
            stft = mag * np.exp(1.j * phase)
    return wav


def get_data_onebyone(wav_files_list,batch_size,data_augment=True):
    count=0
    x=[]
    y=[]
    #print(len(wav_files_list),"wav_files_listwav_files_listwav_files_listwav_files_list")
    random.shuffle(wav_files_list)
    
    for data in wav_files_list:
        #print(data)
        #print(count,"wav_filewav_filewav_file")
        if batch_size>count:
            wav_file_sample = random.choice(wav_files_list)
            rand_num=random.randint(0,1)
            print(rand_num)
            if data_augment and rand_num>0:
                x1,y1=get_mfccs_and_phones_dataAugmentation(data,wav_file_sample)
                #x1,y1=swap(x1,y1)
                #print(x1.shape)
                #swap(x1,y1)
                x.append(tf.expand_dims(x1,0))
                y.append(y1)
            else:
                x1,y1=get_mfccs_and_phones(wav_file=data)
                x.append(tf.expand_dims(x1,0))
                y.append(y1)
                
            count=count+1
                
            
        else:
            
            yield x,y
            count=0
            x=[]
            y=[]

            x1,y1=get_mfccs_and_phones(wav_file=data)
            x.append(tf.expand_dims(x1,0))
            y.append(y1)
                
            count=count+1


def get_data(wav_files_list,batch_size):
    count=0
    x=[]
    y=[]
    #print(len(wav_files_list),"wav_files_listwav_files_listwav_files_listwav_files_list")
    for data in wav_files_list:
        #print(wav_files_list,"wav_filewav_filewav_file")
        if batch_size>count:
            #wav_file_sample = random.choice(wav_files_list)
            #wav_file_sampled = random.choice(wav_files_list)
            #x1,y1=get_mfccs_and_phones(wav_file=wav_file)
            
            x1,y1=get_mfccs_and_phones(wav_file=data)
            x.append(tf.expand_dims(x1,0))
            y.append(y1)
            
            count=count+1
        else:
            #print(wav_files_list,"wav_filewav_filewav_file")
            yield x,y
            count=0
            x=[]
            y=[]

            x1,y1=get_mfccs_and_phones(wav_file=data)
            x.append(tf.expand_dims(x1,0))
            y.append(y1)
                
            count=count+1

def get_data_cls(wav_files_list,batch_size):
    count=0
    x=[]
    y=[]
    #print(len(wav_files_list),"wav_files_listwav_files_listwav_files_listwav_files_list")
    for data in wav_files_list:
        #print(wav_files_list,"wav_filewav_filewav_file")
        if batch_size>count:
            #wav_file_sample = random.choice(wav_files_list)
            #wav_file_sampled = random.choice(wav_files_list)
            #x1,y1=get_mfccs_and_phones(wav_file=wav_file)
            
            x1,y1=get_mfccs_and_phones(wav_file=data)
            x.append(tf.expand_dims(x1,0))
            y.append(y1)
            
            count=count+1
        else:
            #print(wav_files_list,"wav_filewav_filewav_file")
            yield x,y
            count=0
            x=[]
            y=[]

            x1,y1=get_mfccs_and_phones(wav_file=data)
            x.append(tf.expand_dims(x1,0))
            y.append(y1)
                
            count=count+1

def get_data_clip_test(wav_files_list,batch_size,data_augment=True):
    count=0
    x=[]
    y=[]
    #print(len(wav_files_list),"wav_files_listwav_files_listwav_files_listwav_files_list")
    for _ in range(math.floor(len(wav_files_list)/batch_size)):
        #print(count,"wav_filewav_filewav_file")
        if batch_size>count:
            wav_file_sample = random.choice(wav_files_list)
            wav_file_sampled = random.choice(wav_files_list)
            #x1,y1=get_mfccs_and_phones(wav_file=wav_file)
            if data_augment:
                x1,y1=get_mfccs_and_phones_dataAugmentation(wav_file_sample,wav_file_sampled)
                #x1,y1=swap(x1,y1)
                #print(x1.shape)
                #swap(x1,y1)
                x.append(x1)
                y.append(y1)
            else:
                x1,y1=get_mfccs_and_phones(wav_file=wav_file_sample)
                x.append(x1)
                y.append(y1)
                
            count=count+1
        else:
            
            yield x,y
            count=0
            x=[]
            y=[]

def get_data_ss(wav_files_list,batch_size,random_crop=False,bReSample=False):
    count=0
    x=[]
    y=[]
    z=[]
    random.shuffle(wav_files_list)
    #print(len(wav_files_list),"wav_files_listwav_files_listwav_files_listwav_files_list")
    for data in wav_files_list:
        #print(count,"wav_filewav_filewav_file")
        if batch_size>count:
            #wav_file = random.choice(wav_files_list)
            x1,y1,z1=get_mfccs_and_spectrogram(wav_file=data,random_crop=random_crop,bReSample=bReSample)
            x.append(tf.expand_dims(x1,0))
            y.append(tf.expand_dims(y1,0))
            z.append(tf.expand_dims(z1,0))
            count=count+1
        else:
            
            yield x,y,z
            count=0
            x=[]
            y=[]
            z=[]

            x1,y1,z1=get_mfccs_and_spectrogram(wav_file=data,random_crop=random_crop,bReSample=bReSample)
            x.append(tf.expand_dims(x1,0))
            y.append(tf.expand_dims(y1,0))
            z.append(tf.expand_dims(z1,0))
            count=count+1

def get_data_ss_random_choice(wav_files_list,batch_size,random_crop=False,bReSample=False):
    count=0
    x=[]
    y=[]
    z=[]
    random.shuffle(wav_files_list)
    #print(len(wav_files_list),"wav_files_listwav_files_listwav_files_listwav_files_list")
    for _ in range(math.floor(len(wav_files_list)/batch_size)):
        #print(count,"wav_filewav_filewav_file")
        if batch_size>count:
            wav_file = random.choice(wav_files_list)
            x1,y1,z1=get_mfccs_and_spectrogram(wav_file=wav_file,random_crop=random_crop,bReSample=bReSample)
            x.append(tf.expand_dims(x1,0))
            y.append(tf.expand_dims(y1,0))
            z.append(tf.expand_dims(z1,0))
            count=count+1
        else:
            
            yield x,y,z
            count=0
            x=[]
            y=[]
            z=[]
            wav_file = random.choice(wav_files_list)
            x1,y1,z1=get_mfccs_and_spectrogram(wav_file=wav_file,random_crop=random_crop,bReSample=bReSample)
            x.append(tf.expand_dims(x1,0))
            y.append(tf.expand_dims(y1,0))
            z.append(tf.expand_dims(z1,0))
            count=count+1

def swap(x,y):
    count=0
    front=0
    behind=0
    range_phs={}
    #print([i for i in range(0,5)])
    for i in range(0,y.shape[0]):
        if(y[front]!=y[behind]):
            range_phs[count]=[y[front],front,behind]
            count=count+1
            front=behind
        
        behind=i

        if (behind+1)==y.shape[0]:
            range_phs[count]=[y[front],front,y.shape[0]]
            count=count+1
            front=y.shape[0]
        #print(behind)
    #print(range_phs,"前")
    #print(y)
    #print(x[0:40][:].shape)
    keys=list(range_phs.keys())
    
    for j in range(len(keys)):
        for i in range(j+1,len(keys)):
            if(range_phs[keys[j]][0]==range_phs[keys[i]][0]) and j!=i:
                #print(range_phs[keys[j]][0],"keykey")
                front=range_phs[keys[j]][1]
                behind=range_phs[keys[j]][2]
                range_phs[keys[j]][1]=range_phs[keys[i]][1]
                range_phs[keys[j]][2]=range_phs[keys[i]][2]
                range_phs[keys[i]][1]=front
                range_phs[keys[i]][2]=behind
    new_y=[]
    new_x=[]
    for j in range(len(keys)):
        new_y.append(y[range_phs[keys[j]][1]:range_phs[keys[j]][2]])
        new_x.append(x[range_phs[keys[j]][1]:range_phs[keys[j]][2]][:])

    #print(np.concatenate(new_x,axis=0).shape,np.concatenate(new_y).shape)
    return np.concatenate(new_x,axis=0),np.concatenate(new_y)
        #print(range_phs[keys[j]][0]," ",x[range_phs[keys[j]][1]:range_phs[keys[j]][2]][:].shape)
    #print(np.concatenate(new_y))
    #print(np.concatenate(new_x,axis=0))
    #print(range_phs,"后") 
#a,b,c=get_mfccs_and_spectrogram("SA1.WAV")
#d,e=get_mfccs_and_phones("SA1.WAV")
#d1,e1=get_mfccs_and_phones("SA1.WAV")
#print(a.shape)
#print(b.shape)
#print(c.shape)
#print(d.shape)
#print(d1.shape)
#print(e.shape)
