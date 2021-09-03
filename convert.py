import tensorflow as tf
import numpy as np
import math
import sys
import glob
import os
import time
#np.set_printoptions(threshold = 1e6)
from SpeechSynthesizer import speech_synthesizer
from PhoneClassifer import phone_classifer
from utils.load_data import get_data_ss,default_config,phns,denormalize_db,db2amp,spec2wav,inv_preemphasis,save_wave,get_data
from LOAD_SAVE_PARAMS.LOAD_SAVE_PARAMS import save_weight,load_weight

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()

config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
tf.compat.v1.keras.backend.set_session(session)
tf.compat.v1.keras.backend.clear_session()



os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

def convert(model, x_mfccs, y_spec):
    pred_spec,ppgs = model.predict(x_mfccs,dropout_rate=0.0)
    #print(model.loss_ss(me_,y_mel,pred_spec,y_spec),"losssss")
    y_spec=tf.reshape(y_spec,[y_spec.shape[1],y_spec.shape[-1]]).numpy()
    pred_spec=tf.reshape(pred_spec,[pred_spec.shape[1],pred_spec.shape[-1]]).numpy()
    # Denormalizatoin
    pred_spec = denormalize_db(pred_spec, default_config['max_db'], default_config['min_db'])
    y_spec = denormalize_db(y_spec, default_config['max_db'], default_config['min_db'])

    # Db to amp
    pred_spec = db2amp(pred_spec)
    y_spec = db2amp(y_spec)

    # Emphasize the magnitude
    pred_spec = np.power(pred_spec, default_config['emphasis_magnitude'])
    y_spec = np.power(y_spec, default_config['emphasis_magnitude'])
    # Spectrogram to waveform
    audio = spec2wav(pred_spec.T, default_config['n_fft'], default_config['win_length'], default_config['hop_length'],default_config['n_iter'])
                                               default_config['n_iter']), y_spec))
    y_audio = spec2wav(y_spec.T, default_config['n_fft'], default_config['win_length'], default_config['hop_length'],default_config['n_iter'])
    
    # Apply inverse pre-emphasis
    audio = inv_preemphasis(audio, coeff=default_config['preemphasis'])
    y_audio = inv_preemphasis(y_audio, coeff=default_config['preemphasis'])

    return audio, y_audio, ppgs

#tf.summary.audio('A', y_audio, hp.default.sr, max_outputs=hp.convert.batch_size)
#heatmap = np.expand_dims(ppgs, 3)  # channel=1
#tf.summary.image('PPG', heatmap, max_outputs=ppgs.shape[0])
def do_convert(model,params,batch_size):
    acc=[]
    acc_spec=[]
    wav_files_list = glob.glob(default_config["data_convert"])
    #print(wav_files_list)
    iter_data=get_data_ss(wav_files_list,batch_size)
    outputs=[]
    Ys=[]
    los=[]
    count=0
    for x_mfccs,y_spec,y_mel in iter_data:
        e_start=time.time()
        
        X_mfccs=tf.concat(x_mfccs,0)
        
        Y_spec=tf.concat(y_spec,0)
        
        Y_mel=tf.concat(y_mel,0)
        
        X_mfccs=tf.cast(X_mfccs,dtype=tf.float32)
        
        Y_spec=tf.cast(Y_spec,dtype=tf.float32)
  
        audio, y_audio, ppgs = convert(model, X_mfccs,Y_spec)    

        path="save_data"
        
        y_audio=y_audio*32767
        audio=audio*32767
        save_wave(path,"y_audio"+str(count)+".wav",y_audio,16000)
        save_wave(path,"audio"+str(count)+".wav",audio,16000)
        
        e_end=time.time()
        print("一次耗时%f秒"%(e_end-e_start))
        count=count+1
        
    
def return_accuracy(Y,Y_pre):
    num=Y.shape[0]*Y.shape[1]
    
    rowMaxSoft=np.argmax(Y_pre, axis=-1)+1
    rowMax=np.argmax(Y, axis=-1)+1
    
    rowMaxSoft=rowMaxSoft.reshape([1,-1])
    rowMax=rowMax.reshape([1,-1])
    
    nonO=rowMaxSoft-rowMax
   
    exist = (nonO != 0) * 1.0
    factor = np.ones([nonO.shape[1],1])
    res = np.dot(exist, factor)
   
    accuracy=(float(num)-res[0][0])/float(num)
    
    return accuracy
    
if __name__ == "__main__":
    
    batch_size=1

    n_mel=default_config["n_mels"]

    n_spec=(default_config["n_fft"]//2)+1

    input_nums=default_config["n_mfcc"]

    num_hiddens=512#default_config["hidden_units"]#768#768电脑带不动

    num_outputs=default_config["n_mfcc"]

    layer_nums=12

    multi_head=12

    mel_layers=6

    spec_layers=6
    
    max_position_dim=1024

    clip_norm=1.0

    epochs=3000

    isContinue=True
    
    #lr=1e-5#5e-6
    model_pc=phone_classifer(lr=1e-5,input_nums=input_nums,hidden_nums=num_hiddens,output_nums=num_outputs,max_position_dim=max_position_dim,layers_encoder=layer_nums,labels_num=len(phns),multi_head=multi_head)

    params_pc=model_pc.get_params()+model_pc.get_params_vc()

    model_ss=speech_synthesizer(lr=1e-5,mel_in_nums=len(phns),mel_hi_nums=num_hiddens,mel_out_nums=len(phns),mel_layers=mel_layers,n_mel=n_mel,spec_in_nums=n_mel,spec_hi_nums=num_hiddens,spec_out_nums=n_mel,spec_layers=spec_layers,n_spec=n_spec,max_position_dim=max_position_dim,multi_head=multi_head)

    
    
    params_mel=model_ss.get_params_mel()
    params_spec=model_ss.get_params_spec()

    params_ss=params_mel+params_spec

    try:
        load_weight("ckp","params_pc",params_pc)
    except:
        raise("未发现已经训练好的音素分类模型数据")

    model_ss.init_pc(model_pc)#加载pc模型

    if isContinue==True:
        try:
            load_weight("ckp","params_ss",params_ss)
        except:
            raise("未发现已经训练好的语音合成模型数据")


    
    for i in range(epochs):
        with tf.device('/gpu:0'):
            do_convert(model_ss,params_ss,batch_size)
            #save_weight("ckp","params_ss",params_ss)
