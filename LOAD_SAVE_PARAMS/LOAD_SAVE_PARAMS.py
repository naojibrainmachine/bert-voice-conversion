import zipfile
import os
import numpy as np
import tensorflow as tf

def save_weight(path,name,params):
    #name='\\weight_trained.zip'
    path=path+"\\"+name+".zip"
    with zipfile.ZipFile(path,'w') as zipobj:
        for k in range(len(params)):
            filepath="p"+str(k)+".txt"
            np.savetxt(filepath,(params[k].numpy()).reshape(1,-1))
            zipobj.write(filepath)
            os.remove(filepath) 
    zipobj.close()
    
def load_weight(path,name,params):
    #name='\\weight_trained.zip'
    path=path+"\\"+name+".zip"
    with zipfile.ZipFile(path,'r') as zipobj:
        for k in range(len(params)):
            try:
                filepath="p"+str(k)+".txt"
                with zipobj.open(filepath) as f:
                    params[k].assign((np.loadtxt(f,dtype=np.float32)).reshape(params[k].shape))
            except:
                count=0
                for i in params[k].shape:
                    count=count+i
                params[k].assign(tf.reshape(tf.random.truncated_normal(params[k].shape,stddev=tf.math.sqrt(2.0/(count))),shape=params[k].shape))
    zipobj.close()
