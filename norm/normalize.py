import tensorflow as tf
def layer_norm(z,axis=-1,eps=1e-9):
    
    mean=tf.reduce_mean(z,axis=axis,keepdims=True)
    std=tf.math.reduce_std(z,axis=axis,keepdims=True)
    
    return ((z-mean)/tf.math.sqrt(std**2+eps))


def batch_norm(inputs,axis,eps=1e-9):
    #print(inputs.shape,"inputsinputs")
    mean=tf.reduce_mean(inputs,axis=axis,keepdims=True)
    #print(mean,"meanmean")
    std=tf.math.reduce_std(inputs,axis=axis,keepdims=True)
    #print(std,"stdstd")
    #print((inputs-mean)/tf.math.sqrt(std**2+eps),"(inputs-mean)/tf.math.sqrt(std**2+eps)")
    return (inputs-mean)/tf.math.sqrt(std**2+eps)
