import tensorflow as tf
'''
def layer_norm(z,axis=-1):
    ax=[]
    ax2=[z.shape[0]]
    for i in range(len(z.shape)):
        if i is not 0:
            ax.append(i)
            ax2.append(1)
    mean=tf.reduce_mean(z,ax)#这个是错的既不是BN也不是LN。LN应该是再最后一个维度，就是词向量维度进行归一化，就是对单一个的词向量[1,2,3]进行求均值之类的。而BN应该是在两个Batch的每一个样本的相同维度上进行归一化，如[[[1,2,3]][[4,5,6]]],shape=[2,2,3],第一个归一化的是1和4（相同维度）
    mean=tf.reshape(mean,ax2)#这里错在对单单一个数据的所有词向量的相同维度做了归一化
    variance=tf.reduce_mean(tf.math.square(z-mean),ax)
    variance=tf.reshape(variance,ax2)
    output=(z-mean)/(tf.math.sqrt(variance+1e-5))
    return output
'''

def layer_norm(z,axis=-1,eps=1e-9):
    
    mean=tf.reduce_mean(z,axis=axis,keepdims=True)
    std=tf.math.reduce_std(z,axis=axis,keepdims=True)
    
    return ((z-mean)/tf.math.sqrt(std**2+eps))

def add(x,y):
    return x+y

'''
def layer_norm_true(z):
    ax=[]
    ax2=[z.shape[0]]
    for i in range(len(z.shape)):
        if i is not 0:
            ax.append(i)
            ax2.append(1)
    mean=tf.reduce_mean(z,ax)#这个是错的既不是BN也不是LN。LN应该是再最后一个维度，就是词向量维度进行归一化，就是对单一个的词向量[1,2,3]进行求均值之类的。而BN应该是在两个Batch的每一个样本的相同维度上进行归一化，如[[[1,2,3]][[4,5,6]]],shape=[2,2,3],第一个归一化的是1和4（相同维度）
    mean=tf.reshape(mean,ax2)#这里错在对单单一个数据的所有词向量的相同维度做了归一化
    variance=tf.reduce_mean(tf.math.square(z-mean),ax)
    variance=tf.reshape(variance,ax2)
    output=(z-mean)/(tf.math.sqrt(variance+1e-5))
    return output
'''
