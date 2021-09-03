import tensorflow as tf
import math
class feed_forward:
    def __init__(self,hidden_nums,output_nums):
        self.w1=tf.Variable(tf.random.truncated_normal([output_nums,hidden_nums],stddev=tf.math.sqrt(2.0/(hidden_nums+output_nums))))
        self.b1=tf.Variable(tf.random.truncated_normal([hidden_nums]))

        self.w2=tf.Variable(tf.random.truncated_normal([hidden_nums,output_nums],stddev=tf.math.sqrt(2.0/(hidden_nums+output_nums))))
        self.b2=tf.Variable(tf.zeros([output_nums]))

    def __call__(self,z_output_self_attentiont):
        z_output=tf.matmul(z_output_self_attentiont,self.w1)+self.b1
        z_output=self.gelu(z_output)
        z_output=tf.matmul(z_output,self.w2)+self.b2
        return z_output

    def get_params(self):
        params=[]

        params.append(self.w1)
        params.append(self.b1)
        params.append(self.w2)
        params.append(self.b2)

        return params
    def gelu(self,input_tensor):
        #cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / tf.math.sqrt(2.0)))

        cdf2 = 0.5 * (1.0 + tf.nn.tanh((input_tensor+0.044715*tf.math.pow(input_tensor,3))*tf.math.sqrt(2.0/math.pi)))
        
        return input_tensor*cdf2
