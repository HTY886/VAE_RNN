import tensorflow as tf

def batch_to_time_major(inputs,split_size):
    inputs = tf.split(inputs,  num_or_size_splits=split_size ,axis=1)
    inputs = [tf.squeeze(e,axis=1) for e in inputs]
    return inputs
    
    
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)