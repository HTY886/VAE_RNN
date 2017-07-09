import tensorflow as tf

def batch_to_time_major(inputs,split_size):
    inputs = tf.split(inputs,  num_or_size_splits=split_size ,axis=1)
    inputs = [tf.squeeze(e,axis=1) for e in inputs]
    return inputs