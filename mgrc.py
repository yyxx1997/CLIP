import torch
import tensorflow as tf

# default: K=20 and eta = 0.6


def mgmc_sampling(src_embedding, trg_embedding, K, eta):
    batch_size = tf.shape(src_embedding)[0]

    def get_samples(x_vector, y_vector):
        bias_vector = y_vector - x_vector
        W_r = (tf.abs(bias_vector) - tf.reduce_min(tf.abs(bias_vector), axis=1, keepdims=True)) / \
            (tf.reduce_max(tf.abs(bias_vector), 1, keepdims=True) -
             tf.reduce_min(tf.abs(bias_vector), 1, keepdims=True))
        # initializing the set of samples
        R = []
        omega = tf.random.normal(tf.shape(bias_vector), 0, W_r)
        sample = x_vector + tf.multiply(omega, bias_vector)
        R.append(sample)
        for i in range(1, K):
            chain = [tf.expand_dims(item, axis=1) for item in R[:i]]
            average_omega = tf.reduce_mean(tf.concat(chain, axis=1), axis=1)
            omega = eta * tf.random.normal(tf.shape(bias_vector), 0, W_r) + (
                1.0 - eta) * tf.random.normal(tf.shape(bias_vector), average_omega, 1.0)
            sample = x_vector + tf.multiply(omega, bias_vector) 
            R.append(sample)
        return R
    x_sample = get_samples(src_embedding, trg_embedding)
    y_sample = get_samples(trg_embedding, src_embedding)
    return x_sample + y_sample


if __name__ == "__main__":
    src = tf.random.uniform([2, 3], 0.001, 0.95, dtype=tf.float32)
    tgt = tf.random.uniform([2, 3], 0.001, 0.95, dtype=tf.float32)
    results = mgmc_sampling(src, tgt, K=20, eta=0.6)
    ...
