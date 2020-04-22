import tensorflow as tf

def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.reshape(a_C, shape=[m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, n_H * n_W, n_C])

    content_cost = 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled))

    return content_cost

def gram_matrix(A):
    return tf.matmul(A, tf.transpose(A))


def compute_style_cost_layer(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S_unrolled = tf.reshape(tf.transpose(a_S, perm=[0, 3, 1, 2]), shape=[n_C, n_H * n_W])
    a_G_unrolled = tf.reshape(tf.transpose(a_G, perm=[0, 3, 1, 2]), shape=[n_C, n_H * n_W])

    GS = gram_matrix(a_S_unrolled)
    GG = gram_matrix(a_G_unrolled)

    layer_style_cost = 1 / (4 * (n_C ** 2) * (n_H * n_W) ** 2) * tf.reduce_sum(tf.square(GS - GG))
    return layer_style_cost