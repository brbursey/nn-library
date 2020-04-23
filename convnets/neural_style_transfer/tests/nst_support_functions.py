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


def compute_style_cost(model, style_input, generated_input, style_layers, coefs, name='VGG19'):
    style_preprocessed = tf.keras.applications.vgg19.preprocess_input(style_input)
    generated_preprocessed = tf.keras.applications.vgg19.preprocess_input(generated_input)

    style_layer_outputs = [model.get_layer(name).output for name in style_layers]

    style = tf.keras.Model(inputs=[model.input], outputs=[style_layer_outputs], name=name)

    style_outputs = style(style_preprocessed)
    generated_outputs = style(generated_preprocessed)

    style_cost = 0

    for i in range(0, len(style_outputs)):
        a_S = style_outputs[i]
        a_G = generated_outputs[i]

        layer_cost = compute_style_cost_layer(a_S, a_G)
        style_cost += (coefs[i] * layer_cost)

    return style_cost