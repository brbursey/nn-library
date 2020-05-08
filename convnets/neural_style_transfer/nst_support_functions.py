import tensorflow as tf
import numpy as np
import PIL.Image

def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.reshape(a_C, shape=[m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, n_H * n_W, n_C])

    content_cost = 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled))

    return content_cost


def gram_matrix(A):
    return tf.matmul(A, tf.transpose(A))


def compute_style_cost_layer(a_S, a_G):
    m, n_H, n_W, n_C = a_G.shape

    a_S_unrolled = tf.reshape(tf.transpose(a_S, perm=[0, 3, 1, 2]), shape=[n_C, n_H * n_W])
    a_G_unrolled = tf.reshape(tf.transpose(a_G, perm=[0, 3, 1, 2]), shape=[n_C, n_H * n_W])

    GS = gram_matrix(a_S_unrolled)
    GG = gram_matrix(a_G_unrolled)

    layer_style_cost = 1 / (4 * (n_C ** 2) * (n_H * n_W) ** 2) * tf.reduce_sum(tf.square(GS - GG))
    return layer_style_cost


def compute_style_cost(model, style_input, generated_input, style_layers, coefs, name='VGG19'):
    style_layer_outputs = [model.get_layer(layer).output for layer in style_layers]

    style = tf.keras.Model(inputs=[model.input], outputs=style_layer_outputs, name=name)

    style_outputs = style(style_input)
    generated_outputs = style(generated_input)

    style_cost = 0

    for i in range(0, len(style_outputs)):
        a_S = style_outputs[i].numpy()
        a_G = generated_outputs[i].numpy()


        layer_cost = compute_style_cost_layer(a_S, a_G)
        style_cost += (coefs[i] * layer_cost)

    return style_cost

def total_cost(content_cost, style_cost, alpha = 10, beta = 10):
    total_cost = (alpha * content_cost) + (beta * style_cost)
    return total_cost


def reshape_and_normalize_image(image):
    """
    Reshape and normalize the input image (content or style)
    """

    # Reshape image to mach expected input of VGG16
    image = np.reshape(image, ((1,) + image.shape))
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

    # Substract the mean to match the expected input of VGG16
    image = image - MEANS

    return image


def vgg_layers(layers):
    vgg = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False,
                                            input_tensor=tf.keras.Input(shape=(224, 224, 3)))
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layers]

    model = tf.keras.Model([vgg.input], outputs=outputs, name='VGG19')
    return model

def gram(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)