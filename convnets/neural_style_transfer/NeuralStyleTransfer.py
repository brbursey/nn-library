import tensorflow as tf
import numpy as np
from skimage.transform import resize
import sys
sys.path.append('/')
from StyleContentModel import StyleContentModel

class NeuralStyleTransfer():
    def __init__(self, content_image,
                 style_image,
                 content_layers=['block4_conv2'],
                 style_layers = ['block1_conv1',
                                'block2_conv1',
                                'block3_conv1',
                                'block4_conv1',
                                'block5_conv1'],
                 content_weight=5,
                 style_weight=5,
                 total_variation_weight=30,
                 iterations=100
                 ):
        self.content_image = content_image
        self.style_image = style_image
        self.layers = [content_layers, style_layers]
        self.content_weight = content_weight,
        self.style_weight = style_weight,
        self.total_variation_weight = total_variation_weight
        self.iterations = iterations


    def prep_image(self, content_image, style_image):
        content = resize(content_image, (300, 300))
        print(f'\nResizing content image from {content_image.shape} to (300, 300, 3)')
        # content = reshape_and_normalize_image(content)
        content = content.astype(float)
        content_tensor = tf.expand_dims(tf.constant(content), 0)
        print(f'\nExpanding content image to shape {content_tensor.shape}')

        style = resize(style_image, (300, 300))
        print(f'\nResizing style image from {style_image.shape} to (300, 300, 3)')
        # style = reshape_and_normalize_image(style)

        style = style.astype(float)
        style_tensor = tf.expand_dims(tf.constant(style), 0)
        print(f'\nExpanding style image to shape {style_tensor.shape}')

        return content_tensor, style_tensor


    def style_content_loss(self, outputs, style_targets, content_targets):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                               for name in style_outputs.keys()])

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])

        loss = (self.style_weight * style_loss) + (self.content_weight * content_loss)
        return loss


    @tf.function()
    def training_step(self, image, model_targets, optimizer):
        model = model_targets['model']
        style_targets = model_targets['style']
        content_targets = model_targets['content']

        with tf.GradientTape() as tape:
            outputs = model(image)
            loss = self.style_content_loss(outputs, style_targets, content_targets)
            loss += tf.cast(self.total_variation_weight * tf.image.total_variation(image), dtype='float32')

        grad = tape.gradient(loss, image)
        optimizer.apply_gradients([(grad, image)])
        image.assign(image)


    def neural_style_transfer(self):
        prepped_content, prepped_style = self.prep_image(self.content_image, self.style_image)
        content_layers, style_layers = self.layers

        style_content_model = StyleContentModel(style_layers, content_layers)
        style_targets = style_content_model(prepped_style)['style']
        content_targets = style_content_model(prepped_content)['content']

        model_targets = {'model': style_content_model, 'style': style_targets, 'content': content_targets}

        image = tf.Variable(prepped_content)

        opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        # test
        for i in range(1, self.iterations):
            self.training_step(image, model_targets, opt)
            if i % 10 == 0:
                print(f'Training... {self.iterations-i} iterations to go.')
        return image
