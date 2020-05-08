import tensorflow as tf
import numpy as np
import sys
sys.path.append('../')
from nst_support_functions import compute_style_cost

class TestComputeStyleCost(tf.test.TestCase):

    def setUp(self):
        with self.session(use_gpu=True):
            tf.keras.backend.set_floatx('float64')
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.Conv2D(1, (3, 3), activation='relu', input_shape=(4, 4, 1)))
            self.model.add(tf.keras.layers.Flatten())
            self.model.build()
            self.model._name = "MockModel"

            weights = np.array(
                [
                    [[[0]], [[-1]], [[0]]],
                    [[[-1]], [[5]], [[-1]]],
                    [[[0]], [[-1]], [[0]]]
                ])
            bias = np.array([0])
            self.model.set_weights(
                [weights, bias]
            )
            self.input = np.full((1, 4, 4, 1), 2.0)


    def testMockConv2DModel_OneLayer(self):
        with self.session(use_gpu=True):

            output = self.model(self.input)
            expected = np.array([[2.0, 2.0, 2.0, 2.0]])

            self.assertAllEqual(expected, output)


    def testComputeStyleCost(self):
        with self.session(use_gpu=True):
            gen_input = np.full((1, 4, 4, 1), 3.0)
            layers = ['conv2d']
            coefs = [1]

            expected = 6.25
            result = compute_style_cost(self.model, self.input, gen_input, layers, coefs, name="MockModel")

            assert expected == result.numpy()

tf.test.main()