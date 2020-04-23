import tensorflow as tf
import numpy as np

class TestComputeStyleCost(tf.test.TestCase):

    def setUp(self):
        with self.session(use_gpu=True):
            tf.keras.backend.set_floatx('float64')
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.Conv2D(1, (3, 3), activation='relu', input_shape=(4, 4, 1)))
            self.model.add(tf.keras.layers.Flatten())
            self.model.build()
        # self.model.summary()


    def testMockConv2DModel_OneLayer(self):
        with self.session(use_gpu=True):
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
            input = np.full((1, 4, 4, 1), 2.0)

            output = self.model(input)
            expected = np.array([[2.0, 2.0, 2.0, 2.0]])

            self.assertAllEqual(expected, output)


tf.test.main()