from unittest import TestCase
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../')
from nst_support_functions import compute_style_cost_layer

class TestCompute_style_cost_layer(TestCase):

    def testComputeStyleCostLayer_GivenShapes_1x2x2x1(self):

        a_C = tf.constant(np.full((1, 2, 2, 1), 3), dtype=tf.float32)
        a_G = tf.constant(np.full((1, 2, 2, 1), 1), dtype=tf.float32)

        expected = 16
        result = compute_style_cost_layer(a_C, a_G)

        assert expected == result

    def testComputeStyleCostLayer_GivenShapes_1x100x100x32(self):

        a_C = tf.constant(np.full((1, 100, 100, 32), 3), dtype=tf.float32)
        a_G = tf.constant(np.full((1, 100, 100, 32), 2), dtype=tf.float32)

        expected = 6.25
        result = compute_style_cost_layer(a_C, a_G)

        diff = np.absolute(expected - result)
        percent_error = (diff / expected) * 100

        assert percent_error < 0.001
