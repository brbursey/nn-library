from unittest import TestCase
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../')
from nst_support_functions import compute_content_cost


class TestCompute_content_cost(TestCase):

    def testComputeCost_GivenShape_1x2x2x1(self):
        a_C = tf.constant(np.full((1, 2, 2, 1), 2), dtype=tf.float32)
        a_G = tf.constant(np.full((1, 2, 2, 1), 1), dtype=tf.float32)

        expected = 0.25
        result = compute_content_cost(a_C, a_G)

        assert expected == result

    def testComputeCost_GivenShape_1x5x5x1(self):
        a_C = tf.constant(np.full((1, 5, 5, 1), 3), dtype=tf.float32)
        a_G = tf.constant(np.full((1, 5, 5, 1), 1), dtype=tf.float32)

        expected = 1.0
        result = compute_content_cost(a_C, a_G)

        assert expected == result

    def testComputeCost_GivenChannelIsGreaterThanOne(self):
        a_C = tf.constant(np.full((1, 2, 2, 2), 4), dtype=tf.float32)
        a_G = tf.constant(np.full((1, 2, 2, 2), 1), dtype=tf.float32)

        expected = 2.25
        result = compute_content_cost(a_C, a_G)

        assert expected == result

    def testComputeCost_GivenPictureShape(self):
        a_C = tf.constant(np.full((1, 300, 400, 3), 5), dtype=tf.float32)
        a_G = tf.constant(np.full((1, 300, 400, 3), 1), dtype=tf.float32)

        expected = 4
        result = compute_content_cost(a_C, a_G)

        assert expected == result


