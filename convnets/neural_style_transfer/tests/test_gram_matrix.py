from unittest import TestCase
from nst_support_functions import gram_matrix
import tensorflow as tf

class TestGram_matrix(TestCase):

    def test_gram_matrix_given_a_3x3_matrix(self):
        matrix = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9], shape=[3,3], dtype=tf.int32)

        expected = tf.constant([14, 32, 50, 32, 77, 122, 50, 122, 194], shape=[3,3], dtype=tf.int32)
        result = gram_matrix(matrix)

        assert expected.numpy().all() == result.numpy().all()

