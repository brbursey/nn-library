from unittest import TestCase
import skimage.io as io
import numpy as np
import sys
sys.path.append('../')
from nst_support_functions import reshape_and_normalize_image


class TestReshape_and_normalize_image(TestCase):

    def testReshapeAndNormalizeImage(self):
        doggy_image = io.imread('./../images/doggy.jpg')

        expected = (1, 185, 273, 3)
        result = reshape_and_normalize_image(doggy_image).shape

        assert expected == result
