import tensorflow as tf
import numpy as np
import IPython.display as display
import skimage.io as image
import sys
sys.path.append('/')
import NeuralStyleTransfer as nst
from nst_support_functions import tensor_to_image

style_image = image.imread('./images/starrynight.jpg')
content_image = image.imread('./images/louvre.jpg')

generated = nst.NeuralStyleTransfer(content_image=content_image, style_image=style_image)
image = generated.neural_style_transfer()

display.clear_output(wait=True)
display.display(tensor_to_image(image))


