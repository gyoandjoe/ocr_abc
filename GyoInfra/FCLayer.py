__author__ = 'Gyo'

import theano
import theano.tensor as T
import theano.sandbox.cuda.basic_ops as sbcuda
from theano.sandbox.cuda import dnn

class FCLayer(object):
    def __init__(self, layer_name, input_image, initial_filter_values, initial_bias_values):
        """
        :param layer_name:
        :param input_image: Image to be treated
        :param initial_filter_values: Initial values of filter
        :param initial_bias_values: Initial values of bias
        :param activation_function: function that computes the ProductoCruz
        :return:
        """

        self.LayerName = 'FCLayer_' + layer_name



        self.Bias = theano.shared(
            value=initial_bias_values,
            name='Bias_' + str(self.LayerName),
        )

        self.Filter = theano.shared(
            value=initial_filter_values,
            name='Filter_'+str(self.LayerName),
            borrow = True
        )

        self.ProductoCruz = T.dot(input_image, self.Filter) + self.Bias
