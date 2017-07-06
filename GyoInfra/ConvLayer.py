__author__ = 'Gyo'

import theano
from theano.tensor.nnet import conv


class ConvLayer(object):
    def __init__(self,layer_name, input_image,initial_filter_values,filter_shape, image_shape, stride=(1,1)):
        """

        :param layer_name: name of convolutional layer
        :param input_image: input image for convolutional layer
        :param initial_filter_values: initial values when the convolutional layer is created
        :param filter_shape: size of convolutional filter
        :param image_shape: size of input image
        :param stride:
        :return:
        """
        self.LayerName = 'ConvLayer_' + str(layer_name),
        self.Filter = theano.shared(
            value=initial_filter_values,
            name='Filter_' + str(self.LayerName),
            borrow = True
        )




        #flat_image = image.reshape((-1, 1, 1, 600))
        #    filter = filter.reshape((1, 1, 1, -1))   # (num_filters, num_input_channels, filter_rows, filter_columns)



        self.Out = theano.tensor.nnet.conv2d(
            input=input_image,
            filters=self.Filter,
            subsample=stride,
            border_mode='half',
            input_shape=image_shape,
            filter_shape=filter_shape
        )


        self.conved =self.Out
        #conv.conv2d
        """
        self.conved = theano.tensor.nnet.conv2d(
            input=input_image,
            filters=self.Filter,
            subsample=stride,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode='half'
        )
        """

