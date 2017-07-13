__author__ = 'Gyo'
import theano
import theano.tensor as T
import numpy as np
import GyoInfra.ConvLayer as conv_layer
import GyoInfra.DropOutLayer as dropout_layer
import GyoInfra.SoftMaxLayer as softmax_layer
import GyoInfra.FCLayer as fc_layer
import GyoInfra.LogisticRegressionLayer as lr_layer
import GyoInfra.Utiles as utils
from theano.tensor.signal import pool
import theano.sandbox.cuda.basic_ops as sbcuda

"""
Nuestro Raiz agregado es una red neuronal que tiene propiedades como pesos, learningRate y usa una arquitectura, y tiene una entidad que es la arquitectura
Usamos la arquitectura para los propositos de la red neuronal
"""


class DBenGurionArchitecture(object):
    def __init__(self, image_input, batch_size, layers_metaData, initWeights, srng,no_channels_imageInput=1,isTraining=1):
        """
        :param image_input: size: 64x64, channel: 1
        :param conv1_noFilters:
        :return:
        """
        layers_metaData['Conv1_NoFiltersIn'] = no_channels_imageInput
        """
        convolution 1
        kernel: 3x3,
        channel out: 128
        Channel input: 1
        sizeImage input: 64
        ReLU Function
        Out Size: ((64-3 + 1*2) / 1) + 1 = 64
        ((W?F+2P)/S)+1
        W = Input volume size = 64
        F = Filter shape = 3
        S = stride = 1
        P = Padding = 1

        outputShape = batchsize x channel out x sizeImage input x sizeImage input = 10x128x64x64
        """
        conv1_filterShape = (layers_metaData['Conv1_NoFiltersOut'], layers_metaData['Conv1_NoFiltersIn'],
                             layers_metaData['Conv1_sizeKernelW'], layers_metaData['Conv1_sizeKernelH'])
        c1Values = initWeights['conv1Values']
        c1ImageShape = (batch_size, no_channels_imageInput, layers_metaData['Conv1_sizeImgInH'], layers_metaData['Conv1_sizeImgInW'])
        self.conv1 = conv_layer.ConvLayer('conv1Layer', image_input, c1Values, conv1_filterShape, c1ImageShape)
        self.conv_relu_1 = utils.Relu(self.conv1.Out)

        GPUFreeMemoryInBytes = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
        freeGPUMemInGBs = GPUFreeMemoryInBytes / 1024. / 1024 / 1024
        print("Your GPU has %s GBs of free memory" % str(freeGPUMemInGBs))
        """
        convolution
        kernel: 3x3, channel: 128
        ReLU

        """

        conv2_filterShape = (layers_metaData['Conv2_NoFiltersOut'], layers_metaData['Conv2_NoFiltersIn'],
                             layers_metaData['Conv2_sizeKernelW'], layers_metaData['Conv2_sizeKernelH'])
        c2Values = initWeights['conv2Values']
        c2ImageShape = (batch_size, layers_metaData['Conv1_NoFiltersOut'], layers_metaData['Conv2_sizeImgInH'], layers_metaData['Conv2_sizeImgInW'])
        self.conv2 = conv_layer.ConvLayer('conv2Layer', self.conv_relu_1, c2Values, conv2_filterShape, c2ImageShape)
        self.conv_relu_2 = utils.Relu(self.conv2.Out)

        #self.ConvdnnTest=conv_layer.ConvLayer('conv2Layer', self.conv_relu_1, c2Values, conv2_filterShape, c2ImageShape)


        """
        Pool Layer
        OutputShape = (NX128X32X32)
        """
        self.MaxPool_1 = pool.pool_2d(
            input=self.conv_relu_2,
            stride=(2, 2), #stride
            ws =(2, 2),
            mode='max',
            ignore_border=True
        )
        # ((inputSize?kernelSize+2(Padding))/Stride)+1

        conv3_filterShape = (layers_metaData['Conv3_NoFiltersOut'], layers_metaData['Conv3_NoFiltersIn'],
                             layers_metaData['Conv3_sizeKernelW'], layers_metaData['Conv3_sizeKernelH'])
        c3Values = initWeights['conv3Values']
        c3ImageShape = (batch_size, layers_metaData['Conv3_NoFiltersOut'], layers_metaData['Conv3_sizeImgInH'], layers_metaData['Conv3_sizeImgInW'])
        self.conv3 = conv_layer.ConvLayer('conv3Layer', self.MaxPool_1, c3Values, conv3_filterShape, c3ImageShape)
        self.conv_relu_3 = utils.Relu(self.conv3.Out)


        conv4_filterShape = (layers_metaData['Conv4_NoFiltersOut'], layers_metaData['Conv4_NoFiltersIn'],
                             layers_metaData['Conv4_sizeKernelW'], layers_metaData['Conv4_sizeKernelH'])
        c4Values = initWeights['conv4Values']
        c4ImageShape = (batch_size, layers_metaData['Conv4_NoFiltersOut'], layers_metaData['Conv4_sizeImgInH'], layers_metaData['Conv4_sizeImgInW'])
        self.conv4 = conv_layer.ConvLayer('conv4Layer', self.conv_relu_3, c4Values, conv4_filterShape, c4ImageShape)
        self.conv_relu_4 = utils.Relu(self.conv4.Out)


        self.MaxPool_2 = pool.pool_2d(
            input=self.conv_relu_4,
            stride=(2, 2), #stride
            ws =(2, 2),
            mode='max',
            ignore_border=True)

        conv5_filterShape = (layers_metaData['Conv5_NoFiltersOut'], layers_metaData['Conv5_NoFiltersIn'],
                             layers_metaData['Conv5_sizeKernelW'], layers_metaData['Conv5_sizeKernelH'])
        c5Values = initWeights['conv5Values']
        c5ImageShape = (batch_size, layers_metaData['Conv5_NoFiltersOut'], layers_metaData['Conv5_sizeImgInH'], layers_metaData['Conv5_sizeImgInW'])
        self.conv5 = conv_layer.ConvLayer('conv5Layer', self.MaxPool_2, c5Values, conv5_filterShape, c5ImageShape)
        self.conv_relu_5 = utils.Relu(self.conv5.Out)


        """
        outputShape = batchsize x channel out x sizeImage input x sizeImage input = 10x512x32x32
        """
        conv6_filterShape = (layers_metaData['Conv6_NoFiltersOut'], layers_metaData['Conv6_NoFiltersIn'],
                             layers_metaData['Conv6_sizeKernelW'], layers_metaData['Conv6_sizeKernelH'])
        c6Values = initWeights['conv6Values']
        c6ImageShape = (batch_size, layers_metaData['Conv6_NoFiltersOut'], layers_metaData['Conv6_sizeImgInH'], layers_metaData['Conv6_sizeImgInW'])
        self.conv6 = conv_layer.ConvLayer('conv6Layer', self.conv_relu_5, c6Values, conv6_filterShape, c6ImageShape)
        self.conv_relu_6 = utils.Relu(self.conv6.Out)

        """
        outputShape = batchsize x channel out x sizeImage input x sizeImage input = 10x512x8x8
        """
        self.MaxPool_3 = pool.pool_2d(
            input=self.conv_relu_6,
            stride=(2, 2), #stride
            ws =(2, 2),
            mode='max',
            ignore_border=True)


        self.mm =theano.tensor.reshape(self.MaxPool_3, (batch_size, 32768))

        FC1Values = initWeights['FC1Values']
        FC1_BiasInitial_bias_values= initWeights['FC1BiasValues']

        self.FC_1 = fc_layer.FCLayer (
            input_image=self.mm,
            initial_filter_values = FC1Values,
            initial_bias_values=FC1_BiasInitial_bias_values,
            layer_name="FcLayer_1"
        )
        self.FC_relu_1 = utils.Relu(self.FC_1.ProductoCruz)

        self.DO_1 = dropout_layer.DropOutLayer(self.FC_relu_1,srng, (batch_size,layers_metaData['DO1_size_in']),isTraining,0.5)

        FC2Values = initWeights['FC2Values']
        FC2_BiasInitial_bias_values = initWeights['FC2BiasValues']
        self.FC_2 = fc_layer.FCLayer(
            input_image=self.DO_1.output,
            initial_filter_values=FC2Values,
            initial_bias_values=FC2_BiasInitial_bias_values,
            layer_name="FcLayer_2"
        )
        self.FC_relu_2 = utils.Relu(self.FC_2.ProductoCruz)

        self.DO_2 = dropout_layer.DropOutLayer(self.FC_relu_2, srng, (batch_size, layers_metaData['DO2_size_in']), isTraining, 0.5)

        SoftMaxValues = initWeights['SoftMax1Values']
        SoftMaxBiasInitial_bias_values = initWeights['SoftMax1BiasValues']
        self.SoftMax_1 = softmax_layer.SoftMaxLayer(
            input_image=self.DO_2.output,
            initial_filter_values=SoftMaxValues,
            initial_bias_values=SoftMaxBiasInitial_bias_values,
            layer_name="SoftMax_1"
        )
        return

        """
        Input
        size: 64x64, channel: 1
        convolution
        kernel: 3x3, channel: 128
        ReLU

        convolution
        kernel: 3x3, channel: 128
        ReLU

        max pool
        kernel: 2x2
        convolution
        kernel: 3x3, channel: 256
        ReLU

        convolution
        kernel: 3x3, channel: 256
        ReLU

        max pool
        kernel: 2x2
        convolution
        kernel: 3x3, channel: 512
        ReLU

        convolution
        kernel: 3x3, channel: 512
        ReLU

        max pool
        kernel: 2x2
        fully connected
        units: 2048
        ReLU

        dropout
        0.5
        fully connected
        units: 2048
        ReLU

        dropout
        0.5
        softmax
        units: 62
        """
