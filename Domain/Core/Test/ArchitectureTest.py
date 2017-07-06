__author__ = 'Gyo'
import _pickle as cPickle

import numpy as np
import theano
import theano.sandbox.cuda.basic_ops as sbcuda
import theano.tensor as T

import Domain.Arquitectures.DBenGurionArchitecture as arqui

# THEANO_FLAGS='floatX=float32,device=cuda0,lib.cnmem=1'
#THEANO_FLAGS = 'dnn.conv.algo_fwd=none'

x = T.tensor4('x')  # the data is presented as rasterized images
y = T.ivector('y')
"""
borrow = True

conv1SharedValues=theano.shared(np.asarray(np.zeros((128, 1, 3, 3)), dtype=theano.config.floatX),borrow=borrow)
conv2SharedValues=theano.shared(np.asarray(np.zeros((128, 128, 3, 3)), dtype=theano.config.floatX),borrow=borrow)
conv3SharedValues=theano.shared(np.asarray(np.zeros((256, 128, 3, 3)), dtype=theano.config.floatX),borrow=borrow)
conv4SharedValues=theano.shared(np.asarray(np.zeros((256, 256, 3, 3)), dtype=theano.config.floatX),borrow=borrow)
conv5SharedValues=theano.shared(np.asarray(np.zeros((512, 256, 3, 3)), dtype=theano.config.floatX),borrow=borrow)
conv5SharedValues=theano.shared(np.asarray(np.zeros((512, 512, 3, 3)), dtype=theano.config.floatX),borrow=borrow)
FC1Values=theano.shared(np.asarray(np.zeros((131072, 2048)), dtype=theano.config.floatX),borrow=borrow)
FC1BiasValues=theano.shared(np.asarray(np.zeros(((2048))), dtype=theano.config.floatX),borrow=borrow)
FC2Values=theano.shared(np.asarray(np.zeros((2048,2048)), dtype=theano.config.floatX),borrow=borrow)
FC2BiasValues=theano.shared(np.asarray(np.zeros(((2048))), dtype=theano.config.floatX),borrow=borrow)
LR1Values=theano.shared(np.asarray(np.zeros(((2048,62))), dtype=theano.config.floatX),borrow=borrow)
LR1BiasValues=theano.shared(np.asarray(np.zeros(((62))), dtype=theano.config.floatX),borrow=borrow)

initial_weights = {
    "conv1Values":conv1SharedValues,
    "conv2Values":conv2SharedValues,
    "conv3Values":conv3SharedValues,
    "conv4Values":conv4SharedValues,
    "conv5Values": conv5SharedValues,
    "conv6Values": conv5SharedValues,
    "FC1Values": FC1Values,
    "FC1BiasValues":FC1BiasValues,
    "FC2Values": FC2Values,
    "FC2BiasValues": FC2BiasValues,
    "LR1Values": LR1Values,
    "LR1BiasValues":LR1BiasValues
}
"""

initial_weights = {
    "conv1Values": np.asarray(np.zeros((128, 1, 3, 3)), dtype=theano.config.floatX),
    "conv2Values": np.asarray(np.zeros((128, 128, 3, 3)), dtype=theano.config.floatX),
    "conv3Values": np.asarray(np.zeros((256, 128, 3, 3)), dtype=theano.config.floatX),
    "conv4Values": np.asarray(np.zeros((256, 256, 3, 3)), dtype=theano.config.floatX),
    "conv5Values": np.asarray(np.zeros((512, 256, 3, 3)), dtype=theano.config.floatX),
    "conv6Values": np.asarray(np.zeros((512, 512, 3, 3)), dtype=theano.config.floatX),
    "FC1Values":   np.asarray(np.zeros((32768, 2048)), dtype=theano.config.floatX),
    "FC1BiasValues": np.asarray(np.zeros(((2048))), dtype=theano.config.floatX),
    "FC2Values": np.asarray(np.zeros((2048,2048)), dtype=theano.config.floatX),
    "FC2BiasValues": np.asarray(np.zeros(((2048))), dtype=theano.config.floatX),
    "SoftMax1Values": np.asarray(np.zeros(((2048,62))), dtype=theano.config.floatX),
    "SoftMax1BiasValues": np.asarray(np.zeros(((62))), dtype=theano.config.floatX)
}

layers_metaData = {
    'Conv1_NoFiltersOut': 128,
    'Conv1_NoFiltersIn': 1,
    'Conv1_sizeKernelW': 3,
    'Conv1_sizeKernelH': 3,
    'Conv1_sizeImgInH': 64,
    'Conv1_sizeImgInW': 64,

    'Conv2_NoFiltersOut': 128,
    'Conv2_NoFiltersIn': 128,
    'Conv2_sizeKernelW': 3,
    'Conv2_sizeKernelH': 3,
    'Conv2_sizeImgInH': 64,
    'Conv2_sizeImgInW': 64,
    # Pool 1
    'Conv3_NoFiltersOut': 256,
    'Conv3_NoFiltersIn': 128,
    'Conv3_sizeKernelW': 3,
    'Conv3_sizeKernelH': 3,
    'Conv3_sizeImgInH': 32,
    'Conv3_sizeImgInW': 32,

    'Conv4_NoFiltersOut': 256,
    'Conv4_NoFiltersIn': 256,
    'Conv4_sizeKernelW': 3,
    'Conv4_sizeKernelH': 3,
    'Conv4_sizeImgInH': 32,
    'Conv4_sizeImgInW': 32,
    # pool 2
    'Conv5_NoFiltersOut': 512,
    'Conv5_NoFiltersIn': 256,
    'Conv5_sizeKernelW': 3,
    'Conv5_sizeKernelH': 3,
    'Conv5_sizeImgInH': 16,
    'Conv5_sizeImgInW': 16,

    'Conv6_NoFiltersOut': 512,
    'Conv6_NoFiltersIn': 512,
    'Conv6_sizeKernelW': 3,
    'Conv6_sizeKernelH': 3,
    'Conv6_sizeImgInH': 16,
    'Conv6_sizeImgInW': 16,
    # pool 3
    'FC1_NoFiltersOut': 2048,
    'FC1_NoFiltersIn': 512,
    'FC1_sizeKernelW': 3,
    'FC1_sizeKernelH': 3,
    'FC1_sizeImgInH': 8,
    'FC1_sizeImgInW': 8,

    'FC2_NoFiltersIn': 512 * 8 * 8,
    'FC2_NoFiltersOut': 2048,

    'SoftM_NoFiltersIn':2048,
    'SoftM_NoFiltersOut':62,



    'DO1_size_in': 2048,

    'DO2_size_in': 2048,
}



random_droput = np.random.RandomState(1234)
rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999999))
fLoaded = open("D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\TrainSet_full.pkl", 'rb')
rawData = cPickle.load(fLoaded, encoding='latin1')
rawXTrainingDataSet = rawData[0]
rawYTrainingDataSet = rawData[1]

XimgLetras = np.asarray(rawXTrainingDataSet[0:batch_size], dtype=theano.config.floatX).reshape((batch_size, 1, 64, 64))
XimgLetrasShared = theano.shared(XimgLetras)

YimgLetras = np.asarray(rawYTrainingDataSet[0:batch_size], dtype=np.int32)
YimgLetrasShared = theano.shared(YimgLetras)
batch_size=500


size = XimgLetras.shape[0]

# .reshape((self.dataset_size, 1, 28, 28))
# YimgLetras = np.asarray(rawTrainingDataSet[1])

# batch_size = 50000
# img_input =  x #T.reshape(x,(batch_size, 1, 28, 28))
cnn = arqui.DBenGurionArchitecture(
    image_input=x,
    batch_size=batch_size,
    layers_metaData=layers_metaData,
    initWeights=initial_weights,
    srng=rng_droput,
    no_channels_imageInput=1
)
cost = cnn.SoftMax_1.cost_function(y)
weights = [cnn.conv1.Filter,
           cnn.conv2.Filter,
           cnn.conv3.Filter,
           cnn.conv4.Filter,
           cnn.conv5.Filter,
           cnn.conv6.Filter,
           cnn.FC_1.Filter,
           cnn.FC_1.Bias,
           cnn.FC_2.Filter,
           cnn.FC_2.Bias,
           cnn.SoftMax_1.Filter,
           cnn.SoftMax_1.Bias]

#grads = T.grad(cost, weights)
grads = T.grad(cost, weights, disconnected_inputs="raise")
learningRate = T.fscalar()

updates = [
    (param_i, param_i + (learningRate * grad_i))
    for param_i, grad_i in zip(weights, grads)
    ]

#index = T.lscalar()


train_model = theano.function(
    [learningRate],
    cost,
    updates=updates,
    givens={
        x: XimgLetrasShared,
        y: YimgLetrasShared
    }
)


#convTest = theano.function(
#    [],
#    cnn.SoftMax_1.p_y_given_x,  # self.classifier.FC.p_y_given_x,#dropout.output #conv_relu_1,    #ConvdnnTest.conved,
#    givens={
#        x: XimgLetrasShared # [0: 10]
#    }
#)

GPUFreeMemoryInBytes = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
freeGPUMemInGBs = GPUFreeMemoryInBytes / 1024. / 1024 / 1024
print("Your GPU has %s GBs of free memory" % str(freeGPUMemInGBs))
#result = convTest()
result = train_model(0.5)
GPUFreeMemoryInBytes = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
freeGPUMemInGBs = GPUFreeMemoryInBytes / 1024. / 1024 / 1024
print("Your GPU has %s GBs of free memory" % str(freeGPUMemInGBs))
print("ok " + str(result.shape[0]) + ',' + str(result.shape[1]))
print('ok2_ ' + str(result.shape[2]) + ',' + str(result.shape[3]))

# result shape (10L, 128L, 62L, 62L)
