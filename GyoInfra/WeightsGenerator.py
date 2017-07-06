import numpy as np


class WeightsGenerator(object):
    def __init__(self):
        return

    def Generate_normal_distributionValues(shape, initMean=0, initSD=0.001):
        numberWeights = np.prod(shape)
        normalDistributionValues = np.random.normal(initMean, initSD, numberWeights)
        return normalDistributionValues.reshape(shape)

    def Generate_uniform_distributionValues(shape,low_value, high_value):
        numberWeights = np.prod(shape)
        uniformDistributionValues = np.random.uniform(low_value, high_value, numberWeights)
        return uniformDistributionValues.reshape(shape)

