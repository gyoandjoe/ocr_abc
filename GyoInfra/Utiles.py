__author__ = 'Gyo'
import theano


def Relu(x):
    return theano.tensor.switch(x < 0, 0, x)
