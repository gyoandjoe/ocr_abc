import theano.tensor as T
import theano
import numpy as np

class DropOutLayer(object):
    def __init__(self, input, srng, image_shape, is_training,p):
        mask=srng.binomial(n=1,size=image_shape,p=p, dtype=theano.config.floatX)
        self.output = T.switch(T.neq(is_training, 0), np.multiply(input,mask), np.multiply(input,p))
        #np.multiply(input,mask) => Training
        #np.multiply(input, p) => not training