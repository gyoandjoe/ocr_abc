import numpy as np
import theano.tensor as T
import _pickle as cPickle
import theano
import Domain.Preprocessing.DataAugmentation as data_augmentation
from skimage.io import imread, imsave

random_droput = np.random.RandomState(1234)
rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999999))

batch_size = 500
fLoaded = open("D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\TrainSet_full.pkl", 'rb')
rawData = cPickle.load(fLoaded, encoding='latin1')
rawXTrainingDataSet = rawData[0]
rawYTrainingDataSet = rawData[1]

XimgLetras = np.asarray(rawXTrainingDataSet[0:batch_size], dtype=theano.config.floatX).reshape((batch_size, 1, 64, 64))
XimgLetrasShared = theano.shared(XimgLetras)

YimgLetras = np.asarray(rawYTrainingDataSet[0:batch_size], dtype=np.int32)
YimgLetrasShared = theano.shared(YimgLetras)

result = data_augmentation.batch_iterator(XimgLetras, YimgLetras, 50)

i = 0
for batchImageX, batchImageY in result:
    for j in range(batchImageX.shape[0]):
        imgT = batchImageX[j][0]
        imsave(
            'D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\allTrainSet\\img' + str(i) + '_' + str(j) + '.bmp',
            imgT)
    i += 1

print("OK")
