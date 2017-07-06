import numpy as np
import theano.tensor as T
import _pickle as cPickle
import theano
import Domain.Preprocessing.DataAugmentation as data_augmentation
from skimage.io import imread, imsave

random_droput = np.random.RandomState(1234)
rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999999))


fLoaded = open("D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\TrainSet_full3_orig.pkl", 'rb')
rawData = cPickle.load(fLoaded, encoding='latin1')
rawXTrainingDataSet = rawData[0]
rawYTrainingDataSet = rawData[1]
batch_size = rawXTrainingDataSet.shape[0]

XimgLetras = np.asarray(rawXTrainingDataSet[0:batch_size], dtype=theano.config.floatX).reshape((batch_size, 1, 64, 64))


YimgLetras = np.asarray(rawYTrainingDataSet[0:batch_size], dtype=np.int32)


result = (XimgLetras,YimgLetras)

i = 0
for j in range(XimgLetras.shape[0]):
    imgT = XimgLetras[j][0]
    y = YimgLetras[j]


    imsave(
        'D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\allTrainSet\\img' + str(j) + '_' + str(y) + '.bmp',
        imgT)

print("OK")
