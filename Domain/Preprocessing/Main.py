from Domain.Preprocessing import TransformDataSet
from Domain.Preprocessing.ABCToPKL import ABCToPKL
import pandas as pd
import numpy as np
__author__ = 'Gyo'

outTrainDataSet = "D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\TrainSet_full3_orig.pkl"
outTestDataSet = "D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\TestSet_full3_oring.pkl"

ConvToGScale = TransformDataSet.TransformDataSet()

classReferenceFile = "D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\trainLabels_Indexing.csv"
#ConvToGScale.TransformLabels(classReferenceFile) #Descomentar si se desea volver a generar



allDataPlusAgmented = ConvToGScale.GetSubDataTrainPlusAumented(0, 6283,classReferenceFile)
from random import shuffle
# Given list1 and list2
dataX_shuf = []
dataY_shuf = []
#index_shuf = range(allDataPlusAgmented[0].shape[0])
p = np.random.permutation(allDataPlusAgmented[0].shape[0])
#shuffle(index_shuf)
for i in p:
    dataX_shuf.append(allDataPlusAgmented[0][i])
    dataY_shuf.append(allDataPlusAgmented[1][i])

#dataX_shuf=allDataPlusAgmented[0]
#dataY_shuf=allDataPlusAgmented[1]

dataX_shuf_array = np.array(dataX_shuf)
dataY_shuf_array = np.array(dataY_shuf)

trainDataX = dataX_shuf_array[0:8796]#ConvToGScale.GetSubDataTrain(0, 4283,classReferenceFile)
trainDataY = dataY_shuf_array[0:8796]
testDataX = dataX_shuf_array[8796:12566] #ConvToGScale.GetSubDataTrain(4283,6283,classReferenceFile)
testDataY = dataY_shuf_array[8796:12566]

PKLSerializer = ABCToPKL()

PKLSerializer.StartProcess((trainDataX,trainDataY),outTrainDataSet)

PKLSerializer.StartProcess((testDataX,testDataY),outTestDataSet)

print ('ok')
