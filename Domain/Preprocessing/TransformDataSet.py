__author__ = 'Gyo'
import glob

import skimage
import numpy
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import csv
import pandas as pd

from skimage import img_as_ubyte
import Domain.Preprocessing.DataAugmentation as data_augmentation
from skimage.io import imread, imsave

"""
image_file = Image.open(img_path)
    image = image_file.convert('L')  # convert image to monochrome
    image = numpy.array(image)
"""
path = "D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources"
trainFiles = glob.glob(path + "\\train\\*")


class TransformDataSet:
    def __init__(self):
        pass

    def TransformLabels(self,fileReferenceTrainDataOut,fileReferenceTrainData= "D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\trainLabels.csv"):

        r = pd.read_csv(fileReferenceTrainData)
        all = pd.DataFrame({'Index' : range(0, r.Class.unique().size),'Character' : r.Class.unique()})
        all.to_csv(path_or_buf= fileReferenceTrainDataOut,index=False)

    def GetSubDataTrain(self, startRowIndex, subDataSetSize,fileReferenceLabels, fileReferenceTrainData= "D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\trainLabels.csv",divideX255=True):
        trainPath = "D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\trainResized"
        referenceClass = pd.read_csv(fileReferenceLabels)
        #referenceDataTrainData = np.loadtxt(fileReferenceTrainData, delimiter=',', dtype='str', usecols=(0, 1),skiprows=1)
        #referenceDataTrainData=np.genfromtxt(fileReferenceTrainData, dtype=None)

        referenceDataTrainData =  pd.read_csv(fileReferenceTrainData)
        endRowIndex = (startRowIndex + subDataSetSize)
        #referenceDataTrainData = referenceDataTrainData[startRowIndex:endRowIndex, :]
        referenceDataTrainData = referenceDataTrainData.loc[startRowIndex:endRowIndex]

        noRegisters = referenceDataTrainData.shape[0]

        DataX = np.zeros((noRegisters, 64, 64), dtype=np.float64)
        DataY = np.zeros(noRegisters, dtype=np.int64)


        for contador,row in referenceDataTrainData.iterrows():
            imagePath = trainPath + '\\' + str(row["ID"]) + '.bmp'
            image = imread(imagePath)
            imagegrayScale = skimage.img_as_ubyte(rgb2gray(image))
            if (divideX255 == True):
                DataX[contador] = np.asarray(imagegrayScale, dtype=np.float64) / 256
            else:
                DataX[contador] = np.asarray(imagegrayScale, dtype=np.float64)

            yValue = (referenceClass.loc[referenceClass['Character'] == row["Class"]].Index).values[0]

            DataY[contador] = yValue #any(referenceClass.column == 07311954 ) row[1]
            contador += 1
        return (DataX, DataY)

    def GetSubDataTrainPlusAumented(self, startRowIndex, subDataSetSize, fileReferenceLabels,fileReferenceTrainData="D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\trainLabels.csv"):
        originalData = self.GetSubDataTrain(startRowIndex, subDataSetSize, fileReferenceLabels,
                        fileReferenceTrainData,
                        divideX255=True)

        dataAugmented = data_augmentation.batch_iterator(originalData[0].reshape(originalData[0].shape[0],1,originalData[0].shape[1],originalData[0].shape[2]),
                                                         originalData[1],
                                                         originalData[0].shape[0]
                                                         )


        xData = np.array(originalData[0], dtype=np.float32)
        yData = np.array(originalData[1], dtype=np.int32)

        resultAugmentedX = []
        resultAugmentedY = []
        for (batchImageX, batchImageY) in dataAugmented:
            for j in range(batchImageX.shape[0]):
                x_augmented = batchImageX[j][0]
                y_augmented = batchImageY[j]
                resultAugmentedX.append(x_augmented)
                resultAugmentedY.append(y_augmented)

        xDataAugmented = np.array(resultAugmentedX)
        yDataAugmented = np.array(resultAugmentedY, dtype=np.int32)

        allXData = np.concatenate((xData,xDataAugmented))
        allYData = np.concatenate((yData, yDataAugmented))

        return (allXData,allYData)






"""
    for i, nameFile in enumerate(trainFiles):
        image = imread(nameFile)

        imagegrayScale = img_as_ubyte(rgb2gray(image))


        ar = numpy.asarray(imagegrayScale, dtype=int)
        print imagegrayScale
        plt.imshow(imagegrayScale, cmap = plt.get_cmap('gray'))

        plt.show()
        print "ok"



    testFiles = glob.glob(path + "\\test\\*")
    for i, nameFile in enumerate(testFiles):
        image = imread(nameFile)
        imagegrayScale = rgb2gray(image)

"""

# GetTrainData()
