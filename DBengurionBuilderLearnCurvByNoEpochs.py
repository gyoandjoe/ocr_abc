import _pickle as cPickle
import numpy as np
import theano.tensor as T
import theano
from Domain.Arquitectures import DBenGurionOCR
from Domain.Arquitectures import DBenGurionConfig
from Domain.Experiments import ExperimentsRepo
from Domain.Core.Weights.WeigthsService import WeigthsService
from Domain.Core.Weights.WeigthsRepo import WeigthsRepo
from GyoInfra.Logger import Logger
import Domain.Analysis.Analizador as Analizador


__author__ = 'win-g'
#print(theano.config)
idExperiment = 2


weigthsFullPath = 'D:\\Gyo\\Dev\\ocr_abc\\Weights'

experimenRepo = ExperimentsRepo.ExperimentsRepo('BD\\OCR_ABC.db',idExperiment)
logger = Logger(id_experiment=idExperiment,database_name='BD\\OCR_ABC.db')
weigthsRepo = WeigthsRepo( database_name='BD\\OCR_ABC.db',folder_path=weigthsFullPath)


wService = WeigthsService(database_name='BD\\OCR_ABC.db',weights_repo=weigthsRepo)

#Cargamos Train Raw Data
trainDataSetPklFile = "D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\TrainSet_full.pkl"
fTrainLoaded = open(trainDataSetPklFile, 'rb')
rawTrainData = cPickle.load(fTrainLoaded, encoding='latin1')

#Cargamos Validation Raw Data
testDataSetPklFile = "D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\TestSet_full.pkl"
fValLoaded = open(testDataSetPklFile, 'rb')
rawValData = cPickle.load(fValLoaded, encoding='latin1')

batchSize = experimenRepo.ObtenerBatchSize()
batchSize = int(batchSize)


analizador = Analizador.Analizador('BD\\OCR_ABC.db')

analizador.BuildWeigthsErrorAndCost(
    id_experiment=idExperiment,
    bd='BD\\OCR_ABC.db',
    weigths_path=weigthsFullPath,
    layers_metaData=DBenGurionConfig.layers_metaData,
    raw_train_set=rawTrainData,
    train_batch_size=batchSize,
    raw_validation_set=rawValData,
    validation_batch_size=batchSize,
    logger=logger,
    weigths_service=wService,
    experimentsRepo=experimenRepo
)



print('Finish Building')

