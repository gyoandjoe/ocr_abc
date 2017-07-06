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


#print(theano.config)
idExperiment = 1
idWeigths = 1
dataSetPklFile = "D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\TrainSet_full.pkl"
weigthsFullPath = 'D:\\Gyo\\Dev\\ocr_abc\\Weights'

experimenRepo = ExperimentsRepo.ExperimentsRepo('BD\\OCR_ABC.db',idExperiment)
logger = Logger(id_experiment=idExperiment,database_name='BD\\OCR_ABC.db')
weigthsRepo = WeigthsRepo( database_name='BD\\OCR_ABC.db',folder_path=weigthsFullPath)


wService = WeigthsService(database_name='BD\\OCR_ABC.db',weights_repo=weigthsRepo)

#Cargamos Raw Data
fLoaded = open(dataSetPklFile, 'rb')
rawTrainData = cPickle.load(fLoaded, encoding='latin1')


initial_weights =  wService.LoadRawWeigths(idWeigths)

#Variables para hacer dropout
random_droput = np.random.RandomState(1234)
rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999999))

batchSize = experimenRepo.ObtenerBatchSize()
batchSize = int(batchSize)
maxEpoch = experimenRepo.ObtenerMaxEpoch()
wLrDecay = experimenRepo.ObtenerWithLRDecay()
lRate = experimenRepo.ObtenerLearningRate()
saveWFrecuency = experimenRepo.ObtenerFrecuencySaveWeigths()
lrDecayFrecyency = experimenRepo.ObtenerFrecuencyLRDecay()


#Primero de bemos cargar todos los parametros del experimento que se quiere
DBG = DBenGurionOCR.DBenGurionOCR(
            id_experiment = idExperiment,
            layers_metaData = DBenGurionConfig.layers_metaData,
            batch_size = batchSize,
            raw_train_set=rawTrainData,
            logger = logger,
            weigthts_service = wService,
            experimentsRepo=experimenRepo,
            initial_weights=initial_weights,
            max_epochs=maxEpoch,
            with_lr_decay=wLrDecay,
            learning_rate=lRate,
            saveWeigthsFrecuency=saveWFrecuency,
            frecuency_lr_decay=lrDecayFrecyency,
            )

DBG.Train(current_epoch=0,
          id_train='',
          extra_info='')

print('Finish Training')