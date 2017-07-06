from Domain.Core.ExperimentsFactory import ExperimentsFactory
from Domain.Core.Weights import WeigthsRepo, WeigthsService
from GyoInfra import WeightsGenerator, DistTypes
import Domain.Arquitectures.DBenGurionConfig as config

bdFullPath='BD\\OCR_ABC.db'
weigthsFullPath='D:\\Gyo\\Dev\\ocr_abc\\Weights'
trainDataSetFullPath= "D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\TrainSet_full.pkl"
testDataSetFullPath = "D:\\Gyo\\Dev\\OCR\\Kaggle_chars74subset\\resources\\TestSet_full.pkl"

wRepo = WeigthsRepo.WeigthsRepo(database_name=bdFullPath, folder_path=weigthsFullPath)
wService = WeigthsService.WeigthsService(database_name=bdFullPath, weights_repo=wRepo)
wGenerator = WeightsGenerator.WeightsGenerator()

ef= ExperimentsFactory(bdFullPath,wService,wGenerator,trainDataSetFullPath, testDataSetFullPath)

ef.CreateExperiment('',batchSize=250,
                    initialLearningRate=0.1,
                    EpochFrecSaveWeights = 20,
                    withLRDecay = 0
                    )

#ef.CreateNewWeigths(1,DistTypes.DistTypes.uniform,distributionParams=config.distributionParams,layers_metaData=config.layers_metaData)
ef.CreateNewWeigths(1,DistTypes.DistTypes.normal,distributionParams=config.distributionNormalDistParams,layers_metaData=config.layers_metaData)

