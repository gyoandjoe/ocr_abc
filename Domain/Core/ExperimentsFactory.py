import numpy as np
import sqlite3
import GyoInfra.DistTypes as DistTypes
from GyoInfra import WeightsGenerator
from Domain.Core.Weights import WeigthsService
from Domain.Core.Weights import WeigthsRepo
import theano

class ExperimentsFactory(object):
    def __init__(self,dbFile , weigthts_service,weigthts_generator,trainDataSet,testDataSet):
        self.dataBaseFile = dbFile
        self.weigthts_service = weigthts_service
        self.weigthts_generator = weigthts_generator
        self.trainDataSet = trainDataSet
        self.testDataSet = testDataSet
        return

    def CreateExperiment(self, nameExperiment,batchSize = 500,initialLearningRate = 0.1,EpochFrecSaveWeights = 10,withLRDecay = 0):
        conn = sqlite3.connect(self.dataBaseFile)

        c = conn.cursor()

        maxEpoch = 500



        EpochFrecLRDecay = 1


        n=str(batchSize)
        shouldDecreaseNow = 0
        shouldIncreaseNow = 0
        query="INSERT INTO Experiments VALUES (NULL ,'"+str(self.trainDataSet)+"','"+str(self.testDataSet)+"','"+str(batchSize)+"',"+str(initialLearningRate)+", 'Pendiente',0,"+str(maxEpoch)+","+str(EpochFrecSaveWeights)+","+str(withLRDecay)+","+str(EpochFrecLRDecay)+","+str(shouldDecreaseNow)+"," +str(shouldIncreaseNow)+ " )"
        c.execute(query)

        """
        Id,
        TrainDataSetFile,
        TestDataSetFile,
        BatchSize,
        InitialLearningRate,
        Status,
        BatchActual,
        MaxEpoch,
        EpochFrecSaveWeights,
        WithLRDecay,
        EpochFrecLRDecay
        """
        conn.commit()


            #wg =
            #debemos generar los pesos iniciales y guardarlos y generar un primer registro


        # We can also close the connection if we are done with it.
        # Just be sure any changes have been committed or they will be lost.
        conn.close()
        return

    def CreateNewWeigths(self,idExperiment,distributionType, distributionParams,layers_metaData):
        newWeights = self.GetNewWeights(distributionType, distributionParams, layers_metaData)
        self.SaveWeights(newWeights, idExperiment, -1)




    def GetNewWeights(self,distributionType, distributionParams,layers_metaData):
        conv1_filterShape = (layers_metaData['Conv1_NoFiltersOut'], layers_metaData['Conv1_NoFiltersIn'],
                             layers_metaData['Conv1_sizeKernelW'], layers_metaData['Conv1_sizeKernelH'])

        conv2_filterShape = (layers_metaData['Conv2_NoFiltersOut'], layers_metaData['Conv2_NoFiltersIn'],
                             layers_metaData['Conv2_sizeKernelW'], layers_metaData['Conv2_sizeKernelH'])

        conv3_filterShape = (layers_metaData['Conv3_NoFiltersOut'], layers_metaData['Conv3_NoFiltersIn'],
                             layers_metaData['Conv3_sizeKernelW'], layers_metaData['Conv3_sizeKernelH'])

        conv4_filterShape = (layers_metaData['Conv4_NoFiltersOut'], layers_metaData['Conv4_NoFiltersIn'],
                             layers_metaData['Conv4_sizeKernelW'], layers_metaData['Conv4_sizeKernelH'])

        conv5_filterShape = (layers_metaData['Conv5_NoFiltersOut'], layers_metaData['Conv5_NoFiltersIn'],
                             layers_metaData['Conv5_sizeKernelW'], layers_metaData['Conv5_sizeKernelH'])

        conv6_filterShape = (layers_metaData['Conv6_NoFiltersOut'], layers_metaData['Conv6_NoFiltersIn'],
                             layers_metaData['Conv6_sizeKernelW'], layers_metaData['Conv6_sizeKernelH'])

        FC1_filterShape = (
        layers_metaData["FC1_NoFiltersIn"] * layers_metaData["FC1_sizeImgInH"] * layers_metaData["FC1_sizeImgInW"],
        layers_metaData["FC1_NoFiltersOut"])
        FC1Bias_filterShape = ((layers_metaData["FC1_NoFiltersOut"]))

        FC2_filterShape = (layers_metaData["FC2_NoFiltersIn"], layers_metaData["FC2_NoFiltersOut"])
        FC2Bias_filterShape = ((layers_metaData["FC2_NoFiltersOut"]))

        SoftM_filterShape = (layers_metaData['SoftM_NoFiltersIn'], layers_metaData['SoftM_NoFiltersOut'])
        SoftMBias_filterShape = ((layers_metaData['SoftM_NoFiltersOut']))

        if distributionType is DistTypes.DistTypes.uniform:
            conv1Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(conv1_filterShape,
                                                                                               distributionParams["conv1LowValue"],
                                                                                               distributionParams["conv1HighValue"])

            conv2Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(conv2_filterShape,
                                                                                               distributionParams["conv2LowValue"],
                                                                                               distributionParams["conv2HighValue"])

            conv3Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(conv3_filterShape,
                                                                                               distributionParams["conv3LowValue"],
                                                                                               distributionParams["conv3HighValue"])

            conv4Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(conv4_filterShape,
                                                                                               distributionParams["conv4LowValue"],
                                                                                               distributionParams["conv4HighValue"])

            conv5Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(conv5_filterShape,
                                                                                               distributionParams["conv5LowValue"],
                                                                                               distributionParams["conv5HighValue"])

            conv6Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(conv6_filterShape,
                                                                                               distributionParams["conv6LowValue"],
                                                                                               distributionParams["conv6HighValue"])

            fc1Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(FC1_filterShape,
                                                                                             distributionParams["fc1LowValue"],
                                                                                             distributionParams["fc1HighValue"])

            if (distributionParams["FC1BiasInit"] == 1):
                fc1BiasValues = np.ones(FC1Bias_filterShape, dtype=theano.config.floatX)
                print ("Ones in fc1BiasValues")
            else:
                print("Zeros in fc1BiasValues")
                fc1BiasValues = np.zeros(FC1Bias_filterShape, dtype=theano.config.floatX)

            fc2Values = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(FC2_filterShape,
                                                                                             distributionParams["fc2LowValue"],
                                                                                             distributionParams["fc2HighValue"])
            if (distributionParams["FC2BiasInit"] == 1):
                fc2BiasValues = np.ones(FC2Bias_filterShape, dtype=theano.config.floatX)
                print("Ones in fc2BiasValues")
            else:
                print ("Zeros in fc2BiasValues")
                fc2BiasValues = np.zeros(FC2Bias_filterShape, dtype=theano.config.floatX)


            SoftMValues = WeightsGenerator.WeightsGenerator.Generate_uniform_distributionValues(SoftM_filterShape,
                                                                                               distributionParams["SoftMLowValue"],
                                                                                               distributionParams["SoftMHighValue"])
            if (distributionParams["SoftMBiasInit"] == 1):
                SoftMBiasValues = np.ones(SoftMBias_filterShape, dtype=theano.config.floatX)
                print("Ones in SoftMBiasValues")
            else:
                print("Zeros in SoftMBiasValues")
                SoftMBiasValues = np.zeros(SoftMBias_filterShape, dtype=theano.config.floatX)
        elif (distributionType is DistTypes.DistTypes.normal):
            conv1Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(conv1_filterShape,
                                                                                               distributionParams["conv1InitMean"],
                                                                                               distributionParams["conv1InitSD"])

            conv2Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(conv2_filterShape,
                                                                                               distributionParams["conv2InitMean"],
                                                                                               distributionParams["conv2InitSD"])

            conv3Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(conv3_filterShape,
                                                                                               distributionParams["conv3InitMean"],
                                                                                               distributionParams["conv3InitSD"])

            conv4Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(conv4_filterShape,
                                                                                               distributionParams["conv4InitMean"],
                                                                                               distributionParams["conv4InitSD"])

            conv5Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(conv5_filterShape,
                                                                                               distributionParams[
                                                                                                   "conv5InitMean"],
                                                                                               distributionParams[
                                                                                                   "conv5InitSD"])

            conv6Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(conv6_filterShape,
                                                                                               distributionParams[
                                                                                                   "conv6InitMean"],
                                                                                               distributionParams[
                                                                                                   "conv6InitSD"])

            fc1Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(FC1_filterShape,
                                                                                             distributionParams[
                                                                                                 "fc1InitMean"],
                                                                                             distributionParams[
                                                                                                 "fc1InitSD"])
            if (distributionParams["FC1BiasInit"] == 1):
                fc1BiasValues = np.ones(FC1Bias_filterShape, dtype=theano.config.floatX)
                print("Ones en fc1BiasValues")
            else:
                fc1BiasValues = np.zeros(FC1Bias_filterShape, dtype=theano.config.floatX)
                print("Zeros en fc1BiasValues")
            fc2Values = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(FC2_filterShape,
                                                                                             distributionParams[
                                                                                                 "fc2InitMean"],
                                                                                             distributionParams[
                                                                                                 "fc2InitSD"])

            if (distributionParams["FC2BiasInit"] == 1):
                fc2BiasValues = np.ones(FC2Bias_filterShape, dtype=theano.config.floatX)
                print("Ones en fc2BiasValues")
            else:
                fc2BiasValues = np.zeros(FC2Bias_filterShape, dtype=theano.config.floatX)
                print("Zeros en fc2BiasValues")

            SoftMValues = WeightsGenerator.WeightsGenerator.Generate_normal_distributionValues(SoftM_filterShape,
                                                                                               distributionParams["SoftMInitMean"],
                                                                                               distributionParams["SoftMInitSD"])

            if (distributionParams["SoftMBiasInit"] == 1):
                SoftMBiasValues = np.ones(SoftMBias_filterShape, dtype=theano.config.floatX)
                print("Ones en SoftMBiasValues")
            else:
                print("Zeros en SoftMBiasValues")
                SoftMBiasValues = np.zeros(SoftMBias_filterShape, dtype=theano.config.floatX)

        initial_weights = {
            "conv1Values": np.asarray(conv1Values, dtype=theano.config.floatX),
            "conv2Values": np.asarray(conv2Values, dtype=theano.config.floatX),
            "conv3Values": np.asarray(conv3Values, dtype=theano.config.floatX),
            "conv4Values": np.asarray(conv4Values, dtype=theano.config.floatX),
            "conv5Values": np.asarray(conv5Values, dtype=theano.config.floatX),
            "conv6Values": np.asarray(conv6Values, dtype=theano.config.floatX),

            "FC1Values": np.asarray(fc1Values, dtype=theano.config.floatX),
            "FC1BiasValues": np.asarray(fc1BiasValues, dtype=theano.config.floatX),

            "FC2Values": np.asarray(fc2Values, dtype=theano.config.floatX),
            "FC2BiasValues": np.asarray(fc2BiasValues, dtype=theano.config.floatX),

            "SoftMax1Values": np.asarray(SoftMValues, dtype=theano.config.floatX),
            "SoftMax1BiasValues": np.asarray(SoftMBiasValues, dtype=theano.config.floatX)
        }


        return initial_weights

    def SaveWeights(self, weights,idExperiment,learning_rate, epoch=0, batch=0, iteration=0, cost=0, error=0, costVal=0, errorVal=0, costTest=0, errorTest=0):

        hyper_params = "{learning rate: " + str(learning_rate) + "}"
        # = [c1_v,c3_v,fc5v_v,fc5b_v,fc6v,fc6b_v]
        self.weigthts_service.SaveWeights(weights, idExperiment,epoch, batch, iteration, hyper_params, cost, error, costVal,
                                          errorVal, costTest, errorTest)
        return


