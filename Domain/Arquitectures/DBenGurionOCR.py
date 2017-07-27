import theano
import theano.tensor as T
import numpy as np
import Domain.Arquitectures.DBenGurionArchitecture as DBenGurionArchitecture
import _pickle as cPickle
import Domain.Core.LayerEnum as LayerEnum
from theano.tensor.shared_randomstreams import RandomStreams

"""
Este es el modelo principal que se podra entrenar o usar para propositos finales, debe diseñarse teniendo en cuenta que podra tener 2 usos
"""


class DBenGurionOCR(object):
    """
    Constructor para uso productivo
    """
    def __init__(self):
        return

    """
    Constructor para validacion
    """

    @classmethod
    def Validator(self,id_experiment,layers_metaData, batch_size, raw_data_set, logger,weigthts_service, experimentsRepo,initial_weights):
        self.idExperiment = id_experiment
        self.logger = logger
        self.weigthts_service = weigthts_service
        self.experimentsRepo = experimentsRepo

        self.x = T.tensor4('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')

        index = T.lscalar()

        random_droput = np.random.RandomState(1234)
        rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999999))

        rawXDataSet = raw_data_set[0]
        rawYDataSet = raw_data_set[1]
        self.totalDataSize = rawXDataSet.shape[0]
        self.no_batchs_in_data_set = self.totalDataSize // batch_size

        # batch_size = 50000
        # img_input =  x #T.reshape(x,(batch_size, 1, 28, 28))
        self.CNN = DBenGurionArchitecture.DBenGurionArchitecture(
            image_input=self.x,
            batch_size=batch_size,
            layers_metaData=layers_metaData,
            initWeights=initial_weights,
            srng=rng_droput,
            no_channels_imageInput=1,
            isTraining=1
        )

        XimgLetras = np.asarray(rawXDataSet, dtype=theano.config.floatX).reshape(
            (self.totalDataSize, 1, 64, 64))
        XimgLetrasShared = theano.shared(XimgLetras)

        YimgLetras = np.asarray(rawYDataSet, dtype=np.int32)
        YimgLetrasShared = theano.shared(YimgLetras)

        cost = self.CNN.SoftMax_1.negative_log_likelihood(self.y)
        self.evaluate_model_with_cost = theano.function(
            [index],
            cost,
            givens={
                self.x: XimgLetrasShared[index * batch_size: (index + 1) * batch_size],
                self.y: YimgLetrasShared[index * batch_size: (index + 1) * batch_size]
            }
        )

        error = self.CNN.SoftMax_1.errors(self.y)
        self.evaluate_model_with_error = theano.function(
            [index],
            error,
            givens={
                self.x: XimgLetrasShared[index * batch_size: (index + 1) * batch_size],
                self.y: YimgLetrasShared[index * batch_size: (index + 1) * batch_size]
            }
        )
        return self()

    """
    Constructor para Entrenamiento
    """

    @classmethod
    def Trainer(self,id_experiment,layers_metaData, batch_size, raw_train_set, logger, weigthts_service, experimentsRepo,initial_weights,max_epochs,with_lr_decay,learning_rate,saveWeigthsFrecuency,frecuency_lr_decay,p_DropOut):
        self.idExperiment = id_experiment
        self.logger = logger
        self.max_epochs = max_epochs
        self.with_lr_decay =with_lr_decay
        self.learning_rate = float(learning_rate)
        self.weigthts_service = weigthts_service
        self.saveWeigthsFrecuency = saveWeigthsFrecuency
        self.frecuency_lr_decay = frecuency_lr_decay
        self.experimentsRepo = experimentsRepo
        self.theano_rng = RandomStreams(123)

        self.x = T.tensor4('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')
        learningRate = T.fscalar()
        index = T.lscalar()

        random_droput = np.random.RandomState(1234)
        rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999999))

        rawXTrainingDataSet = raw_train_set[0]
        rawYTrainingDataSet = raw_train_set[1]
        self.trainDataSetSize = rawXTrainingDataSet.shape[0]
        self.no_batchs_in_data_set = self.trainDataSetSize // batch_size

        # batch_size = 50000
        # img_input =  x #T.reshape(x,(batch_size, 1, 28, 28))
        self.CNN = DBenGurionArchitecture.DBenGurionArchitecture(
            image_input=self.x,
            batch_size=batch_size,
            layers_metaData=layers_metaData,
            initWeights=initial_weights,
            srng=rng_droput,
            no_channels_imageInput=1,
            isTraining=1,
            pDropOut=p_DropOut
        )

        XimgLetras = np.asarray(rawXTrainingDataSet, dtype=theano.config.floatX).reshape((self.trainDataSetSize, 1, 64, 64))
        XimgLetrasShared = theano.shared(XimgLetras)

        YimgLetras = np.asarray(rawYTrainingDataSet, dtype=np.int32)
        YimgLetrasShared = theano.shared(YimgLetras)

        #cost = self.CNN.SoftMax_1.cost_function(y)
        cost = self.CNN.SoftMax_1.negative_log_likelihood(self.y)

        error =  self.CNN.SoftMax_1.errors(self.y)
        #error = self.CNN.SoftMax_1.(y)

        weights = [self.CNN.conv1.Filter,
                   self.CNN.conv2.Filter,
                   self.CNN.conv3.Filter,
                   self.CNN.conv4.Filter,
                   self.CNN.conv5.Filter,
                   self.CNN.conv6.Filter,
                   self.CNN.FC_1.Filter,
                   self.CNN.FC_1.Bias,
                   self.CNN.FC_2.Filter,
                   self.CNN.FC_2.Bias,
                   self.CNN.SoftMax_1.Filter,
                   self.CNN.SoftMax_1.Bias]

        grads = T.grad(cost, weights, disconnected_inputs="raise")

        updates = [
            (param_i, param_i + (learningRate * grad_i))
            for param_i, grad_i in zip(weights, grads)
        ]

        #errors = self.CNN.SoftMax_1.

        self.train_model = theano.function(
            [index,learningRate],
            cost,
            updates=updates,
            givens={
                self.x: XimgLetrasShared[index * batch_size: (index + 1) * batch_size],
                self.y: YimgLetrasShared[index * batch_size: (index + 1) * batch_size]
            }
        )

        return self()


    def GetWeigthsValuesByLayer(self, layer):
        if layer is LayerEnum.LayerEnum.conv1:
            return np.asarray(self.CNN.conv1.Filter.get_value(),dtype=theano.config.floatX)
        if layer is LayerEnum.LayerEnum.conv2:
            return np.asarray(self.CNN.conv2.Filter.get_value(),dtype=theano.config.floatX)
        if layer is LayerEnum.LayerEnum.conv3:
            return np.asarray(self.CNN.conv3.Filter.get_value(),dtype=theano.config.floatX)
        if layer is LayerEnum.LayerEnum.conv4:
            return np.asarray(self.CNN.conv4.Filter.get_value(),dtype=theano.config.floatX)
        if layer is LayerEnum.LayerEnum.conv5:
            return np.asarray(self.CNN.conv5.Filter.get_value(),dtype=theano.config.floatX)
        if layer is LayerEnum.LayerEnum.conv6:
            return np.asarray(self.CNN.conv6.Filter.get_value(),dtype=theano.config.floatX)
        elif layer is LayerEnum.LayerEnum.FC_1:
            return (np.asarray(self.CNN.FC_1.Filter.get_value(),dtype=theano.config.floatX),np.asarray(self.CNN.FC_1.Bias.get_value(),dtype=theano.config.floatX))
        elif layer is LayerEnum.LayerEnum.FC_2:
            return (np.asarray(self.CNN.FC_2.Filter.get_value(),dtype=theano.config.floatX),np.asarray(self.CNN.FC_2.Bias.get_value(),dtype=theano.config.floatX))
        elif layer is LayerEnum.LayerEnum.SoftMax_1:
            return (np.asarray(self.CNN.SoftMax_1.Filter.get_value(),dtype=theano.config.floatX),np.asarray(self.CNN.SoftMax_1.Bias.get_value(),dtype=theano.config.floatX))

    def Train(self, current_epoch=0, id_train='', extra_info='' ):
        for epoch_index in range(self.max_epochs):
            if epoch_index < current_epoch:  # hacemos esta verificacion pues solo tiene sentido iniciar en una epoca diferente cuando existen pesos iniciales (para reanudar)
                continue

            if epoch_index != 0 and self.with_lr_decay == True and epoch_index % self.frecuency_lr_decay == 0:
                self.learning_rate *= 0.1
            elif self.with_lr_decay == False:
                decreaseNow = self.experimentsRepo.ObtenerDecreaseNow()
                increaseNow = self.experimentsRepo.ObtenerIncreaseNow()
                if decreaseNow == True:
                    self.experimentsRepo.UpdateLearningRate(self.learning_rate)
                    self.experimentsRepo.SetFalseDecreaseNow()
                    self.learning_rate *= 0.1
                    print("Decremento mandatorio, learning rate: " + str(self.learning_rate))
                elif  increaseNow == True:
                    self.experimentsRepo.UpdateLearningRate(self.learning_rate)
                    self.experimentsRepo.SetFalseIncreaseNow()
                    self.learning_rate /= 0.1


            newOrder = self.theano_rng.permutation(n=self.trainDataSetSize, size=(1,)),
            self.x = self.x[newOrder]
            self.y = self.y[newOrder]

            #Recorremos todo el dataset dividido en n Batches
            for batch_index in range(self.no_batchs_in_data_set):
                cost = self.train_model(batch_index, self.learning_rate)
                print("costo: " + str(cost) + " epoca: " + str(epoch_index) + " Batch: " +str(batch_index) +"/"+str(self.no_batchs_in_data_set)+" Learning Rate: " + str(self.learning_rate))
                self.logger.LogTrain(cost, str(epoch_index), str(batch_index), str(self.learning_rate))
                #self.logger.Log(str(cost), "costo", str(epoch_index), str(batch_index), id_train,
                #                "learning rate: " + str(self.learning_rate) + "," + extra_info)
            if (epoch_index + 1) % self.saveWeigthsFrecuency == 0:
                self.SaveWeights(epoch_index, batch_index, -1)

    def SaveWeights(self, epoch, batch, iteration, cost=0, error=0, costVal=0, errorVal=0, costTest=0, errorTest=0):
        allWeiths = {
            "conv1Values": self.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.conv1),
            "conv2Values": self.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.conv2),
            "conv3Values": self.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.conv3),
            "conv4Values": self.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.conv4),
            "conv5Values": self.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.conv5),
            "conv6Values": self.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.conv6),
            "FC1Values": self.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.FC_1)[0],
            "FC1BiasValues": self.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.FC_1)[1],
            "FC2Values": self.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.FC_2)[0],
            "FC2BiasValues": self.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.FC_2)[1],
            "SoftMax1Values": self.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.SoftMax_1)[0],
            "SoftMax1BiasValues": self.GetWeigthsValuesByLayer(LayerEnum.LayerEnum.SoftMax_1)[1]
        }

        hyper_params = "learning rate: " + str(self.learning_rate)
        # = [c1_v,c3_v,fc5v_v,fc5b_v,fc6v,fc6b_v]
        self.weigthts_service.SaveWeights(allWeiths,self.idExperiment, epoch, batch, iteration, hyper_params, cost, error, costVal,
                                          errorVal, costTest, errorTest)


        return


    def CalculateCost(self,noBatchsToEvaluate = -1):

        if noBatchsToEvaluate == -1:
           noBatchsToEvaluate = self.no_batchs_in_data_set
        sumaCost = 0.0
        for batch_index in range(noBatchsToEvaluate):
            cost = self.evaluate_model_with_cost(batch_index)
            print ("calculando costos: costo: "+str(cost)+" en batch: " + str(batch_index))
            sumaCost=sumaCost + cost
        promedio = sumaCost / noBatchsToEvaluate
        return promedio

    def CalculateError(self,noBatchsToEvaluate = -1):

        if noBatchsToEvaluate == -1:
           noBatchsToEvaluate = self.no_batchs_in_data_set

        sumaCost = 0.0
        for batch_index in range(noBatchsToEvaluate):
            error = self.evaluate_model_with_error(batch_index)
            print ("calculando costos: errores: "+str(error)+" en batch: " + str(batch_index))
            sumaCost=sumaCost + error
        promedio = sumaCost / noBatchsToEvaluate
        return promedio
"""
        self.evaluation_model_with_errors = theano.function(
            [index],
            errors,
            givens={
                x: YimgLetrasShared[index * batch_size: (index + 1) * batch_size],
                y: YimgLetrasShared[index * batch_size: (index + 1) * batch_size]
            }
        )
"""

