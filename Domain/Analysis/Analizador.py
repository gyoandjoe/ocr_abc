import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Domain.Analysis.AnalisysRepo as Analisys_Repo
import Domain.Experiments.ExperimentsRepo as ExperimentsRepo
import Domain.Core.Weights.WeigthsService as WeigthsService
import Domain.Core.Weights.WeigthsRepo as WeigthsRepo
import Domain.Arquitectures.DBenGurionOCR as D_BenGurionOCR


class Analizador(object):
    def __init__(self, data_base):
        self.analisys_repo = Analisys_Repo.AnalisysRepo(data_base=data_base )

    def AnalizarInRealTIme(self, id_experiment, velocity_update=0.05):
        #plt.axis([0, 10, 0, 1])
        plt.ion()

        while True:

            registros = self.analisys_repo.ObtenerLogsTraining(id_experiment)
            df = pd.DataFrame(registros, columns=['Id','IdExperiment','FechaRegistro','Costo','TipoLog','EpochIndex','BatchIndex','LearningRate','Contenido']) #,dtype=[('Contenido', np.float64)]
            df.convert_objects(convert_numeric=True)
            df['Costo'] = df['Costo'].astype(np.float64)
            grouped = df.groupby('EpochIndex')
            #for registro in df['contenido'].values:
            #    print registro
            xx=grouped.groups.keys()

            yy=grouped['Costo'].mean().values

            x = np.asarray(list(xx), dtype=int)
            y = np.asarray(yy, dtype=np.float64)
            plt.cla()
            plt.plot(x,y,'-')
            #print y
            #plt.show()


            plt.pause(velocity_update)

    def AnalizarRapidamente(self, id_experiment):
        registros = self.analisys_repo.ObtenerLogsTraining(id_experiment)
        df = pd.DataFrame(registros, columns=['Id','IdExperiment','FechaRegistro','Costo','TipoLog','EpochIndex','BatchIndex','LearningRate','Contenido']) #,dtype=[('Contenido', np.float64)]
        df.convert_objects(convert_numeric=True)
        df['Costo'] = df['Costo'].astype(np.float64)
        grouped = df.groupby('EpochIndex')
        #for registro in df['contenido'].values:
        #    print registro
        xx=grouped.groups.keys()

        yy=grouped['Costo'].mean().values

        x = np.asarray(list(xx), dtype=int)
        y = np.asarray(yy, dtype=np.float64)
        plt.plot(x,y,'-')
            #print y
        plt.show()

    def BuildWeigthsErrorAndCost(self, id_experiment, bd, weigths_path,layers_metaData,train_batch_size,raw_train_set,logger,weigths_service,experimentsRepo,raw_validation_set,validation_batch_size):
        # Calculo total del costo y error por todos los datos, pero por cada conjunto de pesos generados
        #experiment_repo = Experiments.ExperimentsRepo.ExperimentsRepo(bd, id_experiment)
        #weigths_repo = WeigthsRepo.WeigthsRepo(bd, weigths_path)


        weigthsOfExperiment = weigths_service.GetListOfWeightsByIdExperiment(id_experiment)


        print('--------------------------- TEST SET -------------------------------------------------')

        print("Calculando Errores en validationSet y costos en validation set")

        for w in weigthsOfExperiment:
            ws = weigths_service.LoadRawWeigths(w[0])
            DBG_OCR = D_BenGurionOCR.DBenGurionOCR.Validator(
                id_experiment=id_experiment,
                layers_metaData=layers_metaData,
                batch_size=validation_batch_size,
                raw_data_set=raw_validation_set,
                logger=logger,
                weigthts_service=weigths_service,
                experimentsRepo=experimentsRepo,
                initial_weights=ws
            )



            averageError = DBG_OCR.CalculateError()
            weigths_service.UpdateTestErrorWeigth(w[0], averageError)
            print("--------[Test Set] El error promedio es: " + str(averageError))


            averageCost = DBG_OCR.CalculateCost()
            weigths_service.UpdateTestCostWeigth(w[0], averageCost)
            print("--------[Test Set] El costo promedio es: " + str(averageCost))




        print('--------------------------- TRAIN SET -------------------------------------------------')



        # Actualizamos los costos en el dataset entero con cada uno de los pesos de que se han generado durante un determinado experimento
        # Por cada conjunto de pesos obtenidos en el experimento, hacemos el calculo del costo y del error en el train set
        print("Calculando errores en trainSet y costos en trainSet")

        for w in weigthsOfExperiment:
            ws = weigths_service.LoadRawWeigths(w[0])
            DBG_OCR = D_BenGurionOCR.DBenGurionOCR.Validator(
                id_experiment=id_experiment,
                layers_metaData=layers_metaData,
                batch_size=train_batch_size,
                raw_data_set=raw_train_set,
                logger=logger,
                weigthts_service=weigths_service,
                experimentsRepo=experimentsRepo,
                initial_weights=ws
            )

            averageError = DBG_OCR.CalculateError()
            weigths_service.UpdateTrainErrorWeigth(w[0], averageError)
            print("--------[Train Set] El error promedio es: " + str(averageError))


            averageCost = DBG_OCR.CalculateCost()
            weigths_service.UpdateTrainCostWeigth(w[0], averageCost)
            print("--------[Train Set] El costo promedio es: " + str(averageCost))


        print("End Validation :)")


    def GraficarCostosXEpocaXDataSet(self, id_experiment):

        data_weigths = self.analisys_repo.GetWeigthsByXIdExperiment(id_experiment)
        data = pd.DataFrame(data_weigths, columns=['Id',
       'FileName',
       'IdExperiment',
       'FechaRegistro',
       'Epoch',
       'Batch',
       'Iteracion',
       'HyperParams',
       'TrainError',
       'TrainCost',
       'ValidCost',
       'ValidError',
       'TestCost',
       'TestError'])
        epochs = data['Epoch']
        trainCost = data['TrainCost']
        #valCost = data['CostVal']
        testCost = data['TestCost']

        xEpochs = np.asarray(epochs.values,  dtype=int)
        yTrain = np.asarray(trainCost.values, dtype=np.float64)
        #yVal = np.asarray(valCost.values, dtype=np.float64)
        yTest = np.asarray(testCost.values, dtype=np.float64)


        plt.plot(xEpochs,yTrain,'r--')
        #plt.plot(xEpochs,yVal,'bs')
        plt.plot(xEpochs,yTest,'g^')
        plt.title('Cost id experiment(' + str(id_experiment) + ')')
        plt.show()
        return

    def GraficarErrorsXEpocaXDataSet(self, id_experiment):
        data_weigths = self.analisys_repo.GetWeigthsByXIdExperiment(id_experiment)
        data = pd.DataFrame(data_weigths, columns=['Id','FileName',
       'IdExperiment',
       'FechaRegistro',
       'Epoch',
       'Batch',
       'Iteracion',
       'HyperParams',
       'TrainError',
       'TrainCost',
       'ValidCost',
       'ValidError',
       'TestCost',
       'TestError'])
        epochs = data['Epoch']
        trainError = data['TrainError']
        #valError = data['ValidError']
        testError = data['TestError']

        xEpochs = np.asarray(epochs.values,  dtype=int)
        yTrain = np.asarray(trainError.values, dtype=np.float64)
        #yVal = np.asarray(valError.values, dtype=np.float64)
        yTest = np.asarray(testError.values, dtype=np.float64)

        plt.plot(xEpochs,yTrain,'r--')
        #plt.plot(xEpochs,yVal,'bs')
        plt.plot(xEpochs,yTest,'g^')
        plt.title('Error id experiment(' + str(id_experiment)+')')
        plt.show()
        return


    def BuildLearningCurveAnalysisByExamples(self, id_experiment,id_Analisys, bd, id_weigths, folderWeigths,layers_metaData,raw_train_set,train_batch_size,raw_validation_set,validation_batch_size,logger,weigthts_service,experimentsRepo):
        weigths_repo = WeigthsRepo.WeigthsRepo(bd,folderWeigths)
        weigths_service = WeigthsService.WeigthsService(bd, weigths_repo)


        print('--------------------------- Train Set -------------------------------------------------')
        ws=weigths_service.LoadRawWeigths(id_weigths)
        DBG_OCR = D_BenGurionOCR.DBenGurionOCR.Validator(
            id_experiment = id_experiment,
            layers_metaData = layers_metaData,
            batch_size = train_batch_size,
            raw_data_set = raw_train_set,
            logger=logger,
            weigthts_service=weigthts_service,
            experimentsRepo=experimentsRepo,
            initial_weights = ws
        )

        no_batchs = DBG_OCR.totalDataSize // train_batch_size

        ar = Analisys_Repo.AnalisysRepo(data_base=bd)
        for i in range(1, no_batchs):
            l_data_Size = i * train_batch_size  # cantidad de ejemplos que se evaluaran

            print("Calculando Errores en validationSet y costos en Train set")
            averageError = DBG_OCR.CalculateError(noBatchsToEvaluate=i)

            print("--------[Train Set] El error promedio es: " + str(averageError) + " para " + str(l_data_Size) + " ejemplos")
            averageCost = DBG_OCR.CalculateCost(noBatchsToEvaluate=i)

            ar.UpdateLearningCurveErrorXNoExamp(id_Analisys, l_data_Size, DBG_OCR.totalDataSize, averageError, averageCost,'TrainSet')
            print("--------[Train Set] El costo promedio es: " + str(averageCost) + " para " + str(l_data_Size) + " ejemplos")

        print('--------------------------- Validation SET -------------------------------------------------')

        ws=weigths_service.LoadRawWeigths(id_weigths)
        DBG_OCR = D_BenGurionOCR.DBenGurionOCR.Validator(
            id_experiment = id_experiment,
            layers_metaData = layers_metaData,
            batch_size = validation_batch_size,
            raw_data_set = raw_validation_set,
            logger=logger,
            weigthts_service=weigthts_service,
            experimentsRepo=experimentsRepo,
            initial_weights = ws
        )
        no_batchs = DBG_OCR.totalDataSize // validation_batch_size

        ar = Analisys_Repo.AnalisysRepo(data_base=bd)

        for i in range(1, no_batchs):
            l_data_Size = i * validation_batch_size  # cantidad de ejemplos que se evaluaran

            print("Calculando Errores en test set y costos en test set")
            averageError = DBG_OCR.CalculateError(noBatchsToEvaluate=i)

            print("--------[Validation Set] El error promedio es: " + str(averageError) + " para " + str(l_data_Size) + " ejemplos")
            averageCost = DBG_OCR.CalculateCost(noBatchsToEvaluate=i)

            ar.UpdateLearningCurveErrorXNoExamp(id_Analisys, l_data_Size, DBG_OCR.totalDataSize, averageError, averageCost,'ValSet')
            print("--------[Validation Set] El costo promedio es: " + str(averageCost) + " para " + str(l_data_Size) + " ejemplos")

        return

    def GraficarLCXErrors(self, id_analisys):
        # TestSet
        data_raw_testset = self.analisys_repo.GetDataLCXIdAnalisys(id_analisys, "TestSet")
        data_testset = pd.DataFrame(data_raw_testset,
                                    columns=['Id', 'NoExperiments', 'Cost', 'Error', 'TipoDataSet', 'DataSetSize',
                                             'IdLearningCurveAnalysis'])
        testset_experiments = np.asarray(data_testset['NoExperiments'].values, dtype=int)
        testset_errors = np.asarray(data_testset['Error'].values, dtype=np.float64)
        plt.plot(testset_experiments, testset_errors, 'g-')

        # ValSet
        data_raw_valset = self.analisys_repo.GetDataLCXIdAnalisys(id_analisys, "ValSet")
        data_valset = pd.DataFrame(data_raw_valset,
                                   columns=['Id', 'NoExperiments', 'Cost', 'Error', 'TipoDataSet', 'DataSetSize',
                                            'IdLearningCurveAnalysis'])
        valset_experiments = np.asarray(data_valset['NoExperiments'].values, dtype=int)
        valset_errors = np.asarray(data_valset['Error'].values, dtype=np.float64)
        plt.plot(valset_experiments, valset_errors, 'r-')

        # TrainSet
        data_raw_trainset = self.analisys_repo.GetDataLCXIdAnalisys(id_analisys, "TrainSet")
        data_trainset = pd.DataFrame(data_raw_trainset,
                                     columns=['Id', 'NoExperiments', 'Cost', 'Error', 'TipoDataSet', 'DataSetSize',
                                              'IdLearningCurveAnalysis'])
        trainset_experiments = np.asarray(data_trainset['NoExperiments'].values, dtype=int)
        trainset_errors = np.asarray(data_trainset['Error'].values, dtype=np.float64)
        plt.plot(trainset_experiments, trainset_errors, 'b-')

        # valset_costs =np.asarray(data_valset['Cost'].values, dtype=np.float64)
        # plt.plot(valset_experiments,valset_costs,'bs')
        # plt.plot(valset_experiments,valset_errors,'g^')
        plt.title('Error Analysis (' + str(id_analisys) + ')')
        plt.show()

        return

    def GraficarLCXCosts(self, id_analisys):
        # TestSet
        data_raw_testset = self.analisys_repo.GetDataLCXIdAnalisys(id_analisys, "TestSet")
        data_testset = pd.DataFrame(data_raw_testset,
                                    columns=['Id', 'NoExperiments', 'Cost', 'Error', 'TipoDataSet', 'DataSetSize',
                                             'IdLearningCurveAnalysis'])
        testset_experiments = np.asarray(data_testset['NoExperiments'].values, dtype=int)
        testset_costs = np.asarray(data_testset['Cost'].values, dtype=np.float64)
        #plt.plot(testset_experiments, testset_costs, 'r-')

        # ValSet
        data_raw_valset = self.analisys_repo.GetDataLCXIdAnalisys(id_analisys, "ValSet")
        data_valset = pd.DataFrame(data_raw_valset,
                                   columns=['Id', 'NoExperiments', 'Cost', 'Error', 'TipoDataSet', 'DataSetSize',
                                            'IdLearningCurveAnalysis'])
        valset_experiments = np.asarray(data_valset['NoExperiments'].values, dtype=int)
        valset_costs = np.asarray(data_valset['Cost'].values, dtype=np.float64)
        plt.plot(valset_experiments, valset_costs, 'g-')

        # TrainSet
        data_raw_trainset = self.analisys_repo.GetDataLCXIdAnalisys(id_analisys, "TrainSet")
        data_trainset = pd.DataFrame(data_raw_trainset,
                                     columns=['Id', 'NoExperiments', 'Cost', 'Error', 'TipoDataSet', 'DataSetSize',
                                              'IdLearningCurveAnalysis'])
        trainset_experiments = np.asarray(data_trainset['NoExperiments'].values, dtype=int)
        trainset_costs = np.asarray(data_trainset['Cost'].values, dtype=np.float64)
        plt.plot(trainset_experiments, trainset_costs, 'b-')

        plt.title('Cost Analysis (' + str(id_analisys) + ')')
        plt.show()

        return

