import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Domain.Analysis.AnalisysRepo as AnalisysRepo


class Analizador(object):
    def __init__(self, data_base):
        self.analisys_repo = AnalisysRepo(data_base)


    def AnalizarInRealTIme(self, id_experiment,velocity=0.05):
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


            plt.pause(velocity)


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

    def GraficarCostosXEpocaXDataSet(self, id_experiment):
        data_weigths = self.analisys_repo.GetWeigthsByXIdExperiment(id_experiment)
        data = pd.DataFrame(data_weigths,
                            columns=['Id', 'IdExperiment', 'Fecha', 'Epoch', 'Batch', 'Iteration', 'Cost', 'FileName',
                                     'HyperParams', 'Error', 'CostVal', 'ErrorVal', 'CostTest', 'ErrorTest'])
        epochs = data['Epoch']
        trainCost = data['Cost']
        valCost = data['CostVal']
        testCost = data['CostTest']

        xEpochs = np.asarray(epochs.values, dtype=int)
        yTrain = np.asarray(trainCost.values, dtype=np.float64)
        yVal = np.asarray(valCost.values, dtype=np.float64)
        yTest = np.asarray(testCost.values, dtype=np.float64)

        plt.plot(xEpochs, yTrain, 'r--')
        plt.plot(xEpochs, yVal, 'bs')
        plt.plot(xEpochs, yTest, 'g^')
        plt.title('Cost id experiment(' + str(id_experiment) + ')')
        plt.show()
        return

    def GraficarErrorsXEpocaXDataSet(self, id_experiment):
        data_weigths = self.analisys_repo.GetWeigthsByXIdExperiment(id_experiment)
        data = pd.DataFrame(data_weigths,
                            columns=['Id', 'IdExperiment', 'Fecha', 'Epoch', 'Batch', 'Iteration', 'Cost', 'FileName',
                                     'HyperParams', 'Error', 'CostVal', 'ErrorVal', 'CostTest', 'ErrorTest'])
        epochs = data['Epoch']
        trainError = data['Error']
        valError = data['ErrorVal']
        testError = data['ErrorTest']

        xEpochs = np.asarray(epochs.values, dtype=int)
        yTrain = np.asarray(trainError.values, dtype=np.float64)
        yVal = np.asarray(valError.values, dtype=np.float64)
        yTest = np.asarray(testError.values, dtype=np.float64)

        plt.plot(xEpochs, yTrain, 'r--')
        plt.plot(xEpochs, yVal, 'bs')
        plt.plot(xEpochs, yTest, 'g^')
        plt.title('Error id experiment(' + str(id_experiment) + ')')
        plt.show()
        return

    def BuildLearningCurveAnalysisByExamples(self, id_experiment, id_Analisys, bd, id_weigths, folderWeigths):
        experiment_repo = Experiments.ExperimentsRepo(bd, id_experiment)
        weigths_service = WeigthsService.WigthsService(bd, folderWeigths, id_experiment,
                                                       TypeInitWFunctionEnum.TypeInitWFunctionEnum.UniformDistribution)


        '--------------------------- VALIDATION SET -------------------------------------------------'

        totalDataSize = 10000
        lBatcthSize = 500
        no_batchs = totalDataSize // lBatcthSize

        lvt = LenetValidationAndTest(
            dataset_file=experiment_repo.ObtenerArchivoDataSet(),
            batch_size=lBatcthSize,  # 5000 en total
            dataset_size=totalDataSize,
            weigthts_service=weigths_service,
            tipoDataSet=TiposDataSetEnum.TiposDataSetEnum.validationSet
        )
        ar = AnalisysRepo.AnalisysRepo(data_base=bd)
        for i in range(1, no_batchs):
            l_data_Size = i * lBatcthSize  # cantidad de ejemplos que se evaluaran

            print("Calculando Errores en validationSet y costos en validation set")
            averageError = lvt.CalculateError(id_weigths=id_weigths, noBatchsToEvaluate=i)

            print("--------[Validation Set] El error promedio es: " + str(averageError) + " para " + str(l_data_Size) + " ejemplos")
            averageCost = lvt.CalculateCost(id_weigths=id_weigths, noBatchsToEvaluate=i)

            ar.UpdateLearningCurveErrorXNoExamp(id_Analisys, l_data_Size, totalDataSize, averageError, averageCost,
                                                'ValSet')
            print
            "--------[Validation Set] El costo promedio es: " + str(averageCost) + " para " + str(
                l_data_Size) + " ejemplos"

        print
        '--------------------------- TEST SET -------------------------------------------------'

        totalDataSize = 10000
        lBatcthSize = 500
        no_batchs = totalDataSize // lBatcthSize

        lvt = LenetValidationAndTest(
            dataset_file=experiment_repo.ObtenerArchivoDataSet(),
            batch_size=lBatcthSize,  # 5000 en total
            dataset_size=totalDataSize,
            weigthts_service=weigths_service,
            tipoDataSet=TiposDataSetEnum.TiposDataSetEnum.testSet
        )

        ar = AnalisysRepo.AnalisysRepo(data_base=bd)

        for i in xrange(1, no_batchs):
            l_data_Size = i * lBatcthSize  # cantidad de ejemplos que se evaluaran

            print
            "Calculando Errores en test set y costos en test set"
            averageError = lvt.CalculateError(id_weigths=id_weigths, noBatchsToEvaluate=i)

            print
            "--------[Test Set] El error promedio es: " + str(averageError) + " para " + str(lBatcthSize) + " ejemplos"
            averageCost = lvt.CalculateCost(id_weigths=id_weigths, noBatchsToEvaluate=i)

            ar.UpdateLearningCurveErrorXNoExamp(id_Analisys, l_data_Size, totalDataSize, averageError, averageCost,
                                                'TestSet')

            print
            "--------[Test Set] El costo promedio es: " + str(averageCost) + " para " + str(lBatcthSize) + " ejemplos"

        return
        print
        '--------------------------- TRAIN SET -------------------------------------------------'

        totalDataSize = 50000
        lBatcthSize = 500
        no_batchs = totalDataSize // lBatcthSize

        lvt = LenetValidationAndTest(
            dataset_file=experiment_repo.ObtenerArchivoDataSet(),
            batch_size=lBatcthSize,
            dataset_size=totalDataSize,
            weigthts_service=weigths_service,
            tipoDataSet=TiposDataSetEnum.TiposDataSetEnum.trainSet
        )

        ar = AnalisysRepo.AnalisysRepo(data_base=bd)

        for i in xrange(1, no_batchs):
            l_data_Size = i * lBatcthSize  # cantidad de ejemplos que se evaluaran

            print
            "Calculando Errores en Train set y costos en Train set"
            averageError = lvt.CalculateError(id_weigths=id_weigths, noBatchsToEvaluate=i)

            print
            "--------[Train Set] El error promedio es: " + str(averageError) + " para " + str(l_data_Size) + " ejemplos"
            averageCost = lvt.CalculateCost(id_weigths=id_weigths, noBatchsToEvaluate=i)

            ar.UpdateLearningCurveErrorXNoExamp(id_Analisys, l_data_Size, totalDataSize, averageError, averageCost,
                                                'TrainSet')
            print
            "--------[Train Set] El costo promedio es: " + str(averageCost) + " para " + str(l_data_Size) + " ejemplos"

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
        plt.plot(testset_experiments, testset_costs, 'g-')

        # ValSet
        data_raw_valset = self.analisys_repo.GetDataLCXIdAnalisys(id_analisys, "ValSet")
        data_valset = pd.DataFrame(data_raw_valset,
                                   columns=['Id', 'NoExperiments', 'Cost', 'Error', 'TipoDataSet', 'DataSetSize',
                                            'IdLearningCurveAnalysis'])
        valset_experiments = np.asarray(data_valset['NoExperiments'].values, dtype=int)
        valset_costs = np.asarray(data_valset['Cost'].values, dtype=np.float64)
        plt.plot(valset_experiments, valset_costs, 'r-')

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







