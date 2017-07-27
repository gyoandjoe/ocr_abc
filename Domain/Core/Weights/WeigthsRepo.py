import datetime

__author__ = 'win-g'

import _pickle as cPickle
import sqlite3
import os

class WeigthsRepo(object):
    def __init__(self, database_name,folder_path):
        self.database_name = database_name
        self.folder_path = folder_path


    def SaveWeights(self, weights, id_experiment, epoch, batch, iteracion, hyper_params, trainCost = 0,trainError = 0,costVal=0,errorVal=0,costTest =0,errorTest=0):
        fecha = datetime.datetime.now().strftime("%d-%m-%Y %H %M %S")
        fileName = "weights_idExp " + str(id_experiment) + "_fecha " + fecha + "_epoch " + str(epoch)+ "_batch " + str(batch) + "_iter " + str(iteracion) + '.pkl'

        self.folder_path = self.folder_path + "\\" + str(id_experiment)
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        fullName = self.folder_path + "\\" + fileName

        f = open(fullName, 'w+b')
        cPickle.dump(weights, f, protocol=2)
        f.close()

        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "INSERT INTO Weights VALUES (NULL,\'{0}\',{1},\'{2}\',{3},{4},{5},\'{6}\',{7},{8},{9},{10},{11},{12})".format(
            fullName,
            str(id_experiment),
            fecha,
            epoch,
            batch,
            iteracion,
            hyper_params,
            trainError,
            trainCost,
            costVal,
            errorVal,
            costTest,
            errorTest)

        c.execute(query)
        conn.commit()
        conn.close()
        return

    def GetWeithsInfoById(self, id_weigths):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        c.execute("SELECT * FROM Weights WHERE Id = ?",[str(id_weigths)])
        registro = c.fetchone()
        conn.close()

        return registro

    def GetGeigthsByExperimentId(self, id_experiment):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        c.execute("SELECT * FROM Weights WHERE IdExperiment = ?",[str(id_experiment)])
        registros = c.fetchall()
        conn.close()

        return registros

    def UpdateCostForWeigth(self, id_weigth, cost):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "UPDATE Weights SET TrainCost = {0} WHERE Id = {1}".format(cost,id_weigth)
        c.execute(query)
        conn.commit()
        conn.close()
        return

    def UpdateErrorForWeigth(self, id_weigth, error):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "UPDATE Weights SET TrainError = {0} WHERE Id = {1}".format(error,id_weigth)
        c.execute(query)
        conn.commit()
        conn.close()
        return

    def UpdateTestCostForWeigth(self, id_weigth, costTest):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "UPDATE Weights SET TestCost = {0} WHERE Id = {1}".format(costTest,id_weigth)
        c.execute(query)
        conn.commit()
        conn.close()
        return

    def UpdateTestErrorForWeigth(self, id_weigth, errorTest):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "UPDATE Weights SET TestError = {0} WHERE Id = {1}".format(errorTest,id_weigth)
        c.execute(query)
        conn.commit()
        conn.close()
        return

    def UpdateValCostForWeigth(self, id_weigth, costVal):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "UPDATE Weights SET ValidCost = {0} WHERE Id = {1}".format(costVal,id_weigth)
        c.execute(query)
        conn.commit()
        conn.close()
        return

    def UpdateValErrorForWeigth(self, id_weigth, errorVal):
        conn = sqlite3.connect(self.database_name)
        c = conn.cursor()
        query = "UPDATE Weights SET ValidError = {0} WHERE Id = {1}".format(errorVal,id_weigth)
        c.execute(query)
        conn.commit()
        conn.close()
        return