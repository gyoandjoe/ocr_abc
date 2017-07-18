__author__ = 'win-g'
import sqlite3

class AnalisysRepo(object):
    def __init__(self, data_base):
        """
        :rtype : object
        """
        self.Database_Name = data_base

    def ObtenerLogsTraining(self, id_experiment):
        conn = sqlite3.connect(self.Database_Name)
        c = conn.cursor()
        query = "select * from LogTraining where BatchIndex >= 0 and IdExperiment =? "

        c.execute(query, [str(id_experiment)])
        registros= c.fetchall()
        # Save (commit) the changes
        # conn.commit()

        # We can also close the connection if we are done with it.
        # Just be sure any changes have been committed or they will be lost.
        conn.close()
        return registros

    def GetWeigthsByXIdExperiment(self, id_experiment):
        conn = sqlite3.connect(self.Database_Name)
        c = conn.cursor()
        query = "select * from Weights where IdExperiment =?"
        c.execute(query, [str(id_experiment)])
        registros = c.fetchall()
        conn.close()
        return registros


    def CreateLearningCurveAnalysis(self, idExperiment, IdWeigths):
        return

    def UpdateLearningCurveErrorXNoExamp(self,id_Analisys, noExperiments, dataSetSize, error,cost, tipoDataSet):
        conn = sqlite3.connect(self.Database_Name)
        c = conn.cursor()

        query = "INSERT INTO LearningCurveXNoExamp  VALUES (NULL,{0},{1},{2},'{3}',{4},{5})".format(noExperiments,cost,error,tipoDataSet,dataSetSize,id_Analisys)
        c.execute(query)
        conn.commit()
        conn.close()
        return

    def UpdateLearningCurveCostXNoExamp(self,id_Analisys, noExperiments, dataSetSize, cost, tipoDataSet):
        conn = sqlite3.connect(self.Database_Name)
        c = conn.cursor()
        query = "UPDATE LearningCurveXNoExamp SET NoExperiments = {0}, Cost = {1}, TipoDataSet = {2}, DataSetSize = {3} WHERE Id = {4}".format(noExperiments,cost,tipoDataSet,dataSetSize,id_Analisys)

        c.execute(query)
        conn.commit()
        conn.close()
        return

    def GetDataLCXIdAnalisys(self, id_analisys, tipo_data_set):
        conn = sqlite3.connect(self.Database_Name)
        c = conn.cursor()

        query = "select * from LearningCurveXNoExamp where IdLearningCurveAnalysis = ? and TipoDataSet = ? order by DataSetSize"

        c.execute(query, [id_analisys,str(tipo_data_set)])
        registros= c.fetchall()

        conn.close()
        return registros

