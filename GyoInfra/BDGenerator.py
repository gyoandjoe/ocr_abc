__author__ = 'win-g'

import sqlite3



conn = sqlite3.connect('..\BD\OCR_ABC.db')

c = conn.cursor()


c.execute('''CREATE TABLE Experiments
             (Id INTEGER PRIMARY KEY, 
             TrainDataSetFile text, 
             TestDataSetFile text, 
             BatchSize real, 
             InitialLearningRate real, 
             Status text, 
             BatchActual integer,
             MaxEpoch INTEGER,
             EpochFrecSaveWeights INTEGER,
             WithLRDecay INTEGER, 
             EpochFrecLRDecay INTEGER,
             ShouldDecreaseNow INTEGER,
             ShouldIncreaseNow INTEGER
             )''')

c.execute('''CREATE TABLE Weights
             (Id INTEGER PRIMARY KEY,
              FileName text,
             IdExperiment integer, 
             FechaRegistro text, 
             Epoch integer, 
             Batch integer, 
             Iteracion integer,                            
             HyperParams text,              
             TrainError real,
             TrainCost real,
             ValidCost real,
             ValidError real,
             TestCost real,
             TestError real)''')
#['Id','IdExperiment','Fecha','Epoch','Batch','Iteration','Cost', 'FileName' ,'HyperParams','Error','CostVal','ErrorVal','CostTest','ErrorTest']


c.execute('''CREATE TABLE LogExperiment
             (Id INTEGER PRIMARY KEY, 
             IdExperiment integer, 
             FechaRegistro text, 
             Contenido text,  
             TipoLog text, 
             EpochIndex integer, 
             BatchIndex integer,
             InfoExtra text,
             Referencia text)''')

c.execute('''CREATE TABLE LogTraining
             (Id INTEGER PRIMARY KEY, 
             IdExperiment integer, 
             FechaRegistro text, 
             Costo REAL,  
             TipoLog text, 
             EpochIndex integer, 
             BatchIndex integer,
             LearningRate REAL,
             Contenido text
             )''')

#LogExperiment ['Id','IdExperient','Fecha','Contenido','TipoLog','EpochIndex','BatchIndex','ExtraInfo','Referencia']

c.execute('''CREATE TABLE LearningCurveAnalysisXNoExp (Id INTEGER PRIMARY KEY, IdExperiment integer, IdWeigths INTEGER)''')

c.execute('''CREATE TABLE LearningCurveXNoExamp (Id INTEGER PRIMARY KEY,
NoExperiments INTEGER, 
Cost REAL, 
Error REAL, 
TipoDataSet TEXT, 
DataSetSize INTEGER,
IdLearningCurveAnalysis integer)''')


#Agregar intentoId, batch, iteracion

# Save (commit) the changes
conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()


