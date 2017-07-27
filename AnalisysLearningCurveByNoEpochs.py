import Domain.Analysis.Analizador as Analizador

__author__ = 'win-g'

analizador = Analizador.Analizador('BD\\OCR_ABC.db')

#analizador.GraficarCostosXEpocaXDataSet(2)

analizador.GraficarErrorsXEpocaXDataSet(2)