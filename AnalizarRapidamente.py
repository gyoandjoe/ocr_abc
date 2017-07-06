import Domain.Analysis.Analizador as Analizador

__author__ = 'win-g'

analizador = Analizador.Analizador('BD\\OCR_ABC.db')

analizador.AnalizarRapidamente(1)
print ("Fin analisis rapido :)")