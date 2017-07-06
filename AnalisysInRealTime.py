import Domain.Analysis.Analizador as Analizador

__author__ = 'win-g'

analizador = Analizador.Analizador('BD\\OCR_ABC.db')

analizador.AnalizarInRealTIme(1,1.0)
print ("Fin analisis rapido :)")