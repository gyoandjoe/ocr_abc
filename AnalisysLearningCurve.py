import Domain.Analysis.Analizador as Analizador

__author__ = 'win-g'

analizador = Analizador.Analizador('BD\\OCR_ABC.db')

analizador.GraficarLCXCosts(id_analisys=3)
print ("Fin analisis rapido :)")