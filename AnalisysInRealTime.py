import Domain.Analysis.Analizador as Analizador

__author__ = 'win-g'

analizador = Analizador.Analizador('BD\\OCR_ABC.db')

analizador.AnalizarInRealTIme(id_experiment=2,velocity_update=1.0)
print ("Fin analisis rapido :)")