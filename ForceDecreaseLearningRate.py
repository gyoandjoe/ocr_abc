from Domain.Experiments import ExperimentsRepo

idExperiment = 2

experimenRepo = ExperimentsRepo.ExperimentsRepo('BD\\OCR_ABC.db', idExperiment)
experimenRepo.SetTrueDecreaseNow()

