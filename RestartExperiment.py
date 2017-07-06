from Domain.Experiments import ExperimentsRepo

idExperiment = 3

experimenRepo = ExperimentsRepo.ExperimentsRepo('BD\\OCR_ABC.db', idExperiment)
experimenRepo.SetFalseDecreaseNow()
