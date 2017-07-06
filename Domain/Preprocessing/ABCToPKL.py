__author__ = 'Gyo'
import numpy as np
from PIL import Image

import _pickle as cPickle

class ABCToPKL(object):

    def StartProcess(self, serializedData,fullFileName):
        print ('PROCESSING -- Serialize Numpy Array to File: ' + fullFileName)
        f = open(fullFileName, 'w+b')
        cPickle.dump(serializedData, f, protocol=2)
        f.close()
        print ('.............. OK: ' + fullFileName)
        print ("Serialize DataSet -- OK")
        return fullFileName
