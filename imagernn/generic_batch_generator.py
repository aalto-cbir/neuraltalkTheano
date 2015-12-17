import numpy as np
#import gnumpy as gp
gp = np
import code
import time
from numbapro 	import cuda
from imagernn.utils import merge_init_structs, initw, initwG, accumNpDicts
from imagernn.lstm_generator import LSTMGenerator
from imagernn.rnn_generator import RNNGenerator

def decodeGenerator(generator):
  if generator == 'lstm':
    return LSTMGenerator
  if generator == 'rnn':
    return RNNGenerator
  else:
    raise Exception('generator %s is not yet supported' % (base_generator_str,))
