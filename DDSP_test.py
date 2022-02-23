'''
#@title Install and import dependencies

%tensorflow_version 2.x
!pip install -qU ddsp

# Ignore a bunch of deprecation warnings
import warnings
warnings.filterwarnings("ignore")

import ddsp
import ddsp.training
from ddsp.colab.colab_utils import play, specplot, DEFAULT_SAMPLE_RATE
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sample_rate = DEFAULT_SAMPLE_RATE  # 16000
'''