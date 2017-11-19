import numpy as np
import tensorflow as tf

#import forward_model as fm
with tf.variable_scope("backward"):
    import backward_model as bm
