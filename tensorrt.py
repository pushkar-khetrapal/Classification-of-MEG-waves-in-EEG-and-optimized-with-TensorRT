import numpy as np
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import tensorflow.compat.v1.keras.backend as K
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

from tensorflow.python.compiler.tensorrt import trt_convert as tftrt

import copy
import time
from enum import Enum
from optparse import OptionParser
from pdb import set_trace


class EEG_FrozenGraph(object):
    def __init__(self, model):
        shape = (None, 33, 1)  # shape of input

        with K.get_session() as sess:
            place = []
            for i in range(10):   ## creating placeholder for 10 inputs
                place.append(tf.placeholder(tf.float32, shape, "tensor_{}".format(i)))
            K.set_learning_phase(0)
            y_tensor = model(place)  # raw input to create graph
            y_name = [y_tensor[0].name[:-2]]
            graph = sess.graph.as_graph_def()
            graph = tf.graph_util.convert_variables_to_constants(sess, graph, y_name)
            graph = tf.graph_util.remove_training_nodes(graph)
        
        nm = [] ## giving name to every input
        for i in range(10):
            nm.append("tensor_{}".format(i))
            print(nm)
            self.x_name = nm
            self.y_name = y_name
            self.frozen = graph

class TfEngine(object):
    def __init__(self, graph):
        g = tf.Graph()
        with g.as_default():
        a,b,c,d,e,f,q,h,i,j, y = tf.import_graph_def(
            graph_def=graph.frozen, return_elements=graph.x_name + graph.y_name)  ##this return 10 input and 1 output sample
        # setting the 10 input and 1 output
        self.x0_ten = a.outputs[0]
        self.x1_ten = b.outputs[0]
        self.x2_ten = c.outputs[0]
        self.x3_ten = d.outputs[0]
        self.x4_ten = e.outputs[0]
        self.x5_ten = f.outputs[0]
        self.x6_ten = q.outputs[0]
        self.x7_ten = h.outputs[0]
        self.x8_ten = i.outputs[0]
        self.x9_ten = j.outputs[0]
        self.y_tensor = [y.outputs[0]]
        
        config = tf.ConfigProto(gpu_options=
        tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
        allow_growth=True))

        self.sess = tf.Session(graph=g, config=config)

  def infer(self, x): 
        for i in range(10):
            t0 = time.time()
            y0 = self.sess.run(self.y_tensor,
                feed_dict={self.x0_ten: x[0], self.x1_ten: x[1],self.x2_ten: x[2],
                        self.x3_ten: x[3],self.x4_ten: x[4], self.x5_ten: x[5],
                        self.x6_ten: x[6], self.x7_ten: x[7],self.x8_ten: x[8], 
                        self.x9_ten: x[9]})  ## inferencing on frozen graph of tensorflow
        return y0, time.time() - t0

class TftrtEngine(TfEngine):
    def __init__(self, graph, batch_size, precision):
        tftrt_graph = tftrt.create_inference_graph(
        graph.frozen,
        outputs=graph.y_name,
        max_batch_size=batch_size,
        max_workspace_size_bytes=1 << 25,
        precision_mode=precision,
        minimum_segment_size=2)   ## creating the tensorRT inference graph

        opt_graph = copy.deepcopy(graph)
        opt_graph.frozen = tftrt_graph
        super(TftrtEngine, self).__init__(opt_graph)
        #self.batch_size = batch_size

  def infer(self, x):

    # running the graph for 10 times beacause the initial pass is just a warm-up
    for i in range(10):
        t0 = time.time()
        y0 = self.sess.run(self.y_tensor,
            feed_dict={self.x0_ten: x[0], self.x1_ten: x[1],self.x2_ten: x[2],
                    self.x3_ten: x[3],self.x4_ten: x[4], self.x5_ten: x[5],
                    self.x6_ten: x[6], self.x7_ten: x[7],self.x8_ten: x[8], 
                    self.x9_ten: x[9]})   ## inferencing through tensorRT graph

    return y0, time.time() - t0