# coding=utf-8

from __future__ import unicode_literals

import logging
import codecs
import re
import os
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

log = logging.getLogger(__name__)

Label_Line = ["S004001_定期报告", "S004002_重大事项", "S004003_交易提示", "S004004_IPO", "S004005_增发",
              "S004006_配股", "S004007_股权股本", "S004008_一般公告"]


class CNNTitleClassifier(object):
    def __init__(self, cnn_model, cnn_vocabulary):
        self.cnn_model = cnn_model
        self.cnn_vocabulary = cnn_vocabulary
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(self.cnn_vocabulary)

        self.graph = tf.Graph()
        self.session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.session = tf.Session(config=self.session_conf, graph=self.graph)

        with self.graph.as_default():
            with self.session.as_default():
                graph_def = tf.GraphDef()
                with tf.gfile.FastGFile(self.cnn_model, 'rb') as f:
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')
                    self.predictions = self.graph.get_tensor_by_name('output/predictions:0')
                    self.graph.finalize()

    def clean_str(self, s):
        """Clean sentence"""
        string = re.sub(r"\s{2,}", " ", s)
        return string.strip()

    def classify(self, x):
        x_test = [self.clean_str(x)]
        x_test = np.array(list(self.vocab_processor.transform(x_test)))

        with self.graph.as_default():
            with self.session.as_default():
                pre = self.session.run(self.predictions, {'input_x:0': x_test, 'dropout_keep_prob:0': 1.0})
                return Label_Line[pre[0]]

    def classify_batch(self, x, batch_size=100):
        pass


if __name__ == '__main__':
    title_classifier = CNNTitleClassifier(cnn_model="/opt/algorithm/pb_models/title_classify_v35b1e4a2.pb",
                                          cnn_vocabulary="/opt/algorithm/pb_models/cnn_vocabulary_v09b10736.pkl")
    cl_set = '/home/zhwpeng/abc/nlp/notice/data/test_data/'+Label_Line[7]+'.txt'
    with codecs.open(cl_set, "r", encoding='UTF-8') as f:
        data = f.read()
    data = data.split('\n')
    print "quantity of texts to be predict: %d"%len(data)
    c, d = 0, 0
    for line in data:
        result = title_classifier.classify(line)
        # print line, result
        if result == Label_Line[7]:
            c += 1
        elif result == Label_Line[1]:
            d += 1
        else:
            print result
    print 'right predict num: %d'%c
    print '重大事项的预测', d
