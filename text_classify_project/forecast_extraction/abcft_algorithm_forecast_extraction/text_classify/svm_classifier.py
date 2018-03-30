# coding=utf-8

from __future__ import absolute_import

import logging
import re
import os
import jieba
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from abcft_algorithm_grpc import pb_models
from abcft_algorithm_forecast_extraction.text_classify.texttype import TextType, BaseClassifier, \
    Label_Line, Label_Line2

log = logging.getLogger(__name__)


class SVMTextClassifier(BaseClassifier):
    def __init__(self, model, model_hash,  # 研报的性质类型分类器模型
                 vocabulary_path, vocabulary_path_hash,
                 all_features, all_features_hash,
                 selected_features, selected_features_hash,
                 model_in, model_in_hash,  # 研报的行业类型分类器模型
                 all_features_in, all_features_in_hash,
                 selected_features_in, selected_features_in_hash):
        self.vocabulary_path = pb_models.get_model(vocabulary_path, vocabulary_path_hash)
        self.model = pickle.load(open(pb_models.get_model(model, model_hash), 'rb'))
        self.all_features = pickle.load(open(pb_models.get_model(all_features, all_features_hash), 'rb'))
        self.selected_features = np.load(pb_models.get_model(selected_features, selected_features_hash))

        self.model_in = pickle.load(open(pb_models.get_model(model_in, model_in_hash), 'rb'))
        self.all_features_in = pickle.load(open(pb_models.get_model(all_features_in, all_features_in_hash), 'rb'))
        self.selected_features_in = np.load(pb_models.get_model(selected_features_in, selected_features_in_hash))

        # jieba.load_userdict(self.vocabulary_path)
        # jieba.initialize()
        # jieba.set_dictionary(self.vocabulary_path)

    def is_chinese(self, uchar):
        """判断一个unicode是否是汉字"""
        X, Y = [u'\u4e00', u'\u9fa5']  # unicode 前面加u
        if uchar >= X and uchar <= Y:
            return True
        else:
            return False

    def preprocess(self, fulltext, title):
        out_text = []
        stopwords_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'stopwords.txt')
        stopwords = []
        # 获取停用词
        with open(stopwords_path) as fi:
            for line in fi.readlines():
                stopwords.append(line.strip())
        fulltext = fulltext.split('\n')
        for line in fulltext:
        # for line in fulltext.readlines():
            if not len(line.strip()):
                continue
            seg_list = jieba.cut(line)
            for word in seg_list:
                if not self.is_chinese(word):
                    continue
                try:
                    word = word.encode('UTF-8')
                except:
                    continue
                if word not in stopwords:
                    if word != ' ':
                        out_text.append(word)
        fulltext = ' '.join(out_text)

        """Clean sentence"""
        string = re.sub(r"\s{2,}", " ", title)
        return fulltext, string.strip()

    def get_tfidf(self, count_v0, train_texts):
        counts_train = count_v0.fit_transform(train_texts)

        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(counts_train)

        feature_names = count_v0.get_feature_names()  # 关键字
        count_v0_df = pd.DataFrame(counts_train.toarray())
        tfidf_df = pd.DataFrame(tfidf.toarray())
        return count_v0_df, tfidf_df, feature_names

    def get_x_test(self, count_v0_df, feature_names, selected_features):
        count_v0_tfsx_df = count_v0_df.ix[:, selected_features]  # tfidf法筛选后的词向量矩阵
        df_columns = pd.Series(feature_names)[selected_features]

        def guiyi(x):
            x[x > 1] = 1
            return x

        tfidf_df_1 = count_v0_tfsx_df.apply(guiyi)
        return tfidf_df_1, df_columns

    def post_procession(self, predictions):
        pass

    def classify(self, fulltext, title):
        fulltext, _ = self.preprocess(fulltext, title)
        count_text = CountVectorizer(decode_error='replace', vocabulary=self.all_features)  # 特征词库
        count_v0_df, _, feature_names = self.get_tfidf(count_text, [fulltext])  # count_v0_df这是测试文本的词频，后面两个跟训练集保持一致
        tfidf_df_1, _ = self.get_x_test(count_v0_df, feature_names, self.selected_features)  # 筛选后的特征词
        predictions = self.model.predict(tfidf_df_1)
        if predictions[0] != 9:
            # 研报按照行业分类
            count_text_in = CountVectorizer(decode_error='replace', vocabulary=self.all_features_in)  # 不同模型传参不同
            count_v0_df_in, _, feature_names_in = self.get_tfidf(count_text_in, [fulltext])
            tfidf_df_1_in, _ = self.get_x_test(count_v0_df_in, feature_names_in, self.selected_features_in)  # 不同模型传参不同
            predictions_in = self.model_in.predict(tfidf_df_1_in)  # 不同模型传参不同
            return TextType._DESC[Label_Line[predictions[0]]], predictions_in[0]
        else:
            title_classifier = CNNTitleClassifier(cnn_model="title_classify_v35b1e4a2.pb",
                                                  cnn_model_hash="35b1e4a2215346ee85eedc2c32a1c314",
                                                  cnn_vocabulary="cnn_vocabulary_v09b10736.pkl",
                                                  cnn_vocabulary_hash="09b107362b30811de9c686e6cdf198e9")
            label = title_classifier.classify(title, fulltext)
            industry_type = 0
            return label, industry_type

    def classify_batch(self, fulltext, title, batch_size=100):
        pass


class CNNTitleClassifier(BaseClassifier):
    def __init__(self, cnn_model, cnn_model_hash,  # 公告的性质类别分类器模型
                 cnn_vocabulary, cnn_vocabulary_hash):
        self.cnn_model = pb_models.get_model(cnn_model, cnn_model_hash)
        self.cnn_vocabulary = pb_models.get_model(cnn_vocabulary, cnn_vocabulary_hash)
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

    def classify(self, x, text):
        x_test = [self.clean_str(x)]
        x_test = np.array(list(self.vocab_processor.transform(x_test)))

        with self.graph.as_default():
            with self.session.as_default():
                pre = self.session.run(self.predictions, {'input_x:0': x_test, 'dropout_keep_prob:0': 1.0})
                return TextType._DESC[Label_Line2[pre[0]]]

    def classify_batch(self, x, text, batch_size=100):
        pass
