# coding=utf-8

from __future__ import unicode_literals

import codecs
import datetime
from glob import glob
from abcft_algorithm_forecast_extraction.text_classify.svm_classifier import SVMTextClassifier
# from abcft_algorithm_forecast_extraction.text_classify.cnn_classifier import CNNTitleClassifier
from abcft_algorithm_forecast_extraction.text_classify.texttype import TextType


def current_time():
    ct = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    return ct

if __name__ == '__main__':
    text_classifier = SVMTextClassifier(vocabulary_path="abcChinese_ve3c430dc.txt",
                                        vocabulary_path_hash="e3c430dc4b16a790ac687d2c8fa9ae2d",
                                        model="svm_c23v6d6c1568.pkl",
                                        model_hash="6d6c1568af14371007a8347b47868c05",
                                        all_features="all_features_v6cfe2767.pkl",
                                        all_features_hash="6cfe276738e2220a66d982e5a2a113fe",
                                        selected_features="features_index_v45b4dc6b.npy",
                                        selected_features_hash="45b4dc6b78b3010bcdf7918e3743005e",
                                        model_in="svm_c104vdbb90322.pkl",   # 按照行业分类的分类器模型等相关文件
                                        model_in_hash="dbb90322ccc9ae02ea92874be1c820fc",
                                        all_features_in="all_features_va55a8f32.pkl",
                                        all_features_in_hash="a55a8f3233a67868970fa6f58f19be4c",
                                        selected_features_in="features_index_vb998eee7.npy",
                                        selected_features_in_hash="b998eee7e0498193e7d6b797bbc34157"
                                        )
    if 1:
        # textfile = '/home/zhwpeng/abc/nlp/data/0324/word_sep/D001002001/1999085.txt'
        textfile = '/home/abc/pzw/nlp/data/data_industry/test/D001002002/12640895.txt'
        # textfile = '/home/zhwpeng/abc/nlp/data/0324/word_sep/D001002002/1999497.txt'
        # textfile = '/home/zhwpeng/abc/nlp/data/0324/word_sep/D001002003/2001037.txt'
        # textfile = '/home/zhwpeng/abc/nlp/data/0324/word_sep/D001003001/1868377.txt'
        # textfile = '/home/zhwpeng/abc/nlp/data/0324/word_sep/D001004001/20667.txt'
        # textfile = '/home/zhwpeng/abc/nlp/data/0324/word_sep/T004004002/2001903.txt'
        # textfile = '/home/zhwpeng/abc/nlp/data/0324/word_sep/T004019001/2253342.txt'
        # textfile = '/home/zhwpeng/abc/nlp/data/0324/word_sep/OTHER/3418.txt'
        # textfile = '/home/zhwpeng/abc/nlp/data/0324/word_sep/NOTICE/5a2f93b670cff49210652f1b.txt'
        with codecs.open(textfile, "r") as f:
            data = f.read()
        # print data
        # text_result = text_classifier.classify(data, title='股权分置改革的法律意见书')
        text_result, industry_result = text_classifier.classify(data, title='')
        print "一级类别是：", TextType.get_level_one(text_result)
        print "二级类别是：", TextType.get_level_two(text_result)
        print "三级类别是：", TextType.get_level_three(text_result)
        print industry_result
        print "行业类别是：%s" % TextType.get_industry(industry_result)

