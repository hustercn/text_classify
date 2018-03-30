# coding: utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

from abcft_algorithm_forecast_extraction.text_classify.texttype import BaseClassifier


DEFAULT_MODEL = {
    "vocabulary_path": "abcChinese_ve3c430dc.txt",
    "vocabulary_path_hash": "e3c430dc4b16a790ac687d2c8fa9ae2d",
    "model_name": "svm_c23v6d6c1568.pkl",
    "model_hash": "6d6c1568af14371007a8347b47868c05",
    "all_features": "all_features_v6cfe2767.pkl",
    "all_features_hash": "6cfe276738e2220a66d982e5a2a113fe",
    "selected_features": "features_index_v45b4dc6b.npy",
    "selected_features_hash": "45b4dc6b78b3010bcdf7918e3743005e",
    # industry classifier's files 按照行业分类的分类器模型等相关文件
    "model_in_name": "svm_c104vdbb90322.pkl",
    "model_in_hash": "dbb90322ccc9ae02ea92874be1c820fc",
    "all_features_in": "all_features_va55a8f32.pkl",
    "all_features_in_hash": "a55a8f3233a67868970fa6f58f19be4c",
    "selected_features_in": "features_index_vb998eee7.npy",
    "selected_features_in_hash": "b998eee7e0498193e7d6b797bbc34157",
    # cnn title classifier's files
    "cnn_model": "title_classify_v35b1e4a2.pb",
    "cnn_model_hash": "35b1e4a2215346ee85eedc2c32a1c314",
    "cnn_vocabulary": "cnn_vocabulary_v09b10736.pkl",
    "cnn_vocabulary_hash": "09b107362b30811de9c686e6cdf198e9",
}

__version__ = (0, 1, 0)


def get_version():
    return __version__


_Classifier = None  # type: BaseClassifier


def init(**kwargs):
    from abcft_algorithm_forecast_extraction.text_classify.svm_classifier import SVMTextClassifier, CNNTitleClassifier

    global _Classifier
    if _Classifier:
        raise RuntimeError("_Classifier is already initialized")

    model_name = kwargs.get("model_name", DEFAULT_MODEL["model_name"])
    model_hash = kwargs.get("model_hash", DEFAULT_MODEL["model_hash"])

    vocabulary_path = kwargs.get("vocabulary_path", DEFAULT_MODEL["vocabulary_path"])
    vocabulary_path_hash = kwargs.get("vocabulary_path_hash", DEFAULT_MODEL["vocabulary_path_hash"])

    all_features = kwargs.get("all_features", DEFAULT_MODEL["all_features"])
    all_features_hash = kwargs.get("all_features_hash", DEFAULT_MODEL["all_features_hash"])

    selected_features = kwargs.get("selected_features", DEFAULT_MODEL["selected_features"])
    selected_features_hash = kwargs.get("selected_features_hash", DEFAULT_MODEL["selected_features_hash"])

    # industry classifier's files 按照行业分类的分类器模型等相关文件
    model_in_name = kwargs.get("model_in_name", DEFAULT_MODEL["model_in_name"])
    model_in_hash = kwargs.get("model_in_hash", DEFAULT_MODEL["model_in_hash"])

    all_features_in = kwargs.get("all_features_in", DEFAULT_MODEL["all_features_in"])
    all_features_in_hash = kwargs.get("all_features_in_hash", DEFAULT_MODEL["all_features_in_hash"])

    selected_features_in = kwargs.get("selected_features_in", DEFAULT_MODEL["selected_features_in"])
    selected_features_in_hash = kwargs.get("selected_features_in_hash", DEFAULT_MODEL["selected_features_in_hash"])

    _Classifier = SVMTextClassifier(model_name, model_hash,
                                    vocabulary_path, vocabulary_path_hash,
                                    all_features, all_features_hash,
                                    selected_features, selected_features_hash,
                                    model_in_name, model_in_hash,
                                    all_features_in, all_features_in_hash,
                                    selected_features_in, selected_features_in_hash)

    global _Classifier_cnn
    if _Classifier_cnn:
        raise RuntimeError("_Classifier_cnn is already initialized")

    cnn_model = kwargs.get("cnn_model", DEFAULT_MODEL["cnn_model"])
    cnn_model_hash = kwargs.get("cnn_model_hash", DEFAULT_MODEL["cnn_model_hash"])
    cnn_vocabulary = kwargs.get("cnn_vocabulary", DEFAULT_MODEL["cnn_vocabulary"])
    cnn_vocabulary_hash = kwargs.get("cnn_vocabulary_hash", DEFAULT_MODEL["cnn_vocabulary_hash"])

    _Classifier_cnn = CNNTitleClassifier(cnn_model, cnn_model_hash,
                                         cnn_vocabulary, cnn_vocabulary_hash)


def get_classifier():
    if not _Classifier:
        raise RuntimeError("_Classifier is not initialized")
    return _Classifier


def get_classifier_cnn():
    if not _Classifier_cnn:
        raise RuntimeError("_Classifier_cnn is not initialized")
    return _Classifier_cnn
