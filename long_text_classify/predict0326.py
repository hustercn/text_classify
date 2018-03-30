# coding=utf-8

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Yahei Mono']  # 指定默认字体

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def current_time():
    ct = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    return ct


def get_non(df_file):
    """
    # 找到空行
    :param df_file:
    :return: vacant_row = [93, 1487, 1595, 1794, 1985, 2062, 2100]
    """
    vacant_row = []
    for ind, row in df_file.iterrows():
        if not isinstance(row['content'], str) or row['content'] == '\n':
            # print row
            vacant_row.append(ind)
    return vacant_row


def get_data(train_filename="/home/zhwpeng/abc/nlp/data/0321/csvfiles/cleaned_train2.csv",
             test_filename="/home/zhwpeng/abc/nlp/data/0321/csvfiles/cleaned_test.csv"):
    train_df = pd.read_csv(train_filename, compression=None, error_bad_lines=False)
    vr = get_non(train_df)
    if vr is not None:
        train_df.drop(vr, inplace=True)

    if test_filename is not "":
        test_df = pd.read_csv(test_filename, compression=None, error_bad_lines=False)
        vr_test = get_non(test_df)
        if vr_test is not None:
            test_df.drop(vr_test, inplace=True)
        content_df = train_df.append(test_df, ignore_index=True)
    else:
        content_df = train_df

    # shuffle data
    content_df = shuffle(content_df)

    all_texts = content_df['content']
    all_labels = content_df['type']
    # print "新闻文本数量：", len(all_texts), len(all_labels)
    # print "每类新闻的数量：\n", all_labels.value_counts()

    print "沪深股研报文本数量：", len(all_texts), len(all_labels)
    print "每类研报的数量：\n", all_labels.value_counts()
    return all_texts, all_labels


def get_count(all_texts, max_features=10000, save=True, vocab_dir='train_tmp/vocab_dir/'):
    """训练集的关键词 vocabulary 需要保存或被引用"""
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
    if not os.path.exists(vocab_dir + 'vocabulary.pkl'):
        count = CountVectorizer(decode_error='replace', max_features=max_features)
        counts_all = count.fit_transform(all_texts)
        print 'the shape of all datas is ' + repr(counts_all.shape)
        if save:
            with open(vocab_dir + 'vocabulary.pkl', 'wb') as f:  # save vocabulary
                pickle.dump(count.vocabulary_, f)
    else:
        with open(vocab_dir + 'vocabulary.pkl', 'rb') as f:  # load vocabulary
            vocab = pickle.load(f)
        count = CountVectorizer(decode_error='replace', vocabulary=vocab)
    return count


def get_tfidf(count_v0, train_texts):
    counts_train = count_v0.fit_transform(train_texts)

    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(counts_train)
    print 'tfidf:', type(tfidf), tfidf.shape

    feature_names = count_v0.get_feature_names()  # 关键字
    count_v0_df = pd.DataFrame(counts_train.toarray())
    tfidf_df = pd.DataFrame(tfidf.toarray())
    return count_v0_df, tfidf_df, feature_names


def guiyi(x):
    x[x > 1] = 1
    return x


def select_index_and_get_x(count_v0_df, tfidf_df, feature_names,
                           all_labels, tfidf_features=5000, chi2_features=500,
                           features_index_by_ch2_dir='train_tmp/features_index_by_ch2/'):
    """通过tfidf第一次降维，通过卡方检测第二次降维
    :param count_v0_df:
    :param tfidf_df:
    :param feature_names: 所有的特征值（关键词）
    :param all_labels:
    :param tfidf_features: tfidf之后的维度
    :param chi2_features: 卡方之后的维度
    :return:
    """
    features_index_by_tfidf = tfidf_df.sum(axis=0).sort_values(ascending=False)[:tfidf_features].index  # tfidf值降序取前5000
    print 'features_index_by_tfidf:', len(features_index_by_tfidf)

    # # 训练集的tfidf_df筛选后的特征词的index保存，后面测试使用
    # features_index_by_tfidf_dir = 'train_tmp/features_index_by_tfidf/'
    # if not os.path.exists(features_index_by_tfidf_dir):
    #     os.makedirs(features_index_by_tfidf_dir)
    # np.save(features_index_by_tfidf_dir + 'features_index_by_tfidf.npy', np.array(features_index_by_tfidf))
    # print 'saved features_index_by_tfidf successfully!'

    count_v0_tfsx_df = count_v0_df.ix[:, features_index_by_tfidf]   # tfidf筛选后的词向量矩阵
    df_columns = pd.Series(feature_names)[features_index_by_tfidf]
    print 'df_columns shape', df_columns.shape

    tfidf_df_1 = count_v0_tfsx_df.apply(guiyi)
    tfidf_df_1.columns = df_columns
    le = preprocessing.LabelEncoder()
    tfidf_df_1['label'] = le.fit_transform(all_labels)

    ch2 = SelectKBest(chi2, k=chi2_features)
    nolabel_feature = [x for x in tfidf_df_1.columns if x not in ['label']]
    ch2_sx_np = ch2.fit_transform(tfidf_df_1[nolabel_feature], tfidf_df_1['label'])
    label_np = np.array(tfidf_df_1['label'])

    # 训练集的卡方筛选后的特征词的index保存，后面测试使用
    if not os.path.exists(features_index_by_ch2_dir):
        os.makedirs(features_index_by_ch2_dir)
    index_tf = np.array(features_index_by_tfidf)
    a = ch2.get_support()  # 卡方检测之后的index真假矩阵
    features_index_by_ch2 = []
    for ke, v in enumerate(a):
        if v:
            features_index_by_ch2.append(index_tf[ke])
    print features_index_by_ch2
    # np.save(ch2_score_dir + 'ch2_score.npy', ch2.scores_)
    np.save(features_index_by_ch2_dir + 'features_index_by_ch2.npy', np.array(features_index_by_ch2))
    print 'saved features_index_by_ch2 successfully!'

    x, y = ch2_sx_np, label_np
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    return x_train, x_test, y_train, y_test


def get_x_test(count_v0_df, features_index_by_ch2_dir='train_tmp/features_index_by_ch2/'):
    """
    :param count_v0_df: 测试集的tfidf矩阵
    :param feature_names: 所有的特征词
    :param features_index_by_ch2_dir: 训练过程中保存的卡方检测筛选后的特征维度路径
    :return:
    """
    selected_features_index = np.load(features_index_by_ch2_dir + 'features_index_by_ch2.npy')  # 导入训练集tfidf筛选后的维度索引
    count_v0_tfsx_df = count_v0_df.ix[:, selected_features_index]  # tfidf法筛选后的词向量矩阵

    tfidf_df_1 = count_v0_tfsx_df.apply(guiyi)

    return tfidf_df_1


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    import itertools

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", fontsize=16,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    if 0:
        fig1, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center", fontsize=16,
                     color="white" if cm[i, j] > thresh else "black")
        ax.set_ylabel(u'真实标签')
        ax.set_xlabel(u'预测标签')

        fig1.savefig('confusion.png')
        plt.close()


if __name__ == '__main__':
    classes = ['NOTICE', 'T004001001', 'T004001002', 'D001002001', 'D001002002', 'D001002003', 'T004021008',
               'D001003001', 'D001003002', 'D001003003', 'T004022018', 'D001004001', 'D001004002', 'D001004003',
               'T004023007', 'T004004002', 'T004004005', 'T004004001', 'T004004004', 'T004004003', 'T004019001',
               'T004019003', 'OTHER']

    ch_cla = [u"宏观简报", u"宏观定期", u"宏观深度", u"策略简报", u"策略定期", u"策略深度",  u"行业简报",
              u"行业定期", u"行业深度", u"公告", u"其他类别", u"晨会纪要", u"早间资讯", u"公司调研",
              u"公司点评", u"新股研究", u"会议纪要", u"公司深度", u"新三板公司", u"新三板市场", u"海外经济",
              u"全球策略", u"行业调研"]

    cla = {
        "D001002001": 0,
        "D001002002": 1,
        "D001002003": 2,
        "D001003001": 3,
        "D001003002": 4,
        "D001003003": 5,
        "D001004001": 6,
        "D001004002": 7,
        "D001004003": 8,
        "NOTICE": 9,
        "OTHER": 10,
        "T004001001": 11,
        "T004001002": 12,
        "T004004001": 13,
        "T004004002": 14,
        "T004004003": 15,
        "T004004004": 16,
        "T004004005": 17,
        "T004019001": 18,
        "T004019003": 19,
        "T004021008": 20,
        "T004022018": 21,
        "T004023007": 22
    }
    base_dir = "/home/zhwpeng/abc/nlp/data/0324/"
    print '(1) 读取需要被分类的文档...'
    print current_time()
    # unknown_texts = pd.read_csv(base_dir+'dataset/OTHER.csv', compression=None, error_bad_lines=False)
    test_texts = pd.read_csv(base_dir+'csvfiles/test.csv', compression=None, error_bad_lines=False)
    vr = get_non(test_texts)
    if vr is not None:
        test_texts.drop(vr, inplace=True)
    unknown_texts = test_texts['content']
    unknown_labels = list(test_texts['type'])
    y_true = [cla[lab] for lab in unknown_labels]
    # print unknown_texts.head()
    train_tmp_dir = 'train_tmp1/'
    print '(2) 读取训练集保存的关键字对象...'
    print current_time()
    count_v0 = get_count(all_texts=[], max_features=20000, save=True, vocab_dir=base_dir+train_tmp_dir+'vocab_dir/')
    count_v0_df, _, feature_names = get_tfidf(count_v0, unknown_texts)  # count_v0_df这是测试文本的词频，后面两个跟训练集保持一致

    print '(3) 将被预测文本向量化...'
    print current_time()
    # 训练过程保存的特征维度（经过卡方检验筛选后）
    tfidf_df_1 = get_x_test(count_v0_df, features_index_by_ch2_dir=base_dir+train_tmp_dir+'features_index_by_ch2/')

    print '(4) 引入模型并预测类别...'
    print current_time()
    with open(base_dir+train_tmp_dir+'new_model/svm_c23v01.pkl', 'rb') as fr:
        model = pickle.load(fr)
    unknown_texts_preds = model.predict(tfidf_df_1)
    print len(unknown_texts_preds)
    # print unknown_texts_preds[:10]

    pred_labels = [sorted(cla.keys())[pre] for pre in unknown_texts_preds]
    cm = confusion_matrix(unknown_labels, pred_labels, sorted(cla.keys()))
    plot_confusion_matrix(cm, ch_cla)
    # print "混淆矩阵", confusion_matrix(y_true, unknown_texts_preds, unknown_labels)
    # print "分类报告", classification_report(y_true, unknown_texts_preds, unknown_labels)
    print current_time()

    if 0:
        # le = preprocessing.LabelEncoder()
        # le.fit(classes)
        # print list(le.classes_)

        NOTICE, T004001001, T004001002, D001002001, D001002002, D001002003, T004021008, \
        D001003001, D001003002, D001003003, T004022018, D001004001, D001004002, D001004003, \
        T004023007, T004004002, T004004005, T004004001, T004004004, T004004003, T004019001,\
        T004019003, OTHER = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        for pred_result in unknown_texts_preds:
            if pred_result == 0:
                D001002001 += 1
            elif pred_result == 1:
                D001002002 += 1
            elif pred_result == 2:
                D001002003 += 1
            elif pred_result == 3:
                D001003001 += 1
            elif pred_result == 4:
                D001003002 += 1
            elif pred_result == 5:
                D001003003 += 1
            elif pred_result == 6:
                D001004001 += 1
            elif pred_result == 7:
                D001004002 += 1
            elif pred_result == 8:
                D001004003 += 1
            elif pred_result == 9:
                NOTICE += 1
            elif pred_result == 10:
                OTHER += 1
            elif pred_result == 11:
                T004001001 += 1
            elif pred_result == 12:
                T004001002 += 1
            elif pred_result == 13:
                T004004001 += 1
            elif pred_result == 14:
                T004004002 += 1
            elif pred_result == 15:
                T004004003 += 1
            elif pred_result == 16:
                T004004004 += 1
            elif pred_result == 17:
                T004004005 += 1
            elif pred_result == 18:
                T004019001 += 1
            elif pred_result == 19:
                T004019003 += 1
            elif pred_result == 20:
                T004021008 += 1
            elif pred_result == 21:
                T004022018 += 1
            elif pred_result == 22:
                T004023007 += 1
        print 'notice recall is :', (NOTICE), (len(unknown_texts_preds))
        print 'T004001001 recall is :', (T004001001), (len(unknown_texts_preds))
        print 'T004001002 recall is :', (T004001002), (len(unknown_texts_preds))
        print 'D001002001 recall is :', (D001002001), (len(unknown_texts_preds))
        print 'D001002002 recall is :', (D001002002), (len(unknown_texts_preds))
        print 'D001002003 recall is :', (D001002003), (len(unknown_texts_preds))
        print 'T004021008 recall is :', (T004021008), (len(unknown_texts_preds))
        print 'D001003001 recall is :', (D001003001), (len(unknown_texts_preds))
        print 'D001003002 recall is :', (D001003002), (len(unknown_texts_preds))
        print 'D001003003 recall is :', (D001003003), (len(unknown_texts_preds))
        print 'T004022018 recall is :', (T004022018), (len(unknown_texts_preds))
        print 'D001004001 recall is :', (D001004001), (len(unknown_texts_preds))
        print 'D001004002 recall is :', (D001004002), (len(unknown_texts_preds))
        print 'D001004003 recall is :', (D001004003), (len(unknown_texts_preds))
        print 'T004023007 recall is :', (T004023007), (len(unknown_texts_preds))
        print 'T004004002 recall is :', (T004004002), (len(unknown_texts_preds))
        print 'T004004005 recall is :', (T004004005), (len(unknown_texts_preds))
        print 'T004004001 recall is :', (T004004001), (len(unknown_texts_preds))
        print 'T004004004 recall is :', (T004004004), (len(unknown_texts_preds))
        print 'T004004003 recall is :', (T004004003), (len(unknown_texts_preds))
        print 'T004019001 recall is :', (T004019001), (len(unknown_texts_preds))
        print 'T004019003 recall is :', (T004019003), (len(unknown_texts_preds))
        print 'OTHER recall is :', (OTHER), (len(unknown_texts_preds))
        print current_time()
