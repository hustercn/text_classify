# coding=utf-8

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
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
    if 0:
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
        plt.savefig('confusion_matrix_test.png')
        # plt.show()

    fig1, ax = plt.subplots(figsize=(100, 100))
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", fontsize=16,
                 color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('true labels')
    ax.set_xlabel('predictoin labels')

    fig1.savefig('confusion1.png')
    plt.close()


if __name__ == '__main__':
    classes = ['110100', '110200', '110300', '110400', '110500', '110600', '110700', '110800', '210100', '210200',
               '210300', '210400', '220100', '220200', '220300', '220400', '220500', '220600', '230100', '240200',
               '240300', '240400', '240500', '270100', '270200', '270300', '270400', '270500', '280100', '280200',
               '280300', '280400', '330100', '330200', '340300', '340400', '350100', '350200', '360100', '360200',
               '360300', '360400', '370100', '370200', '370300', '370400', '370500', '370600', '410100', '410200',
               '410300', '410400', '420100', '420200', '420300', '420400', '420500', '420600', '420700', '420800',
               '430100', '430200', '450200', '450300', '450400', '450500', '460100', '460200', '460300', '460400',
               '460500', '480100', '490100', '490200', '490300', '510100', '610100', '610200', '610300', '620100',
               '620200', '620300', '620400', '620500', '630100', '630200', '630300', '630400', '640100', '640200',
               '640300', '640400', '640500', '650100', '650200', '650300', '650400', '710100', '710200', '720100',
               '720200', '720300', '730100', '730200']
    # base_dir = "/home/zhwpeng/abc/nlp/data/data_industry/"
    base_dir = "/home/abc/pzw/nlp/data/data_industry/"
    print '(1) 读取需要被分类的文档...'
    print current_time()
    test_texts = pd.read_csv(base_dir+'csvfiles/test.csv', compression=None, error_bad_lines=False)
    vr = get_non(test_texts)
    if vr is not None:
        test_texts.drop(vr, inplace=True)
    unknown_texts = test_texts['content']
    y_true = list(test_texts['type'])
    y_true = [str(i) for i in y_true]
    print 'y true', y_true[:10]
    # unknown_labels = list(test_texts['type'])
    # y_true = [cla[lab] for lab in unknown_labels]
    # print unknown_texts.head()
    train_tmp_dir = 'train_tmp1/'
    print '(2) 读取训练集保存的关键字对象...'
    print current_time()
    count_v0 = get_count(all_texts=[], max_features=20000, save=True, vocab_dir=base_dir+train_tmp_dir+'vocab_dir/')
    count_v0_df, _, feature_names = get_tfidf(count_v0, unknown_texts)  # count_v0_df这是测试文本的词频，feature_names跟训练集保持一致

    print '(3) 将被预测文本向量化...'
    print current_time()
    # 训练过程保存的特征维度（经过卡方检验筛选后）
    tfidf_df_1 = get_x_test(count_v0_df, features_index_by_ch2_dir=base_dir+train_tmp_dir+'features_index_by_ch2/')

    print '(4) 引入模型并预测类别...'
    print current_time()
    with open(base_dir+train_tmp_dir+'new_model/svm_c104v01.pkl', 'rb') as fr:
        model = pickle.load(fr)
    unknown_texts_preds = model.predict(tfidf_df_1)
    print len(unknown_texts_preds)
    # print unknown_texts_preds[:10]

    pred_labels = [classes[pre] for pre in unknown_texts_preds]
    print 'pred labels', pred_labels[:10]
    cm = confusion_matrix(y_true, pred_labels, classes)
    cm_df = pd.DataFrame(cm, columns=classes, index=classes)
    cm_df.to_csv('confusion1.csv')
    plot_confusion_matrix(cm, classes)
    # print "混淆矩阵", confusion_matrix(y_true, unknown_texts_preds, unknown_labels)
    # print "分类报告", classification_report(y_true, unknown_texts_preds, unknown_labels)
    print current_time()

