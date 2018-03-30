# coding=utf-8

import os
import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


def current_time():
    ct = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    return ct


def get_non(df_file):
    """
    # 找到空行
    :param df_file:
    :return: vacant_row = [93, 1487, 1595]
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
    train_df.info(memory_usage='deep')
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
    count_v0_df.info(memory_usage='deep')
    tfidf_df = pd.DataFrame(tfidf.toarray())
    tfidf_df.info(memory_usage='deep')
    return count_v0_df, tfidf_df, feature_names


def guiyi(x):
    x[x > 1] = 1
    return x


def select_index_and_get_x(count_v0_df, tfidf_df, feature_names, classes,
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

    count_v0_tfsx_df = count_v0_df.ix[:, features_index_by_tfidf]   # tfidf筛选后的词向量矩阵
    df_columns = pd.Series(feature_names)[features_index_by_tfidf]
    print 'df_columns shape', df_columns.shape

    tfidf_df_1 = count_v0_tfsx_df.apply(guiyi)
    tfidf_df_1.columns = df_columns
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    tfidf_df_1['label'] = le.transform(all_labels)

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
    np.save(features_index_by_ch2_dir + 'features_index_by_ch2.npy', np.array(features_index_by_ch2))
    print 'saved features_index_by_ch2 successfully!'

    x, y = ch2_sx_np, label_np
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    return x, y


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


if __name__ == '__main__':
    # classes = ['NOTICE', 'T004001001', 'T004001002', 'D001002001', 'D001002002', 'D001002003', 'T004021008',
    #            'D001003001', 'D001003002', 'D001003003', 'T004022018', 'D001004001', 'D001004002', 'D001004003',
    #            'T004023007', 'T004004002', 'T004004005', 'T004004001', 'T004004004', 'T004004003', 'T004019001',
    #            'T004019003', 'OTHER']

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

    train_flag = 1
    if train_flag:
        base_dir = "/home/abc/pzw/nlp/data/data_industry/"
        print '(1) 数据准备...'
        print current_time()
        all_texts, all_labels = get_data(train_filename=base_dir+"csvfiles/train.csv", test_filename="")
        print '(2) 计算词频、tfidf矩阵、卡方检验筛选特征维度...'
        train_tmp_dir = 'train_tmp/'
        print current_time()
        count_v0 = get_count(all_texts, max_features=20000, save=True, vocab_dir=base_dir+train_tmp_dir+'vocab_dir/')
        count_v0_df, tfidf_df, feature_names = get_tfidf(count_v0, all_texts)
        X, y = select_index_and_get_x(count_v0_df, tfidf_df, feature_names, classes,
                                      all_labels, tfidf_features=5000, chi2_features=3000,
                                      features_index_by_ch2_dir=base_dir+train_tmp_dir+'features_index_by_ch2/')
        print 'X shape:', X.shape
        print 'y shape:', y.shape

        print '(3) 采用k折交叉验证训练和验证SVM分类器...'
        print current_time()
        kfold = 0
        if kfold:
            skf = StratifiedKFold(y, n_folds=5)
            y_pre = y.copy()
            for train_index, test_index in skf:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                svclf = SVC(kernel='linear')
                svclf.fit(X_train, y_train)
                y_pre[test_index] = svclf.predict(X_test)
            print '准确率为 %.6f' % (np.mean(y_pre == y))
            print y_pre.shape

        if not kfold:
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

            svclf = SVC(kernel='linear')
            svclf.fit(x_train, y_train)
            if not os.path.exists(base_dir+train_tmp_dir+'new_model/'):
                os.makedirs(base_dir+train_tmp_dir+'new_model/')
            with open(base_dir+train_tmp_dir+'new_model/svm_c104v01.pkl', 'wb') as fw:
                pickle.dump(svclf, fw)

            print '(4) 评价分类器...'
            print current_time()
            preds = svclf.predict(x_test)
            num = 0
            preds = preds.tolist()
            for i, pred in enumerate(preds):
                if int(pred) == int(y_test[i]):
                    num += 1
            print 'SVM precision_score:' + str(float(num) / len(preds))
            print current_time()

