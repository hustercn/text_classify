# coding=utf-8

import re
import os
import sys
import csv
csv.field_size_limit(sys.maxsize)
import jieba
import chardet
jieba.load_userdict('abcChinese.txt')

import pyhanlp
import zipfile
import logging
import requests
import datetime
import zhon.hanzi
import numpy as np
from tqdm import tqdm
from glob import glob
from bosonnlp import BosonNLP


def current_time():
    """
    获取当前时间：年月日时分秒
    :return:
    """
    ct = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    return ct


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    X, Y = [u'\u4e00', u'\u9fa5']  # unicode 前面加u
    if uchar >= X and uchar <= Y:
        return True
    else:
        return False


def separate_words_v2(infile, outfile):
    """
    :param infile:
    :param outfile:
    :return:
    用法示例:
    # >>>separate_words("/home/zhwpeng/data/THUCNews/财经/798977.txt", "798977_test.txt")

    """
    try:
        outf = open(outfile, 'w')
        inf = open(infile, 'r')
        for line in inf.readlines():
            if not len(line.strip()):
                continue
            seg_list = jieba.cut(line)

            """此处需要循环每个单词编码为utf-8，jieba.cut将结果转为了unicode编码，
            直接write(space.join(seg_list))会报编码错误"""
            for word in seg_list:
                if not is_chinese(word):
                    continue
                try:
                    word = word.encode('UTF-8')
                except:
                    continue
                if word not in stopwords:
                    if word != ' ':
                        outf.write(word)
                        outf.write(' ')
        outf.write('\n')
        outf.close()
        inf.close()
    except:
        pass


def extract_test_and_train_set(filepath, train_file, test_file):
    # 报错，折叠

    try:
        test_f = open(test_file, 'a')
        train_f = open(train_file, 'a')
        try:
            with open(filepath) as f:
                is_title_line = True
                for line in f.readlines():
                    if is_title_line:
                        is_title_line = False
                        continue
                    if not len(line):
                        continue
                    if np.random.random() <= 0.2:
                        test_f.write(line)
                    else:
                        train_f.write(line)
        except:
            print "IO ERROR"
        finally:
            test_f.close()
            train_f.close()
    except:
        print "can not open file"


if __name__ == '__main__':
    # 原始数据存放位置
    # base_data_dir = '/home/zhwpeng/abc/classify/yanbao_stock_a/'
    # base_data_dir = '/home/zhwpeng/abc/nlp/data/0324/raw/'
    base_data_dir = '/home/abc/pzw/nlp/data/data_industry/raw/'
    # 分词后的数据存放位置
    # separated_word_file_dir = "/home/zhwpeng/abc/nlp/data/0324/word_sep/"
    separated_word_file_dir = "/home/abc/pzw/nlp/data/data_industry/word_sep/"
    if not os.path.exists(separated_word_file_dir):
        os.makedirs(separated_word_file_dir)

    # 在txt中添加文本类型后存放位置
    # dataset_dir = "/home/zhwpeng/abc/nlp/data/0324/dataset/"
    dataset_dir = "/home/abc/pzw/nlp/data/data_industry/dataset/"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # types = ['NOTICE', 'OTHER']
    # types = ['NOTICE', 'T004001001', 'T004001002', 'D001002001', 'D001002002', 'D001002003', 'T004021008',
    #          'D001003001', 'D001003002', 'D001003003', 'T004022018', 'D001004001', 'D001004002', 'D001004003',
    #          'T004023007', 'T004004002', 'T004004005', 'T004004001', 'T004004004', 'T004004003', 'T004019001',
    #          'T004019003', 'OTHER']

    types = ['110100', '110200', '110300', '110400', '110500', '110600', '110700', '110800', '210100', '210200',
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

    flags = [1, 0]

    if flags[0]:
        print ("第一步：分词同时去除停用词和空格")
        # 第一步：分词
        print current_time()
        stop_words_file = "stopwords.txt"
        stopwords = []
        # 获取停用词
        with open(stop_words_file) as fi:
            for line in fi.readlines():
                stopwords.append(line.strip())
        print current_time()
        for ty in tqdm(types):
            txt_dirs = glob(base_data_dir + ty + '/*')
            # print ty, len(txt_dirs)
            output_dir = separated_word_file_dir + ty
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for ii, txt_dir in enumerate(txt_dirs):
                # if ii < 3000:
                    # print txt_dir.split('/')[-1]
                separate_words_v2(txt_dir, output_dir + '/' + txt_dir.split('/')[-1])
        print current_time()

    if flags[1]:
        print ("第二步：保存数据到csv文件中")
        # 第三步：保存数据到csv文件中
        print 'current time is', current_time()
        # for ty in tqdm(types[:5]):
        for ty in tqdm(types):
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            path = os.path.join(dataset_dir, ty + '.csv')
            csvfile = open(path, 'w')
            fieldnames = ['type', 'content']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            # txt_dirs = glob("word_s2/" + ty + '/*')  # 小批量测试代码
            txt_dirs = glob(separated_word_file_dir + ty + '/*')
            for i in txt_dirs:
                # print i
                label = os.path.dirname(i).split('/')[-1]
                content = open(i).read()
                writer.writerow({'type': label, 'content': content})
            csvfile.close()
        print 'current time is', current_time()

    if flags[1]:
        print ("第三步：随机抽样，获取训练集和测试集")
        # 第四步：随机抽样，获取训练集和测试集
        print 'current time is', current_time()
        # csv_dir = "/home/zhwpeng/abc/nlp/data/0324/csvfiles/"
        csv_dir = "/home/abc/pzw/nlp/data/data_industry/csvfiles/"
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        csv_train = open(os.path.join(csv_dir, "train.csv"), 'a')
        csv_test = open(os.path.join(csv_dir, "test.csv"), 'a')
        fieldnames = ['type', 'content']
        writer_tr = csv.DictWriter(csv_train, fieldnames=fieldnames)
        writer_te = csv.DictWriter(csv_test, fieldnames=fieldnames)
        writer_tr.writeheader()
        writer_te.writeheader()

        cfs = glob(dataset_dir + '/*.csv')
        for cf in cfs:
            print 'extract file:', cf
            try:
                cfi = open(cf, 'r')
                reader = csv.DictReader(cfi)
                for r in reader:
                    # print r['type']
                    # writer_tr.writerow({'type': r['type'], 'content': r['content']})
                    if np.random.random() <= 0.2:
                        writer_te.writerow({'type': r['type'], 'content': r['content']})
                    else:
                        writer_tr.writerow({'type': r['type'], 'content': r['content']})
            except:
                print "IO ERROR"
        print 'current time is', current_time()

