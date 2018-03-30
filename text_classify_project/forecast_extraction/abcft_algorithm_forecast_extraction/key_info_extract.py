#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
_______________________________________________________________________________

                       Created on '2018-03-13 10:46'

                               @author: xdliu

                            All rights reserved.
********************************************************************************
"""
from __future__ import unicode_literals, print_function

import re
import copy
import logging

import cv2
import numpy as np
import unicodecsv as csv
import os

from abcft_algorithm_forecast_extraction.text_classify import classifier
from abcft_algorithm_forecast_extraction.text_classify.texttype import TextType
from utils import Rect, get_fake_table, TableFormatter

log = logging.getLogger(__name__)

org_name_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             'org_name.csv')
re_limit = 100
# 发布机构名库
with open(org_name_path, 'r') as f:
    csv_reader = csv.reader(f)
    orgs = list(csv_reader)
    long_name = [o[0] for o in orgs if len(o) == 2]
    short_name = [o[1] for o in orgs if len(o) == 2]
    chunk_num = len(long_name) // re_limit + 1
    org_name_pattern_list = []
    for c in range(chunk_num - 1):
        end_cursor = (c + 1) * re_limit
        _pattern = '|'.join(long_name[c * re_limit: end_cursor])
        org_name_pattern_list.append(re.compile(_pattern))
        if c == chunk_num - 2 and len(long_name) > end_cursor:
            _pattern = '|'.join(long_name[end_cursor:])
    orgs_dict = {}
    for l, s in orgs:
        orgs_dict[l] = s

# 预测评级的EPS、PE匹配文本
EPS_PE_RE = re.compile(r"(?<=。)[^。]+预[^。]+EPS[^。]+PE[^。]+。?[^。]+评级*。|"
                       r"(?<=。)[^。]+预[^。]+EPS[^。]+。?[^。]+评级*。")

# 年份匹配的数字，两位或者数字
YEAR_RE = re.compile(u"(?<=[\u2E80-\u9FFF~/\-])\d{2}(?=\D)|(?<=\D)\d{4}(?!["
                     u"\d\.])")

# EPS的数字，一般为两位小数点的数
EPS_RE = re.compile(r'(?<=\D)\d\.+\d{1,2}(?=\D)')

# PE对应的数字，一般为一到三位整数或者，两位小数点的小数
PE_RE = re.compile(r'(?<=\D)\d{1,3}(?=\D)\.*\d*(?![\d年])')

# 带有指定年份的PE匹配字符串
PE_YEAR_RE = re.compile(u"(?<=[\u2E80-\u9FFF~-])\d{4}(?=年)|"
                        u"(?<=[\u2E80-\u9FFF~-])\d{2}(?=年)")

# 当前价，目标价，上次目标价等价格信息匹配文本
PRICE_RE = re.compile(
    r"(?<=，)[^，]*目标价[^，]+\d+\.*\d*[^，]+元[^，]*(?=，)|"
    r"(?<=，)[^，]*现价[^，]+\d+\.*\d*[^，]+元[^，]*(?=，)|"
    r"(?<=，)[^，]*当前价[^，]+\d+\.*\d*[^，]+元[^，]*(?=，)|"
    r"(?<=，)[^，]*当前股价[^，]+\d+\.*\d*[^，]+元[^，]*(?=，)|"
    r"(?<=，)[^，]*当前股价[^，]+\d+\.*\d*[^，]+RMB[^，]*(?=，)|"
    r"(?<=，)[^，]*最新收盘价[^，]+\d+\.*\d*[^，]+元[^，]*(?=，)"
)

# 目标价
TARGET_PRICE_RE = re.compile(r"(?<!上次)目标价[^元，]+\d+\.*\d*[^元，]+元|"
                             r"(?<!上次)目标价[^元，]+\d+\.*\d*[- ]*RMB(?!当前股价])")
DATA_TARGET_PRICE = re.compile(r"(?<=\D)\d+\.*\d*(?=\D)")

# 上次目标价
PRE_TARGET_PRICE_RE = re.compile(r"上次目标价[^元，]+\d+\.*\d*[^元，]+元|"
                                 r"上次目标价[^元，]+\d+\.*\d*[- ]*RMB(?!当前股价])")
DATA_PRE_TARGET_PRICE = DATA_TARGET_PRICE

# 当前价
CURRENT_PRICE_RE = re.compile(r"现价(?!）)[^元，]+\d+\.*\d*[^元，]+元|"
                              r"当前价[^元，]+\d+\.*\d*[^元，]+元|"
                              r"当前股价[^元，]+\d+\.*\d*[^元，]+元|"
                              r"当前股价[^元，]+\d+\.*\d*[^元，]+RMB|"
                              r"最新收盘价[^元，].+\d+\.*\d*[^元，]+元|")
DATA_CURRENT_PRICE = DATA_TARGET_PRICE

# 发布日期
PUB_DATE_RE = re.compile(r"\d{4}[-年/]\d{1,2}[-月/]\d{1,2}[日]*")

# 发布机构
PUB_COMPANY_NAME = re.compile(r"(?<=声明)[^声明证券]{1,25}证券[^公司]{1,15}公司|"
                              r"(?<=条款)[^条款证券]{1,25}证券[^公司]{1,15}公司")

# 证券代码和证券名称
SEC_RE = re.compile(u"(?<=[ —\n])[\u2E80-\u9FFF]{2,4}[(（]\d{6}[\.SZ ]*[)）]|"
                    u"(?<=[ —\n])[\u2E80-\u9FFF]{2,4} +股票代码[:：]\d{6}[\.SZ ]*")

# 给标题检测用的短语证券代码检测
SEC_RE_SHORT = re.compile(u"[\u2E80-\u9FFF]{2,4}[(（]\d{6}[\.SZ ]*[)）]|"
                          u"[\u2E80-\u9FFF]{2,4} +股票代码[:：]\d{6}[\.SZ ]*")

# EMAIL及电话及证券从业资格证编码及姓氏
EMAIL_RE = re.compile(r"[a-zA-Z\d]+@[a-zA-Z\d\.]+")
CERT_ID = re.compile(r"S\d{13}")
TEL = re.compile(r"(?<=[:： ])\d{11}|"
                 r"(?<=[:： ])0\d{2,3}[\-]?\d{7,8}")
FAMILY_NAME = re.compile(u""
                         u"[赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严"
                         u"华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤"
                         u"花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐"
                         u"于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄"
                         u"米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻"
                         u"强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万"
                         u"支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔"
                         u"吉钮龚程嵇邢滑裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松"
                         u"井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾"
                         u"暴甘钭厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲邰"
                         u"从鄂索咸籍赖卓蔺屠蒙池乔阴鬱胥能苍双闻莘党翟谭贡劳逄姬"
                         u"申扶堵冉宰郦雍卻璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏"
                         u"柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满"
                         u"弘匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾"
                         u"辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖"
                         u"益桓公万俟"
                         u"(司马)(上官)(欧阳)(夏侯)(诸葛)(闻人)"
                         u"(东方)(赫连)(皇甫(尉迟(公羊(澹台(公冶)"
                         u"(宗政)(濮阳)(淳于)(单于)(太叔)(申屠)"
                         u"(公孙)(仲孙)(轩辕)(令狐)(钟离)(宇文)(长孙)"
                         u"(慕容)(鲜于)(闾丘)(司徒)(司空)(丌官)"
                         u"(司寇)(仉督)(子车)(颛孙)(端木)(巫马)(公西)"
                         u"(漆雕)(乐正)(壤驷)(公良)(拓跋)(夹谷)"
                         u"(宰父)(谷梁)(晋楚)(闫法)(汝鄢)(涂钦)(段干)"
                         u"(百里)(东郭)(南门)(呼延)(归海)(羊舌)"
                         u"(微生)(岳帅)(缑亢)(况郈)(有琴)(梁丘)(左丘)"
                         u"(东门)(西门)(商牟)(佘佴)(伯赏)(南宫)"
                         u"(墨哈)(谯笪)(年爱)(阳佟)(第五)(言福)]+"
                         u"[\u2E80-\u9FFF]{1,3}")

# name blacklist
NAME_BLACKLIST = re.compile(u'公司概要|证书编号')
log = logging.getLogger(__name__)

'''
def draw_text(img, texts, x, y, color, font_path):
    """
    画出所有的文本
    :param img:
    :param texts:
    :param x:
    :param y:
    :param color:
    :param font_path:
    :return:
    """
    if len(texts) > 0:
        color = texts[0].get('color', '#000000')
        font_size = texts[0].get('font_size', 8.0)
        text = [t.get('text', '') for t in texts]
        draw_chinese = PutText(font_path)
        # draw_chinese.draw_text(img, text)
    else:
        return img
'''


def draw_bbox(img, bboxes):
    """
    画出所有的bbox
    :param img:
    :param bboxes:
    :return:
    """
    for bbox in bboxes:
        img = cv2.rectangle(img, (int(bbox.left), int(bbox.top)),
                            (int(bbox.right), int(bbox.bottom)),
                            color=(0, 255, 0),
                            thickness=1)
    return img


def get_page_id(data):
    indexes = set([d.get('pageIndex', 0) for d in data])
    return indexes


def split_page(data):
    pages = []
    for i in get_page_id(data):
        pages.append([d for d in data if d.get('pageIndex', 0) == i])
    return pages


def get_page_size(page, offset=50):
    rects = [Rect((p['area']['x'], p['area']['y'],
                   p['area']['w'], p['area']['h']))
             for p in page]
    bottom_right = [[r.bottom, r.right] for r in rects]
    bottom_right = np.array(bottom_right)
    max_br = bottom_right.max(axis=0) + offset
    for idx, (p, rect) in enumerate(zip(page, rects)):
        page[idx]['rect'] = rect

    return int(max_br[0]), int(max_br[1]), page


def draw_parse_result(data):
    """

    :param data: List,
    """
    pages = split_page(data)
    for page in pages:
        h, w, rects = get_page_size(page)
        img = np.ones((h, w), dtype=np.float32) * 255
        img = img.astype(dtype=np.uint8)
        img = draw_bbox(img, rects)
        cv2.imshow('bbox', img)
        cv2.waitKey()


def deal_years(years):
    """
    获取字符串中的年份信息，并自动填充，比如2017~2019填充为2017,2018,2019
    :param years:
    :return:
    """
    result = []
    if len(years) > 0 and isinstance(years[0], unicode):
        years = [years]
    else:
        return result
    years = [y for y in years if len(y) > 0]
    if len(years) == 1:
        year_list = years[0]
        if len(year_list) > 0:
            lens = [len(y) for y in year_list]  # 所有候选年份的长度
            if len(set(lens)) == 1:
                result = year_list
            elif set(lens) == {4, 2}:
                # 例如2017-19->['2017', '19'],去除4为的前两位，加在两位的前面
                year_4_prefix = [y[:2] for y in year_list if len(y) == 4]
                if len(set(year_4_prefix)) == 1:
                    for idx, y in enumerate(year_list):
                        if len(y) == 2:
                            year_list[idx] = year_4_prefix[0] + y
                    result = year_list
                else:
                    log.warn('cannot deal multi preffix of years')
            else:
                log.warn('cannot deal year length quantity that '
                         'greater '
                         'than 2')
    else:
        log.warning('cannot deal with year proposal length bigger than'
                    ' 1 or equal to 0')
    if len(result) > 0:
        year_int = [int(y) for y in result]
        # 确定两个四位年份是挨着的，如果不是，则线性填充
        if len(year_int) == 2 and (year_int[1] - year_int[0]) > 1:
            year_int = list(range(year_int[0], year_int[1] + 1))
            result = [str(y) for y in year_int]

    return result


def eps_pe(data):
    """
    提取文本中的预测评级信息，EPS和PE信息
    :param data: 全文分好段落的字符串
    :return:
    """
    ret_years = []
    ret_eps = []
    ret_pe = []
    pe_years = []
    result = EPS_PE_RE.findall(data)  # 找到预测评级的长字符串
    result = list(set(result))  # 去重
    if len(result) == 0:
        log.debug("no pattern found")
    for proposal in result:
        log.debug("                     "
                  "++++++++++++++++++++----------------------+++++++++++"
                  "+++++++++++++++++++++++++++++++++")
        # 替换多个空格为单个空格，英文符号为中文符号
        proposal = re.sub(' +', ' ', proposal)
        proposal = re.sub(',', '，', proposal)
        proposal = re.sub(';', '；', proposal)
        proposal = re.sub('；', ',', proposal)
        dot_sents = proposal.split('。')  # 按照句号分句
        comma_sents = []
        for dot_sent in dot_sents:
            # 句子内部按照逗号继续分成短句子
            comma_sents.extend(dot_sent.split('，'))
        phrase_sents = []
        for comma_sent in comma_sents:
            # 短句子中按照年，预EPS等关键词分开，成为一个个短语，方便后面的数据提取
            if not re.search(r'PE', comma_sent):
                comma_sent = re.sub(r'年', '年*-*-*', comma_sent)
            comma_sent = re.sub(r'预', '*-*-*预', comma_sent)
            comma_sent = re.sub(r'EPS', '*-*-*EPS', comma_sent)
            phrase_sents.extend(comma_sent.split('*-*-*'))
        for phrase in phrase_sents:
            # 在短语中找到年份，EPS，PE信息
            if re.search(r"预.*年*", phrase) and not re.search(r'PE', phrase):
                years = YEAR_RE.findall(phrase)
                years = deal_years(years)
                if len(years) > 0:
                    ret_years.append(years)
                    log.debug("years: %s" % years)
                else:
                    log.info('years: NOT found!')
            elif re.search(r'EPS', phrase):
                all_eps = EPS_RE.findall(phrase)
                if len(all_eps) > 0:
                    ret_eps.append(all_eps)
                    log.debug("EPS: %s" % all_eps)
                else:
                    log.info('EPS: NOT found!')
            elif re.search(r'PE', phrase):
                pe_year = PE_YEAR_RE.findall(phrase)
                all_pe = PE_RE.findall(phrase)
                if len(pe_year) > 0 and len(all_pe) > 0:
                    pe_years.append(pe_year)
                    log.debug('pe year belongs to: %s' % pe_year)
                    all_pe = '~'.join(all_pe)
                    ret_pe.append([all_pe])
                    log.debug('PE is: %s' % all_pe)
                elif len(pe_year) == 0 and len(all_pe) > 0:
                    ret_pe.append(all_pe)
                    log.debug("PE: %s" % all_pe)
                else:
                    log.info("PE: NOT found!")
            # 可以继续补充营业收入，归母利润，增速等关键信息

        log.debug(proposal)
    # 处理带有年份的PE
    # 目前所有的值均为二维数组，优先处理年份，当年份数量大于两个时，输出warning，
    # 并只使用第一个
    if len(ret_years) > 1:
        ret_years = ret_years[:1]
        logging.warning('got year proposal more than one')
    elif len(ret_years) == 0:
        logging.warning('no years found')

    # 此时确保了年份长度为1，再来考虑EPS
    # 当EPS为1时和EPS长度为2时两种情况可以搞，其他时候不可以。
    if len(ret_eps) >= 2:
        log.warning('got eps proposal more than 2: %s' % ret_eps)
        ret_eps = ret_eps[:2]
        if ret_eps[0] == ret_eps[1]:
            ret_eps = [ret_eps[0]]

    # 当PE带有年份信息的时候，需要验证是几年
    if len(ret_pe) >= 2:
        ret_pe = ret_pe[:2]
        if ret_pe[0] == ret_pe[1]:
            ret_pe = [ret_pe[0]]
    new_pe = []
    if len(pe_years) == 1:
        if len(pe_years[0]) != len(ret_years[0]):
            tmp = []
            for y_ in ret_years[0]:
                if y_ == pe_years[0] and len(ret_pe[0]) == 1:
                    tmp.append(ret_pe[0][0])
                else:
                    tmp.append(u'')
            new_pe.append(tmp)
    elif len(pe_years) == 0:
        if len(ret_pe) > 0:
            new_pe = ret_pe
        else:
            new_pe = []
    else:
        new_pe = []

    eps_pe_template = {
        "years": ret_years,
        "eps": ret_eps,  # 有可能是两组
        "pe": new_pe  # 有可能是两组
    }

    return eps_pe_template


def parsing_price(data_str):
    """
    目标价，当前价，上次目标价等信息
    :param data_str:
    """
    # 变量定义
    ret_target_price = []
    ret_current_price = []
    ret_pre_target_price = []
    # 替换所有的应为逗号为中文逗号，多余的空格替换为一个空格,也可以用\s+
    data_str = re.sub(r',', '，', data_str)
    data_str = re.sub(r' +', ' ', data_str)
    result = PRICE_RE.findall(data_str)
    result = [s for s in result if len(s) > 0]
    if len(result) > 0:
        result = list(set(result))  # 去重

    log.debug("               +++++++++++++price "
              "proposal+++++++++++++++++++++++++")
    log.debug(result)
    # 在结果里面寻找目标价等关键字
    for proposal in result:
        # 目标价
        tp = TARGET_PRICE_RE.findall(proposal)
        tp = [s for s in tp if len(s) > 0]  # 去空结果
        # if len(tp) > 0 and isinstance(tp[0], str):
        #     tp = [tp]
        if len(tp) > 0:
            tp = list(set(tp))
            log.debug('                  target price phrase: %s' % tp)
            for t_ in tp:
                t_ = re.sub(r"RMB", "RMB*-*-*", t_)  # 利用特殊字符分短句
                t_ = re.sub(r"元", "元*-*-*", t_)
                split_t = t_.split("*-*-*")
                for sub_split_t in split_t:
                    target_price = DATA_TARGET_PRICE.findall(sub_split_t)
                    if len(target_price) > 0:
                        log.debug(
                            '                      目标价: %s' % target_price)
                        ret_target_price.extend(target_price)
        else:
            log.debug('                      target price phrase: %s' %
                      tp)

        # 上次目标价
        ptp = PRE_TARGET_PRICE_RE.findall(proposal)
        ptp = [s for s in ptp if len(s) > 0]
        if len(ptp) > 0:
            ptp = list(set(ptp))
            log.debug('                  pre target phrase: %s' % ptp)
            for t_ in ptp:
                t_ = re.sub(r"RMB", "RMB*-*-*", t_)  # 利用特殊字符分短句
                t_ = re.sub(r"元", "元*-*-*", t_)
                split_t = t_.split("*-*-*")
                for sub_split_t in split_t:
                    pre_target_price = DATA_PRE_TARGET_PRICE.findall(
                        sub_split_t)
                    if len(pre_target_price) > 0:
                        log.debug('                           上次目标价： %s' %
                                  pre_target_price)
                        ret_pre_target_price.extend(pre_target_price)

        else:
            log.debug('                  pre target phrase: %s' % ptp)

        # 当前价
        cp = CURRENT_PRICE_RE.findall(proposal)
        cp = [s for s in cp if len(s) > 0]
        if len(cp) > 0:
            cp = list(set(cp))
            log.debug(
                '                          current price phrase: %s' % cp)
            for t_ in cp:
                t_ = re.sub(r"RMB", "RMB*-*-*", t_)
                t_ = re.sub(r"元", "元*-*-*", t_)
                split_t = t_.split("*-*-*")
                for sub_split_t in split_t:
                    current_price = DATA_CURRENT_PRICE.findall(sub_split_t)
                    if len(current_price) > 0:
                        log.debug(
                            '                      当前价: %s' % current_price)
                        ret_current_price.extend(current_price)
    price_template = {
        "target_price": ret_target_price[0] if len(ret_target_price) == 1 else
        u'',
        "current_price": ret_current_price[0] if len(ret_current_price) == 1
        else
        u'',
        "previous_target_price": ret_pre_target_price[0] if len(
            ret_pre_target_price) == 1
        else u''
    }
    return price_template


def parse_publish_info(data_str):
    # 发布日期
    proposal_date = PUB_DATE_RE.search(data_str)
    if proposal_date:
        proposal_date = proposal_date.group()
    else:
        proposal_date = ''

    # 发布机构
    org_name = ''
    proposal_com_names = PUB_COMPANY_NAME.findall(data_str)
    org_names = []
    for org_pattern in org_name_pattern_list:
        for proposal_com_str in proposal_com_names:
            org_name = org_pattern.search(proposal_com_str)
            if org_name:
                org_name = org_name.group()
                org_name = orgs_dict.get(org_name, org_name)
                org_names.append(org_name)
                break
    org_names = list(set(org_names))
    if len(org_names) > 0:
        org_name = org_names[0]
    # for p in proposal_com_names:
    #     log.info("机构名称候选: %s" % p)
    return {
        "publish_date": proposal_date,
        "publish_name": org_name
    }


def security_info(data_str):
    # 证券信息
    proposal_sec_info = SEC_RE.search(data_str)
    if proposal_sec_info:
        proposal_sec_info = proposal_sec_info.group()
        if u'股票代码' in proposal_sec_info:
            proposal_sec_info = proposal_sec_info.split(' ')
            sec_name = proposal_sec_info[0]
            sec_id = re.search(u"\d{6}[\.SZ ]*", proposal_sec_info[1])
            if sec_id:
                sec_id = sec_id.group()
            else:
                sec_id = ''
        else:
            proposal_sec_info = re.sub('（', '(', proposal_sec_info)
            proposal_sec_info = re.sub('[）)]', '', proposal_sec_info)
            proposal_sec_info = proposal_sec_info.split('(')
            sec_name, sec_id = proposal_sec_info[:2]

    else:
        sec_name = ''
        sec_id = ''

    return {
        "security_name": sec_name,
        "security_id": sec_id
    }


def get_title(paragraph, top_k=5):
    paragraph = sorted(
        paragraph, key=lambda p: max([t.get('font_size')
                                        for t in p.get('texts')]), reverse=True)
    proposal = paragraph[:top_k]
    proposal = [[''.join([t.get('text') for t in p.get('texts')]),
                 max([t.get('font_size') for t in p.get('texts')])] for p in
                proposal]
    font_sizes = [p[1] for p in proposal]
    if len(set(font_sizes)) != len(font_sizes):
        new_proposal = []
        tmp_font_size = None
        for s, f in proposal:
            if tmp_font_size and f == tmp_font_size:
                new_proposal[-1] = new_proposal[-1] + s
            else:
                new_proposal.append(s)
            tmp_font_size = f
    else:
        new_proposal = proposal

    new_proposal = [p for p in new_proposal if SEC_RE_SHORT.search(p) is None
                    and len(p) > 8]
    # longest_len = max([len(s) for s in proposal])
    # title = [s for s in proposal if len(s) == longest_len]
    title = new_proposal[0]

    return {
        "title": title
    }


class Contact(object):

    def __init__(self, name):
        self.name = name
        self.tel = ''
        self.email = ''
        self.cert_id = ''

    def reset(self):
        self.name = ''
        self.tel = ''
        self.email = ''
        self.cert_id = ''

    def is_empty(self):
        if self.cert_id == '' and self.email == '' and self.cert_id == '':
            return True
        return False


def get_contact_from_column(column_str, th):
    contact_list = []
    if len(column_str) > th:
        column_str = sorted(column_str, key=lambda p: p.get('rect').center_y)
        name_exist = False
        for p in column_str:
            text = ':'.join([t.get('text') for t in p.get('texts')])

            name = FAMILY_NAME.findall(text)
            name = [n for n in name if not NAME_BLACKLIST.search(n)]
            tel = TEL.search(text)
            email = EMAIL_RE.search(text)
            cert_id = CERT_ID.search(text)

            if len(name) > 0:
                if len(contact_list) == 0:
                    c = Contact(name[0])
                    contact_list.append(c)
                else:
                    contact_list[-1] = copy.deepcopy(contact_list[-1])
                    contact_list.append(c)
                    c.reset()
                    c.name = name[0]
                name_exist = True
            if name_exist and tel:
                c.tel = tel.group()

            if name_exist and email:
                c.email = email.group()

            if name_exist and cert_id:
                c.cert_id = cert_id.group()
        contact_list = [c for c in contact_list if not c.is_empty()]
    return contact_list


def split_first_page(paragraph):
    pages = split_page(paragraph)
    first_page = pages[0]
    h, w, first_page = get_page_size(first_page, offset=50)
    return h, w, first_page


def get_contact_info(paragraph, th=10):
    # 把文本分为两栏
    # 首先拿到页面大小
    # 完全在中线一边的Bbox，如果数量足够多，则考虑有可能含有分析师信息
    # 另外一边的正文则用于寻找EPS等信息了
    h, w, first_page = split_first_page(paragraph)
    split_line = w / 2

    column_right = [p for p in first_page
                    if p.get('rect').left > split_line
                    and p.get('rect').right > split_line]
    column_left = [p for p in first_page
                   if p.get('rect').left < split_line
                   and p.get('rect').right < split_line]
    if len(column_left) > th:
        contact_list = get_contact_from_column(column_left, th)
    else:
        contact_list = get_contact_from_column(column_right, th)

    return {
        'analyst_name': [[c.name] for c in contact_list],
        'tel': [[c.tel] for c in contact_list],
        'email': [[c.email] for c in contact_list],
        'cert_id': [[c.cert_id] for c in contact_list]
    }


def merge_dict(x, y):
    import sys
    ver = sys.version_info[0]
    if ver == 2:
        z = x.copy()
        z.update(y)
    else:
        pass
    # o       z = {**x, **y}
    return z


def merge_dict_list(dict_list):
    new_dict = dict_list[0]
    for idx in range(len(dict_list)):
        if idx > 0:
            new_dict = merge_dict(new_dict, dict_list[idx])
    return new_dict


def report_classify(long_str):
    report_type = classifier.get_classifier().classify(long_str)
    report_type = TextType.cn_desc(report_type)
    return {
        "report_type": report_type
    }


def parse_forecast(long_str, paragraph, fmt='html'):
    # unicode str
    if not isinstance(long_str, unicode):
        raise TypeError('Only support unicode str')
    # 标题
    report_type = report_classify(long_str)
    title = get_title(paragraph)
    contacts = get_contact_info(paragraph)
    eps_and_pe = eps_pe(long_str)
    price = parsing_price(long_str)
    pub_date = parse_publish_info(long_str)
    sec_info = security_info(long_str)
    all_info = merge_dict_list([report_type, title, contacts, eps_and_pe, price,
                                pub_date, sec_info])
    fake_table = get_fake_table(all_info)
    # fake_table = fake_table.extend(get_fake_table(eps_and_pe))
    # 后处理，把单行的合并到一起
    row_lengths = [len(r) for r in fake_table]
    max_row_len = max(row_lengths)
    for row in fake_table:
        col_diff = max_row_len - len(row)
        if col_diff > 0 and len(row) > 0:
            row[-1]['merge_right'] = col_diff
    table_formatter = TableFormatter(fake_table)
    return table_formatter.write(fmt)


if __name__ == '__main__':
    import glob

    file_dir = "/home/xdliu/workspace/data/hdd/split1"
    for file_path in glob.glob(file_dir + "/*.txt"):
        # file_path = file_dir + "/2258091.txt"

        log.info(file_path)
        with open(file_path) as f:
            # data = json.load(f)
            data = f.readlines()
        for idx, d in enumerate(data):
            if isinstance(d, str):
                data[idx] = d.decode('utf-8')  # for python2
        data = u''.join(data)
        data = re.sub('\n', '', data)
        # draw_parse_result(data)
        # try:
        fake_t = parse_forecast(data)
        pass
        # except Exception as e:
        #     logging.exception('parse failed')
        #     continue
