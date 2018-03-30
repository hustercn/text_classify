#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
_______________________________________________________________________________

                       Created on '2018-03-13 11:19'

                               @author: xdliu

                            All rights reserved.
********************************************************************************
"""
import copy
import json

from six import text_type
import numpy as np

from xml.etree import ElementTree


class TableFormatter(object):
    def __init__(self, table):
        "table cell: CellVecVec"
        self.table = table
        self._text = None
        self._json = None
        self._xml = None
        self._html = None

    def write(self, fmt=None):
        _write = {
            "text": self.write_text,
            "json": self.write_json,
            "xml": self.write_xml,
            "html": self.write_html
        }.get(fmt, self.write_raw)
        return _write()

    def write_raw(self):
        ll = []
        for cc in self.table:
            l = []
            for c in cc:
                l.append((c.x, c.y, c.w, c.h, c.get('text'), c.get('row_idx'), c.get('col_idx'), c.get('merge_right'), c.get('merge_down')))
            ll.append(l)
        return ll

    def write_text(self):
        if self._text:
            return self._text

        ll = []
        for cc in self.table:
            l = []
            for c in cc:
                t = c.get('text').replace('"', '""')
                l.append('"{}"'.format(t))
            ll.append(",".join(l))
        self._text = "\n".join(ll)
        return self._text

    def write_json(self):
        if self._json:
            return self._json

        jj = []
        for cc in self.table:
            j = []
            for c in cc:
                j.append(c.kv())
            jj.append(j)
        r = json.dumps(jj)
        self._json = text_type(r)
        return self._json

    def write_xml(self):
        if self._xml:
            return self._xml

        workbook = ElementTree.Element("Workbook")
        workbook.set("xmlns", "urn:schemas-microsoft-com:office:spreadsheet")
        workbook.set("xmlns:o", "urn:schemas-microsoft-com:office:office")
        workbook.set("xmlns:x", "urn:schemas-microsoft-com:office:excel")
        workbook.set("xmlns:ss", "urn:schemas-microsoft-com:office:spreadsheet")
        workbook.set("xmlns:html", "http://www.w3.org/TR/REC-html40")

        styles = ElementTree.Element("Styles")
        style = ElementTree.Element("Style")
        style.set("ss:ID", "Default")
        style.set("ss:Name", "Normal")
        alignment = ElementTree.Element("Alignment")
        alignment.set("ss:Horizontal", "Center")
        alignment.set("ss:Vertical", "Center")
        alignment.set("ss:WrapText", "1")
        font = ElementTree.Element("Font")
        font.set("ss:FontName", "宋体")
        font.set("ss:Color", "#000000")
        font.set("ss:Size", "11")
        style.append(alignment)
        style.append(font)
        styles.append(style)

        workbook.append(styles)

        worksheet = ElementTree.Element("Worksheet")
        worksheet.set("ss:Name", "Microsoft Excel")

        ele_table = ElementTree.Element("Table")

        for cc in self.table:
            ele_row = ElementTree.Element("Row")
            idx = 0
            for c in cc:
                ele_cell = ElementTree.Element("Cell")
                ele_cell.set("x", text_type(c.x))
                ele_cell.set("y", text_type(c.y))
                ele_cell.set("w", text_type(c.w))
                ele_cell.set("h", text_type(c.h))
                if c.get('merge_right') > 0:
                    ele_cell.set("ss:MergeAcross", text_type(c.get('merge_right')))
                if c.get('merge_down') > 0:
                    ele_cell.set("ss:MergeDown", text_type(c.get('merge_down')))
                if c.get('col_idx') != idx:
                    ele_cell.set("ss:Index", text_type(c.get('col_idx') + 1))
                    idx = c.get('col_idx')
                idx += 1

                ele_data = ElementTree.Element("Data")
                ele_data.set("ss:Type", "String")
                ele_data.text = c.get('text')

                ele_cell.append(ele_data)
                ele_row.append(ele_cell)
            ele_table.append(ele_row)

        worksheet.append(ele_table)
        workbook.append(worksheet)

        class dummy:
            pass

        data = []
        xmlfile = dummy()
        xmlfile.write = data.append
        ElementTree.ElementTree(workbook).write(xmlfile, encoding="utf-8")
        datax = [b'<?xml version="1.0" encoding="utf-8"?>', b'<?mso-application progid="Excel.Sheet"?>']
        for d in data:
            if d.startswith(b"<"):
                d = b"\n" + d
            datax.append(d)
        r = b"".join(datax)
        self._xml = r.decode("utf-8")
        return self._xml


    def write_html(self):
        if self._html:
            return self._html

        def get_max_col(table):
            m = 0
            for row in table:
                x = 0
                for c in row:
                    x += 1
                    x += c.get('merge_right')
                if m < x:
                    m = x
            return m

        col_max = get_max_col(self.table)
        skip = set()

        ele_table = ElementTree.Element("table")
        ele_table.set("class", "table table-sm table-bordered table-striped")  # for bootstrap

        row_idx = 0
        for row in self.table:
            ele_tr = ElementTree.Element("tr")
            col_idx = 0
            for c in row:

                while col_idx < c.get('col_idx'):
                    if (row_idx, col_idx) not in skip:
                        ele_td = ElementTree.Element("td")
                        ele_td.text = ""
                        ele_tr.append(ele_td)
                    col_idx += 1

                ele_td = ElementTree.Element("td")
                if c.get('merge_right') > 0:
                    ele_td.set("colspan", text_type(c.get('merge_right') + 1))
                    col_idx += c.get('merge_right')
                if c.get('merge_down') > 0:
                    ele_td.set("rowspan", text_type(c.get('merge_down') + 1))
                    for i in range(1, c.get('merge_down') + 1):
                        for j in range(c.get('merge_right') + 1):
                            skip.add((c.get('row_idx') + i, c.get('col_idx') + j))
                ele_td.text = c.get('text')
                ele_tr.append(ele_td)
                col_idx += 1

            while col_idx < col_max:
                if (row_idx, col_idx) not in skip:
                    ele_td = ElementTree.Element("td")
                    ele_td.text = ""
                    ele_tr.append(ele_td)
                col_idx += 1

            row_idx += 1
            ele_table.append(ele_tr)

        r = ElementTree.tostring(ele_table, encoding="utf-8", method="html")
        self._html = r.decode("utf-8")
        return self._html


class Rect(object):
    """基本矩形框类"""
    x = None
    y = None
    w = None
    h = None
    angle = 0
    rotated_rect = None

    def __init__(self, rect):
        if len(rect) == 3:
            self.angle = rect[2]
            self.rx, self.ry = rect[0]
            self.rw, self.rh = rect[1]
            self.rotated_rect = tuple(rect)
            points = cv2.boxPoints(rect)
            self.x, self.y, self.w, self.h = cv2.boundingRect(points)
        else:
            self.x, self.y, self.w, self.h = rect
            self.angle = 0
            self.rotated_rect = (
                (self.center_x, self.center_y), (self.w, self.h), 0)

    @property
    def left(self):
        """
        矩形框的左边界
        :return: 矩形框的左边界
        """
        return self.x

    @property
    def top(self):
        """
        矩形框的上边界
        :return: 矩形框的上边界
        """
        return self.y

    @property
    def right(self):
        """
        矩形框的右边界
        :return: 矩形框的右边界
        """
        return self.x + self.w

    @property
    def bottom(self):
        """
        矩形的下边界
        :return: 矩形的下边界
        """
        return self.y + self.h

    @property
    def center_x(self):
        """
        矩形框的中心x坐标
        :return: 矩形框的中心x坐标
        """
        return self.x + self.w / 2

    @property
    def center_y(self):
        """
        矩形框的中心y坐标
        :return: 矩形框的中心y坐标
        """
        return self.y + self.h / 2

    def union(self, rect2):
        """合并矩形"""
        self.x = min(self.x, rect2.x)
        self.y = min(self.y, rect2.y)
        self.w = max(self.right, rect2.right) - self.x
        self.h = max(self.bottom, rect2.bottom) - self.y

    def extend(self, dw, dh):
        """扩大矩形"""
        self.w += dw
        self.h += dh
        return self

    def move(self, dx, dy):
        """移动矩形"""
        self.x += dx
        self.y += dy
        return self

    def intersects(self, rect2):
        """判断矩形是否相交"""
        return rect2.right > self.x and \
               rect2.bottom > self.y and \
               rect2.x < self.right and \
               rect2.y < self.bottom

    def copy(self):
        return Rect((self.x, self.y, self.w, self.h))

    @property
    def area(self):
        return self.w * self.h

    @property
    def rect_3d(self):
        """
        返回类似于opencv RotatedRect结构的矩形框tuple
        :return:
        """
        return [self.x, self.y], [self.w, self.h], 0

    def __str__(self):
        return "x: %d, y: %d, w: %d, h:%d" % (self.x, self.y, self.w, self.h)

'''
class PutText(object):
    def __init__(self, ttf):
        self._face = freetype.Face(ttf)

    def draw_text(self, image, pos, text, text_size, text_color):
        """
        draw chinese(or not) text with ttf
        :param image:     image(numpy.ndarray) to draw text
        :param pos:       where to draw text
        :param text:      the context, for chinese should be unicode type
        :param text_size: text size
        :param text_color:text color
        :return:          image
        """
        self._face.set_char_size(text_size * 64)
        metrics = self._face.size
        ascender = metrics.ascender / 64.0

        # descender = metrics.descender/64.0
        # height = metrics.height/64.0
        # linegap = height - ascender + descender
        ypos = int(ascender)

        # if not isinstance(text, unicode):
        #     text = text.decode('utf-8')
        img = self.draw_string(image, pos[0], pos[1] + ypos, text, text_color)
        return img

    def draw_string(self, img, x_pos, y_pos, text, color):
        """
        draw string
        :param x_pos: text x-postion on img
        :param y_pos: text y-postion on img
        :param text:  text (unicode)
        :param color: text color
        :return:      image
        """
        prev_char = 0
        pen = freetype.Vector()
        pen.x = x_pos << 6  # div 64
        pen.y = y_pos << 6

        hscale = 1.0
        matrix = freetype.Matrix(int(hscale) * 0x10000, int(0.2 * 0x10000),
                                 int(0.0 * 0x10000), int(1.1 * 0x10000))
        cur_pen = freetype.Vector()
        pen_translate = freetype.Vector()

        image = copy.deepcopy(img)
        for cur_char in text:
            self._face.set_transform(matrix, pen_translate)

            self._face.load_char(cur_char)
            kerning = self._face.get_kerning(prev_char, cur_char)
            pen.x += kerning.x
            slot = self._face.glyph
            bitmap = slot.bitmap

            cur_pen.x = pen.x
            cur_pen.y = pen.y - slot.bitmap_top * 64
            self.draw_ft_bitmap(image, bitmap, cur_pen, color)

            pen.x += slot.advance.x
            prev_char = cur_char

        return image

    def draw_ft_bitmap(self, img, bitmap, pen, color):
        """
        draw each char
        :param bitmap: bitmap
        :param pen:    pen
        :param color:  pen color e.g.(0,0,255) - red
        :return:       image
        """
        x_pos = pen.x >> 6
        y_pos = pen.y >> 6
        cols = bitmap.width
        rows = bitmap.rows

        glyph_pixels = bitmap.buffer

        for row in range(rows):
            for col in range(cols):
                if glyph_pixels[row * cols + col] != 0:
                    img[y_pos + row][x_pos + col][0] = color[0]
                    img[y_pos + row][x_pos + col][1] = color[1]
                    img[y_pos + row][x_pos + col][2] = color[2]
'''


def table_cell(text, row_id, col_id):
    """
    表格的一个cell
    :param text:
    :param row_id:
    :param col_id:
    :return:
    """
    return {'text': text,
            'row_idx': row_id,
            'col_idx': col_id,
            'merge_down': 0,
            'merge_right': 0, 'x': 0, 'y': 0, 'w': 1, 'h': 1}


'''
def get_fake_table(kv_pair):
    """
    增加一行表格
    :param kv_pair:
    :param row_id:
    :return:
    """
    table_row = []
    row_id = 0
    for k, v in kv_pair.items():
        table_row.append(table_cell(k, row_id, 0))
        if isinstance(v, list):
            for sub_v in v:
                for col_id, sub_sub_v in enumerate(sub_v):
                    table_row.append(table_cell(sub_sub_v, row_id, col_id+1))
                row_id += 1
        else:
            table_row.append(table_cell(v, row_id, 1))
            row_id += 1
    return table_row
'''


def get_fake_table(kv_pair):
    order_list = ['publish_date', 'publish_name', 'report_type',
                  'title',
                  'security_name', 'security_id',
                  'target_price', 'current_price',
                  'years', 'eps', 'pe',
                  'analyst_name', 'tel', 'email', 'cert_id']
    black_list = ['previous_target_price']
    desc_ = {'years': u"年份",
             'eps': u"EPS",
             'pe': u"PE",
             'target_price': u"目标价",
             'pre_target_price': u"上次目标价",
             'current_price': u"当前价",
             'publish_date': u"发布日期",
             'publish_name': u"发布机构",
             'security_name': u"股票名称",
             'security_id': u"股票代码",
             'title': u"标题",
             'analyst_name': u"作者姓名",
             'tel': u"联系电话",
             'email': u"电子邮件",
             'cert_id': u"执业证书编号",
             'report_type': u"报告类型"}
    for bl in black_list:
        kv_pair.pop(bl)
    table = []
    row_id = 0
    for k in order_list:
        table_row = [table_cell(desc_.get(k), row_id, 0)]  # 把key先加进去
        v = kv_pair.get(k, [['']])
        if isinstance(v, list):
            for sub_v in v:
                for col_id, sub_sub_v in enumerate(sub_v):
                    table_row.append(table_cell(sub_sub_v, row_id, col_id+1))
                row_id += 1
        else:
            table_row.append(table_cell(v, row_id, 1))
            row_id += 1
        table.append(table_row)
    return table


if __name__ == '__main__':
    # just for test
    import cv2

    line = '你好'
    img = np.zeros([300, 300, 3])

    color_ = (0, 255, 0)  # Green
    pos = (3, 3)
    text_size = 24

    # ft = put_chinese_text('wqy-zenhei.ttc')
    # ft = PutText('msyh.ttf')
    # image = ft.draw_text(img, pos, line, text_size, color_)

    # cv2.imshow('ss', image)
    cv2.waitKey(0)
