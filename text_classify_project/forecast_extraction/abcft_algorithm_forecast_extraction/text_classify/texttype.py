# coding: utf-8
from __future__ import unicode_literals


class TextType(object):
    # 18个单类, 1个其他有意义类
    Notice              = 0  # 公告
    Report              = 1  # 研报
    Morning_Minutes     = 2  # 晨会纪要
    Morning_News        = 3  # 早间资讯
    Macro_Presentation  = 4  # 宏观简报
    Macro_Regular       = 5  # 宏观定期
    Macro_Depth         = 6  # 宏观深度
    Overseas_Economy    = 7  # 海外经济
    Strategy_Briefing   = 8  # 策略简报
    Regular_Strategy    = 9  # 策略定期
    Strategy_Depth      = 10  # 策略深度
    Global_Strategy     = 11  # 全球策略
    Industry_Briefing   = 12  # 行业简报
    Regular_Industry    = 13  # 行业定期
    Industry_Depth      = 14  # 行业深度
    Industry_Research   = 15  # 行业调研
    Company_Reviews     = 16  # 公司点评
    Company_Depth       = 17  # 公司深度
    Company_Call        = 18  # 公司调研
    Meeting_Minutes     = 19  # 会议纪要
    New_Share_Research  = 20  # 新股研究
    OTC_Company         = 21  # 新三板公司
    OTC_Market          = 22  # 新三板市场
    Other               = 23  # 其他类别

    Periodic_Notice     = 50  # 定期报告
    Major_Notice        = 51  # 重大事项
    Trading_Tips        = 52  # 交易提示
    IPO_Notice          = 53  # IPO
    Addition_Notice     = 54  # 增发
    Allotment           = 55  # 配股
    Share_Capital       = 56  # 股权股本
    General_Notice      = 57  # 一般公告

    _DESC = {
        Notice: "Notice",
        Report: "Report",
        Morning_Minutes: "Morning_Minutes",
        Morning_News: "Morning_News",
        Macro_Presentation: "Macro_Presentation",
        Macro_Regular: "Macro_Regular",
        Macro_Depth: "Macro_Depth",
        Overseas_Economy: "Overseas_Economy",
        Strategy_Briefing: "Strategy_Briefing",
        Regular_Strategy: "Regular_Strategy",
        Strategy_Depth: "Strategy_Depth",
        Global_Strategy: "Global_Strategy",
        Industry_Briefing: "Industry_Briefing",
        Regular_Industry: "Regular_Industry",
        Industry_Depth: "Industry_Depth",
        Industry_Research: "Industry_Research",
        Company_Reviews: "Company_Reviews",
        Company_Depth: "Company_Depth",
        Company_Call: "Company_Call",
        Meeting_Minutes: "Meeting_Minutes",
        New_Share_Research: "New_Share_Research",
        OTC_Company: "OTC_Company",
        OTC_Market: "OTC_Market",
        Other: "Other",
        Periodic_Notice: "Periodic_Notice",
        Major_Notice: "Major_Notice",
        Trading_Tips: "Trading_Tips",
        IPO_Notice: "IPO_Notice",
        Addition_Notice: "Addition_Notice",
        Allotment: "Allotment",
        Share_Capital: "Share_Capital",
        General_Notice: "General_Notice"
    }

    _LEVEL3 = {
        "Morning_Minutes": "晨会纪要",
        "Morning_News": "早间资讯",
        "Macro_Presentation": "宏观简报",
        "Macro_Regular": "宏观定期",
        "Macro_Depth": "宏观深度",
        "Overseas_Economy": "海外经济",
        "Strategy_Briefing": "策略简报",
        "Regular_Strategy": "策略定期",
        "Strategy_Depth": "策略深度",
        "Global_Strategy": "全球策略",
        "Industry_Briefing": "行业简报",
        "Regular_Industry": "行业定期",
        "Industry_Depth": "行业深度",
        "Industry_Research": "行业调研",
        "Company_Reviews": "公司点评",
        "Company_Depth": "公司深度",
        "Company_Call": "公司调研",
        "Meeting_Minutes": "会议纪要",
        "New_Share_Research": "新股研究",
        "OTC_Company": "新三板公司",
        "OTC_Market": "新三板市场",
        "Other": "其他类别",
    }

    _LEVEL2 = {
        "S004001": "晨会报告",
        "S004021": "宏观研究",
        "S004022": "投资策略",
        "S004023": "行业研究",
        "S004004": "公司研究",
        "S004019": "新三板研究",

        "Periodic_Notice": "定期报告",
        "Major_Notice": "重大事项",
        "Trading_Tips": "交易提示",
        "IPO_Notice": "IPO",
        "Addition_Notice": "增发",
        "Allotment": "配股",
        "Share_Capital": "股权股本",
        "General_Notice": "一般公告"
    }

    _INDUS = {
        '110100': '种植业', '110200': '渔业', '110300': '林业', '110400': '饲料', '110500': '农产品加工',
        '110600': '农业综合', '110700': '畜禽养殖', '110800': '动物保健', '210100': '石油开采', '210200': '煤炭开采',
        '210300': '其他采掘', '210400': '采掘服务', '220100': '石油化工', '220200': '化学原料', '220300': '化学制品',
        '220400': '化学纤维', '220500': '塑料', '220600': '橡胶', '230100': '钢铁', '240200': '金属非金属材料',
        '240300': '工业金属', '240400': '黄金', '240500': '稀有金属', '270100': '半导体', '270200': '元件',
        '270300': '光学光电子', '270400': '其他电子', '270500': '电子制造', '280100': '汽车整车', '280200': '汽车零部件',
        '280300': '汽车服务', '280400': '其他交运设备', '330100': '白色家电', '330200': '视听器材', '340300': '饮料制造',
        '340400': '食品加工', '350100': '纺织制造', '350200': '服装家纺', '360100': '造纸', '360200': '包装印刷',
        '360300': '家用轻工', '360400': '其他轻工制造', '370100': '化学制药', '370200': '中药', '370300': '生物制品',
        '370400': '医药商业', '370500': '医疗器械', '370600': '医疗服务', '410100': '电力', '410200': '水务',
        '410300': '燃气', '410400': '环保工程及服务', '420100': '港口', '420200': '高速公路', '420300': '公交',
        '420400': '航空运输', '420500': '机场', '420600': '航运', '420700': '铁路运输', '420800': '物流',
        '430100': '房地产开发', '430200': '园区开发', '450200': '贸易', '450300': '一般零售', '450400': '专业零售',
        '450500': '商业物业经营', '460100': '景点', '460200': '酒店', '460300': '旅游综合', '460400': '餐饮',
        '460500': '其他休闲服务', '480100': '银行', '490100': '证券', '490200': '保险', '490300': '多元金融',
        '510100': '综合', '610100': '水泥制造', '610200': '玻璃制造', '610300': '其他建材', '620100': '房屋建设',
        '620200': '装修装饰', '620300': '基础建设', '620400': '专业工程', '620500': '园林工程', '630100': '电机',
        '630200': '电气自动化设备', '630300': '电源设备', '630400': '高低压设备', '640100': '通用机械', '640200': '专用设备',
        '640300': '仪器仪表', '640400': '金属制品', '640500': '运输设备', '650100': '航天装备', '650200': '航空装备',
        '650300': '地面兵装', '650400': '船舶制造', '710100': '计算机设备', '710200': '计算机应用', '720100': '文化传媒',
        '720200': '营销传播', '720300': '互联网传媒', '730100': '通信运营', '730200': '通信设备'
    }

    @staticmethod
    def desc(ttype):
        return TextType._DESC.get(ttype)

    @staticmethod
    def is_report(ttype):
        """一级分类"""
        if isinstance(ttype, (tuple, list)):
            ttype = ttype[0]
        return ttype in [
            TextType._DESC.get(TextType.Morning_Minutes),
            TextType._DESC.get(TextType.Morning_News),
            TextType._DESC.get(TextType.Macro_Presentation),
            TextType._DESC.get(TextType.Macro_Regular),
            TextType._DESC.get(TextType.Macro_Depth),
            TextType._DESC.get(TextType.Overseas_Economy),
            TextType._DESC.get(TextType.Strategy_Briefing),
            TextType._DESC.get(TextType.Regular_Strategy),
            TextType._DESC.get(TextType.Strategy_Depth),
            TextType._DESC.get(TextType.Global_Strategy),
            TextType._DESC.get(TextType.Industry_Briefing),
            TextType._DESC.get(TextType.Regular_Strategy),
            TextType._DESC.get(TextType.Industry_Depth),
            TextType._DESC.get(TextType.Industry_Research),
            TextType._DESC.get(TextType.Company_Reviews),
            TextType._DESC.get(TextType.Company_Depth),
            TextType._DESC.get(TextType.Company_Call),
            TextType._DESC.get(TextType.Meeting_Minutes),
            TextType._DESC.get(TextType.New_Share_Research),
            TextType._DESC.get(TextType.OTC_Company),
            TextType._DESC.get(TextType.OTC_Market),
        ]

    @staticmethod
    def is_notice(ttype):
        """一级分类"""
        if isinstance(ttype, (tuple, list)):
            ttype = ttype[0]
        return ttype in [
            TextType._DESC.get(TextType.Periodic_Notice),
            TextType._DESC.get(TextType.Major_Notice),
            TextType._DESC.get(TextType.Trading_Tips),
            TextType._DESC.get(TextType.IPO_Notice),
            TextType._DESC.get(TextType.Addition_Notice),
            TextType._DESC.get(TextType.Allotment),
            TextType._DESC.get(TextType.Share_Capital),
            TextType._DESC.get(TextType.General_Notice)
        ]

    @staticmethod
    def is_morning_report(ttype):
        """研报二级分类-S004001"""
        if isinstance(ttype, (tuple, list)):
            ttype = ttype[0]
        return ttype in [
            TextType._DESC.get(TextType.Morning_Minutes),
            TextType._DESC.get(TextType.Morning_News)
        ]

    @staticmethod
    def is_macro_report(ttype):
        """研报二级分类-S004021"""
        if isinstance(ttype, (tuple, list)):
            ttype = ttype[0]
        return ttype in [
            TextType._DESC.get(TextType.Macro_Presentation),
            TextType._DESC.get(TextType.Macro_Regular),
            TextType._DESC.get(TextType.Macro_Depth),
            TextType._DESC.get(TextType.Overseas_Economy)
        ]

    @staticmethod
    def is_investment_strategy(ttype):
        """研报二级分类-S004022"""
        if isinstance(ttype, (tuple, list)):
            ttype = ttype[0]
        return ttype in [
            TextType._DESC.get(TextType.Strategy_Briefing),
            TextType._DESC.get(TextType.Regular_Strategy),
            TextType._DESC.get(TextType.Strategy_Depth),
            TextType._DESC.get(TextType.Global_Strategy)
        ]

    @staticmethod
    def is_industry_report(ttype):
        """研报二级分类-S004023"""
        if isinstance(ttype, (tuple, list)):
            ttype = ttype[0]
        return ttype in [
            TextType._DESC.get(TextType.Industry_Briefing),
            TextType._DESC.get(TextType.Regular_Industry),
            TextType._DESC.get(TextType.Industry_Depth),
            TextType._DESC.get(TextType.Industry_Research)
        ]

    @staticmethod
    def is_company_report(ttype):
        """研报二级分类-S004004"""
        if isinstance(ttype, (tuple, list)):
            ttype = ttype[0]
        return ttype in [
            TextType._DESC.get(TextType.Company_Reviews),
            TextType._DESC.get(TextType.Company_Depth),
            TextType._DESC.get(TextType.Company_Call),
            TextType._DESC.get(TextType.Meeting_Minutes),
            TextType._DESC.get(TextType.New_Share_Research)
        ]

    @staticmethod
    def is_OTC_report(ttype):
        """研报二级分类-S004019"""
        if isinstance(ttype, (tuple, list)):
            ttype = ttype[0]
        return ttype in [
            TextType._DESC.get(TextType.OTC_Company),
            TextType._DESC.get(TextType.OTC_Market)
        ]

    @staticmethod
    def get_level_one(ttype):
        """一级分类"""
        if TextType.is_report(ttype):
            return "研报"
        elif TextType.is_notice(ttype):
            return "公告"
        else:
            return "其他"

    @staticmethod
    def get_level_two(ttype):
        """研报二级分类"""
        if TextType.is_morning_report(ttype):
            return TextType._LEVEL2["S004001"]
        elif TextType.is_macro_report(ttype):
            return TextType._LEVEL2["S004021"]
        elif TextType.is_investment_strategy(ttype):
            return TextType._LEVEL2["S004022"]
        elif TextType.is_industry_report(ttype):
            return TextType._LEVEL2["S004023"]
        elif TextType.is_company_report(ttype):
            return TextType._LEVEL2["S004004"]
        elif TextType.is_OTC_report(ttype):
            return TextType._LEVEL2["S004019"]
        elif TextType.is_notice(ttype):
            return TextType._LEVEL2[ttype]
        else:
            return "其他类别暂无二级分类"

    @staticmethod
    def get_level_three(ttype):
        """研报三级分类"""
        if TextType.is_report(ttype):
            return TextType._LEVEL3[ttype]
        else:
            return "公告或其他类别暂无三级分类"

    @staticmethod
    def get_industry(ttype):
        """研报行业分类"""
        return TextType._INDUS[sorted(TextType._INDUS.keys())[ttype]]


class IndustryType(object):
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
    _INDUS = {
        '110100': '种植业', '110200': '渔业', '110300': '林业', '110400': '饲料', '110500': '农产品加工',
        '110600': '农业综合', '110700': '畜禽养殖', '110800': '动物保健', '210100': '石油开采', '210200': '煤炭开采',
        '210300': '其他采掘', '210400': '采掘服务', '220100': '石油化工', '220200': '化学原料', '220300': '化学制品',
        '220400': '化学纤维', '220500': '塑料', '220600': '橡胶', '230100': '钢铁', '240200': '金属非金属材料',
        '240300': '工业金属', '240400': '黄金', '240500': '稀有金属', '270100': '半导体', '270200': '元件',
        '270300': '光学光电子', '270400': '其他电子', '270500': '电子制造', '280100': '汽车整车', '280200': '汽车零部件',
        '280300': '汽车服务', '280400': '其他交运设备', '330100': '白色家电', '330200': '视听器材', '340300': '饮料制造',
        '340400': '食品加工', '350100': '纺织制造', '350200': '服装家纺', '360100': '造纸', '360200': '包装印刷',
        '360300': '家用轻工', '360400': '其他轻工制造', '370100': '化学制药', '370200': '中药', '370300': '生物制品',
        '370400': '医药商业', '370500': '医疗器械', '370600': '医疗服务', '410100': '电力', '410200': '水务',
        '410300': '燃气', '410400': '环保工程及服务', '420100': '港口', '420200': '高速公路', '420300': '公交',
        '420400': '航空运输', '420500': '机场', '420600': '航运', '420700': '铁路运输', '420800': '物流',
        '430100': '房地产开发', '430200': '园区开发', '450200': '贸易', '450300': '一般零售', '450400': '专业零售',
        '450500': '商业物业经营', '460100': '景点', '460200': '酒店', '460300': '旅游综合', '460400': '餐饮',
        '460500': '其他休闲服务', '480100': '银行', '490100': '证券', '490200': '保险', '490300': '多元金融',
        '510100': '综合', '610100': '水泥制造', '610200': '玻璃制造', '610300': '其他建材', '620100': '房屋建设',
        '620200': '装修装饰', '620300': '基础建设', '620400': '专业工程', '620500': '园林工程', '630100': '电机',
        '630200': '电气自动化设备', '630300': '电源设备', '630400': '高低压设备', '640100': '通用机械', '640200': '专用设备',
        '640300': '仪器仪表', '640400': '金属制品', '640500': '运输设备', '650100': '航天装备', '650200': '航空装备',
        '650300': '地面兵装', '650400': '船舶制造', '710100': '计算机设备', '710200': '计算机应用', '720100': '文化传媒',
        '720200': '营销传播', '720300': '互联网传媒', '730100': '通信运营', '730200': '通信设备'
    }


Label_Line = [
    TextType.Macro_Presentation,
    TextType.Macro_Regular,
    TextType.Macro_Depth,
    TextType.Strategy_Briefing,
    TextType.Regular_Strategy,
    TextType.Strategy_Depth,
    TextType.Industry_Briefing,
    TextType.Regular_Strategy,
    TextType.Industry_Depth,
    TextType.Notice,
    TextType.Other,
    TextType.Morning_Minutes,
    TextType.Morning_News,
    TextType.Company_Call,
    TextType.Company_Reviews,
    TextType.New_Share_Research,
    TextType.Meeting_Minutes,
    TextType.Company_Depth,
    TextType.OTC_Company,
    TextType.OTC_Market,
    TextType.Overseas_Economy,
    TextType.Global_Strategy,
    TextType.Industry_Research,
]

Label_Line2 = [
    TextType.Periodic_Notice,
    TextType.Major_Notice,
    TextType.Trading_Tips,
    TextType.IPO_Notice,
    TextType.Addition_Notice,
    TextType.Allotment,
    TextType.Share_Capital,
    TextType.General_Notice
]


class BaseClassifier(object):
    def classify(self, textfile, title):
        raise NotImplemented

    def classify_batch(self, textfile, title, batch_size):
        raise NotImplemented
