# 导入函数库
import threading

from jqdatasdk import *
import statsmodels.api as sm
import time
import numpy as np
from datetime import datetime, date, timedelta

auth('17318519489','519489')

class Context:
    N=0

context = Context()
# 首次运行判断
context.init = True

# 初始化函数，设定基准等等
def initialize(index):
    set_parameter(index)
    print('%s 开盘开始' % str(context.previous_date))
    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
    # 开盘前运行
    before_market_open()
    # 开盘时运行
    market_open()
    print('%s 收盘结束'%str(context.previous_date))


def set_parameter(index):
    # 设置RSRS指标中N, M的值
    # 统计周期
    context.N = 18
    # 统计样本长度
    context.M = 1100
    # 持仓股票数
    context.stock_num = 1
    # 风险参考基准
    context.security = '000300.XSHG'
    # 记录策略运行天数
    context.days = 0
    # 买入阈值
    context.buy = 0.7
    context.sell = -0.7
    # 用于记录回归后的beta值，即斜率
    context.ans = []
    # 用于计算被决定系数加权修正后的贝塔值
    context.ans_rightdev = []
    # 双均线短线
    context.short_d = 10
    # 双均线长线
    context.long_d = 30
    context.previous_date = date.today() + timedelta(days=-index)

    context.total_value=50000
    context.positions={}

    # 计算2005年1月5日至回测开始日期的RSRS斜率指标
    prices = get_price(context.security, '2005-01-05', context.previous_date.strftime("%Y-%m-%d %H:%M:%S"), 'daily', None, True, 'none')
    highs = prices.high
    lows = prices.low
    context.ans = []
    for i in range(len(highs))[context.N:]:
        data_high = highs.iloc[i - context.N + 1:i + 1]
        data_low = lows.iloc[i - context.N + 1:i + 1]
        X = sm.add_constant(data_low)
        model = sm.OLS(data_high, X)
        results = model.fit()
        context.ans.append(results.params[1])
        # 计算r2
        context.ans_rightdev.append(results.rsquared)
    print('set parameter end')


## 开盘前运行函数
def before_market_open():
    # 输出运行时间
    print('函数运行时间(before_market_open)：%s'%str(context.previous_date))
    context.days += 1



## 开盘时运行函数
def market_open():
    security = context.security
    # 填入各个日期的RSRS斜率值
    beta = 0
    r2 = 0
    if context.init:
        context.init = False
    else:
        # RSRS斜率指标定义
        prices = get_price(security, (context.previous_date + timedelta(days = -context.N)).strftime("%Y-%m-%d %H:%M:%S"), context.previous_date.strftime("%Y-%m-%d %H:%M:%S"), 'daily', None, False, 'none')
        highs = prices.high
        lows = prices.low
        X = sm.add_constant(lows)
        model = sm.OLS(highs, X)
        beta = model.fit().params[1]
        context.ans.append(beta)
        # 计算r2
        r2 = model.fit().rsquared
        context.ans_rightdev.append(r2)

    # 计算标准化的RSRS指标
    # 计算均值序列
    section = context.ans[-context.M:]
    # 计算均值序列
    mu = np.mean(section)
    # 计算标准化RSRS指标序列
    sigma = np.std(section)
    zscore = (section[-1] - mu) / sigma
    # 计算右偏RSRS标准分
    zscore_rightdev = zscore * beta * r2
    print("time: %s, zscore_rightdev: %f"%(str(context.previous_date),zscore_rightdev))
    # 如果上一时间点的RSRS斜率大于买入阈值, 则全仓买入
    if zscore_rightdev > context.buy:
        # 记录这次买入
        print("市场风险在合理范围")
        # 满足条件运行交易
        trade_func()
    # 如果上一时间点的RSRS斜率小于卖出阈值, 则空仓卖出
    elif (zscore_rightdev < context.sell) and (len(context.positions.keys()) > 0):
        # 记录这次卖出
        print("市场风险过大，保持空仓状态")
        # 卖出所有股票,使这只股票的最终持有量为0
        for s in context.positions.keys():
            # order_target(s, 0)
            print("清仓%s"%s)


#策略选股买卖部分
def trade_func():
    #获取股票池
    df = get_fundamentals(query(valuation.code,valuation.pb_ratio,indicator.roe,valuation.pe_ratio))
    #进行pb,roe大于0筛选
    df = df[(df['roe']>0) & (df['pb_ratio']>0) & (df['pe_ratio']<30)].sort_values('pb_ratio')
    #以股票名词作为index
    df.index = df['code'].values
    #取roe倒数
    df['1/roe'] = 1/df['roe']
    #获取综合得分
    df['point'] = df[['pb_ratio','1/roe']].rank().T.apply(f_sum)
    #按得分进行排序，取指定数量的股票
    df = df.sort_values('point')[:context.stock_num]
    pool = df.index
    print('总共选出%s只股票'%len(pool))
    #得到每只股票应该分配的资金
    cash = context.total_value/len(pool)
    #获取已经持仓列表
    hold_stock = context.positions.keys()
    #卖出不在持仓中的股票
    for s in hold_stock:
        if s not in pool:
            #order_target(s,0)
            print("卖出%s"%s)
            del context.positions[s]
    #买入股票
    for s in pool:
        if s not in hold_stock:
            #order_target_value(s,cash)
            print("买入股票：%s，买入资金：%s"%(s,cash))
            context.positions.update(s=cash)

#打分工具
def f_sum(x):
    return sum(x)


def timer():
    index=85
    while True:
        index-=1
        if index==0:
            break;
        initialize(index)  # 此处为要执行的任务
        print("=======================================")
        time.sleep(1)

timer = threading.Timer(1, timer).start()