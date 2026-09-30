"""
update:2025/12/10
for: adv class
author:Nick_Ni
"""

import pandas as pd 
import numpy as np
from scipy import stats
import statsmodels.api as sm
from tqdm import *
import os
from pathlib import Path
from joblib import Parallel, delayed
import pickle
import time

# from rqdatac import *
# from rqfactor import *
# # from rqfactor import Factor
# from rqfactor.extension import *

# import rqdatac

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib


# 关闭通知
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)




# plt.style.use('default')
# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False

#  plt.legend(bbox_to_anchor = (1.05,0),loc = 3,borderaxespad = 0)


# # 算子 update

# # 算子券池阈
# def universe_mask():
#     mask = pd.DataFrame({
#         k: dict.fromkeys(v, True)
#         for k, v in rqdatac.index_components(index_item, start_date='2010-01-01', end_date=time.strftime('%F')).items()
#     }).T
#     mask.fillna(False, inplace=True)
#     return mask

# def mask_universe(df):
#     mask = universe_mask()
#     mask = mask.reindex(index=df.index, columns=df.columns, fill_value=False)
#     return df.where(mask)

# def MASK_UNIVERSE(f):
#     return UnaryCrossSectionalFactor(mask_universe, f)


# # 截面减均值
# def DEMEAN_UNIVERSE(f):
#     return DEMEAN(MASK_UNIVERSE(f))


# # 截面标准化
# def zscore_universe(df):
#     mask = universe_mask()
#     mask = mask.reindex(index=df.index, columns=df.columns, fill_value=False)
#     df = df.where(mask)
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)

#     std = df.std(axis=1)
#     mean = df.mean(axis=1)
#     df = df.sub(mean, axis=0).div(std, axis=0)

#     df[std == 0] = 0
#     return df

# def CS_ZSCORE_UNIVERSE(f):
#     return UnaryCrossSectionalFactor(zscore_universe, f)




# 新建文件夹
def create_dir_not_exist(path):
    # 若不存在该路径则自动生成
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass


# 动态券池
def INDEX_FIX(start_date,end_date,index_item):
    """
    :param start_date: 开始日 -> str
    :param end_date: 结束日 -> str 
    :param index_item: 指数代码 -> str 
    :return index_fix: 动态因子值 -> df_unstack
    """
    
    index = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in index_components(index_item,start_date=start_date,end_date=end_date).items()])).T

    # 构建动态股票池 
    index_fix = index.unstack().reset_index().iloc[:,-2:]
    index_fix.columns = ['date','stock']
    index_fix.date = pd.to_datetime(index_fix.date)
    index_fix['level'] = True
    index_fix.dropna(inplace = True)
    index_fix = index_fix.set_index(['date','stock']).level.unstack()
    index_fix.fillna(False,inplace = True)

    return index_fix

# 1.1 新股过滤（新股财报完整性确实，没有完整的4个财报季）
def get_new_stock_filter(stock_list,date_list, newly_listed_threshold = 252):
    newly_listed_threshold = 252
    # 获取上市日期
    listed_date_list = [rqdatac.instruments(stock).listed_date for stock in stock_list] 
    # 获取上市后的第252个交易日（新股和老股的分界点）
    newly_listed_window = pd.Series(index=stock_list, 
                                    data=[pd.to_datetime(rqdatac.get_next_trading_date(listed_date, n=newly_listed_threshold)) for listed_date in listed_date_list]) 
    # 防止分割日在研究日之后，后续填充不存在
    for k,v in enumerate(newly_listed_window):
        if v > date_list[-1]:
            newly_listed_window.iloc[k] = date_list[-1]

    # 标签新股，构建过滤表格
    newly_listed_window.index.names = ['order_book_id']
    newly_listed_window = newly_listed_window.to_frame('date')
    newly_listed_window['signal'] = True
    newly_listed_window = newly_listed_window.reset_index().set_index(['date','order_book_id']).signal.unstack('order_book_id').reindex(index=date_list)
    newly_listed_window = newly_listed_window.shift(-1).bfill().fillna(False)

    return newly_listed_window


# 1.2 st过滤（风险警示标的默认不进行研究）
def get_st_filter(stock_list,date_list):
    # 当st时返回1，非st时返回0
    st_filter = rqdatac.is_st_stock(stock_list,date_list[0],date_list[-1]).reindex(columns=stock_list,index = date_list)
    st_filter = st_filter.shift(-1).fillna(method = 'ffill')

    return st_filter

# 1.3 停牌过滤 （无法交易）
def get_suspended_filter(stock_list,date_list):
    # 当停牌时返回1，非停牌时返回0
    suspended_filter = rqdatac.is_suspended(stock_list,date_list[0],date_list[-1]).reindex(columns=stock_list,index=date_list)
    suspended_filter = suspended_filter.shift(-1).fillna(method = 'ffill')

    return suspended_filter

# 1.4 涨停过滤 （开盘无法买入）
def get_limit_up_filter(stock_list,date_list):
    # 涨停时返回为1,非涨停返回为0    
    price = rqdatac.get_price(stock_list,date_list[0],date_list[-1],adjust_type='none',fields = ['open','limit_up'])
    df = (price['open'] == price['limit_up']).unstack('order_book_id').shift(-1).fillna(False)

    return df

# 2 数据清洗函数 -----------------------------------------------------------

# 2.1 MAD:中位数去极值
# MAD:中位数去极值
def filter_extreme_MAD(series,n = 3 * 1.4826): 
    median = series.median()
    new_median = ((series - median).abs()).median()
    return series.clip(median - n*new_median,median + n*new_median)

# 2.1 MAD:中位数去极值
def mad(df,n = 3 * 1.4826):
    # 离群值处理
    df = df.apply(lambda x :filter_extreme_MAD(x,n = n), axis=1)

    return df


# 2.2 标准化（去量纲）
def standardize(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


# 2.3 行业市值中性化
def neutralization(df,order_book_ids,index_item = '',industry_type = 'zx'):

    """
    :param df: 因子值 -> df_unstack
    :param order_book_ids: 股票队列 -> list
    :param index_item: 指数名称 -> str
    :return df_result: 残差因资质 -> df_unstack
    """

    datetime_period = df.index.tolist()
    start = datetime_period[0].strftime("%F")
    end = datetime_period[-1].strftime("%F")
    #获取行业/市值暴露度
    try:
        # 获取存储数据
        df_industry_market = pd.read_pickle(f'tmp/df_industry_market_{industry_type}_{index_item}_{start}_{end}.pkl')
    except:
        # 获取市值暴露度
        market_cap = execute_factor(LOG(Factor('market_cap_3')),order_book_ids,start,end).stack().to_frame('market_cap')
        # 获取行业暴露度
        industry_df = pd.get_dummies(get_industry_exposure(order_book_ids,
                                                           datetime_period,
                                                           industry_type))
        # 合并市值行业暴露度
        industry_df['market_cap'] = market_cap
        df_industry_market = industry_df
        df_industry_market.index.names = ['datetime','order_book_id']
        df_industry_market.dropna(inplace = True)
        create_dir_not_exist('tmp')
        df_industry_market.to_pickle(f'tmp/df_industry_market_{industry_type}_{index_item}_{start}_{end}.pkl')

    df_industry_market['factor'] = df.stack()
    df_industry_market.dropna(subset = 'factor',inplace = True)
    
    # OLS回归取残差 剥离行业市值风格的影响（残差 -> 新因子）

    # 方法一：普通 ==================

    # df_result = pd.DataFrame(columns = order_book_ids,index = datetime_period)
    # for i in tqdm(datetime_period):
    #     try:
    #         df_day = df_industry_market.loc[i]
    #         x = df_day.iloc[:,:-1]   # 市值/行业
    #         y = df_day.iloc[:,-1]    # 因子值
    #         df_result.loc[i] = sm.OLS(y.astype(float),x.astype(float),hasconst=False, missing='drop').fit().resid   # 回顾计算残差
    #     except:
    #         pass
    
    # 方法二: 多线程 ==================

    def get_ols_mult(df_industry_market,i):
        try:
            df_day = df_industry_market.loc[i]
            x = df_day.iloc[:,:-1]   # 市值/行业
            y = df_day.iloc[:,-1]    # 因子值
            resid = sm.OLS(y.astype(float),x.astype(float),hasconst=False, missing='drop').fit().resid
        except:
            resid = np.array([np.nan])
        return resid

    df_result = Parallel(n_jobs=2)(delayed(get_ols_mult)(df_industry_market,i) for i in datetime_period)
    df_result = pd.DataFrame(df_result,index = datetime_period).sort_index(axis = 1)

    df_result = df_result.dropna(how = 'all',axis = 1)                 # 删除整列为空
    df_result.index.names = ['datetime']

    return df_result


# 2.3.1 获取行业暴露度举证
def get_industry_exposure(order_book_ids,datetime_period,industry_type = 'zx'):
    
    """
    :param order_book_ids: 股票池 -> list
    :param datetime_period: 研究日 -> list
    :param industry_type: 行业分类标准 二选一 中信/申万 zx/sw -> str
    :return result: 虚拟变量 -> df_unstack
    """

    if industry_type not in ['zx','sw']:
        return print("select on from ['zx','sw']")
    
    # 获取行业特征数据
    if industry_type == 'zx':
        industry_map_dict = rqdatac.client.get_client().execute('__internal__zx2019_industry')
        # 构建个股/行业map
        df = pd.DataFrame(industry_map_dict, columns=["first_industry_name", "order_book_id", "start_date"])
        df.sort_values(["order_book_id", "start_date"], ascending=True, inplace=True)
        df = df.pivot(index="start_date", columns="order_book_id", values="first_industry_name").ffill()
    else:
        industry_map_dict = rqdatac.client.get_client().execute('__internal__shenwan_industry')
        df = pd.DataFrame(industry_map_dict, columns=["index_name", "order_book_id",'version', "start_date"])
        df = df[df.version == 2]
        df = df.drop_duplicates()
        df = df.set_index(['start_date','order_book_id']).index_name.unstack('order_book_id').ffill()

    # 匹配交易日
    date_list_base = pd.to_datetime(get_trading_dates(get_previous_trading_date(df.index[0],2),df.index[-1]))
    df.index = date_list_base.take(date_list_base.searchsorted(df.index, side='right') - 1)
    # 切片所需日期
    df = df.reset_index().drop_duplicates(subset = ['index']).set_index('index')
    df = df.reindex(index = date_list_base).ffill().reindex(index = datetime_period).ffill()
    inter_stock_list = list(set(df.columns) & set(order_book_ids))
    df = df[inter_stock_list].sort_index(axis = 1)
    
    #生成行业虚拟变量
    return df.stack()



# 2.4 因子清洗
def data_clean(df,index_fix,index_item = '',industry_type = 'zx'):
    """
    :param df: 因子值 -> df_unstack
    :param index_fix: 动态券池 -> df_unstack
    :param index_item: 指数代码 -> str
    :param industry_type: 行业分类标准 二选一 中信/申万 zx/sw -> str
    :return df: 清洗后因子值 -> df_unstack
    """
    stock_list = index_fix.columns.tolist()
    start_date = index_fix.index[0].strftime('%F')
    end_date = index_fix.index[-1].strftime('%F')
    date_list = pd.to_datetime(index_fix.index.tolist())
    try:
        combo_mask = pd.read_pickle(f'tmp/combo_mask_{index_item}_{start_date}_{end_date}.pkl')
    except:
        # 新股过滤
        new_stock_filter = get_new_stock_filter(stock_list,date_list)
        # st过滤
        st_filter = get_st_filter(stock_list,date_list)
        # 停牌过滤
        suspended_filter = get_suspended_filter(stock_list,date_list)

        combo_mask = (new_stock_filter.astype(int) 
                    + st_filter.astype(int)
                    + suspended_filter.astype(int)
                    + (~index_fix).astype(int)) == 0

        create_dir_not_exist('tmp')
        combo_mask.to_pickle(f'tmp/combo_mask_{index_item}_{start_date}_{end_date}.pkl')
    
    df = df.mask(~combo_mask).dropna(how = 'all',axis = 1)
    # 离群值 标准化 中性化 标准化
    df = standardize(neutralization(standardize(mad(df)),stock_list,index_item,industry_type))
    df = df.apply(lambda x: x.astype(float))

    try:
        limit_up_filter = pd.read_pickle(f'tmp/limit_up_filter_{index_item}_{start_date}_{end_date}.pkl')
    except:
        # 涨停过滤
        limit_up_filter = get_limit_up_filter(stock_list,date_list)
        limit_up_filter.to_pickle(f'tmp/limit_up_filter_{index_item}_{start_date}_{end_date}.pkl')
    
    df = df.mask(limit_up_filter)
    
    return df


# IC计算  
def Quick_Factor_Return_N_IC(df,n,index_item = '000985.XSHG',name = '',Rank_IC = True):

    """
    :param df: 因子值 -> unstack
    :param n: 调仓日 -> int
    :param index_item: 券池 -> str
    :param name: 因子名称 -> str
    :param True/False: Rank_ic/Normal_ic -> bool
    :return result: ic序列 -> series
    :return report: ic报告 -> dataframe
    """
    order_book_ids = df.columns.tolist()
    datetime_period = df.index.tolist()
    start = datetime_period[0].strftime('%F')
    end = datetime_period[-1].strftime('%F')
    try:
        close = pd.read_pickle(f'tmp/close_{index_item}_{start}_{end}.pkl')
    except:
        index_fix = INDEX_FIX(start,end,index_item)
        order_book_ids = index_fix.columns.tolist()
        close = get_price(order_book_ids, start_date=start, end_date=end,frequency='1d',fields='close').close.unstack('order_book_id')
        create_dir_not_exist('tmp')
        close.to_pickle(f'tmp/close_{index_item}_{start}_{end}.pkl')

    return_n = close.pct_change(n).shift(-n)

    if Rank_IC == True:
        result = df.corrwith(return_n,axis = 1,method='spearman').dropna(how = 'all')
    else:
        result = df.corrwith(return_n,axis = 1,method='pearson').dropna(how = 'all')

    t_stat,_ = stats.ttest_1samp(result, 0)

    report = {'name': name,
    'IC mean':round(result.mean(),4),
    'IC std':round(result.std(),4),
    'IR':round(result.mean()/result.std(),4),
    'IR_ly':round(result[-252:].mean()/result[-252:].std(),4),
    'IC>0':round(len(result[result>0].dropna())/len(result),4),
    'ABS_IC>2%':round(len(result[abs(result) > 0.02].dropna())/len(result),4),
    't_stat':round(t_stat,4),
    }

    print(report)
    report = pd.DataFrame([report])

    return result,report



# 买入队列构建
def get_buy_list(df,top_tpye = 'rank',rank_n = 100,quantile_q = 0.8):
    """
    :param df: 因子值 -> dataframe/unstack
    :param top_tpye: 选择买入队列方式，从['rank','quantile']选择一种方式 -> str
    :param rank_n: 值最大的前n只的股票 -> int
    :param quantile_q: 值最大的前n分位数的股票 -> float
    :return df: 买入队列 -> dataframe/unstack
    """

    if top_tpye == 'rank':
        df = df.rank(axis  = 1,ascending=False) <= rank_n
    elif top_tpye == 'quantile':
        df = df.sub(df.quantile(quantile_q,axis = 1),axis = 0) > 0
    else:
        print("select one from ['rank','quantile']")

    df = df.astype(int)
    df = df.replace(0,np.nan).dropna(how = 'all',axis = 1)
    
    return df

# 获取本地开盘价
def get_bar(df, adjust='none'):
    """
    :param df: 权重表，使用其日期索引和股票列对齐行情。
    :param adjust: 保留以兼容旧调用，不改变本地文件的复权口径。
    :return: open_price.pkl 中的开盘价，缺失日期或股票保留 NaN。
    """
    price_path = Path(__file__).resolve().with_name('open_price.pkl')
    price_open = pd.read_pickle(price_path)
    return price_open.reindex(index=df.index, columns=df.columns)


# 回测框架
def backtest(df_weight, change_n = 20, cash = 10000 * 10000, tax = 0.0005, other_tax = 0.0001, commission = 0.0002, min_fee = 5, cash_interest_yield = 0.02):
    change_day = sorted(set(df_weight.index.tolist()[::change_n] + [df_weight.index[-1]]))
    return _backtest_on_dates(df_weight, change_day, cash, tax, other_tax,
                              commission, min_fee, cash_interest_yield)


def backtest_month(df_weight, change_month='月初', cash=10000 * 10000,
                   tax=0.0005, other_tax=0.0001, commission=0.0002,
                   min_fee=5, cash_interest_yield=0.02):
    """按月初、月中或月末调仓，返回与backtest相同的账户DataFrame。

    change_month: '月初' / '月中' / '月末'，也支持 start / mid / end。
    月中为每月15日之后（不含15日）的首个交易日。
    使用benchmark.pkl完整日期索引作为交易日历，权重缺失日为空仓信号。
    首个调仓日前持有现金；不在本函数中移位权重。
    最后一个交易日仅作为期末估值日，不进行无后续持有期的调仓。
    交易日历须覆盖所研究月份；文件末尾不足整月时无法判断真实月末。
    """
    aliases = {'月初': 'start', '月中': 'mid', '月末': 'end',
               'start': 'start', 'mid': 'mid', 'end': 'end'}
    if change_month not in aliases:
        raise ValueError("change_month 必须为 '月初'、'月中'或'月末'")
    if df_weight.empty or not isinstance(df_weight.index, pd.DatetimeIndex):
        raise ValueError('df_weight 必须非空，并使用DatetimeIndex')
    if not df_weight.index.is_unique or not df_weight.index.is_monotonic_increasing:
        raise ValueError('df_weight 日期必须升序且无重复')
    calendar = pd.DatetimeIndex(pd.read_pickle(
        Path(__file__).resolve().with_name('benchmark.pkl')).index).sort_values().unique()
    if not df_weight.index.isin(calendar).all():
        raise ValueError('权重日期不在本地benchmark交易日历内')
    mode = aliases[change_month]
    eligible_calendar = calendar[calendar.day > 15] if mode == 'mid' else calendar
    grouped_dates = eligible_calendar.to_series().groupby(eligible_calendar.to_period('M'))
    scheduled = pd.DatetimeIndex(grouped_dates.max() if mode == 'end' else grouped_dates.min())
    dates = calendar[(calendar >= df_weight.index[0]) & (calendar <= df_weight.index[-1])]
    weights = df_weight.reindex(dates)
    # 期末日为估值终点；只保留在终点之前能实际持有的调仓日。
    trade_dates = scheduled[(scheduled >= dates[0]) & (scheduled < dates[-1])]
    daily_interest = (1 + cash_interest_yield) ** (1 / 252) - 1
    initial_cash_curve = cash * (1 + daily_interest) ** np.arange(len(dates))
    account = pd.DataFrame({
        'total_account_asset': initial_cash_curve,
        'holding_market_cap': np.zeros(len(dates)),
        'cash_account': initial_cash_curve,
    }, index=dates).round(2)
    if len(trade_dates):
        first_trade = trade_dates[0]
        first_pos = dates.get_loc(first_trade)
        available_cash = float(initial_cash_curve[first_pos])
        change_days = trade_dates.tolist() + [dates[-1]]
        traded = _backtest_on_dates(
            weights.loc[first_trade:], change_days, available_cash, tax, other_tax,
            commission, min_fee, cash_interest_yield, initial_record=False)
        account.loc[traded.index, traded.columns] = traded
    account.attrs['rebalance_dates'] = trade_dates.tolist()
    account.attrs['rebalance_rule'] = change_month
    return account


def _backtest_on_dates(df_weight, change_day, cash, tax, other_tax,
                       commission, min_fee, cash_interest_yield, initial_record=True):

    # 基础参数
    inital_cash = cash                                                                                                            # 起始资金
    stock_holding_num_hist = 0                                                                                                    # 初始化持仓       
    buy_cost = other_tax + commission                                                                                             # 买入交易成本
    sell_cost = tax + other_tax + commission                                                                                      # 卖出交易成本
    cash_interest_daily = (1 + cash_interest_yield) ** (1/252) - 1                                                                # 现金账户利息(日)
    account = pd.DataFrame(index = df_weight.index,columns=['total_account_asset','holding_market_cap','cash_account'], dtype=float)           # 账户信息存储
    price_open = get_bar(df_weight)                                                                                              # 读取本地开盘价
    price_open_pre = price_open.copy()                                                                                           # 使用同一份行情，不额外复权
    stock_round_lot = pd.Series(100, index=df_weight.columns, dtype=int)                                                          # 每只股票固定按100股整手计算

    # 滚动计算
    for i in tqdm(range(0,len(change_day)-1)):
        start_date = change_day[i]
        end_date = change_day[i+1]

        # 获取给定权重
        df_weight_temp = df_weight.loc[start_date].dropna()
        stock_list_temp = df_weight_temp.index.tolist()
        # 计算个股持股数量 = 向下取整(给定权重 * 可用资金 // 最小买入股数) * 最小买入股数
        stock_holding_num = ((df_weight_temp 
                            * cash 
                            / (price_open.loc[start_date,stock_list_temp] * (1 + sell_cost))        # 预留交易费用
                            // stock_round_lot.loc[stock_list_temp]) 
                            * stock_round_lot.loc[stock_list_temp])

        # 仓位变动      
        ## 防止相减为空 & 剔除无变动
        stock_holding_num_change = stock_holding_num.sub(stock_holding_num_hist,fill_value = 0).replace(0,np.nan).dropna()
        # 获取期间价格
        price_open_temp = price_open.loc[start_date]           # 引入完整券池
        
        # 计算交易成本 (可设置万一免五)
        def calc_fee(x,min_fee):
            if x < 0:
                fee_temp = -1 * x * sell_cost                                                                                       # 印花税 + 过户费等 + 佣金
            else:
                fee_temp = x * buy_cost                                                                                             # 过户费等 + 佣金
            # 最低交易成本限制
            if fee_temp > min_fee:
                return fee_temp
            else:
                return min_fee

        transaction_costs = ((price_open_temp
                            * stock_holding_num_change)).apply(lambda x: calc_fee(x,min_fee)).sum()
        
        adjust_ratio = price_open_pre.loc[start_date:end_date]/price_open_pre.loc[start_date]                                        # 复权比率
        adjust_ratio_price = adjust_ratio.T.mul(price_open_temp,axis = 0).dropna(how = 'all').T                                      # 除权价格复权
        
        # 计算期间市值 （交易手续费在现金账户计提）
        holding_market_cap = (adjust_ratio_price * stock_holding_num).sum(axis =1)
        cash_account = cash - transaction_costs - holding_market_cap.loc[start_date]
        cash_account = pd.Series([cash_account * ((1 + cash_interest_daily)**(i+1)) for i in range(0,len(holding_market_cap))],
                                index = holding_market_cap.index)
        total_account_asset = holding_market_cap + cash_account
        
        # 将当前持仓存入 
        stock_holding_num_hist = stock_holding_num
        # 下一期期初可用资金
        cash = total_account_asset.loc[end_date]

        account.loc[start_date:end_date,'total_account_asset'] = round(total_account_asset,2)
        account.loc[start_date:end_date,'holding_market_cap'] = round(holding_market_cap,2)
        account.loc[start_date:end_date,'cash_account'] = round(cash_account,2)

    # 使用权重表首日作为初始资金基准（持仓为0），不新增前一交易日。
    if initial_record:
        account.loc[df_weight.index[0]] = [inital_cash, 0, inital_cash]
    account = account.sort_index()
    
    return account



def get_benchmark(df, benchmark='000300.XSHG', benchmark_type='mcw'):
    """
    从本文件同目录的 benchmark.pkl 读取沪深300价格，并对齐账户日期。
    benchmark、benchmark_type 保留以兼容旧调用，基准固定为 000300.XSHG。
    返回价格而非收益率，由绩效分析中的 pct_change 统一计算逐日收益。
    """
    benchmark_path = Path(__file__).resolve().with_name('benchmark.pkl')
    benchmark_price = pd.read_pickle(benchmark_path)
    return benchmark_price[['000300.XSHG']].reindex(df.index)



# 回测绩效指标绘制
def get_performance_analysis(account_result,benchmark_index = '000300.XSHG',benchmark_type = 'mcw'):
    
    rf = 0.03

    # 加入基准    
    performance = pd.concat([account_result['total_account_asset'].to_frame('strategy'),
                             get_benchmark(account_result,benchmark_index,benchmark_type)],axis = 1)
    performance = performance.astype(float)
    benchmark_index = performance.columns[1]  
    performance_net = performance.pct_change(fill_method=None).dropna(how = 'all')
    performance_cumnet = (1 + performance_net).cumprod()
    performance_cumnet['alpha'] = performance_cumnet['strategy']/performance_cumnet[benchmark_index]
    performance_cumnet = performance_cumnet.fillna(1)

    # 指标计算
    performance_pct = performance_cumnet.pct_change().dropna()

    # 策略收益
    strategy_name,benchmark_name,alpha_name = performance_cumnet.columns.tolist() 
    Strategy_Final_Return = performance_cumnet[strategy_name].iloc[-1] - 1

    # 策略年化收益
    Strategy_Annualized_Return_EAR = (1 + Strategy_Final_Return) ** (252/len(performance_cumnet)) - 1

    # 基准收益
    Benchmark_Final_Return = performance_cumnet[benchmark_name].iloc[-1] - 1

    # 基准年化收益
    Benchmark_Annualized_Return_EAR = (1 + Benchmark_Final_Return) ** (252/len(performance_cumnet)) - 1

    # alpha 
    ols_result = sm.OLS(performance_pct[strategy_name] * 252 - rf, sm.add_constant(performance_pct[benchmark_name] * 252 - rf)).fit()
    Alpha = ols_result.params.iloc[0]

    # beta
    Beta = ols_result.params.iloc[1]

    # beta_2 = np.cov(performance_pct[strategy_name],performance_pct[benchmark_name])[0,1]/performance_pct[benchmark_name].var()
    # 波动率
    Strategy_Volatility = performance_pct[strategy_name].std() * np.sqrt(252)

    # 夏普
    Strategy_Sharpe = (Strategy_Annualized_Return_EAR - rf)/Strategy_Volatility

    # 下行波动率
    strategy_ret = performance_pct[strategy_name]
    Strategy_Down_Volatility = strategy_ret[strategy_ret < 0].std() * np.sqrt(252)

    # sortino
    Sortino = (Strategy_Annualized_Return_EAR - rf)/Strategy_Down_Volatility
    
    # 跟踪误差
    Tracking_Error = (performance_pct[strategy_name] - performance_pct[benchmark_name]).std() * np.sqrt(252)

    # 信息比率
    Information_Ratio = (Strategy_Annualized_Return_EAR - Benchmark_Annualized_Return_EAR)/Tracking_Error

    # 最大回测
    i = np.argmax((np.maximum.accumulate(performance_cumnet[strategy_name]) 
                    - performance_cumnet[strategy_name])
                    /np.maximum.accumulate(performance_cumnet[strategy_name]))
    j = np.argmax(performance_cumnet[strategy_name][:i])
    Max_Drawdown = (1-performance_cumnet[strategy_name].iloc[i]/performance_cumnet[strategy_name].iloc[j])

    # 卡玛比率
    Calmar = (Strategy_Annualized_Return_EAR)/Max_Drawdown

    # 超额收益
    Alpha_Final_Return = performance_cumnet[alpha_name].iloc[-1] - 1

    # 超额年化收益
    Alpha_Annualized_Return_EAR = (1 + Alpha_Final_Return) ** (252/len(performance_cumnet)) - 1

    # 超额波动率
    Alpha_Volatility = performance_pct[alpha_name].std() * np.sqrt(252)

    # 超额夏普
    Alpha_Sharpe = (Alpha_Annualized_Return_EAR - rf)/Alpha_Volatility

    # 超额最大回撤
    i = np.argmax((np.maximum.accumulate(performance_cumnet[alpha_name]) 
                    - performance_cumnet[alpha_name])
                    /np.maximum.accumulate(performance_cumnet[alpha_name]))
    j = np.argmax(performance_cumnet[alpha_name][:i])
    Alpha_Max_Drawdown = (1-performance_cumnet[alpha_name].iloc[i]/performance_cumnet[alpha_name].iloc[j])

    # 胜率
    performance_pct['win'] = performance_pct[alpha_name] > 0
    Win_Ratio = performance_pct['win'].value_counts().loc[True] / len(performance_pct)

    # 盈亏比
    profit_lose = performance_pct.groupby('win')[alpha_name].mean()
    Profit_Lose_Ratio = abs(profit_lose[True]/profit_lose[False])
    

    result = {
        '策略累计收益':round(Strategy_Final_Return,4),
        '策略年化收益': round(Strategy_Annualized_Return_EAR,4),
        '基准累计收益':round(Benchmark_Final_Return,4),
        '基准年化收益': round(Benchmark_Annualized_Return_EAR,4),
        '阿尔法':round(Alpha,4),
        '贝塔':round(Beta,4),
        '波动率':round(Strategy_Volatility,4),
        '夏普比率':round(Strategy_Sharpe,4),
        '下行波动率':round(Strategy_Down_Volatility,4),
        '索提诺比率':round(Sortino,4),
        '跟踪误差':round(Tracking_Error,4),
        '信息比率':round(Information_Ratio,4),
        '最大回撤':round(Max_Drawdown,4),
        '卡玛比率': round(Calmar,4),
        '超额累计收益':round(Alpha_Final_Return,4),
        '超额年化收益': round(Alpha_Annualized_Return_EAR,4),
        '超额波动率':round(Alpha_Volatility,4),
        '超额夏普':round(Alpha_Sharpe,4),
        '超额最大回撤':round(Alpha_Max_Drawdown,4),
        '胜率':round(Win_Ratio,4),
        '盈亏比':round(Profit_Lose_Ratio,4)

    }
    

    return performance_cumnet,result


def get_benchmark1(df,benchmark = '000985.XSHG'):
    """
    :param df: 买入队列 -> dataframe/unstack
    :param benchmark: 基准指数 -> str
    :return ret: 基准的逐日收益 -> dataframe
    """
    start_date = get_previous_trading_date(df.index.min(),1).strftime('%F')
    end_date = df.index.max().strftime('%F')
    price_open = get_price([benchmark],start_date,end_date,fields=['open']).open.unstack('order_book_id')
    
    return price_open


def backtest1(df_weight, change_n = 20, cash = 10000 * 1000, tax = 0.0005, other_tax = 0.0001, commission = 0.0002, min_fee = 5, cash_interest_yield = 0.02):

    # 基础参数
    inital_cash = cash                                                                                                            # 起始资金
    stock_holding_num_hist = 0                                                                                                    # 初始化持仓       
    buy_cost = other_tax + commission                                                                                             # 买入交易成本
    sell_cost = tax + other_tax + commission                                                                                      # 卖出交易成本
    cash_interest_daily = (1 + cash_interest_yield) ** (1/252) - 1                                                                # 现金账户利息(日)
    account = pd.DataFrame(index = df_weight.index,columns=['total_account_asset','holding_market_cap','cash_account'], dtype=float)           # 账户信息存储
    price_open = get_bar(df_weight)                                                                                               # 获取开盘价格数据
    stock_round_lot = pd.Series(100, index=df_weight.columns, dtype=int)  # 每只股票固定按100股整手计算
    change_day = sorted(set(df_weight.index.tolist()[::change_n] + [df_weight.index[-1]]))                                        # 调仓日期

    # 滚动计算
    for i in tqdm(range(0,len(change_day)-1)):
        start_date = change_day[i]
        end_date = change_day[i+1]

        # 获取给定权重
        df_weight_temp = df_weight.loc[start_date].dropna()
        stock_list_temp = df_weight_temp.index.tolist()
        # 计算个股持股数量 = 向下取整(给定权重 * 可用资金 // 最小买入股数) * 最小买入股数
        stock_holding_num = ((df_weight_temp 
                            * cash 
                            / (price_open.loc[start_date,stock_list_temp] * (1 + sell_cost))        # 预留交易费用
                            // stock_round_lot.loc[stock_list_temp]) 
                            * stock_round_lot.loc[stock_list_temp])

        # 仓位变动
        stock_holding_num_change = stock_holding_num - stock_holding_num_hist
        # 获取期间价格
        price_open_temp = price_open.loc[start_date:end_date,stock_list_temp]
        # 计算交易成本 (可设置万一免五)
        def calc_fee(x,min_fee):
            if x < 0:
                fee_temp = x * sell_cost                                                                                            # 印花税 + 过户费等 + 佣金
            else:
                fee_temp = x * buy_cost                                                                                             # 过户费等 + 佣金
            # 最低交易成本限制
            if fee_temp > min_fee:
                return fee_temp
            else:
                return min_fee

        transaction_costs = ((price_open_temp.loc[start_date] 
                            * stock_holding_num_change)).apply(lambda x: calc_fee(x,min_fee)).sum()
        # 计算期间市值 （交易手续费在现金账户计提）
        holding_market_cap = (price_open_temp * stock_holding_num).sum(axis =1)
        cash_account = cash - transaction_costs - holding_market_cap.loc[start_date]
        cash_account = pd.Series([cash_account*((1 + cash_interest_daily)**(i+1)) for i in range(0,len(holding_market_cap))],
                                index = holding_market_cap.index)
        total_account_asset = holding_market_cap + cash_account
        
        # 将当前持仓存入 
        stock_holding_num_hist = stock_holding_num
        # 下一期期初可用资金
        cash = total_account_asset.loc[end_date]

        account.loc[start_date:end_date,'total_account_asset'] = round(total_account_asset,2)
        account.loc[start_date:end_date,'holding_market_cap'] = round(holding_market_cap,2)
        account.loc[start_date:end_date,'cash_account'] = round(cash_account,2)

    # 使用权重表首日作为初始资金基准（持仓为0），不新增前一交易日。
    account.loc[df_weight.index[0]] = [inital_cash, 0, inital_cash]
    account = account.sort_index()
    
    return account



def get_performance_analysis1(account_result,name = ' ',rf = 0.03,benchmark_index = '000300.XSHG',benchmark_type = 'mcw'):

    # 去掉首次持仓前的等待期，保留前一条账户记录作为净值1的起点。
    account_result = account_result.sort_index()
    holding = pd.to_numeric(account_result['holding_market_cap'], errors='raise')
    active_positions = np.flatnonzero(holding.fillna(0).ne(0).to_numpy())
    if len(active_positions) == 0:
        raise ValueError('账户没有实际持仓，暂无可绘制的策略运行区间')
    account_result = account_result.iloc[max(int(active_positions[0]) - 1, 0):].copy()

    # 加入基准    
    performance = pd.concat([account_result['total_account_asset'].to_frame('strategy'),
                             get_benchmark(account_result,benchmark_index,benchmark_type)],axis = 1)
    performance = performance.astype(float)
    benchmark_index = performance.columns[1]  # 使用本地文件实际返回的沪深300列名
    performance_net = performance.pct_change(fill_method=None)
    performance_net.iloc[0] = 0.0
    performance_cumnet = (1 + performance_net).cumprod()
    performance_cumnet['alpha'] = performance_cumnet['strategy']/performance_cumnet[benchmark_index]
    performance_cumnet = performance_cumnet.fillna(1)

    # 指标计算
    performance_pct = performance_cumnet.pct_change().dropna()

    # 策略收益
    strategy_name,benchmark_name,alpha_name = performance_cumnet.columns.tolist() 
    Strategy_Final_Return = performance_cumnet[strategy_name].iloc[-1] - 1

    # 策略年化收益
    Strategy_Annualized_Return_EAR = (1 + Strategy_Final_Return) ** (252/len(performance_cumnet)) - 1

    # 基准收益
    Benchmark_Final_Return = performance_cumnet[benchmark_name].iloc[-1] - 1

    # 基准年化收益
    Benchmark_Annualized_Return_EAR = (1 + Benchmark_Final_Return) ** (252/len(performance_cumnet)) - 1

    # alpha 
    ols_result = sm.OLS(performance_pct[strategy_name] * 252 - rf, sm.add_constant(performance_pct[benchmark_name] * 252 - rf)).fit()
    Alpha = ols_result.params.iloc[0]

    # beta
    Beta = ols_result.params.iloc[1]

    # beta_2 = np.cov(performance_pct[strategy_name],performance_pct[benchmark_name])[0,1]/performance_pct[benchmark_name].var()
    # 波动率
    Strategy_Volatility = performance_pct[strategy_name].std() * np.sqrt(252)

    # 夏普
    Strategy_Sharpe = (Strategy_Annualized_Return_EAR - rf)/Strategy_Volatility

    # 下行波动率
    strategy_ret = performance_pct[strategy_name]
    Strategy_Down_Volatility = strategy_ret[strategy_ret < 0].std() * np.sqrt(252)

    # sortino
    Sortino = (Strategy_Annualized_Return_EAR - rf)/Strategy_Down_Volatility
    
    # 跟踪误差
    Tracking_Error = (performance_pct[strategy_name] - performance_pct[benchmark_name]).std() * np.sqrt(252)

    # 信息比率
    Information_Ratio = (Strategy_Annualized_Return_EAR - Benchmark_Annualized_Return_EAR)/Tracking_Error

    # 最大回测
    i = np.argmax((np.maximum.accumulate(performance_cumnet[strategy_name]) 
                    - performance_cumnet[strategy_name])
                    /np.maximum.accumulate(performance_cumnet[strategy_name]))
    j = np.argmax(performance_cumnet[strategy_name][:i])
    
    Max_Drawdown = (1-performance_cumnet[strategy_name].iloc[i]/performance_cumnet[strategy_name].iloc[j])

    # 卡玛比率
    Calmar = (Strategy_Annualized_Return_EAR)/Max_Drawdown

    # 超额收益
    Alpha_Final_Return = performance_cumnet[alpha_name].iloc[-1] - 1

    # 超额年化收益
    Alpha_Annualized_Return_EAR = (1 + Alpha_Final_Return) ** (252/len(performance_cumnet)) - 1

    # 超额波动率
    Alpha_Volatility = performance_pct[alpha_name].std() * np.sqrt(252)

    # 超额夏普
    Alpha_Sharpe = (Alpha_Annualized_Return_EAR - rf)/Alpha_Volatility

    # 超额最大回测
    i = np.argmax((np.maximum.accumulate(performance_cumnet[alpha_name]) 
                    - performance_cumnet[alpha_name])
                    /np.maximum.accumulate(performance_cumnet[alpha_name]))
    j = np.argmax(performance_cumnet[alpha_name][:i])
    Alpha_Max_Drawdown = (1-performance_cumnet[alpha_name].iloc[i]/performance_cumnet[alpha_name].iloc[j])

    # 胜率
    performance_pct['win'] = performance_pct[alpha_name] > 0
    Win_Ratio = performance_pct['win'].value_counts().loc[True] / len(performance_pct)

    # 盈亏比
    profit_lose = performance_pct.groupby('win')[alpha_name].mean()
    Profit_Lose_Ratio = abs(profit_lose[True]/profit_lose[False])
    


    result = {
        '策略累计收益':round(Strategy_Final_Return,4),
        '策略年化收益': round(Strategy_Annualized_Return_EAR,4),
        '基准累计收益':round(Benchmark_Final_Return,4),
        '基准年化收益': round(Benchmark_Annualized_Return_EAR,4),
        '阿尔法':round(Alpha,4),
        '贝塔':round(Beta,4),
        '波动率':round(Strategy_Volatility,4),
        '夏普比率':round(Strategy_Sharpe,4),
        '下行波动率':round(Strategy_Down_Volatility,4),
        '索提诺比率':round(Sortino,4),
        '跟踪误差':round(Tracking_Error,4),
        '信息比率':round(Information_Ratio,4),
        '最大回撤':round(Max_Drawdown,4),
        '卡玛比率': round(Calmar,4),
        '超额累计收益':round(Alpha_Final_Return,4),
        '超额年化收益': round(Alpha_Annualized_Return_EAR,4),
        '超额波动率':round(Alpha_Volatility,4),
        '超额夏普':round(Alpha_Sharpe,4),
        '超额最大回测':round(Alpha_Max_Drawdown,4),
        '胜率':round(Win_Ratio,4),
        '盈亏比':round(Profit_Lose_Ratio,4)

    }
    


    # 回测图绘制
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    x = performance_cumnet.index
    y1 = performance_cumnet['strategy']
    y2 = performance_cumnet[benchmark_index]
    y3 = performance_cumnet['alpha']


    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(x, y1, label='strategy',color = 'darkred')
    ax.plot(x, y2, label=benchmark_index)
    ax.plot(x, y3, label='alpha')
    plt.title(name)

    # 调整子图的布局，留出空间给表格
    fig.subplots_adjust(left=0.08, right=0.70, bottom=0.12, top=0.92)
    # 创建一个额外的空白子图
    # 添加表格
    cell_text =  [['指标','数值']] + [list(result.items())][0]
    table = ax.table(cellText=cell_text, loc='right')

    # 调整表格的大小
    #table.scale(0.7, 0.7)

    # 设置单元格的属性
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 对第一行进行处理
            cell.set_text_props(fontsize=10, ha='center', va='center')  # 居中对齐
        elif j == 0:  # 对第一列进行处理
            cell.set_text_props(fontsize=10, ha='left', va='center')  # 左对齐
        else:
            cell.set_text_props(fontsize=10, ha='right', va='center')  # 右对齐

    # 设置行高
    for i in range(len(cell_text)):
        table._cells[(i, 0)].set_height(0.0454)
        table._cells[(i, 1)].set_height(0.0453)
    table.auto_set_column_width([0, 1])
    table.auto_set_font_size(False)

    # 显示图例
    ax.legend()
    plt.show()

    return performance_cumnet


# 累计ic图
def cumic(name,ic_df):
    """
    :param name: 因子名称 -> list 
    :param ic_df: ic序列表 -> dataframe 
    :return fig: 累计ic图 -> plot
    """
    ic_df[name].cumsum().plot(figsize = (len(name)/2,len(name)/4))

# 热力图    
def hot_corr(name,ic_df):
    """
    :param name: 因子名称 -> list 
    :param ic_df: ic序列表 -> dataframe 
    :return fig: 热力图 -> plt
    """
    ax = plt.subplots(figsize=(len(name), len(name)))#调整画布大小
    ax = sns.heatmap(ic_df[name].corr(),vmin=0.4, square=True, annot= True,cmap = 'Blues')   #annot=True 表示显示系数
    plt.title('Factors_IC_CORRELATION')
    # 设置刻度字体大小
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)



# 相关性过滤
def corr_filter(pass_icir_factor,factor_performance,low):
    # 计算各因子指标序列相关性
    ic_summary = factor_performance[pass_icir_factor].mean()/factor_performance[pass_icir_factor].std()
    pass_icir_factor = ic_summary.abs().sort_values(ascending=False).index.tolist()
    corr_df = factor_performance[pass_icir_factor].corr()

    # 相关性过滤
    length = corr_df.shape[1]
    pass_list = []
    for i in range(length):
        try:
            pass_list.append(corr_df.index.tolist()[0])
            corr_df = corr_df[(abs(corr_df.iloc[:, 0]) < low)]
            corr_df = corr_df[corr_df.index.tolist()]
        except:
            break
    
    return pass_list


# 2.5 Barra 风格中性化
def barra_style_neutralization(factor_df, barra_exposure, exposure_lag=0, style_cols=None, min_obs=100):
    """
    :param factor_df: 待中性化因子宽表，index=日期、columns=股票代码
    :param barra_exposure: Barra 风格暴露度长表，MultiIndex=(日期, 股票代码)
    :param exposure_lag: Barra 暴露度滞后交易日数；选股信号通常设为 0
    :param style_cols: 需要剥离的 Barra 风格因子list；None 表示使用全部列
    :param min_obs: 单日最少有效回归样本数
    :return result: Barra 风格中性化后的因子宽表
    :return report: 每日回归样本数、R2、回归状态等报告
    """
    if not isinstance(factor_df, pd.DataFrame) or factor_df.empty:
        raise ValueError("factor_df must be a non-empty wide DataFrame")
    if not isinstance(barra_exposure, pd.DataFrame) or not isinstance(barra_exposure.index, pd.MultiIndex):
        raise ValueError("barra_exposure must be a DataFrame with a 2-level MultiIndex")
    if factor_df.columns.has_duplicates or barra_exposure.index.duplicated().any():
        raise ValueError("factor_df columns and barra_exposure index must be unique")
    if not isinstance(exposure_lag, int) or exposure_lag < 0:
        raise ValueError("exposure_lag must be a non-negative integer")

    style_cols = barra_exposure.columns.tolist() if style_cols is None else list(style_cols)
    missing_cols = set(style_cols) - set(barra_exposure.columns)
    if missing_cols:
        raise ValueError(f"style_cols not found in barra_exposure: {missing_cols}")

    factor_long = factor_df.stack().rename("factor").to_frame()
    factor_long.index = factor_long.index.set_names(["date", "order_book_id"])

    exposure = barra_exposure.loc[:, style_cols].copy()
    exposure.index = exposure.index.set_names(["date", "order_book_id"])
    exposure = exposure.sort_index()

    result = pd.DataFrame(np.nan, index=factor_df.index, columns=factor_df.columns, dtype=float)
    report_cols = ["date", "input_count", "regression_count", "n_styles", "matrix_rank", "r2", "status"]

    if factor_long.empty:
        return result, pd.DataFrame(columns=report_cols).set_index("date")

    if exposure_lag == 0:
        aligned_exposure = exposure.reindex(factor_long.index)
        aligned_exposure.index = factor_long.index
    else:
        exposure_dates = exposure.index.get_level_values(0).unique().sort_values()
        if exposure_lag >= len(exposure_dates):
            raise ValueError("exposure_lag is too large for available Barra dates")

        previous_date_map = dict(zip(exposure_dates[exposure_lag:], exposure_dates[:-exposure_lag]))
        source_dates = factor_long.index.get_level_values(0).map(previous_date_map)
        valid_source = source_dates.notna()

        aligned_exposure = pd.DataFrame(np.nan, index=factor_long.index, columns=style_cols)
        source_index = pd.MultiIndex.from_arrays(
            [source_dates[valid_source], factor_long.index.get_level_values(1)[valid_source]],
            names=["date", "order_book_id"],
        )
        aligned_exposure.loc[valid_source, :] = exposure.reindex(source_index).to_numpy()

    reg_data = factor_long.join(aligned_exposure, how="left")
    report_rows = []

    for date, day_data in reg_data.groupby(level=0, sort=False):
        input_count = len(day_data)
        day_data = day_data.dropna(subset=["factor", *style_cols])
        day_data = day_data.loc[np.isfinite(day_data.to_numpy(dtype=float)).all(axis=1)]
        valid_count = len(day_data)

        if valid_count < min_obs:
            report_rows.append({"date": date, "input_count": input_count, "regression_count": valid_count,
                                "n_styles": 0, "matrix_rank": np.nan, "r2": np.nan,
                                "status": "insufficient_observations"})
            continue

        y = day_data["factor"].to_numpy(dtype=float)
        x_raw = day_data[style_cols].to_numpy(dtype=float)
        usable_style = x_raw.std(axis=0) > 1e-12

        if not usable_style.any():
            report_rows.append({"date": date, "input_count": input_count, "regression_count": valid_count,
                                "n_styles": 0, "matrix_rank": np.nan, "r2": np.nan,
                                "status": "no_varying_styles"})
            continue

        x = x_raw[:, usable_style]
        if valid_count < max(min_obs, x.shape[1] + 2):
            report_rows.append({"date": date, "input_count": input_count, "regression_count": valid_count,
                                "n_styles": x.shape[1], "matrix_rank": np.nan, "r2": np.nan,
                                "status": "insufficient_observations"})
            continue

        design = np.column_stack([np.ones(valid_count), x])
        coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients

        ss_total = np.square(y - y.mean()).sum()
        r2 = np.nan if ss_total <= 0 else 1 - np.square(residual).sum() / ss_total

        stock_ids = day_data.index.get_level_values(1)
        result.loc[date, stock_ids] = residual

        report_rows.append({"date": date, "input_count": input_count, "regression_count": valid_count,
                            "n_styles": x.shape[1], "matrix_rank": rank, "r2": r2,
                            "status": "success"})

    report = pd.DataFrame(report_rows, columns=report_cols).set_index("date")
    return result, report



# 联合剥离行业、对数市值和 Barra 风格暴露。
def all_neutralization(factor_df, order_book_ids, barra_exposure, index_item='', industry_type='zx', risk_lag=0, style_cols=None, min_obs=100):
    """
    :param factor_df: 待中性化因子宽表，index=日期、columns=股票代码
    :param order_book_ids: 全样本股票代码列表，通常传 factor_df.columns.tolist()
    :param barra_exposure: Barra 暴露度长表，MultiIndex=(日期, 股票代码)
    :param index_item: 股票池/指数标识，用于缓存文件名
    :param industry_type: 行业分类，例如 zx
    :param risk_lag: 风险暴露滞后期数；GP 信号且后续权重 shift(1) 时通常设为 0
    :param style_cols: 需要剥离的 Barra 风格列；None 表示使用全部 Barra 列
    :param min_obs: 单日最少有效回归样本数
    :return result: 联合中性化后的因子宽表
    :return report: 每日回归报告
    """
    if not isinstance(factor_df, pd.DataFrame) or factor_df.empty:
        raise ValueError("factor_df must be a non-empty wide DataFrame")
    if not isinstance(barra_exposure, pd.DataFrame) or not isinstance(barra_exposure.index, pd.MultiIndex):
        raise ValueError("barra_exposure must be a DataFrame with a 2-level MultiIndex")
    if factor_df.columns.has_duplicates or barra_exposure.index.duplicated().any():
        raise ValueError("factor_df columns and barra_exposure index must be unique")
    if not isinstance(risk_lag, int) or risk_lag < 0:
        raise ValueError("risk_lag must be a non-negative integer")

    style_cols = barra_exposure.columns.tolist() if style_cols is None else list(style_cols)
    missing_cols = set(style_cols) - set(barra_exposure.columns)
    if missing_cols:
        raise ValueError(f"style_cols not found in barra_exposure: {missing_cols}")

    dates = pd.Index(factor_df.index).unique().sort_values()
    if len(dates) == 0:
        raise ValueError("factor_df index is empty")

    start, end = dates[0].strftime("%F"), dates[-1].strftime("%F")
    cache_path = f"tmp/df_industry_market_{industry_type}_{index_item}_{start}_{end}.pkl"

    try:
        base_exposure = pd.read_pickle(cache_path)
    except FileNotFoundError:
        market_cap = execute_factor(
            LOG(Factor("market_cap_3")),
            order_book_ids,
            start,
            end,
        ).stack().to_frame("market_cap")

        industry_df = pd.get_dummies(
            get_industry_exposure(
                order_book_ids,
                dates.tolist(),
                industry_type,
            )
        )

        base_exposure = industry_df.join(market_cap, how="left")
        base_exposure.index = base_exposure.index.set_names(["date", "order_book_id"])
        base_exposure = base_exposure.dropna(subset=["market_cap"])

        create_dir_not_exist("tmp")
        base_exposure.to_pickle(cache_path)

    factor_long = factor_df.stack().rename("factor").to_frame()
    factor_long.index = factor_long.index.set_names(["date", "order_book_id"])

    base_exposure = base_exposure.copy()
    base_exposure.index = base_exposure.index.set_names(["date", "order_book_id"])
    base_exposure = base_exposure.sort_index()

    barra_data = barra_exposure.loc[:, style_cols].copy()
    barra_data.index = barra_data.index.set_names(["date", "order_book_id"])
    barra_data = barra_data.sort_index()

    industry_cols = [col for col in base_exposure.columns if col != "market_cap"]
    if not industry_cols:
        raise ValueError("no industry dummy columns found in base exposure data")

    result = pd.DataFrame(np.nan, index=factor_df.index, columns=factor_df.columns, dtype=float)
    report_cols = ["date", "input_count", "regression_count", "n_industries", "n_styles", "matrix_rank", "r2", "status"]

    if factor_long.empty:
        return result, pd.DataFrame(columns=report_cols).set_index("date")

    if risk_lag == 0:
        aligned_base = base_exposure.reindex(factor_long.index)
        aligned_barra = barra_data.reindex(factor_long.index)
        aligned_base.index, aligned_barra.index = factor_long.index, factor_long.index
    else:
        if risk_lag >= len(dates):
            raise ValueError("risk_lag is too large for factor_df dates")

        previous_date_map = dict(zip(dates[risk_lag:], dates[:-risk_lag]))
        source_dates = factor_long.index.get_level_values(0).map(previous_date_map)
        valid_source = source_dates.notna()

        aligned_base = pd.DataFrame(np.nan, index=factor_long.index, columns=base_exposure.columns)
        aligned_barra = pd.DataFrame(np.nan, index=factor_long.index, columns=style_cols)

        source_index = pd.MultiIndex.from_arrays(
            [source_dates[valid_source], factor_long.index.get_level_values(1)[valid_source]],
            names=["date", "order_book_id"],
        )

        aligned_base.loc[valid_source, :] = base_exposure.reindex(source_index).to_numpy()
        aligned_barra.loc[valid_source, :] = barra_data.reindex(source_index).to_numpy()

    reg_cols = industry_cols + ["market_cap"] + style_cols
    reg_data = factor_long.join(aligned_base, how="left").join(aligned_barra, how="left")
    report_rows = []

    for date, day_data in reg_data.groupby(level=0, sort=False):
        input_count = len(day_data)
        day_data = day_data.dropna(subset=["factor", *reg_cols])
        day_data = day_data.loc[np.isfinite(day_data.to_numpy(dtype=float)).all(axis=1)]
        day_data = day_data.loc[day_data[industry_cols].sum(axis=1) > 0]
        valid_count = len(day_data)

        if valid_count < min_obs:
            report_rows.append({"date": date, "input_count": input_count, "regression_count": valid_count,
                                "n_industries": 0, "n_styles": 0, "matrix_rank": np.nan,
                                "r2": np.nan, "status": "insufficient_observations"})
            continue

        industry_x = day_data[industry_cols].to_numpy(dtype=float)
        industry_active = industry_x.sum(axis=0) > 0

        continuous_cols = ["market_cap", *style_cols]
        continuous_x = day_data[continuous_cols].to_numpy(dtype=float)
        continuous_active = continuous_x.std(axis=0) > 1e-12

        x = np.column_stack([
            industry_x[:, industry_active],
            continuous_x[:, continuous_active],
        ])

        if x.shape[1] == 0 or valid_count < max(min_obs, x.shape[1] + 1):
            report_rows.append({"date": date, "input_count": input_count, "regression_count": valid_count,
                                "n_industries": industry_active.sum(), "n_styles": 0,
                                "matrix_rank": np.nan, "r2": np.nan,
                                "status": "insufficient_regression_variables"})
            continue

        y = day_data["factor"].to_numpy(dtype=float)

        # 不加显式截距：全部有效行业哑变量之和等于 1，已包含截距空间。
        coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ coefficients

        ss_total = np.square(y - y.mean()).sum()
        r2 = np.nan if ss_total <= 0 else 1 - np.square(residual).sum() / ss_total

        stock_ids = day_data.index.get_level_values(1)
        result.loc[date, stock_ids] = residual

        report_rows.append({"date": date, "input_count": input_count, "regression_count": valid_count,
                            "n_industries": industry_active.sum(),
                            "n_styles": max(continuous_active.sum() - 1, 0),
                            "matrix_rank": rank, "r2": r2, "status": "success"})

    report = pd.DataFrame(report_rows, columns=report_cols).set_index("date")
    return result, report



def plot_performance(performance_result,col='strategy',save_path=None):
    # =========================
    # 数据
    # =========================
    df = performance_result.copy()
    df.index = pd.to_datetime(df.index)

    nav = df[col].astype(float)
    drawdown = nav / nav.cummax() - 1

    # 月收益
    monthly_nav = nav.resample('M').last()
    monthly_ret = monthly_nav.pct_change().dropna()

    monthly_ret_df = monthly_ret.to_frame('ret')
    monthly_ret_df['year'] = monthly_ret_df.index.year
    monthly_ret_df['month'] = monthly_ret_df.index.month

    monthly_ret_pivot = monthly_ret_df.pivot(
        index='year',
        columns='month',
        values='ret'
    ).reindex(columns=range(1, 13))

    # 年收益
    yearly_nav = nav.resample('Y').last()
    yearly_ret = yearly_nav.pct_change().dropna()
    yearly_ret.index = yearly_ret.index.year

    # =========================
    # 作图
    # =========================
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(20, 11)
    )

    # =========================
    # 1. 年度收益
    # =========================
    ax = axes[0, 0]

    colors = [
        '#d62728' if v >= 0 else '#2ca02c'
        for v in yearly_ret
    ]

    bars = ax.bar(
        yearly_ret.index.astype(str),
        yearly_ret.values,
        color=colors,
        alpha=0.85
    )

    for bar, value in zip(
            bars,
            yearly_ret.values):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f'{value:.2%}',
            ha='center',
            va='bottom' if value >= 0 else 'top'
        )

    ax.axhline(0, color='black')
    ax.set_title(f'{col}年度收益分析')
    ax.grid(alpha=0.3)

    # =========================
    # 2. 月收益热力图
    # =========================
    ax = axes[0, 1]

    heat = monthly_ret_pivot.copy()

    vmax = np.nanmax(
        np.abs(heat.values)
    )

    norm = mcolors.TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax
    )

    im = ax.imshow(
        heat.values,
        cmap='RdBu_r',
        norm=norm,
        aspect='auto'
    )

    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(
        len(heat.index)-0.5,
        -0.5
    )

    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(
        [f'{i}月' for i in range(1, 13)]
    )

    ax.set_yticks(
        np.arange(len(heat.index))
    )
    ax.set_yticklabels(
        heat.index
    )

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):

            value = heat.iloc[i, j]

            if pd.notna(value):
                ax.text(
                    j,
                    i,
                    f'{value:.1%}',
                    ha='center',
                    va='center'
                )

    ax.set_title(f'{col}月度收益热力图')

    # =========================
    # 3. 月收益柱状图
    # =========================
    ax = axes[1, 0]

    monthly_plot = monthly_ret_pivot.T

    x = np.arange(12)
    years = monthly_plot.columns.tolist()
    n_year = len(years)
    width = 0.8 / n_year

    cmap = plt.cm.tab10

    for i, year in enumerate(years):

        ax.bar(
            x + i * width,
            monthly_plot[year],
            width,
            label=str(year),
            color=cmap(i % 10),
            alpha=0.8
        )

    ax.set_xticks(
        x + width * (n_year - 1) / 2
    )
    ax.set_xticklabels(
        [f'{i}月' for i in range(1, 13)]
    )

    ax.axhline(0, color='black')
    ax.legend(ncol=3)
    ax.grid(alpha=0.3)
    ax.set_title(f'{col}月度收益分析')

    # =========================
    # 4. 净值+回撤
    # =========================
    ax = axes[1, 1]

    ax.fill_between(
        drawdown.index,
        drawdown.values,
        0,
        color='lightskyblue',
        alpha=0.45,
        label='回撤'
    )

    ax2 = ax.twinx()

    ax2.plot(
        nav.index,
        nav.values,
        color='red',
        linewidth=2.5,
        label='净值'
    )

    ax.set_ylabel('回撤')
    ax2.set_ylabel('净值')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax2.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper left'
    )

    ax.set_title(f'{col}净值与回撤')

    plt.tight_layout()

    plt.savefig(
        col,
        dpi=600,
        bbox_inches='tight'
    )

    plt.show()
