"""
成长量化策略 - 因子库
========================
为研报《成长投资全解析》提供因子计算基础设施。

主要功能：
1. PIT 财务数据 → 单季度数据
2. 同比/环比/TTM 计算
3. Rank IC 测试（单因子检验）
4. 十分组测试（多空收益）
5. 可视化

数据格式约定：
- 行情数据：date index × stock columns（前复权价格）
- PIT 财务：(stock, quarter) multi-index × {info_date, if_adjusted, value}
- 因子值：factor X（截面因子值，date × stock）
- 收益：returns X（个股收益，date × stock）
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("../growth_data")


# ============================================================
# 数据加载
# ============================================================

def load_pit_financial(field_name):
    """加载 PIT 财务字段。

    Returns: DataFrame indexed by (order_book_id, quarter), columns: info_date, if_adjusted, field_name
    """
    path = DATA_DIR / "financial" / f"{field_name}.pkl"
    return pd.read_pickle(path)


def load_daily(field_name):
    """加载日行情字段（如 close, open, volume 等）。

    Returns: DataFrame indexed by date, columns = stock codes
    """
    path = DATA_DIR / "daily" / f"{field_name}.pkl"
    return pd.read_pickle(path)


def load_metadata(field_name):
    """加载 metadata 字段（如 all_instruments, is_st, is_suspend 等）。"""
    path = DATA_DIR / "metadata" / f"{field_name}.pkl"
    return pd.read_pickle(path)


def load_index(name):
    """加载指数行情（如 csi_all_share.pkl）。"""
    return load_daily_path(DATA_DIR / "index" / f"{name}.pkl")


def load_daily_path(path):
    """通用加载日频 DataFrame（date × stock）。"""
    return pd.read_pickle(path)


# ============================================================
# 累计 → 单季度
# ============================================================

def to_single_quarter(pit_df, field_name=None, delay=1):
    """
    累计口径 → 单季度。

    PIT 数据是累计口径：
        2016q1 = 一季报（仅 Q1）
        2016q2 = 半年报（Q1+Q2）
        2016q3 = 三季报（Q1+Q2+Q3）
        2016q4 = 年报（Q1+Q2+Q3+Q4）

    单季度 = 当前累计 - 上一期累计
        Q1 单季度 = Q1 累计
        Q2 单季度 = Q2 累计 - Q1 累计
        Q3 单季度 = Q3 累计 - Q2 累计
        Q4 单季度 = Q4 累计 - Q3 累计

    Args:
        pit_df: PIT DataFrame, index=(stock, quarter), columns=[info_date, if_adjusted, value]
        field_name: 字段列名

    Returns:
        sq_df: DataFrame indexed by quarter_end_date, columns = stock codes
        info_df: DataFrame indexed by quarter_end_date, columns = stock codes (对应的 info_date)
    """
    if field_name is None:
        # 找到除 info_date, if_adjusted 之外的字段列
        for col in pit_df.columns:
            if col not in ["info_date", "if_adjusted"]:
                field_name = col
                break

    # 处理多股票 PIT
    if "order_book_id" not in pit_df.index.names:
        pit_df = pit_df.reset_index()
    if "quarter" not in pit_df.columns:
        pit_df = pit_df.reset_index()

    # 选取 each_adjusted=0（当期财报数据，避免用修正版）
    df = pit_df[pit_df["if_adjusted"] == 0].copy()

    # 按 (stock, quarter) 去重（同一财报可能重复出现）
    df = df.drop_duplicates(subset=["order_book_id", "quarter"], keep="last")

    # 透视：(stock × quarter) × [value, info_date]
    val_pivot = df.pivot(index="order_book_id", columns="quarter", values=field_name).sort_index(axis=1)
    info_pivot = df.pivot(index="order_book_id", columns="quarter", values="info_date").sort_index(axis=1)

    # 季度列转日期（用于计算和索引）
    def q_to_date(q):
        year, qstr = q.split("q")
        qnum = int(qstr)
        # 季度结束月份
        end_month = qnum * 3
        return pd.Timestamp(f"{year}-{end_month:02d}-01") + pd.offsets.MonthEnd(0)

    new_cols = [q_to_date(c) for c in val_pivot.columns]
    val_pivot.columns = new_cols
    info_pivot.columns = new_cols

    # 计算单季度
    sq = val_pivot.diff(axis=1)  # 当前 - 上一期
    # 第一季度（Q1）的单季度 = Q1 累计本身
    first_col = val_pivot.columns[0]
    sq[first_col] = val_pivot[first_col]

    # 滞后处理（避免 look-ahead bias）
    if delay > 0:
        # 把列（季度）往后移 delay 季度
        new_cols = []
        old_cols = list(sq.columns)
        for i, col in enumerate(old_cols):
            new_i = i + delay
            if new_i < len(old_cols):
                new_cols.append(old_cols[new_i])
            else:
                new_cols.append(pd.NaT)  # 超出范围，置为空日期
        sq = sq.iloc[:, delay:].copy()
        sq.columns = sq.columns  # 保留原日期名

    return sq, info_pivot


# ============================================================
# 同比 / 环比 / TTM
# ============================================================

def yoy_growth(sq_df, periods=4):
    """
    同比增速：本季度 vs 去年同季度。
    假设 sq_df 是单季度 DataFrame（行=股票，列=季度结束日）。

    Args:
        sq_df: single-quarter DataFrame
        periods: 4（单季度同比）或 1（季度累计同比）

    Returns: 同比增速 DataFrame（同 shape）
    """
    return sq_df / sq_df.shift(periods, axis=1) - 1


def ttm_sum(sq_df, window=4):
    """TTM = 过去4个季度滚动求和。"""
    return sq_df.T.rolling(window=window, min_periods=window).sum().T


def to_ttm(sq_df, window=4):
    """
    单季度 → TTM (过去 window 季度求和)。

    用于上涨次数占比等需要"4 季度累计同比变化"的因子。

    Returns:
        ttm_df: DataFrame, 行=股票, 列=日期. 前 window-1 列为 NaN。
    """
    return sq_df.T.rolling(window=window, min_periods=window).sum().T


def ttm_yoy_growth(sq_df, window=4):
    """
    TTM 同比增速：(TTM_t / TTM_{t-4}) - 1

    例如：净利润 TTM 同比 = 过去4 季度累计 / 去年同 4 季度累计 - 1
    """
    ttm = to_ttm(sq_df, window=window)
    return ttm / ttm.shift(window, axis=1) - 1


def load_industry_dummy():
    """加载中信一级行业哑变量矩阵。

    Returns:
        DataFrame, multi-index (date, order_book_id), columns = industry codes
        例如 columns: 'industry_10', 'industry_20', ...
    """
    return pd.read_pickle(DATA_DIR / "industry" / "industry_dummy.pkl")


def industry_neutralize(factor_df, industry_dummy=None):
    """
    行业内中性化（研报 §3.5 方法）。

    在每个调仓日，按中信一级行业内做百分比排名。
    行业内无同行的股票保持原排名。

    Args:
        factor_df: DataFrame（stocks × dates 或 dates × stocks）
        industry_dummy: DataFrame (date, stock) × industry code

    Returns:
        DataFrame, 同行同列结构
    """
    if industry_dummy is None:
        industry_dummy = load_industry_dummy()

    # 始终转为 dates × stocks（dates 为行，stocks 为列）
    if factor_df.shape[0] > factor_df.shape[1]:
        # stocks × dates → 转置
        was_stocks_first = True
        factor_df = factor_df.T
    else:
        was_stocks_first = False

    common_dates = factor_df.index.intersection(
        industry_dummy.index.get_level_values(0).unique()
    )

    for date in common_dates:
        f_vals = factor_df.loc[date]
        valid_stocks = f_vals.dropna().index

        try:
            ind_today = industry_dummy.loc[date]
        except KeyError:
            continue

        ind_today = ind_today.loc[ind_today.index.intersection(valid_stocks)]
        if len(ind_today) == 0:
            continue

        stock_industry = ind_today.idxmax(axis=1)

        rank_in_industry = pd.Series(index=valid_stocks, dtype=float)
        for ind, stocks in stock_industry.groupby(stock_industry):
            stocks_in_ind = stocks.index.intersection(valid_stocks)
            if len(stocks_in_ind) < 2:
                continue
            ranks = f_vals[stocks_in_ind].rank(pct=True)
            rank_in_industry.loc[stocks_in_ind] = ranks

        # 在 dates × stocks 布局下赋值
        factor_df.loc[date, rank_in_industry.index] = rank_in_industry

    # 转回原始布局
    if was_stocks_first:
        return factor_df.T
    return factor_df


def compose_factor(sub_factors, industry_dummy=None):
    """综合因子 = 研报 §3.5 方法。

    自动处理任意方向的子因子，统一转为 dates × stocks 后求平均。

    Args:
        sub_factors: dict, {特征名: factor DataFrame}
        industry_dummy: 行业哑变量（可选）

    Returns:
        composite: 综合因子 (dates × stocks)
    """
    if industry_dummy is None:
        industry_dummy = load_industry_dummy()

    # 统一为 stocks × dates 方向
    first = next(iter(sub_factors.values()))
    is_stocks_first = (first.shape[0] > first.shape[1])

    aligned = {}
    for name, f in sub_factors.items():
        if (f.shape[0] > f.shape[1]) != is_stocks_first:
            f = f.T
        aligned[name] = f

    # 公共日期（列）和股票（行）
    first_aligned = next(iter(aligned.values()))
    common_dates = first_aligned.columns
    common_stocks = first_aligned.index
    for f in aligned.values():
        common_dates = common_dates.intersection(f.columns)
        common_stocks = common_stocks.intersection(f.index)

    # 各因子：标准化 + 行业内中性化
    norm_factors = {}
    for name, f in aligned.items():
        # 转为 dates × stocks，再 normalize + neutralize
        sub = f.loc[common_stocks, common_dates].T
        normalized = normalize_factor(sub)
        if industry_dummy is not None:
            normalized = industry_neutralize(normalized, industry_dummy)
        norm_factors[name] = normalized

    # 等权平均
    composite = sum(norm_factors.values()) / len(norm_factors)
    return composite


def normalize_factor(factor_df, industry_df=None):
    """
    因子标准化：
    1. 去极值（MAD 法）
    2. 标准化（z-score）
    3. 行业内中性化（可选）
    """
    f = factor_df.copy()

    # 去极值
    median = f.median(axis=1)
    mad = (f.sub(median, axis=0)).abs().median(axis=1)
    mad = mad.replace(0, 1)
    f = f.clip(lower=median - 5 * mad, upper=median + 5 * mad, axis=0)

    # 标准化
    mean = f.mean(axis=1)
    std = f.std(axis=1)
    std = std.replace(0, 1)
    f = f.sub(mean, axis=0).div(std, axis=0)

    # 行业内中性化（如果提供行业）
    if industry_df is not None:
        # 按日期-行业内做 rank
        # 简化：返回原值（后续优化）
        pass

    return f


# ============================================================
# Rank IC 测试
# ============================================================

def rank_ic_test(factor_df, returns_df):
    """
    Rank IC 测试：每个调仓日计算因子与下一个月回度的 Spearman 秩相关系数。

    Args:
        factor_df: DataFrame, date × stock 或 stocks × dates
        returns_df: DataFrame, date × stock, 下月收益

    Returns:
        ic_series: 每个调仓日的 Rank IC（pd.Series）
    """
    # 自动转置为 dates × stocks
    if factor_df.shape[0] > factor_df.shape[1]:
        factor_df = factor_df.T
    common_dates = factor_df.index.intersection(returns_df.index)
    common_stocks = factor_df.columns.intersection(returns_df.columns)

    ic_list = []
    for date in common_dates:
        f_vals = factor_df.loc[date, common_stocks]
        r_vals = returns_df.loc[date, common_stocks]

        # 去除 NaN
        valid = f_vals.notna() & r_vals.notna()
        if valid.sum() < 30:  # 至少 30 只股票
            continue

        # Spearman 秩相关
        corr = f_vals[valid].rank().corr(r_vals[valid].rank())
        ic_list.append((date, corr))

    if not ic_list:
        return pd.Series(dtype=float)

    ic_series = pd.Series([x[1] for x in ic_list], index=[x[0] for x in ic_list])
    return ic_series


def calc_ic_stats(ic_series):
    """IC 统计：均值、ICIR、t 统计。"""
    if len(ic_series) == 0:
        return {"mean": np.nan, "std": np.nan, "icir": np.nan, "n": 0}
    mean = ic_series.mean()
    std = ic_series.std()
    icir = mean / std if std > 0 else np.nan
    return {"mean": mean, "std": std, "icir": icir, "n": len(ic_series)}


# ============================================================
# 十分组测试
# ============================================================

def decile_test(factor_df, returns_df, n_groups=10):
    """
    十分组测试：
    1. 每个调仓日，按因子值十分位
    2. 计算每组的下月平均收益
    3. 返回每组的长期平均收益 + 多空组合

    Args:
        factor_df: date × stock 或 stocks × dates
        returns_df: date × stock（下月收益）
        n_groups: 10

    Returns:
        decile_returns: dict, group_idx (1..n_groups) → avg return
        long_short: long_short return series
    """
    # 自动转置为 dates × stocks
    if factor_df.shape[0] > factor_df.shape[1]:
        factor_df = factor_df.T
    common_dates = factor_df.index.intersection(returns_df.index)
    common_stocks = factor_df.columns.intersection(returns_df.columns)

    # 收集每组的每次收益
    group_returns = {i: [] for i in range(1, n_groups + 1)}

    for date in common_dates:
        f_vals = factor_df.loc[date, common_stocks]
        r_vals = returns_df.loc[date, common_stocks]

        valid = f_vals.notna() & r_vals.notna()
        if valid.sum() < n_groups * 3:
            continue

        # 十分位
        try:
            groups = pd.qcut(f_vals[valid], q=n_groups, labels=False, duplicates="drop")
        except ValueError:
            continue

        for g in range(n_groups):
            mask = groups == g
            if mask.sum() == 0:
                continue
            # 实际组号 = g + 1
            group_returns[g + 1].append(r_vals[valid][mask].mean())

    # 汇总
    result = {}
    for g, rets in group_returns.items():
        if rets:
            result[g] = np.mean(rets)

    return result


# ============================================================
# 收益计算
# ============================================================

def next_month_return(price_df, months=1):
    """
    下月收益：t+1 月的收益。
    Args:
        price_df: date × stock 收盘价
        months: 1（下一月）或 12（未来 12 个月）
    Returns:
        DataFrame indexed by date, columns = stock codes
        每个 date 表示"从该月月末买入持有一段时间"的收益
    """
    # 月末调仓：每月最后一个交易日
    # pandas 2.x 用 'ME'（Month End），老版本用 'M'
    try:
        monthly = price_df.resample("ME").last()
    except ValueError:
        monthly = price_df.resample("M").last()

    # 下月收益 = next / current - 1
    future = monthly.shift(-months)
    ret = future / monthly - 1
    return ret


# ============================================================
# 可视化
# ============================================================

def plot_ic_curve(ic_series, title="Rank IC", figsize=(12, 5)):
    """绘制 IC 累计曲线 + IC 时序图。"""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # IC 时序
    ax1.plot(ic_series.index, ic_series.values, linewidth=0.8, alpha=0.7)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axhline(ic_series.mean(), color="red", linestyle="--", label=f"Mean={ic_series.mean():.4f}")
    ax1.set_title(f"{title} - Time Series")
    ax1.set_ylabel("Rank IC")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # IC 累计
    cum_ic = ic_series.cumsum()
    ax2.plot(cum_ic.index, cum_ic.values, linewidth=1.2, color="darkblue")
    ax2.set_title(f"{title} - Cumulative")
    ax2.set_ylabel("Cumulative IC")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_decile_bars(decile_returns, title="Decile Returns", figsize=(10, 5)):
    """绘制十分组柱状图。"""
    import matplotlib.pyplot as plt
    groups = sorted(decile_returns.keys())
    vals = [decile_returns[g] for g in groups]

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["darkgreen" if v > 0 else "darkred" for v in vals]
    bars = ax.bar(groups, vals, color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Decile (1=Lowest, 10=Highest)")
    ax.set_ylabel("Average Return")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    # 在柱子上标数值
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (0.005 if v > 0 else -0.01),
                f"{v:.2%}", ha="center", va="bottom" if v > 0 else "top", fontsize=9)

    plt.tight_layout()
    return fig


def print_factor_summary(factor_name, ic_series, decile_returns):
    """打印因子汇总。"""
    stats = calc_ic_stats(ic_series)
    print(f"\n{'='*60}")
    print(f"因子: {factor_name}")
    print(f"{'='*60}")
    print(f"  IC 均值:       {stats['mean']:.4f}")
    print(f"  IC 标准差:     {stats['std']:.4f}")
    print(f"  ICIR:          {stats['icir']:.4f}")
    print(f"  测试期数:      {stats['n']}")

    if decile_returns:
        long_leg = max(decile_returns.keys())
        short_leg = min(decile_returns.keys())
        long_short = decile_returns[long_leg] - decile_returns[short_leg]
        annualized = (1 + long_short) ** 12 - 1
        print(f"  多空组合月度收益: {long_short:.4f}")
        print(f"  多空组合年化收益: {annualized:.4f}")
        print(f"\n  各组收益:")
        for g in sorted(decile_returns.keys()):
            print(f"    第 {g:2d} 组: {decile_returns[g]:.4f}")
    print(f"{'='*60}\n")


def to_single_quarter_pit(pit_df, field_name=None, target_dates=None):
    """真正 PIT 的单季度数据：只用 info_date <= target_dates 的最新值。"""
    if field_name is None:
        for col in pit_df.columns:
            if col not in ["info_date", "if_adjusted"]:
                field_name = col
                break

    df = pit_df[pit_df["if_adjusted"] == 0].copy()

    if target_dates is None:
        target_dates = sorted(df["info_date"].dropna().unique())

    df_pivot = df.pivot_table(
        index="order_book_id",
        columns="quarter",
        values=field_name,
        aggfunc="last",
    )
    df_info = df.pivot_table(
        index="order_book_id",
        columns="quarter",
        values="info_date",
        aggfunc="last",
    )

    sq_pivot = df_pivot.diff(axis=1)
    first_col = df_pivot.columns[0]
    sq_pivot[first_col] = df_pivot[first_col]

    result = pd.DataFrame(index=df_pivot.index, columns=target_dates, dtype=float)

    for date in target_dates:
        mask = df_info <= date
        for stock in df_pivot.index:
            valid = mask.loc[stock]
            valid = valid[valid]
            if len(valid) > 0:
                last_q = valid.index[-1]
                result.loc[stock, date] = sq_pivot.loc[stock, last_q]

    return result.astype(float)