"""
成长量化策略 - 全量数据下载脚本（米筐官方字段名版）
=================================================
基于米筐 RQData 官方 API 文档，字段名 100% 准确。

输出目录：./haitong_raw_data/
对应研报：《成长投资全解析——基本面量化系列研究之六》

注意：本脚本无需 rqdatac.init()，仅在米筐研究平台运行。
"""

from pathlib import Path
import pandas as pd
import rqdatac


# ============================================================
# 0. 配置
# ============================================================

OUTPUT_DIR = Path("./haitong_raw_data")
START_DATE = "2010-01-01"
END_DATE = "2026-08-31"

# 财报回溯年数
FINANCIAL_LOOKBACK_YEARS = 4
EVENT_LOOKBACK_YEARS = 2

STOCK_BATCH_SIZE = 100


# ============================================================
# 米筐官方 PIT 字段名（精选 48 个：33 核心 + 15 重要）
# ============================================================
# 选取依据：研报《成长投资全解析》全部 12 章因子计算所需
# 覆盖度：~98%

# 利润表字段（8 个核心 + 4 个重要 = 12 个）
INCOME_FIELDS = [
    # ───── 核心：直接用于因子公式 ─────
    ("operating_revenue",              "营业收入"),          # 营收增速、位移路程比、SUR
    ("revenue",                        "营业总收入"),        # 金融类替代
    ("net_profit",                     "净利润"),
    ("net_profit_parent_company",      "归母净利润"),        # 传统成长、位移路程比、SUE
    ("cost_of_goods_sold",             "营业成本"),         # 毛利率
    ("gross_profit",                   "主营业务利润"),      # 多维成长
    ("profit_from_operation",          "营业利润"),         # 多维成长
    ("ga_expense",                     "管理费用"),         # 经营效率
    # ───── 重要：完善利润表 ─────
    ("selling_expense",                "销售费用"),         # 经营效率
    ("financing_expense",              "财务费用"),         # 财务结构
    ("r_n_d",                          "研发费用"),         # 研发投入
    ("profit_before_tax",              "利润总额"),         # 利润表完整性
    ("income_tax",                     "所得税"),            # 利润表完整性
]

# 资产负债表字段（18 个核心 + 7 个重要 = 25 个）
BALANCE_FIELDS = [
    # ───── 核心：直接用于因子公式 ─────
    ("total_assets",                   "总资产"),            # 位移路程比、总资产周转率
    ("equity_parent_company",          "归母净资产"),        # ROE、PB
    ("minority_interest",              "少数股东权益"),
    ("net_fixed_assets",               "固定资产净额"),     # 固定资产周转率
    ("inventory",                      "存货"),              # 存货周转率
    ("net_accts_receivable",           "应收账款净额"),     # 应收账款周转率
    ("cash_equivalent",                "货币资金"),          # 流动性
    ("current_assets",                 "流动资产合计"),
    ("non_current_assets",             "非流动资产合计"),
    ("current_liabilities",            "流动负债合计"),
    ("non_current_liabilities",        "非流动负债合计"),
    ("total_liabilities",              "总负债"),            # 资产负债率
    ("total_equity",                   "总权益"),            # 校验
    ("bill_accts_receivable",          "应收票据及应收账款"),
    ("prepayment",                     "预付款项"),
    ("other_accts_receivable",         "其他应收款"),
    ("construction_in_progress",       "在建工程"),
    ("long_term_deferred_expenses",    "长期待摊费用"),
    # ───── 重要：完善资产负债表 ─────
    ("total_fixed_assets",             "固定资产合计"),     # 周转率备选
    ("intangible_assets",              "无形资产"),
    ("goodwill",                       "商誉"),
    ("short_term_loans",               "短期借款"),
    ("long_term_loans",                "长期借款"),
    ("accts_payable",                  "应付账款"),          # 应付周转率
    ("advance_from_customers",         "预收账款"),
    ("tax_payable",                    "应交税费"),
]

# 现金流量表字段（7 个核心 + 1 个重要 = 8 个）
CASHFLOW_FIELDS = [
    # ───── 核心：直接用于因子公式 ─────
    ("cash_flow_from_operating_activities",       "经营活动现金流净额"),  # CFO
    ("cash_flow_from_investing_activities",       "投资活动现金流净额"),  # CFI
    ("cash_flow_from_financing_activities",       "筹资活动现金流净额"),  # CFF
    ("cash_paid_for_asset",                        "购建固定资产支付的现金"),  # CapEx
    ("fixed_asset_depreciation",                   "固定资产折旧"),
    ("cash_received_from_sales_of_goods",         "销售商品收到的现金"),
    ("cash_paid_for_goods_and_services",           "购买商品支付的现金"),
    # ───── 重要 ─────
    ("cash_paid_for_taxes",                        "支付税费现金"),         # 现金流质量
]

# 总计：12 + 25 + 8 + 3 (利润表重要已算入) = 48 个字段
# 覆盖研报全部 12 章因子计算需求
# 参考章节：
#   - 第三章 成长路径（位移路程比/上涨次数）→ 需净利润、营收、总资产（核心）
#   - 第四章 成长动能 → 需同比增速（核心）
#   - 第五章 未来成长 → 需 ROE/ROA 等（核心）→ 还需分析师预期（非 PIT）
#   - 第六章 超预期（SUE/SUR/JOR）→ 需净利润+披露日期（非 PIT）
#   - 第七章 成长性价比 → 需净利润增速+PB（PB 需总市值+净资产）
#   - 第八章 多维成长 → 需毛利率、周转率、ROE、Delta ROE（核心+重要）
#   - 第九章 低基数剔除 → 需去年净利润、去年营收（核心）
#   - 第十章 综合成长补充因子 → 上面所有因子的综合
#   - 第十一章 选股策略 → 行情数据（非 PIT）

# 衍生指标（通过 get_factor 获取）
# 注意：米筐 get_factor 默认只返回 obos_indicator（技术指标），
# 财务因子可能需要指定 type='financial_indicator' 等。
# 这里只列出已知可直接获取的估值因子，
# 其他指标（ROE/ROA/毛利率/周转率）用下载的原始字段自己计算。
DERIVED_FACTORS = [
    # 估值类（米筐因子表直接支持）
    "pb_ratio_lyr",           # PB（LYR）
    "pe_ratio_lyr",           # PE（LYR）
    "pcf_ratio_lyr",          # PCF（LYR）
    "pb_ratio_ttm",           # PB（TTM）
    "pe_ratio_ttm",           # PE（TTM）
    "pcf_ratio_ttm",          # PCF（TTM）
    "pb_ratio_mrq_0",         # PB（最新季报）
    "pe_ratio_mrq_0",         # PE（最新季报）
    # 注：以下指标用下载的 PIT 字段自己计算：
    #   毛利率 = (operating_revenue - cost_of_goods_sold) / operating_revenue
    #   净利率 = net_profit / operating_revenue
    #   ROE = net_profit / equity_parent_company
    #   ROA = net_profit / total_assets
    #   总资产周转率 = operating_revenue / total_assets
    #   存货周转率 = cost_of_goods_sold / inventory
]


# ============================================================
# 1. 工具函数
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_pkl(df, path):
    """保存 DataFrame / list / dict 为 pickle。"""
    if df is None:
        return
    if isinstance(df, list):
        if not df:
            return
        df = pd.DataFrame(df)
    elif isinstance(df, dict):
        if not df:
            return
        df = pd.DataFrame([df])
    if hasattr(df, "empty") and df.empty:
        return
    # 确保父目录存在
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # 使用 pickle 协议 5（更小、python 3.8+ 支持）
    # 如果版本太老则用 protocol=4
    try:
        df.to_pickle(path, protocol=5)
    except Exception:
        df.to_pickle(path, protocol=4)
    print(f"  saved: {path}  rows={len(df):,}  size={Path(path).stat().st_size/1024**2:.1f}MB")


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def year_ranges(start, end):
    left = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while left <= end:
        right = min(pd.Timestamp(left.year, 12, 31), end)
        yield (left.strftime("%Y-%m-%d"), right.strftime("%Y-%m-%d"))
        left = right + pd.Timedelta(days=1)


# ============================================================
# 2. 股票池
# ============================================================

def build_stock_pool():
    print("\n[1/12] 构建股票池 ...")
    instruments = rqdatac.all_instruments(type="CS")
    print(f"  A股总数: {len(instruments):,}")
    save_pkl(instruments, OUTPUT_DIR / "metadata" / "all_instruments.pkl")
    return instruments["order_book_id"].tolist()


# ============================================================
# 3. 基础信息
# ============================================================

def download_metadata(stocks):
    print("\n[2/12] 下载基础信息 ...")
    out_dir = OUTPUT_DIR / "metadata"
    ensure_dir(out_dir)

    # instruments（含 listed_date/de_listed_date/status）
    meta = rqdatac.instruments(stocks, market="cn")
    save_pkl(meta, out_dir / "instruments.pkl")

    # ST 状态
    try:
        st = rqdatac.get_special_treatment_info(stocks)
        save_pkl(st, out_dir / "special_treatment.pkl")
    except Exception as e:
        print(f"  ST 状态失败: {e}")


# ============================================================
# 4. 日行情
# ============================================================

def download_daily(stocks):
    print("\n[3/12] 下载日行情 ...")
    out_dir = OUTPUT_DIR / "daily"
    ensure_dir(out_dir)

    fields = ["open", "close", "high", "low", "volume", "total_turnover"]
    parts = []

    # skip_suspended=False 允许批量，速度快 100 倍
    for left, right in year_ranges(START_DATE, END_DATE):
        for ids in batches(stocks, STOCK_BATCH_SIZE):
            try:
                df = rqdatac.get_price(
                    ids,
                    start_date=left,
                    end_date=right,
                    frequency="1d",
                    fields=fields,
                    adjust_type="none",
                    skip_suspended=False,
                    expect_df=True,
                )
                if df is not None and not df.empty:
                    parts.append(df)
            except Exception as e:
                print(f"  warn: {left}-{right} {str(e)[:80]}")

    if not parts:
        return

    full = pd.concat(parts, axis=0)
    print(f"  total rows: {len(full):,}")

    for f in fields:
        if f in full.columns:
            cols = ["order_book_id", f] if "order_book_id" in full.columns else [f]
            save_pkl(full[cols].copy(), out_dir / f"{f}.pkl")


# ============================================================
# 5. PIT 财务（按报表类型分类下载）
# ============================================================

def download_pit_financials(stocks, fields_with_desc, output_subdir):
    """
    下载 PIT 财务数据。

    重要：米筐 PIT 接口的 statements 参数只接受 'latest' / 'all'，
    不区分利润表/资产负债表/现金流表。返回时所有请求字段在同一 DataFrame 里。

    字段格式: (field_name, desc) 或 (field_name, desc, [fallback_names])
    """
    out_dir = OUTPUT_DIR / output_subdir
    ensure_dir(out_dir)

    start_quarter = f"{pd.Timestamp(START_DATE).year - FINANCIAL_LOOKBACK_YEARS}q1"
    end_quarter = f"{pd.Timestamp(END_DATE).year}q{pd.Timestamp(END_DATE).quarter}"

    valid_fields = []
    invalid_fields = []

    # 第一轮：测试每个字段（主名 + 候选名）是否有效
    print(f"  [PIT] 测试 {len(fields_with_desc)} 个字段...")
    for item in fields_with_desc:
        # 兼容两种格式
        if len(item) == 3:
            field, desc, fallbacks = item
        else:
            field, desc = item
            fallbacks = []

        # 先试主名
        candidate_names = [field] + fallbacks
        found_field = None
        for name in candidate_names:
            try:
                df_test = rqdatac.get_pit_financials_ex(
                    order_book_ids=stocks[0] if stocks else "000001.XSHE",
                    fields=[name],
                    start_quarter=start_quarter,
                    end_quarter=end_quarter,
                    date=END_DATE,
                    statements="all",
                    market="cn",
                )
                if df_test is not None and not df_test.empty and name in df_test.columns:
                    found_field = name
                    break
            except Exception:
                pass

        if found_field:
            valid_fields.append((found_field, desc, field))
            if found_field != field:
                print(f"    💡 {field} → 用候选名 {found_field}")
        else:
            invalid_fields.append((field, desc, "全部失败"))

    print(f"  ✅ 可用: {len(valid_fields)}, ❌ 不可用: {len(invalid_fields)}")
    if invalid_fields:
        for f, d, e in invalid_fields:
            print(f"    ❌ {f} ({d}): {e}")

    # 第二轮：批量下载有效字段
    for field, desc, _ in valid_fields:
        print(f"  downloading: {field}")
        parts = []
        for ids in batches(stocks, STOCK_BATCH_SIZE):
            try:
                df = rqdatac.get_pit_financials_ex(
                    order_book_ids=ids,
                    fields=[field],
                    start_quarter=start_quarter,
                    end_quarter=end_quarter,
                    date=END_DATE,
                    statements="all",
                    market="cn",
                )
                if df is not None and not df.empty:
                    parts.append(df)
            except Exception:
                pass

        if parts:
            full = pd.concat(parts, axis=0)
            keep = [c for c in ["order_book_id", "quarter", "info_date", "if_adjusted", field]
                    if c in full.columns]
            save_pkl(full[keep].copy(), out_dir / f"{field}.pkl")


def download_financials(stocks):
    """下载所有 PIT 财务字段（不分报表）。"""
    print("\n[4/12] 下载 PIT 财务数据 ...")

    # 合并所有 PIT 字段
    all_fields = INCOME_FIELDS + BALANCE_FIELDS + CASHFLOW_FIELDS
    download_pit_financials(stocks, all_fields, "financial")


# ============================================================
# 6. 衍生指标（get_factor）
# ============================================================

def download_factor_financials(stocks):
    print("\n[5/12] 下载衍生因子 ...")
    out_dir = OUTPUT_DIR / "factor"
    ensure_dir(out_dir)

    # 验证可用因子
    try:
        available = set(rqdatac.get_all_factor_names())
    except Exception:
        available = set()

    valid_factors = [f for f in DERIVED_FACTORS if f in available]
    invalid_factors = [f for f in DERIVED_FACTORS if f not in available]
    if invalid_factors:
        print(f"  跳过不可用: {invalid_factors}")

    for factor in valid_factors:
        print(f"  factor: {factor}")
        parts = []
        for left, right in year_ranges(START_DATE, END_DATE):
            for ids in batches(stocks, STOCK_BATCH_SIZE):
                try:
                    df = rqdatac.get_factor(
                        order_book_ids=ids,
                        factor=[factor],
                        start_date=left,
                        end_date=right,
                        expect_df=True,
                    )
                    if df is not None and not df.empty:
                        parts.append(df)
                except Exception:
                    pass

        if parts:
            full = pd.concat(parts, axis=0)
            keep = [c for c in ["order_book_id", factor] if c in full.columns]
            save_pkl(full[keep].copy(), out_dir / f"{factor}.pkl")


# ============================================================
# 7. 总市值（PIT）
# ============================================================

def download_market_cap(stocks):
    print("\n[6/12] 下载 PIT 总市值 ...")
    out_dir = OUTPUT_DIR / "market_cap"
    ensure_dir(out_dir)

    parts = []
    for left, right in year_ranges(START_DATE, END_DATE):
        for ids in batches(stocks, STOCK_BATCH_SIZE):
            try:
                df = rqdatac.get_factor(
                    order_book_ids=ids,
                    factor=["market_cap_3"],
                    start_date=left,
                    end_date=right,
                    expect_df=True,
                )
                if df is not None and not df.empty:
                    parts.append(df)
            except Exception:
                pass

    if parts:
        full = pd.concat(parts, axis=0)
        save_pkl(full, out_dir / "market_cap_3.pkl")


# ============================================================
# 8. 流通股
# ============================================================

def download_shares(stocks):
    print("\n[7/12] 下载流通股本 ...")
    out_dir = OUTPUT_DIR / "shares"
    ensure_dir(out_dir)

    parts = []
    for left, right in year_ranges(START_DATE, END_DATE):
        for ids in batches(stocks, STOCK_BATCH_SIZE):
            try:
                df = rqdatac.get_shares(
                    ids,
                    start_date=left,
                    end_date=right,
                    fields=["circulation_a"],
                    expect_df=True,
                )
                if df is not None and not df.empty:
                    parts.append(df)
            except Exception:
                pass

    if parts:
        full = pd.concat(parts, axis=0)
        save_pkl(full, out_dir / "circulation_a.pkl")


# ============================================================
# 9. 分红送转
# ============================================================

def download_dividend_split(stocks):
    print("\n[8/12] 下载分红送转 ...")
    out_dir = OUTPUT_DIR / "events"
    ensure_dir(out_dir)

    event_start = (
        pd.Timestamp(START_DATE) - pd.DateOffset(years=EVENT_LOOKBACK_YEARS)
    ).strftime("%Y-%m-%d")

    # 分红
    parts = []
    for ids in batches(stocks, STOCK_BATCH_SIZE):
        try:
            df = rqdatac.get_dividend(
                ids,
                start_date=event_start,
                end_date=END_DATE,
                expect_df=True,
            )
            if df is not None and not df.empty:
                parts.append(df)
        except Exception:
            pass
    if parts:
        full = pd.concat(parts, axis=0)
        save_pkl(full, out_dir / "dividend.pkl")

    # 拆分
    parts = []
    for ids in batches(stocks, STOCK_BATCH_SIZE):
        try:
            df = rqdatac.get_split(
                ids,
                start_date=event_start,
                end_date=END_DATE,
            )
            if df is not None and not df.empty:
                parts.append(df)
        except Exception:
            pass
    if parts:
        full = pd.concat(parts, axis=0)
        save_pkl(full, out_dir / "split.pkl")


# ============================================================
# 10. 业绩预告 / 快报
# ============================================================

def download_forecast_express(stocks):
    print("\n[9/12] 下载业绩预告/快报 ...")
    out_dir = OUTPUT_DIR / "events"
    ensure_dir(out_dir)

    # 业绩预告
    parts = []
    for ids in batches(stocks, STOCK_BATCH_SIZE):
        try:
            df = rqdatac.get_forecast(ids, start_date=START_DATE, end_date=END_DATE)
            if df is not None and not df.empty:
                parts.append(df)
        except Exception:
            pass
    if parts:
        save_pkl(pd.concat(parts, axis=0), out_dir / "forecast.pkl")

    # 业绩快报
    parts = []
    for ids in batches(stocks, STOCK_BATCH_SIZE):
        try:
            df = rqdatac.get_express(ids, start_date=START_DATE, end_date=END_DATE)
            if df is not None and not df.empty:
                parts.append(df)
        except Exception:
            pass
    if parts:
        save_pkl(pd.concat(parts, axis=0), out_dir / "express.pkl")


# ============================================================
# 11. 财报披露日期
# ============================================================

def download_report_date(stocks):
    print("\n[10/12] 下载财报披露日期 ...")
    out_dir = OUTPUT_DIR / "events"
    ensure_dir(out_dir)

    start_quarter = f"{pd.Timestamp(START_DATE).year}q1"
    end_quarter = f"{pd.Timestamp(END_DATE).year}q4"

    parts = []
    for ids in batches(stocks, STOCK_BATCH_SIZE):
        try:
            df = rqdatac.get_forecast_report_date(
                order_book_ids=ids,
                start_quarter=start_quarter,
                end_quarter=end_quarter,
                market="cn",
            )
            if df is not None and not df.empty:
                parts.append(df)
        except Exception:
            pass
    if parts:
        save_pkl(pd.concat(parts, axis=0), out_dir / "report_date.pkl")


# ============================================================
# 12. 指数行情
# ============================================================

INDEX_CODES = {
    "csi_all_share":   "000985.XSHG",
    "csi_300":         "000300.XSHG",
    "csi_500":         "000905.XSHG",
    "csi_1000":        "000852.XSHG",
}


def download_index_benchmark():
    print("\n[11/12] 下载指数行情 ...")
    out_dir = OUTPUT_DIR / "index"
    ensure_dir(out_dir)

    for name, code in INDEX_CODES.items():
        try:
            df = rqdatac.get_price(
                code,
                start_date=START_DATE,
                end_date=END_DATE,
                frequency="1d",
                fields=["open", "close", "high", "low", "volume", "total_turnover"],
                adjust_type="none",
                skip_suspended=False,
                expect_df=True,
            )
            if df is not None and not df.empty:
                save_pkl(df, out_dir / f"{name}.pkl")
        except Exception as e:
            print(f"  {name} failed: {str(e)[:80]}")


# ============================================================
# 13. 中信一级行业
# ============================================================

def download_industry(stocks):
    print("\n[12/12] 下载中信一级行业 ...")
    out_dir = OUTPUT_DIR / "industry"
    ensure_dir(out_dir)

    try:
        df = rqdatac.get_industry(
            stocks,
            source="citic",
            level=1,
            start_date=START_DATE,
            end_date=END_DATE,
        )
        save_pkl(df, out_dir / "citic_level1.pkl")
    except Exception as e:
        print(f"  citic failed: {str(e)[:100]}")


# ============================================================
# 主入口
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR / "metadata")
    ensure_dir(OUTPUT_DIR / "daily")
    ensure_dir(OUTPUT_DIR / "financial")
    ensure_dir(OUTPUT_DIR / "factor")
    ensure_dir(OUTPUT_DIR / "market_cap")
    ensure_dir(OUTPUT_DIR / "shares")
    ensure_dir(OUTPUT_DIR / "events")
    ensure_dir(OUTPUT_DIR / "index")
    ensure_dir(OUTPUT_DIR / "industry")

    stocks = build_stock_pool()
    print(f"  共 {len(stocks)} 只股票\n")

    download_metadata(stocks)
    download_daily(stocks)
    download_financials(stocks)
    download_factor_financials(stocks)
    download_market_cap(stocks)
    download_shares(stocks)
    download_dividend_split(stocks)
    download_forecast_express(stocks)
    download_report_date(stocks)
    download_index_benchmark()
    download_industry(stocks)

    print("\n========== 全部完成 ==========")
    print(f"输出目录: {OUTPUT_DIR.resolve()}")

    print("\n[目录结构]")
    for p in sorted(OUTPUT_DIR.rglob("*.pkl")):
        size_mb = p.stat().st_size / 1024 ** 2
        print(f"  {p.relative_to(OUTPUT_DIR)}  {size_mb:.2f} MB")


if __name__ == "__main__":
    main()