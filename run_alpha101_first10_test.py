# -*- coding: utf-8 -*-
"""
Alpha101 前 10 因子单因子检验（聚宽研究环境）
=============================================
对 Alpha#1~#10 做标准单因子检验：
  1) 多周期 RankIC 检验（持有周期 1/5/10/20 个交易日）
  2) 五分组分层回测 + 多空组合（因子 t 日收盘可得，t+1 日成交口径）
  3) 两个口径：原始因子 / 行业+市值中性化因子
  4) 两个区间分开展示：训练期 2015.01-2023.12 / 验证期 2024.01-2025.12
     （为后续做 101 因子机器学习合成预留训练/验证隔离，避免数据窥探）

【使用方法】
  1. 将 alpha101_operators.py、alpha101_factors.py、本文件三个文件
     上传到聚宽研究环境（同一目录）
  2. 整段运行本文件
  3. 结果：控制台打印统计表格 + 图片展示；csv 保存到研究环境当前目录 results/ 下

【当前简化处理（第一版，先看信号强度）】
  - 股票池：中证800 当前成分股（静态池，存在幸存者偏差；
    如需动态成分股可用 get_index_stocks(index, date) 逐期获取，留待扩展）
  - 分层回测未扣交易成本、未做涨停/停牌可交易性过滤（IC 检验为第一参考指标）
  - 行业映射使用当前申万行业（历史上行业会变更，影响较小）
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jqdata import *
from alpha101_operators import *
from alpha101_factors import FACTOR_FUNCS

plt.rcParams['font.sans-serif'] = ['SimHei']      # 中文显示
plt.rcParams['axes.unicode_minus'] = False

# ------------------------- 配置 -------------------------
INDEX_CODE  = '000906.XSHG'                 # 中证800
FETCH_START = '2014-06-01'                  # 数据起点（预留最长 60 日窗口的预热）
TRAIN_START, TRAIN_END = '2015-01-01', '2023-12-31'
VALID_START, VALID_END = '2024-01-01', '2025-12-31'
HORIZONS   = [1, 5, 10, 20]                 # IC 检验的持有周期（交易日）
N_GROUPS   = 5                              # 分层回测分组数
SEC_BATCH  = 200                            # 拉行情时分批次（防单次请求过大）
MIN_HIST   = 120                            # 剔除上市不满 120 个交易日的次新
RESULT_DIR = 'results'                      # 研究环境内结果目录


# ------------------------- 数据获取 -------------------------
def fetch_price(stocks, start, end):
    """分股票批次拉日线行情。返回 dict: 字段 -> DataFrame(日期×股票)，停牌日为 NaN。

    兼容性说明：聚宽研究环境（旧版，Python3.6 / pandas 0.23）中
    get_price(panel=False) 对多标的的返回结构有问题（日期混入数据区，
    dtype 变成 datetime64，导致 pct_change 报错）。
    因此改用 panel=True（旧环境原生 Panel 格式），fill_paused=False 让停牌日为 NaN。
    """
    fields = ['open', 'close', 'high', 'low', 'volume', 'money', 'avg']
    out = {f: [] for f in fields}
    for i in range(0, len(stocks), SEC_BATCH):
        batch = stocks[i:i + SEC_BATCH]
        try:
            p = get_price(batch, start_date=start, end_date=end,
                          frequency='daily', fields=fields,
                          skip_paused=False, fq='pre', panel=True, fill_paused=False)
        except TypeError:
            # 极老版本没有 fill_paused 参数时的兜底
            p = get_price(batch, start_date=start, end_date=end,
                          frequency='daily', fields=fields,
                          skip_paused=False, fq='pre', panel=True)
        for f in fields:
            out[f].append(p[f].astype(float))
    for f in fields:
        out[f] = pd.concat(out[f], axis=1).sort_index()
    return out


def fetch_market_cap(stocks, start, end):
    """逐年拉市值表（避免单次请求过大），返回 DataFrame(日期×股票)"""
    parts = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        s = max(start, '%d-01-01' % y)
        e = min(end, '%d-12-31' % y)
        if s > e:
            continue
        df = get_valuation(stocks, s, e, fields=['market_cap'])
        parts.append(df)
    mcap = pd.concat(parts, axis=0)
    mcap = mcap[~mcap.index.duplicated(keep='first')].sort_index()
    return mcap


def fetch_st(stocks, start, end):
    """ST 状态表，返回 DataFrame(日期×股票)，True=ST"""
    return get_extras('is_st', stocks, start_date=start, end_date=end)


def build_valid_mask(price, st):
    """可交易性过滤：上市满 MIN_HIST 个交易日 且 非 ST"""
    close = price['close']
    volume = price['volume']
    listed = volume.notna().cumsum() >= MIN_HIST
    st_flag = st.reindex(close.index, columns=close.columns).fillna(False)
    return listed & (~st_flag)


# ------------------------- 因子与中性化 -------------------------
def neutralize(factor, industry_map, log_mcap):
    """行业去均值 + 对数市值回归残差（两步法，按天横截面）。
    返回与 factor 相同 shape 的 DataFrame（无行业映射的股票被剔除）。"""
    ind = pd.Series(industry_map).reindex(factor.columns)
    valid_cols = ind.dropna().index
    f = factor[valid_cols]
    ind_v = ind[valid_cols]
    # 第一步：行业内横截面去均值
    f = f - f.groupby(ind_v, axis=1).transform('mean')
    # 第二步：对 ln 市值做逐日线性回归取残差（先中心化，闭式解）
    lm = log_mcap[valid_cols]
    f_c = f.sub(f.mean(axis=1), axis=0)
    lm_c = lm.sub(lm.mean(axis=1), axis=0)
    beta = (f_c * lm_c).sum(axis=1) / (lm_c ** 2).sum(axis=1)
    resid = f_c - beta.values[:, None] * lm_c
    return resid


def _monthly_return(ret_series):
    """日收益序列按月聚合为月度收益（兼容 pandas 1.x 'M' 与 2.x 'ME'）"""
    try:
        m = (1 + ret_series).resample('ME').prod() - 1
    except (ValueError, KeyError):
        m = (1 + ret_series).resample('M').prod() - 1
    return m


def forward_returns(close, h):
    """持有 h 日的前向收益（t 日收盘可得因子，t+1 日成交，持有到 t+1+h）：
    在 t 日位置的值 = close[t+1+h] / close[t+1] - 1"""
    return close.shift(-(h + 1)).div(close.shift(-1)) - 1


# ------------------------- 检验统计 -------------------------
def rank_ic_series(factor, fwd_ret):
    """逐日 Spearman RankIC（等价于先按天对两边做横截面分位排序再算 Pearson）"""
    fr = factor.rank(axis=1, pct=True)
    rr = fwd_ret.rank(axis=1, pct=True)
    f_c = fr.sub(fr.mean(axis=1), axis=0)
    r_c = rr.sub(rr.mean(axis=1), axis=0)
    num = (f_c * r_c).sum(axis=1)
    den = ((f_c ** 2).sum(axis=1) * (r_c ** 2).sum(axis=1)) ** 0.5
    return num / den


def ic_stats(ic, start, end):
    """某区间内 IC 序列的统计指标"""
    seg = ic.loc[start:end].dropna()
    if len(seg) == 0:
        return {'IC均值': np.nan, 'ICIR': np.nan, 'IC>0占比': np.nan, 't值': np.nan}
    mean, std = seg.mean(), seg.std()
    return {
        'IC均值': mean,
        'ICIR': mean / std if std > 0 else np.nan,
        'IC>0占比': (seg > 0).mean(),
        't值': mean / std * np.sqrt(len(seg)) if std > 0 else np.nan,
    }


def layered_backtest(factor, ret1, n_groups=N_GROUPS):
    """每日按因子五分位分组，计算组内 t+1 日收益，输出：
    nets: dict 组号 -> 累计净值；ls_net: 多空组合(第5组-第1组)累计净值"""
    pct = factor.rank(axis=1, pct=True)
    g = np.ceil(pct * n_groups).where(pct.notna())
    nets = {}
    for grp in range(1, n_groups + 1):
        r = ret1.where(g == grp).mean(axis=1)          # 组内等权，mean 自动跳过 NaN
        nets[grp] = (1 + r.fillna(0)).cumprod()
    ls_ret = ret1.where(g == n_groups).mean(axis=1) - ret1.where(g == 1).mean(axis=1)
    ls_net = (1 + ls_ret.fillna(0)).cumprod()
    return nets, ls_net


def net_stats(net, start, end):
    """净值曲线在区间内的绩效指标"""
    seg = net.loc[start:end].dropna()
    if len(seg) < 30:
        return (np.nan,) * 4
    ret = seg.pct_change().dropna()
    ann = (seg.iloc[-1] / seg.iloc[0]) ** (252 / len(seg)) - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    mdd = (seg / seg.cummax() - 1).min()
    monthly = _monthly_return(ret)
    win = (monthly > 0).mean()
    return ann, sharpe, mdd, win


# ------------------------- 主流程 -------------------------
def main():
    print('== 拉取中证800 成分股 ==')
    stocks = get_index_stocks(INDEX_CODE)
    print('股票数: %d' % len(stocks))

    print('== 拉取行情（%s ~ %s）==' % (FETCH_START, VALID_END))
    price = fetch_price(stocks, FETCH_START, VALID_END)
    price['returns'] = price['close'].pct_change()
    price['adv20'] = price['money'].rolling(20).mean()
    print('行情面板: %s' % str(price['close'].shape))

    print('== 拉取市值 / ST / 行业 ==')
    mcap = fetch_market_cap(stocks, FETCH_START, VALID_END)
    log_mcap = np.log(mcap.replace(0, np.nan)).reindex(
        price['close'].index, columns=price['close'].columns)
    st = fetch_st(stocks, FETCH_START, VALID_END)
    industry_map = get_industry(stocks)   # dict: code -> 申万行业名
    valid_mask = build_valid_mask(price, st)

    print('== 计算前 10 个因子 ==')
    data = {k: price[k] for k in
            ['open', 'close', 'high', 'low', 'volume', 'money', 'avg', 'returns', 'adv20']}
    factors = {}
    for name, func in FACTOR_FUNCS.items():
        f = func(data)
        if not isinstance(f, pd.DataFrame):
            f = pd.DataFrame(f, index=price['close'].index, columns=price['close'].columns)
        factors[name] = f.where(valid_mask)          # 不可交易处置 NaN
        print('  %s 覆盖率 %.1f%%' % (name, f.notna().mean().mean() * 100))

    print('== 计算前向收益 ==')
    fwd = {h: forward_returns(price['close'], h) for h in HORIZONS}
    ret1 = forward_returns(price['close'], 1)        # t+1 对 t 的收益（分层回测用）

    os.makedirs(RESULT_DIR, exist_ok=True)
    ic_rows, ls_rows = [], []

    for name, fac in factors.items():
        # ---- 中性化版本 ----
        fac_neu = neutralize(fac, industry_map, log_mcap)
        fac_neu = fac_neu.where(valid_mask.reindex(fac_neu.columns, axis=1))

        # ---- IC 检验（两个口径 × 两个区间 × 多周期）----
        for h in HORIZONS:
            for ver, f in [('raw', fac), ('neu', fac_neu)]:
                ic = rank_ic_series(f, fwd[h])
                for period, (ps, pe) in [('train', (TRAIN_START, TRAIN_END)),
                                         ('valid', (VALID_START, VALID_END))]:
                    st_ = ic_stats(ic, ps, pe)
                    ic_rows.append({'factor': name, 'version': ver, 'period': period,
                                    'horizon': h, **st_})
        ic_df = pd.DataFrame(ic_rows)
        ic_df.to_csv(os.path.join(RESULT_DIR, 'ic_stats.csv'),
                     index=False, encoding='utf-8-sig')

        # ---- 分层回测（1 日持有，t+1 成交口径）----
        for ver, f in [('raw', fac), ('neu', fac_neu)]:
            nets, ls_net = layered_backtest(f, ret1)
            for period, (ps, pe) in [('train', (TRAIN_START, TRAIN_END)),
                                     ('valid', (VALID_START, VALID_END))]:
                ann, sharpe, mdd, win = net_stats(ls_net, ps, pe)
                ls_rows.append({'factor': name, 'version': ver, 'period': period,
                                '年化收益': ann, '夏普': sharpe,
                                '最大回撤': mdd, '月胜率': win})
                # 分组单调性：各组年化收益
                for grp, net in nets.items():
                    gann, _, _, _ = net_stats(net, ps, pe)
                    ls_rows[-1]['G%d' % grp] = gann
        ls_df = pd.DataFrame(ls_rows)
        ls_df.to_csv(os.path.join(RESULT_DIR, 'layered_stats.csv'),
                     index=False, encoding='utf-8-sig')

        # ---- 作图：累计 IC + 多空净值（训练/验证分线）----
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        ic10 = rank_ic_series(fac_neu, fwd[10])
        ax = axes[0]
        ax.plot(ic10.loc[TRAIN_START:TRAIN_END].cumsum(), label='训练期')
        ax.plot(ic10.loc[VALID_START:VALID_END].cumsum(), label='验证期')
        ax.axhline(0, color='grey', lw=0.6)
        ax.set_title('%s 累计RankIC(10日,中性化)' % name)
        ax.legend()
        ax = axes[1]
        _, ls_net = layered_backtest(fac_neu, ret1)
        ax.plot(ls_net.loc[TRAIN_START:TRAIN_END], label='训练期')
        ax.plot(ls_net.loc[VALID_START:VALID_END], label='验证期')
        ax.set_title('%s 多空净值(中性化,日调仓未扣费)' % name)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, '%s.png' % name), dpi=100)
        plt.show()

    # ---- 汇总打印 ----
    print('\n================ RankIC 统计（多周期） ================')
    pivot = ic_df.pivot_table(index=['factor', 'version', 'horizon'],
                              columns='period', values=['IC均值', 'ICIR', 'IC>0占比'])
    print(pivot.round(4))
    print('\n================ 分层回测多空绩效（1日持有） ================')
    print(ls_df.round(4))
    print('\n结果文件已保存到研究环境 %s/ 目录' % RESULT_DIR)


if __name__ == '__main__':
    main()
