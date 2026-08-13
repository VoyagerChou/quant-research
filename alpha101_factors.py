# -*- coding: utf-8 -*-
"""
Alpha101 因子公式 Alpha#1 ~ Alpha#10
=====================================
公式来源：《101 Formulaic Alphas》(Kakushadze 2015) 附录 A.1。
中文逐因子详解见仓库根目录《101_Formulaic_Alphas_中文翻译.md》。

数据约定：每个因子函数接收 data (dict[str, DataFrame])，其中包含：
    open, close, high, low, volume : 日线 OHLCV（前复权）
    money    : 成交额（元），用于计算 adv{d}
    avg      : 成交均价 = money/volume，即论文中的 vwap
    returns  : 日收盘到收盘收益率（close.pct_change()）
    adv20    : 20 日平均成交额（money 的 20 日滚动均值）

输出：DataFrame（index=日期, columns=股票代码）

注意：np.where / 运算产生的 ndarray 已统一包装回 DataFrame（_wrap 辅助函数）。
"""
import numpy as np
import pandas as pd
from alpha101_operators import (
    rank, delay, delta, ts_sum, ts_std, ts_min, ts_max,
    ts_rank, ts_argmax, correlation, signedpower,
)


def _wrap(arr, template):
    """把 ndarray 包装回与 template 相同 index/columns 的 DataFrame"""
    if isinstance(arr, pd.DataFrame):
        return arr
    return pd.DataFrame(arr, index=template.index, columns=template.columns)


def alpha001(data):
    """
    (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
    收益为负时取 20 日收益标准差，否则取收盘价；平方后找近 5 天峰值所在天数，横截面排序，中心化。
    """
    close = data['close']
    returns = data['returns']
    cond = np.where(returns < 0, ts_std(returns, 20), close)
    cond = _wrap(cond, returns)
    x = signedpower(cond, 2.0)          # 两分支均非负，等价于 x^2
    return rank(ts_argmax(x, 5)) - 0.5


def alpha002(data):
    """
    (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    对数成交量 2 日变化排名与日内涨幅排名的 6 日相关性，取负（放量上涨看空）。
    """
    close, open_ = data['close'], data['open']
    volume = data['volume'].replace(0, np.nan)
    logv = np.log(volume)
    return -1 * correlation(rank(delta(logv, 2)), rank((close - open_) / open_), 6)


def alpha003(data):
    """
    (-1 * correlation(rank(open), rank(volume), 10))
    开盘价排名与成交量排名的 10 日相关性，取负。
    """
    return -1 * correlation(rank(data['open']), rank(data['volume']), 10)


def alpha004(data):
    """
    (-1 * Ts_Rank(rank(low), 9))
    最低价横截面排名在近 9 天的时间序列分位，取负。
    """
    return -1 * ts_rank(rank(data['low']), 9)


def alpha005(data):
    """
    (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    开盘相对 10 日均价（avg）偏离排名 × 收盘相对当日均价偏离排名的绝对值取负。
    """
    close, open_ = data['close'], data['open']
    vwap = data['avg']                  # 聚宽 avg 字段 = 成交均价 ≈ VWAP
    term1 = rank(open_ - ts_sum(vwap, 10) / 10.0)
    term2 = -1 * np.abs(rank(close - vwap))
    return term1 * term2


def alpha006(data):
    """
    (-1 * correlation(open, volume, 10))
    开盘价与成交量的 10 日相关性，取负。
    """
    return -1 * correlation(data['open'], data['volume'], 10)


def alpha007(data):
    """
    ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
    放量（量>20日均额）时：|7日变化| 近60天时序分位取负 × 7日变化方向；缩量时输出 -1。
    """
    close = data['close']
    volume = data['volume']
    adv20 = data['adv20']
    d7 = delta(close, 7)
    inner = -1 * ts_rank(np.abs(d7), 60) * np.sign(d7)
    out = np.where(adv20 < volume, inner, -1.0)
    out = _wrap(out, close)
    # adv20 未定义（滚动窗口不足）时输出 NaN，避免错误落入默认分支
    return out.where(adv20.notna())


def alpha008(data):
    """
    (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
    近5日开盘和×收益和相对其10天前值的增量，横截面排序取负。
    """
    open_ = data['open']
    returns = data['returns']
    x = ts_sum(open_, 5) * ts_sum(returns, 5)
    return -1 * rank(x - delay(x, 10))


def alpha009(data):
    """
    ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1)
      : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    近5天单边上涨/下跌时顺势取当日变化，震荡时取当日变化的反向（趋势+反转切换）。
    """
    close = data['close']
    d1 = delta(close, 1)
    mn, mx = ts_min(d1, 5), ts_max(d1, 5)
    out = np.where(mn > 0, d1, np.where(mx < 0, d1, -1 * d1))
    out = _wrap(out, close)
    # 滚动窗口不足时输出 NaN，避免错误落入"震荡"分支
    return out.where(mn.notna())


def alpha010(data):
    """
    rank( 与 alpha009 同构，窗口 4 天，外层加横截面排序 )
    """
    close = data['close']
    d1 = delta(close, 1)
    mn, mx = ts_min(d1, 4), ts_max(d1, 4)
    out = np.where(mn > 0, d1, np.where(mx < 0, d1, -1 * d1))
    out = rank(_wrap(out, close))
    return out.where(mn.notna())


# 因子注册表：后续扩展到全部 101 个时，在 alpha101_factors.py 中
# 实现 alpha011~alpha101 并加入此表即可
FACTOR_FUNCS = {
    'alpha001': alpha001,
    'alpha002': alpha002,
    'alpha003': alpha003,
    'alpha004': alpha004,
    'alpha005': alpha005,
    'alpha006': alpha006,
    'alpha007': alpha007,
    'alpha008': alpha008,
    'alpha009': alpha009,
    'alpha010': alpha010,
}
