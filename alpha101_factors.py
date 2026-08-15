# -*- coding: utf-8 -*-
"""
Alpha101 因子公式 Alpha#1 ~ Alpha#101
=====================================
公式来源：《101 Formulaic Alphas》(Kakushadze 2015) 附录 A.1。
中文逐因子详解见仓库根目录《101_Formulaic_Alphas_中文翻译.md》。

数据约定：每个因子函数接收 data (dict[str, DataFrame])，包含：
    open, close, high, low, volume : 日线 OHLCV（前复权，float32）
    money    : 成交额（元），用于计算 adv{d}
    avg      : 成交均价 = money/volume，即论文中的 vwap
    returns  : 日收盘到收盘收益率（close.pct_change()）
    adv{d}   : d 日平均成交额（money 的 d 日滚动均值），d ∈ {5,10,15,20,30,40,50,60,81,120,150,180}
    cap      : 总市值
    ind_l1   : 申万一级行业映射（pd.Series, index=股票代码）
    ind_l2   : 申万二级行业映射（pd.Series, index=股票代码）

行业层级近似（WorldQuant IndClass → A股申万）：
    IndClass.sector       → 申万一级（ind_l1）
    IndClass.industry     → 申万一级（ind_l1）
    IndClass.subindustry  → 申万二级（ind_l2）

注意：np.where / 运算产生的 ndarray 已统一包装回 DataFrame（_wrap 辅助函数）。
      公式中的 min(x, y)/max(x, y)（两个表达式）是逐元素取小/大（np.minimum/np.maximum）；
      min(x, d)/max(x, d)（表达式+天数）按论文定义为时间序列最小/最大（ts_min/ts_max）。
"""
import sys
import types as _types
import pandas as pd
# 防御性获取 numpy（原因见 alpha101_operators.py 顶部注释）：
# 不依赖 import 语句的绑定结果，直接从 sys.modules 取真实模块对象。
_np_mod = sys.modules.get('numpy')
if _np_mod is None or not isinstance(_np_mod, _types.ModuleType):
    _np_mod = __import__('numpy')          # 兜底：真正加载一次
np = _np_mod
from alpha101_operators import (
    rank, delay, delta, ts_sum, ts_product, ts_std, ts_min, ts_max,
    ts_rank, ts_argmax, ts_argmin, correlation, covariance, scale,
    signedpower, decay_linear, indneutralize,
)


def _wrap(arr, template):
    """把 ndarray 包装回与 template 相同 index/columns 的 DataFrame"""
    if isinstance(arr, pd.DataFrame):
        return arr
    return pd.DataFrame(arr, index=template.index, columns=template.columns)


# ============ Alpha#1 ~ #10 ============
def alpha001(data):
    close, returns = data['close'], data['returns']
    cond = np.where(returns < 0, ts_std(returns, 20), close)
    x = signedpower(_wrap(cond, returns), 2.0)
    return rank(ts_argmax(x, 5)) - 0.5


def alpha002(data):
    close, open_ = data['close'], data['open']
    logv = np.log(data['volume'].replace(0, np.nan))
    return -1 * correlation(rank(delta(logv, 2)), rank((close - open_) / open_), 6)


def alpha003(data):
    return -1 * correlation(rank(data['open']), rank(data['volume']), 10)


def alpha004(data):
    return -1 * ts_rank(rank(data['low']), 9)


def alpha005(data):
    close, open_ = data['close'], data['open']
    vwap = data['avg']
    return rank(open_ - ts_sum(vwap, 10) / 10.0) * (-1 * np.abs(rank(close - vwap)))


def alpha006(data):
    return -1 * correlation(data['open'], data['volume'], 10)


def alpha007(data):
    close, volume, adv20 = data['close'], data['volume'], data['adv20']
    d7 = delta(close, 7)
    inner = -1 * ts_rank(np.abs(d7), 60) * np.sign(d7)
    out = np.where(adv20 < volume, inner, -1.0)
    return _wrap(out, close).where(adv20.notna())


def alpha008(data):
    x = ts_sum(data['open'], 5) * ts_sum(data['returns'], 5)
    return -1 * rank(x - delay(x, 10))


def alpha009(data):
    close = data['close']
    d1 = delta(close, 1)
    mn, mx = ts_min(d1, 5), ts_max(d1, 5)
    out = np.where(mn > 0, d1, np.where(mx < 0, d1, -1 * d1))
    return _wrap(out, close).where(mn.notna())


def alpha010(data):
    close = data['close']
    d1 = delta(close, 1)
    mn, mx = ts_min(d1, 4), ts_max(d1, 4)
    out = rank(_wrap(np.where(mn > 0, d1, np.where(mx < 0, d1, -1 * d1)), close))
    return out.where(mn.notna())


# ============ Alpha#11 ~ #30 ============
def alpha011(data):
    close, volume = data['close'], data['volume']
    vc = data['avg'] - close
    return (rank(ts_max(vc, 3)) + rank(ts_min(vc, 3))) * rank(delta(volume, 3))


def alpha012(data):
    return np.sign(delta(data['volume'], 1)) * (-1 * delta(data['close'], 1))


def alpha013(data):
    return -1 * rank(covariance(rank(data['close']), rank(data['volume']), 5))


def alpha014(data):
    return (-1 * rank(delta(data['returns'], 3))) * correlation(data['open'], data['volume'], 10)


def alpha015(data):
    return -1 * ts_sum(rank(correlation(rank(data['high']), rank(data['volume']), 3)), 3)


def alpha016(data):
    return -1 * rank(covariance(rank(data['high']), rank(data['volume']), 5))


def alpha017(data):
    close, volume, adv20 = data['close'], data['volume'], data['adv20']
    return ((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))
            * rank(ts_rank(volume / adv20, 5)))


def alpha018(data):
    close, open_ = data['close'], data['open']
    return -1 * rank((ts_std(np.abs(close - open_), 5) + (close - open_))
                     + correlation(close, open_, 10))


def alpha019(data):
    close, returns = data['close'], data['returns']
    return (-1 * np.sign((close - delay(close, 7)) + delta(close, 7))) \
        * (1 + rank(1 + ts_sum(returns, 250)))


def alpha020(data):
    open_, close, high, low = data['open'], data['close'], data['high'], data['low']
    return ((-1 * rank(open_ - delay(high, 1))) * rank(open_ - delay(close, 1))
            * rank(open_ - delay(low, 1)))


def alpha021(data):
    close, volume, adv20 = data['close'], data['volume'], data['adv20']
    ma8, sd8 = ts_sum(close, 8) / 8.0, ts_std(close, 8)
    ma2 = ts_sum(close, 2) / 2.0
    out = np.where(ma8 + sd8 < ma2, -1.0,
                   np.where(ma2 < ma8 - sd8, 1.0,
                            np.where(volume / adv20 >= 1, 1.0, -1.0)))
    return _wrap(out, close)


def alpha022(data):
    return -1 * (delta(correlation(data['high'], data['volume'], 5), 5)
                 * rank(ts_std(data['close'], 20)))


def alpha023(data):
    high = data['high']
    return _wrap(np.where(ts_sum(high, 20) / 20.0 < high, -1 * delta(high, 2), 0.0), high)


def alpha024(data):
    close = data['close']
    x = delta(ts_sum(close, 100) / 100.0, 100) / delay(close, 100)
    out = np.where(x <= 0.05, -1 * (close - ts_min(close, 100)), -1 * delta(close, 3))
    return _wrap(out, close)


def alpha025(data):
    returns, adv20 = data['returns'], data['adv20']
    return rank((-1 * returns) * adv20 * data['avg'] * (data['high'] - data['close']))


def alpha026(data):
    return -1 * ts_max(correlation(ts_rank(data['volume'], 5), ts_rank(data['high'], 5), 5), 3)


def alpha027(data):
    x = ts_sum(correlation(rank(data['volume']), rank(data['avg']), 6), 2) / 2.0
    return _wrap(np.where(rank(x) > 0.5, -1.0, 1.0), x)


def alpha028(data):
    return scale((correlation(data['adv20'], data['low'], 5)
                  + (data['high'] + data['low']) / 2.0) - data['close'])


def alpha029(data):
    close, returns = data['close'], data['returns']
    x = rank(rank(scale(np.log(ts_sum(ts_min(rank(rank(-1 * rank(delta(close - 1.0, 5)))), 2), 1)))))
    x = rank(x)
    return ts_min(x, 5) + ts_rank(delay(-1 * returns, 6), 5)


def alpha030(data):
    close, volume = data['close'], data['volume']
    d1 = np.sign(close - delay(close, 1)) + np.sign(delay(close, 1) - delay(close, 2)) \
        + np.sign(delay(close, 2) - delay(close, 3))
    return (1.0 - rank(d1)) * ts_sum(volume, 5) / ts_sum(volume, 20)


# ============ Alpha#31 ~ #50 ============
def alpha031(data):
    close, adv20 = data['close'], data['adv20']
    return (rank(rank(rank(decay_linear(-1 * rank(rank(delta(close, 10))), 10))))
            + rank(-1 * delta(close, 3))
            + np.sign(scale(correlation(adv20, data['low'], 12))))


def alpha032(data):
    close, avg = data['close'], data['avg']
    return scale(ts_sum(close, 7) / 7.0 - close) \
        + 20 * scale(correlation(avg, delay(close, 5), 230))


def alpha033(data):
    return rank(-1 * (1 - data['open'] / data['close']))


def alpha034(data):
    close, returns = data['close'], data['returns']
    return rank((1 - rank(ts_std(returns, 2) / ts_std(returns, 5)))
                + (1 - rank(delta(close, 1))))


def alpha035(data):
    close, high, low, volume, returns = data['close'], data['high'], data['low'], data['volume'], data['returns']
    return (ts_rank(volume, 32) * (1 - ts_rank((close + high) - low, 16))
            * (1 - ts_rank(returns, 32)))


def alpha036(data):
    open_, close, volume, returns, adv20 = data['open'], data['close'], data['volume'], data['returns'], data['adv20']
    return (2.21 * rank(correlation(close - open_, delay(volume, 1), 15))
            + 0.7 * rank(open_ - close)
            + 0.73 * rank(ts_rank(delay(-1 * returns, 6), 5))
            + rank(np.abs(correlation(data['avg'], adv20, 6)))
            + 0.6 * rank((ts_sum(close, 200) / 200.0 - open_) * (close - open_)))


def alpha037(data):
    open_, close = data['open'], data['close']
    return rank(correlation(delay(open_ - close, 1), close, 200)) + rank(open_ - close)


def alpha038(data):
    return (-1 * rank(ts_rank(data['close'], 10))) * rank(data['close'] / data['open'])


def alpha039(data):
    close, volume, adv20, returns = data['close'], data['volume'], data['adv20'], data['returns']
    return (-1 * rank(delta(close, 7) * (1 - rank(decay_linear(volume / adv20, 9))))) \
        * (1 + rank(ts_sum(returns, 250)))


def alpha040(data):
    return (-1 * rank(ts_std(data['high'], 10))) * correlation(data['high'], data['volume'], 10)


def alpha041(data):
    return (data['high'] * data['low']) ** 0.5 - data['avg']


def alpha042(data):
    avg = data['avg']
    return rank(avg - data['close']) / rank(avg + data['close'])


def alpha043(data):
    return ts_rank(data['volume'] / data['adv20'], 20) * ts_rank(-1 * delta(data['close'], 7), 8)


def alpha044(data):
    return -1 * correlation(data['high'], rank(data['volume']), 5)


def alpha045(data):
    close, volume = data['close'], data['volume']
    return -1 * (rank(ts_sum(delay(close, 5), 20) / 20.0) * correlation(close, volume, 2)
                 * rank(correlation(ts_sum(close, 5), ts_sum(close, 20), 2)))


def alpha046(data):
    close = data['close']
    x = (delay(close, 20) - delay(close, 10)) / 10.0 - (delay(close, 10) - close) / 10.0
    out = np.where(x > 0.25, -1.0, np.where(x < 0, 1.0, -1.0 * (close - delay(close, 1))))
    return _wrap(out, close)


def alpha047(data):
    close, high, volume, adv20 = data['close'], data['high'], data['volume'], data['adv20']
    return ((rank(1.0 / close) * volume / adv20)
            * (high * rank(high - close)) / (ts_sum(high, 5) / 5.0)
            - rank(data['avg'] - delay(data['avg'], 5)))


def alpha048(data):
    close = data['close']
    d1 = delta(close, 1)
    x = correlation(d1, delta(delay(close, 1), 1), 250) * d1 / close
    return indneutralize(x, data['ind_l2']) / ts_sum((d1 / delay(close, 1)) ** 2, 250)


def alpha049(data):
    close = data['close']
    x = (delay(close, 20) - delay(close, 10)) / 10.0 - (delay(close, 10) - close) / 10.0
    out = np.where(x < -0.1, 1.0, -1.0 * (close - delay(close, 1)))
    return _wrap(out, close)


def alpha050(data):
    return -1 * ts_max(rank(correlation(rank(data['volume']), rank(data['avg']), 5)), 5)


# ============ Alpha#51 ~ #70 ============
def alpha051(data):
    close = data['close']
    x = (delay(close, 20) - delay(close, 10)) / 10.0 - (delay(close, 10) - close) / 10.0
    out = np.where(x < -0.05, 1.0, -1.0 * (close - delay(close, 1)))
    return _wrap(out, close)


def alpha052(data):
    low, returns, volume = data['low'], data['returns'], data['volume']
    return ((-1 * ts_min(low, 5) + delay(ts_min(low, 5), 5))
            * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220.0)
            * ts_rank(volume, 5))


def alpha053(data):
    close, low, high = data['close'], data['low'], data['high']
    return -1 * delta(((close - low) - (high - close)) / (close - low), 9)


def alpha054(data):
    open_, close, low, high = data['open'], data['close'], data['low'], data['high']
    return -1 * ((low - close) * (open_ ** 5)) / ((low - high) * (close ** 5))


def alpha055(data):
    close, low, high, volume = data['close'], data['low'], data['high'], data['volume']
    return -1 * correlation(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))),
                            rank(volume), 6)


def alpha056(data):
    returns, cap = data['returns'], data['cap']
    return -1 * (rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3))
                 * rank(returns * cap))


def alpha057(data):
    close, avg = data['close'], data['avg']
    return -1 * ((close - avg) / decay_linear(rank(ts_argmax(close, 30)), 2))


def alpha058(data):
    return -1 * ts_rank(decay_linear(
        correlation(indneutralize(data['avg'], data['ind_l1']), data['volume'], 3.92795),
        7.89291), 5.50322)


def alpha059(data):
    avg, volume = data['avg'], data['volume']
    x = avg * 0.728317 + avg * (1 - 0.728317)   # = avg
    return -1 * ts_rank(decay_linear(
        correlation(indneutralize(x, data['ind_l1']), volume, 4.25197), 16.2289), 8.19648)


def alpha060(data):
    close, low, high, volume = data['close'], data['low'], data['high'], data['volume']
    return -1 * (2 * scale(rank(((close - low) - (high - close)) / (high - low) * volume))
                 - scale(rank(ts_argmax(close, 10))))


def alpha061(data):
    avg = data['avg']
    return rank(avg - ts_min(avg, 16.1219)) < rank(correlation(avg, data['adv180'], 17.9282))


def alpha062(data):
    open_, high, low, avg = data['open'], data['high'], data['low'], data['avg']
    return (rank(correlation(avg, ts_sum(data['adv20'], 22.4101), 9.91009))
            < rank((rank(open_) + rank(open_)) < (rank((high + low) / 2.0) + rank(high)))) * -1


def alpha063(data):
    close, avg, open_ = data['close'], data['avg'], data['open']
    return (rank(decay_linear(delta(indneutralize(close, data['ind_l1']), 2.25164), 8.22237))
            - rank(decay_linear(correlation(avg * 0.318108 + open_ * (1 - 0.318108),
                                            ts_sum(data['adv180'], 37.2467), 13.557), 12.2883))) * -1


def alpha064(data):
    open_, low, high, avg = data['open'], data['low'], data['high'], data['avg']
    return (rank(correlation(ts_sum(open_ * 0.178404 + low * (1 - 0.178404), 12.7054),
                             ts_sum(data['adv120'], 12.7054), 16.6208))
            < rank(delta((high + low) / 2.0 * 0.178404 + avg * (1 - 0.178404), 3.69741))) * -1


def alpha065(data):
    open_, avg = data['open'], data['avg']
    return (rank(correlation(open_ * 0.00817205 + avg * (1 - 0.00817205),
                             ts_sum(data['adv60'], 8.6911), 6.40374))
            < rank(open_ - ts_min(open_, 13.635))) * -1


def alpha066(data):
    low, open_, high, avg = data['low'], data['open'], data['high'], data['avg']
    return (rank(decay_linear(delta(avg, 3.51013), 7.23052))
            + ts_rank(decay_linear((low - avg) / (open_ - (high + low) / 2.0), 11.4157), 6.72611)) * -1


def alpha067(data):
    high, avg = data['high'], data['avg']
    return (rank(high - ts_min(high, 2.14593))
            ** rank(correlation(indneutralize(avg, data['ind_l1']),
                                indneutralize(data['adv20'], data['ind_l2']), 6.02936))) * -1


def alpha068(data):
    close, low, high = data['close'], data['low'], data['high']
    return (ts_rank(correlation(rank(high), rank(data['adv15']), 8.91644), 13.9333)
            < rank(delta(close * 0.518371 + low * (1 - 0.518371), 1.06157))) * -1


def alpha069(data):
    close, avg = data['close'], data['avg']
    return (rank(ts_max(delta(indneutralize(avg, data['ind_l1']), 2.72412), 4.79344))
            ** ts_rank(correlation(close * 0.490655 + avg * (1 - 0.490655),
                                   data['adv20'], 4.92416), 9.0615)) * -1


def alpha070(data):
    avg = data['avg']
    return (rank(delta(avg, 1.29456))
            ** ts_rank(correlation(indneutralize(data['close'], data['ind_l1']),
                                   data['adv50'], 17.8256), 17.9171)) * -1


# ============ Alpha#71 ~ #90 ============
def alpha071(data):
    close, low, open_, avg = data['close'], data['low'], data['open'], data['avg']
    return np.fmax(
        ts_rank(decay_linear(correlation(ts_rank(close, 3.43976), ts_rank(data['adv180'], 12.0647), 18.0175),
                             4.20501), 15.6948),
        ts_rank(decay_linear(rank((low + open_) - (avg + avg)) ** 2, 16.4662), 4.4388))


def alpha072(data):
    high, low, avg, volume = data['high'], data['low'], data['avg'], data['volume']
    return (rank(decay_linear(correlation((high + low) / 2.0, data['adv40'], 8.93345), 10.1519))
            / rank(decay_linear(correlation(ts_rank(avg, 3.72469), ts_rank(volume, 18.5188), 6.86671),
                                2.95011)))


def alpha073(data):
    open_, low, avg = data['open'], data['low'], data['avg']
    return np.fmax(
        rank(decay_linear(delta(avg, 4.72775), 2.91864)),
        ts_rank(decay_linear(-1 * delta(open_ * 0.147155 + low * (1 - 0.147155), 2.03608)
                             / (open_ * 0.147155 + low * (1 - 0.147155)), 3.33829), 16.7411)) * -1


def alpha074(data):
    close, high, avg, volume = data['close'], data['high'], data['avg'], data['volume']
    return (rank(correlation(close, ts_sum(data['adv30'], 37.4843), 15.1365))
            < rank(correlation(rank(high * 0.0261661 + avg * (1 - 0.0261661)),
                               rank(volume), 11.4791))) * -1


def alpha075(data):
    avg, low, volume = data['avg'], data['low'], data['volume']
    return rank(correlation(avg, volume, 4.24304)) \
        < rank(correlation(rank(low), rank(data['adv50']), 12.4413))


def alpha076(data):
    avg, low = data['avg'], data['low']
    return np.fmax(
        rank(decay_linear(delta(avg, 1.24383), 11.8259)),
        ts_rank(decay_linear(ts_rank(correlation(indneutralize(low, data['ind_l1']),
                                                 data['adv81'], 8.14941), 19.569), 17.1543), 19.383)) * -1


def alpha077(data):
    high, low = data['high'], data['low']
    return np.fmin(
        rank(decay_linear((high + low) / 2.0 - data['avg'], 20.0451)),
        rank(decay_linear(correlation((high + low) / 2.0, data['adv40'], 3.1614), 5.64125)))


def alpha078(data):
    low, avg, volume = data['low'], data['avg'], data['volume']
    return (rank(correlation(ts_sum(low * 0.352233 + avg * (1 - 0.352233), 19.7428),
                             ts_sum(data['adv40'], 19.7428), 6.83313))
            ** rank(correlation(rank(avg), rank(volume), 5.77492)))


def alpha079(data):
    close, open_, avg = data['close'], data['open'], data['avg']
    x = rank(delta(indneutralize(close * 0.60733 + open_ * (1 - 0.60733), data['ind_l1']), 1.23438))
    y = rank(correlation(ts_rank(avg, 3.60973), ts_rank(data['adv150'], 9.18637), 14.6644))
    # indneutralize 会剔除无行业股票，比较前对齐列
    return x.reindex(columns=y.columns) < y


def alpha080(data):
    open_, high = data['open'], data['high']
    return (rank(np.sign(delta(indneutralize(open_ * 0.868128 + high * (1 - 0.868128),
                                             data['ind_l1']), 4.04545)))
            ** ts_rank(correlation(high, data['adv10'], 5.11456), 5.53756)) * -1


def alpha081(data):
    avg, volume = data['avg'], data['volume']
    return (rank(np.log(ts_product(rank(rank(correlation(avg, ts_sum(data['adv10'], 49.6054), 8.47743)) ** 4),
                                   14.9655)))
            < rank(correlation(rank(avg), rank(volume), 5.07914))) * -1


def alpha082(data):
    open_, volume = data['open'], data['volume']
    return np.fmin(
        rank(decay_linear(delta(open_, 1.46063), 14.8717)),
        ts_rank(decay_linear(correlation(indneutralize(volume, data['ind_l1']), open_, 17.4842),
                             6.92131), 13.4283)) * -1


def alpha083(data):
    close, high, low, volume, avg = data['close'], data['high'], data['low'], data['volume'], data['avg']
    amp = (high - low) / (ts_sum(close, 5) / 5.0)
    return (rank(delay(amp, 2)) * rank(rank(volume))) / (amp / (avg - close))


def alpha084(data):
    avg, close = data['avg'], data['close']
    return signedpower(ts_rank(avg - ts_max(avg, 15.3217), 20.7127), delta(close, 4.96796))


def alpha085(data):
    high, close, low, volume = data['high'], data['close'], data['low'], data['volume']
    return (rank(correlation(high * 0.876703 + close * (1 - 0.876703), data['adv30'], 9.61331))
            ** rank(correlation(ts_rank((high + low) / 2.0, 3.70596), ts_rank(volume, 10.1595), 7.11408)))


def alpha086(data):
    close, avg = data['close'], data['avg']
    return (ts_rank(correlation(close, ts_sum(data['adv20'], 14.7444), 6.00049), 20.4195)
            < rank(close - avg)) * -1


def alpha087(data):
    close, avg = data['close'], data['avg']
    return np.fmax(
        rank(decay_linear(delta(close * 0.369701 + avg * (1 - 0.369701), 1.91233), 2.65461)),
        ts_rank(decay_linear(np.abs(correlation(indneutralize(data['adv81'], data['ind_l1']),
                                                close, 13.4132)), 4.89768), 14.4535)) * -1


def alpha088(data):
    open_, low, high, close = data['open'], data['low'], data['high'], data['close']
    return np.fmin(
        rank(decay_linear((rank(open_) + rank(low)) - (rank(high) + rank(close)), 8.06882)),
        ts_rank(decay_linear(correlation(ts_rank(close, 8.44728), ts_rank(data['adv60'], 20.6966), 8.01266),
                             6.65053), 2.61957))


def alpha089(data):
    low, avg = data['low'], data['avg']
    return (ts_rank(decay_linear(correlation(low, data['adv10'], 6.94279), 5.51607), 3.79744)
            - ts_rank(decay_linear(delta(indneutralize(avg, data['ind_l1']), 3.48158), 10.1466), 15.3012))


def alpha090(data):
    close, low = data['close'], data['low']
    return (rank(close - ts_max(close, 4.66719))
            ** ts_rank(correlation(indneutralize(data['adv40'], data['ind_l2']), low, 5.38375), 3.21856)) * -1


# ============ Alpha#91 ~ #101 ============
def alpha091(data):
    close, volume, avg = data['close'], data['volume'], data['avg']
    return (ts_rank(decay_linear(decay_linear(correlation(indneutralize(close, data['ind_l1']),
                                                          volume, 9.74928), 16.398), 3.83219), 4.8667)
            - rank(decay_linear(correlation(avg, data['adv30'], 4.01303), 2.6809))) * -1


def alpha092(data):
    high, low, close, open_ = data['high'], data['low'], data['close'], data['open']
    return np.fmin(
        ts_rank(decay_linear(((high + low) / 2.0 + close < low + open_).astype(float), 14.7221), 18.8683),
        ts_rank(decay_linear(correlation(rank(low), rank(data['adv30']), 7.58555), 6.94024), 6.80584))


def alpha093(data):
    close, avg = data['close'], data['avg']
    return (ts_rank(decay_linear(correlation(indneutralize(avg, data['ind_l1']),
                                             data['adv81'], 17.4193), 19.848), 7.54455)
            / rank(decay_linear(delta(close * 0.524434 + avg * (1 - 0.524434), 2.77377), 16.2664)))


def alpha094(data):
    avg = data['avg']
    return (rank(avg - ts_min(avg, 11.5783))
            ** ts_rank(correlation(ts_rank(avg, 19.6462), ts_rank(data['adv60'], 4.02992), 18.0926), 2.70756)) * -1


def alpha095(data):
    open_, high, low = data['open'], data['high'], data['low']
    return rank(open_ - ts_min(open_, 12.4105)) \
        < ts_rank(rank(correlation(ts_sum((high + low) / 2.0, 19.1351),
                                   ts_sum(data['adv40'], 19.1351), 12.8742)) ** 5, 11.7584)


def alpha096(data):
    avg, volume, close = data['avg'], data['volume'], data['close']
    return np.fmax(
        ts_rank(decay_linear(correlation(rank(avg), rank(volume), 3.83878), 4.16783), 8.38151),
        ts_rank(decay_linear(ts_argmax(correlation(ts_rank(close, 7.45404),
                                                   ts_rank(data['adv60'], 4.13242), 3.65459), 12.6556),
                             14.0365), 13.4143)) * -1


def alpha097(data):
    low, avg = data['low'], data['avg']
    return (rank(decay_linear(delta(indneutralize(low * 0.721001 + avg * (1 - 0.721001), data['ind_l1']),
                                     3.3705), 20.4523))
            - ts_rank(decay_linear(ts_rank(correlation(ts_rank(low, 7.87871),
                                                       ts_rank(data['adv60'], 17.255), 4.97547),
                                           18.5925), 15.7152), 6.71659)) * -1


def alpha098(data):
    avg, open_ = data['avg'], data['open']
    return (rank(decay_linear(correlation(avg, ts_sum(data['adv5'], 26.4719), 4.58418), 7.18088))
            - rank(decay_linear(ts_rank(ts_argmin(correlation(rank(open_), rank(data['adv15']), 20.8187),
                                                  8.62571), 6.95668), 8.07206)))


def alpha099(data):
    high, low, volume = data['high'], data['low'], data['volume']
    return (rank(correlation(ts_sum((high + low) / 2.0, 19.8975),
                             ts_sum(data['adv60'], 19.8975), 8.8136))
            < rank(correlation(low, volume, 6.28259))) * -1


def alpha100(data):
    close, low, high, volume, adv20 = data['close'], data['low'], data['high'], data['volume'], data['adv20']
    clv = ((close - low) - (high - close)) / (high - low) * volume
    t1 = 1.5 * scale(indneutralize(indneutralize(rank(clv), data['ind_l2']), data['ind_l2']))
    t2 = scale(indneutralize(correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30)),
                             data['ind_l2']))
    return -1 * (t1 - t2) * (volume / adv20)


def alpha101(data):
    close, open_, high, low = data['close'], data['open'], data['high'], data['low']
    return (close - open_) / ((high - low) + 0.001)


# 因子注册表（全部 101 个）
FACTOR_FUNCS = {}
for _i in range(1, 102):
    FACTOR_FUNCS['alpha%03d' % _i] = globals()['alpha%03d' % _i]
