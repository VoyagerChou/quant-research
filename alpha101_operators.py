# -*- coding: utf-8 -*-
"""
Alpha101 算子库（pandas 向量化实现）
=====================================
与《101 Formulaic Alphas》(Kakushadze 2015) 附录 A.2 的算子定义一一对应。
中文翻译说明见仓库根目录《101_Formulaic_Alphas_中文翻译.md》附录 A.2。

约定：
- 所有函数输入/输出均为 pandas.DataFrame，index=日期，columns=股票代码
- rank() 为横截面排序（按天，对当天所有股票排序，输出 0~1 分位）
- ts_* 为时间序列算子（逐股票滚动窗口，要求窗口内有 min_periods 个观测才计算）
- 论文约定：非整数天数一律向下取整 floor(d)（本库所有 d 参数都用 int() 强制）
- NaN 处理：滚动窗口不足则输出 NaN；rank 自动跳过 NaN
"""
import sys
import types as _types
import pandas as pd
# 防御性获取 numpy：历史上调试时发现旧版聚宽研究环境配合旧版 numpy 时，
# np.log 作用在字符串数据上会报 "'str' object has no attribute 'log'"，
# 极易误判为 numpy 被污染。这里统一从 sys.modules 取真实模块对象，
# 不依赖 import 语句的绑定结果（在任何环境下都安全）。
_np_mod = sys.modules.get('numpy')
if _np_mod is None or not isinstance(_np_mod, _types.ModuleType):
    _np_mod = __import__('numpy')          # 兜底：真正加载一次
np = _np_mod


def rank(x):
    """横截面排序：每天对所有股票按 x 值排序，输出 0~1 分位。
    pandas 的 rank 默认跳过 NaN，即 NaN 保持为 NaN。"""
    return x.rank(axis=1, pct=True)


def delay(x, d):
    """x 在 d 天前的值"""
    return x.shift(int(d))


def delta(x, d):
    """今天值 - d 天前值"""
    return x - x.shift(int(d))


def ts_sum(x, d):
    """过去 d 天的时间序列求和"""
    d = int(d)
    return x.rolling(d, min_periods=d).sum()


def ts_product(x, d):
    """过去 d 天的时间序列乘积（向量化 shift 实现）"""
    d = int(d)
    out = x * 1.0
    for i in range(1, d):
        out = out * x.shift(i)
    return out.where(x.rolling(d, min_periods=d).count() >= d)


def ts_std(x, d):
    """过去 d 天的滚动时间序列标准差（总体标准差 ddof=0）"""
    d = int(d)
    return x.rolling(d, min_periods=d).std(ddof=0)


def ts_min(x, d):
    """过去 d 天的时间序列最小值"""
    d = int(d)
    return x.rolling(d, min_periods=d).min()


def ts_max(x, d):
    """过去 d 天的时间序列最大值"""
    d = int(d)
    return x.rolling(d, min_periods=d).max()


def ts_rank(x, d):
    """过去 d 天的时间序列排序分位：最新一天的值在窗口内处于什么分位（0~1）。
    向量化实现：统计窗口内（不含当天）小于等于当天值的个数，加 1 除以 d。
    与论文定义（rank of latest value in window）一致，平局取严格小于计数。"""
    d = int(d)
    s = pd.DataFrame(1.0, index=x.index, columns=x.columns)
    for i in range(1, d):
        s = s + (x > x.shift(i)).astype(float)
    s = s / d
    # 窗口不足 d 天时输出 NaN（前 d-1 天无定义）
    s = s.where(x.rolling(d).count() >= d)
    return s


def ts_argmax(x, d):
    """过去 d 天最大值出现在第几天（返回 0..d-1，0=当天，d-1=d 天前）。
    向量化实现：滚动最大值 + 从近到远找第一次等于最大值的位置。"""
    d = int(d)
    mx = x.rolling(d, min_periods=d).max()
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for i in range(d):
        eq = (x.shift(i) == mx) & out.isna()
        out = out.mask(eq, float(i))
    return out


def ts_argmin(x, d):
    """过去 d 天最小值出现在第几天（返回 0..d-1）。向量化实现同 ts_argmax。"""
    d = int(d)
    mn = x.rolling(d, min_periods=d).min()
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for i in range(d):
        eq = (x.shift(i) == mn) & out.isna()
        out = out.mask(eq, float(i))
    return out


def correlation(x, y, d):
    """x 与 y 过去 d 天的时间序列相关系数"""
    d = int(d)
    return x.rolling(d, min_periods=d).corr(y)


def covariance(x, y, d):
    """x 与 y 过去 d 天的时间序列协方差"""
    d = int(d)
    return x.rolling(d, min_periods=d).cov(y)


def scale(x, a=1.0):
    """整体缩放：使每天横截面的 sum(abs(x)) = a（默认 1）。
    保持每天股票间的相对大小，消除量纲。"""
    return x.div(x.abs().sum(axis=1), axis=0).mul(a)


def signedpower(x, a):
    """保留符号的幂：|x|^a，符号与 x 相同（论文 SignedPower(x, a) = x^a，按符号幂实现）"""
    return np.sign(x) * np.abs(x) ** a


def decay_linear(x, d):
    """过去 d 天的线性衰减加权移动平均。
    权重为 d, d-1, ..., 1（最近一天权重最大），缩放使权重和为 1。
    向量化 shift 实现（避免 rolling.apply 在 101 因子全量计算时的性能问题）。"""
    d = int(d)
    w = np.arange(d, 0, -1, dtype=float)
    w = w / w.sum()
    out = x * w[0]
    for i in range(1, d):
        out = out + x.shift(i) * w[i]
    return out.where(x.rolling(d, min_periods=d).count() >= d)


def indneutralize(x, g):
    """按分组 g 做横截面去均值（行业中性化）。
    g: 与 x 的列对齐的 pandas.Series，index=股票代码，value=分组标签（如申万行业名）。
    在每个组内对每天的横截面值减去该组的横截面均值。
    注意：分组标签为 NaN 的股票列会被剔除（返回的列中不含这些股票）。
    实现用显式循环（避免 DataFrame.groupby(axis=1) 在 pandas 2.2+/3.x 中被移除的问题）。"""
    gmap = g.reindex(x.columns)
    valid_cols = gmap.dropna().index
    x = x[valid_cols]
    gmap = gmap[valid_cols]
    out = x.copy()
    for label in gmap.unique():
        cols = gmap[gmap == label].index
        if len(cols) == 1:
            out[cols] = 0.0            # 组内仅一只股票，去均值后为 0
        else:
            out[cols] = x[cols].sub(x[cols].mean(axis=1), axis=0)
    return out
