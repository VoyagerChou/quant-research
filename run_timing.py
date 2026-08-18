# -*- coding: utf-8 -*-
"""
趋势强度敞口择时（本地运行）
============================
基于六因子合成（comp_w）的敞口择时研究。
依据：regime 诊断发现——反转因子的 ICIR 随趋势强度单调下降且不对称
（深跌端 0.39 > 大涨端 0.15），因此：下跌/中性状态满敞口、强上涨状态降敞口。

信号：comp_w 的日度截面得分（来自 run_composite 的 pred_comp_w.csv）
状态：带符号趋势强度 s_t = 市值加权指数/MA20 - 1 的滚动分位（500 日窗口，
      只用当期及过去信息，防前视）
规则（不对称）：p<=p1 敞口1.0 | p1<p<=p2 敞口e_mid | p>p2 敞口e_high
参数（p1,p2,e_mid,e_high）在训练段网格搜索，按 Calmar 选优；测试段固定参数只跑一次。

成本模型：前 20%/后 20% 等权多空组合，t+1 成交；
  每日单边换手 = 0.5*Σ|Δw|（含敞口变化引起的调仓），成本 = 换手*双边费率。

输出（results_timing_<指数>/）：timing_compare.csv + exposure_optimal.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'D:\Quant\Quant_research\code')
import jqdata
import alpha101_test_core as core

OUT = 'results_timing_%s' % jqdata._get_active_index()
COST = 0.001          # 双边费率（千一）
QUANTILE_WIN = 500    # 趋势强度滚动分位窗口
TRAIN = ('2015-01-01', '2023-12-31')
TEST = ('2024-01-01', '2025-12-31')
TOP_PCT = 0.2         # 多空各取前/后 20%


def load_comp_w():
    """加载本指数的 comp_w 预测（run_composite 输出 comp_w_z.csv），
    并应用有效掩码（成员/ST/价格有效）——历史 CSV 可能未带掩码，这里兜底。"""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'results_composite_%s' % jqdata._get_active_index())
    f = os.path.join(d, 'comp_w_z.csv')
    if not os.path.isfile(f):
        raise IOError('未找到 %s，请先运行 run_composite.py' % f)
    p = pd.read_csv(f, index_col=0)
    p.index = pd.to_datetime(p.index)
    # 兜底掩码：非成分股/ST/停牌日置 NaN，避免排名被填 0 的列污染
    jqdata._load()
    vm = core.build_valid_mask(jqdata._price, jqdata._st, jqdata._member)
    vm = vm.reindex(index=p.index, columns=p.columns)
    p = p.where(vm)
    return p


def build_state():
    """市值加权指数 + 趋势强度 + 滚动分位（t 日算，t+1 生效）"""
    jqdata._load()
    close = jqdata._price['close']
    member = jqdata._member
    mcap = jqdata._mcap
    ret = close.pct_change()
    w = mcap.where(member)
    w = w.div(w.sum(axis=1), axis=0)
    idx_ret = (ret * w).sum(axis=1)
    idx_close = (1 + idx_ret.fillna(0)).cumprod()
    s = idx_close / idx_close.rolling(20).mean() - 1        # 带符号趋势强度
    # 滚动分位：过去 QUANTILE_WIN 日中 s 的百分位（不含当天），再 shift 1 天
    pct = s.rolling(QUANTILE_WIN, min_periods=120).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() * 100, raw=False)
    pct = pct.shift(1)
    return idx_ret, pct


def signal_to_weights(signal, exp):
    """信号 → 多空权重矩阵（含敞口）：前 TOP_PCT 做多、后 TOP_PCT 做空，组内等权"""
    n = signal.notna().sum(axis=1)
    rp = signal.rank(axis=1, pct=True)
    w = ((rp >= 1 - TOP_PCT).astype(float) - (rp <= TOP_PCT).astype(float)) \
        * exp.values[:, None] / (TOP_PCT * n).values[:, None]
    return w.where(signal.notna(), 0.0)


def portfolio_ret(signal, exp, stock_ret, cost):
    """多空组合日度收益（t 日权重、t+1 成交），扣换手成本"""
    w = signal_to_weights(signal, exp)
    w_prev = w.shift(1).fillna(0.0)
    turnover = 0.5 * (w - w_prev).abs().sum(axis=1)
    ret = (w_prev * stock_ret).sum(axis=1)
    return ret - turnover * cost, turnover


def exposure_rule(p, p1, p2, e_mid, e_high):
    """不对称敞口：p<=p1 满敞口；p1<p<=p2 用 e_mid；>p2 用 e_high"""
    exp = pd.Series(1.0, index=p.index)
    exp[p > p1] = e_mid
    exp[p > p2] = e_high
    return exp


def metrics(net, start, end):
    seg = net.loc[start:end].dropna()
    if len(seg) < 60:
        return np.nan, np.nan, np.nan, np.nan
    r = seg.pct_change().dropna()
    ann = (seg.iloc[-1] / seg.iloc[0]) ** (252 / len(seg)) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    mdd = (seg / seg.cummax() - 1).min()
    calmar = ann / abs(mdd) if mdd < 0 else np.nan
    return ann, sharpe, mdd, calmar


def main():
    os.makedirs(OUT, exist_ok=True)
    print('== 择时研究 | 指数 %s | 双边费率 %.4f ==' % (jqdata._get_active_index(), COST))
    signal = load_comp_w()
    idx_ret, pct = build_state()
    pct = pct.reindex(signal.index)
    jqdata._load()
    stock_ret = jqdata._price['close'].pct_change().reindex(columns=signal.columns)

    rows = []
    # 基线：不择时
    exp0 = pd.Series(1.0, index=signal.index)
    r, to = portfolio_ret(signal, exp0, stock_ret, COST)
    net = (1 + r.fillna(0)).cumprod()
    for pname, (ps, pe) in [('train', TRAIN), ('test', TEST)]:
        ann, sh, mdd, cal = metrics(net, ps, pe)
        rows.append({'rule': 'baseline', 'period': pname, '年化': ann, '夏普': sh,
                     '最大回撤': mdd, 'Calmar': cal, '日均换手': to.mean()})

    # 网格搜索（训练段 Calmar 选优）
    best_cal = -np.inf
    best = None
    for p1 in (0.5, 0.6):
        for p2 in (0.8, 0.9):
            for em, eh in [(0.75, 0.5), (0.6, 0.3)]:
                exp = exposure_rule(pct, p1, p2, em, eh)
                r, to = portfolio_ret(signal, exp, stock_ret, COST)
                net = (1 + r.fillna(0)).cumprod()
                ann, sh, mdd, cal = metrics(net, TRAIN[0], TRAIN[1])
                print('  网格 p1=%.1f p2=%.1f em=%.2f eh=%.2f | 训练Calmar=%.2f'
                      % (p1, p2, em, eh, cal))
                if not np.isnan(cal) and cal > best_cal:
                    best_cal, best = cal, (p1, p2, em, eh)
    print('最优参数: p1=%.1f p2=%.1f e_mid=%.2f e_high=%.2f | 训练Calmar=%.2f'
          % (best + (best_cal,)))
    p1o, p2o, emo, eho = best
    exp = exposure_rule(pct, p1o, p2o, emo, eho)
    r, to = portfolio_ret(signal, exp, stock_ret, COST)
    net = (1 + r.fillna(0)).cumprod()
    for pname, (ps, pe) in [('train', TRAIN), ('test', TEST)]:
        ann, sh, mdd, cal = metrics(net, ps, pe)
        rows.append({'rule': 'timing(最优)', 'period': pname, '年化': ann, '夏普': sh,
                     '最大回撤': mdd, 'Calmar': cal, '日均换手': to.mean()})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'timing_compare.csv'), index=False, encoding='utf-8-sig')
    exp.to_csv(os.path.join(OUT, 'exposure_optimal.csv'), encoding='utf-8-sig')
    print()
    print('===== 对比表（训练/测试）=====')
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
