# -*- coding: utf-8 -*-
"""
2026 最终验证：14因子 comp_eq / comp_w 在 1000 池的表现
========================================================
【用途】用从未参与过挖掘/调参/验证的 2026 数据，最终检验 14 因子
comp_eq（等权）与 comp_w（滚动ICIR加权）的真实效果。

【防前视纪律（重要）】
  - 复合参数 100% 由 2025-12-31 及以前的信息决定：
      * comp_eq：等权 + 方向符号固定为训练期 IC 符号（全 +1，已核实）
      * comp_w：滚动 ICIR 权重，天然只含 t-1 及以前信息（与历史运行一致）
  - 2026 段只计算信号并评估，不重新拟合任何参数
  - 本脚本结果即"最终验证"，不做任何迭代

【数据】需先完成：
  1. 聚宽运行 export_data_2026.py → 下载到 data/2026_000852/
  2. 本地运行 merge_2026.py（拼接+校准，历史部分不动）

输出（research_notes/1000池_2026最终验证.csv）：
  逐月 IC/ICIR/收益/夏普/回撤/胜率 + 2026 整段汇总
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'D:\Quant\Quant_research\code')
from jqdata import *
import jqdata
import alpha101_test_core as core
from alpha101_factors import FACTOR_FUNCS

OUT = r'D:\Quant\Quant_research\research_notes\1000池_2026最终验证.csv'
FACTORS = ['alpha003', 'alpha006', 'alpha019', 'alpha015', 'alpha021', 'alpha026',
           'alpha040', 'alpha012', 'alpha045', 'alpha018', 'alpha024',
           'alpha069', 'alpha088', 'alpha044']
# 训练期（截至2025-12-31）方向符号：全部 +1（已核实，固定参数，2026 不再调整）
SIGNS = {f: 1 for f in FACTORS}
VALID26 = ('2026-01-01', '2026-12-31')
DATA_END = '2026-08-18'   # 本地数据实际终点（拼接后）


def build_2026_signals():
    """计算 14 因子中性化 z-score（与历史完全同口径），返回面板"""
    stocks, snap_list = core.fetch_index_members(core.INDEX_CODE, core.FETCH_START, DATA_END)
    price = core.fetch_price(stocks, core.FETCH_START, DATA_END)
    price['returns'] = price['close'].pct_change()
    mcap = core.fetch_market_cap(stocks, core.FETCH_START, DATA_END)
    log_mcap = np.log(mcap.replace(0, np.nan)).reindex(
        price['close'].index, columns=price['close'].columns)
    st = core.fetch_st(stocks, core.FETCH_START, DATA_END)
    member_mask = core.build_member_mask(snap_list, price['close'].index, stocks, DATA_END)
    raw_ind = get_industry(stocks)
    industry_map = {k: core._extract_industry_name(v) for k, v in raw_ind.items()}
    industry_l2 = {k: core._extract_industry_level(v, 'sw_l2') for k, v in raw_ind.items()}
    valid_mask = core.build_valid_mask(price, st, member_mask)

    base = {k: price[k] for k in ['open', 'close', 'high', 'low', 'volume', 'avg', 'returns']}
    money = price.pop('money')
    cap_df = mcap.reindex(price['close'].index, columns=price['close'].columns)
    data = core._LazyData(base, money, cap_df,
                          pd.Series(industry_map), pd.Series(industry_l2))

    z_panels = {}
    for name in FACTORS:
        f = FACTOR_FUNCS[name](data)
        if not isinstance(f, pd.DataFrame):
            f = pd.DataFrame(f, index=price['close'].index, columns=price['close'].columns)
        f = f.where(valid_mask)
        f_neu = core.neutralize(f, industry_map, log_mcap)
        sd = f_neu.std(axis=1).replace(0, np.nan)
        z = f_neu.sub(f_neu.mean(axis=1), axis=0).div(sd, axis=0).clip(-5, 5)
        z_panels[name] = z.astype(np.float32)
    return z_panels, price, valid_mask


def build_comp(z_panels, price, valid_mask):
    """comp_eq（等权固定符号）+ comp_w（滚动ICIR加权，只用历史）"""
    names = list(z_panels.keys())
    idx = z_panels[names[0]].index
    cols = z_panels[names[0]].columns

    fwd1 = core.forward_returns(price['close'], 1)
    fwd1_r = fwd1.rank(axis=1, pct=True).astype(np.float32)
    ic_series = {}
    for n, f in z_panels.items():
        f_r = f.rank(axis=1, pct=True).astype(np.float32)
        ic_series[n] = core.rank_ic_from_ranked(f_r, fwd1_r)
    ic_df = pd.DataFrame(ic_series)
    ic_mean = ic_df.rolling(252, min_periods=60).mean().shift(1)
    ic_std = ic_df.rolling(252, min_periods=60).std().shift(1)
    icir = ic_mean / ic_std.replace(0, np.nan)
    w = icir.abs().div(icir.abs().sum(axis=1).replace(0, np.nan), axis=0)

    # comp_eq：方向符号固定（SIGNS），等权
    aligned = {n: z_panels[n].mul(SIGNS[n], axis=0) for n in names}
    z_arr = np.nan_to_num(np.stack([aligned[n].values for n in names])
                           .astype(np.float32), nan=0.0)
    comp_eq = pd.DataFrame(np.nanmean(z_arr, axis=0), index=idx, columns=cols)
    # comp_w：滚动 ICIR 权重（shift(1) 已含）
    wsum = icir.abs().sum(axis=1)
    comp_w_vals = np.einsum('tn,nts->ts', w.fillna(0).values, z_arr)
    comp_w_vals = np.where(np.isnan(wsum.values)[:, None], np.nan, comp_w_vals)
    comp_w = pd.DataFrame(comp_w_vals, index=idx, columns=cols)
    return comp_eq.where(valid_mask.reindex(columns=cols)), \
        comp_w.where(valid_mask.reindex(columns=cols))


def monthly_report(signal, price_close, valid_range):
    """2026 逐月 IC/ICIR/多空绩效（与验证期脚本同口径）"""
    fwd1 = core.forward_returns(price_close, 1).reindex(columns=signal.columns)
    p_r = signal.rank(axis=1, pct=True).astype(np.float32)
    ic = core.rank_ic_from_ranked(p_r, fwd1.rank(axis=1, pct=True).astype(np.float32))

    TOP = 0.2
    n = signal.notna().sum(axis=1)
    w = ((p_r >= 1 - TOP).astype(float) - (p_r <= TOP).astype(float)) / (TOP * n).values[:, None]
    w = w.where(signal.notna(), 0.0)
    ls_ret = (w * fwd1).sum(axis=1)

    seg_ic = ic.loc[valid_range[0]:valid_range[1]].dropna()
    seg_ls = ls_ret.reindex(seg_ic.index)
    avg_n = n.loc[valid_range[0]:valid_range[1]].mean()
    print('2026 共 %d 个交易日 | 信号日均有效 %.0f 股' % (len(seg_ic), avg_n))

    rows = []
    for ym, g in seg_ic.groupby(seg_ic.index.to_period('M')):
        gl = seg_ls.reindex(g.index)
        net = (1 + gl.fillna(0)).cumprod()
        rows.append({'月份': str(ym), 'IC': g.mean(), 'ICIR': g.mean() / g.std() if g.std() > 0 else np.nan,
                     'IC>0占比': (g > 0).mean(),
                     '多空月收益': net.iloc[-1] - 1,
                     '多空夏普': gl.mean() / gl.std() * np.sqrt(21) if gl.std() > 0 else np.nan,
                     '多空回撤': (net / net.cummax() - 1).min(),
                     '日胜率': (gl > 0).mean()})
    return pd.DataFrame(rows)


def main():
    jqdata._load()
    d = jqdata._dates
    if d[-1] < pd.Timestamp('2026-01-01'):
        print('!! 数据仍截至 %s，请先完成 2026 导出与拼接：' % d[-1].date())
        print('   1) 聚宽运行 export_data_2026.py，下载到 data/2026_000852/')
        print('   2) 本地运行 merge_2026.py')
        return
    print('数据覆盖: %s ~ %s（含 2026 年 %d 个交易日）'
          % (d[0].date(), d[-1].date(), (d >= '2026-01-01').sum()))

    print('== 计算 14 因子中性化 z-score（与历史同口径）==')
    z_panels, price, valid_mask = build_2026_signals()
    print('== 构建 comp_eq / comp_w（参数固定，2026 不拟合）==')
    comp_eq, comp_w = build_comp(z_panels, price, valid_mask)

    print()
    print('############ 2026 最终验证（此前从未使用的数据）############')
    for cname, c in [('comp_eq', comp_eq), ('comp_w', comp_w)]:
        print()
        print('===== %s =====' % cname)
        m = monthly_report(c, price['close'], VALID26)
        print(m.round(4).to_string(index=False))
        # 汇总：直接对多空日收益序列计算
        m = monthly_report(c, price['close'], VALID26)
        print(m.round(4).to_string(index=False))
        # 多空日收益序列（与 monthly_report 同口径）
        fwd1x = core.forward_returns(price['close'], 1).reindex(columns=c.columns)
        p_rx = c.rank(axis=1, pct=True).astype(np.float32)
        nx = c.notna().sum(axis=1)
        wx = ((p_rx >= 1 - 0.2).astype(float) - (p_rx <= 0.2).astype(float)) \
            / (0.2 * nx).values[:, None]
        wx = wx.where(c.notna(), 0.0)
        lsx = (wx * fwd1x).sum(axis=1).loc[VALID26[0]:VALID26[1]].dropna()
        net = (1 + lsx).cumprod()
        ann = (net.iloc[-1]) ** (252 / len(lsx)) - 1
        sharpe = lsx.mean() / lsx.std() * np.sqrt(252) if lsx.std() > 0 else np.nan
        mdd = (net / net.cummax() - 1).min()
        win = (lsx > 0).mean()
        print('2026 汇总: 年化=%+.1f%% | 夏普=%.2f | 回撤=%.1f%% | 日胜率=%.1f%%'
              % (ann * 100, sharpe, mdd * 100, win * 100))
        m.to_csv(OUT.replace('.csv', '_%s.csv' % cname), index=False, encoding='utf-8-sig')
    print()
    print('对比（历史验证期 2024-2025）:')
    print('  comp_eq: ICIR_h1 0.550 / h5 0.758 | 夏普 2.52 | 年化 26.5%')
    print('  comp_w : ICIR_h1 0.551 / h5 0.748 | 夏普 2.22 | 年化 23.8%')
    print()
    print('注：以上为多空组合（前20%多/后20%空，t+1成交，未扣费）')


if __name__ == '__main__':
    main()
