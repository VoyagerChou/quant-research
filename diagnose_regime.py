# -*- coding: utf-8 -*-
"""
六因子复合的 regime 诊断（本地运行）
====================================
回答："这六个因子的 alpha 到底跟什么市场状态有关？"

方法：把 7 个实时可得的市场状态代理（趋势强度/动量/波动率/换手/涨跌比/
涨停数/截面离散度）按时间分位切成 5 档，统计各档内信号的日度 IC 分布
（IC均值/ICIR/IC>0占比）。若某代理的高档与低档 IC 差异显著，说明该状态
是因子表现的驱动维度，后续择时才"有据可依"；若全部区分不开，则说明
alpha 与常见市场状态无关，择时无门——结论本身就是收获。

信号对象：
  - comp_w（六因子 ICIR 加权合成，最优线性）
  - 六个单因子（alpha003/006/015/019/021/026）

注意：分箱使用全样本分位数（描述性诊断，非交易回测，允许使用全样本信息）。

输出（results_10yrs_000300/regime_diagnosis/）：
  - regime_diagnosis.csv   长表：代理 × 档位 × 信号 × IC统计
  - regime_summary.csv     宽表：代理 × 信号 的高档-低档 IC 差（效应方向与幅度）
  - proxy_ic_corr.csv      |IC|（信号强度）与各代理的秩相关
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'D:\Quant\Quant_research\code')
import jqdata
import alpha101_test_core as core

BASE = r'D:\Quant\Quant_research\results_10yrs_000300'
OUT = os.path.join(BASE, 'regime_diagnosis')
os.makedirs(OUT, exist_ok=True)


def main():
    # ---------- 1. 数据 ----------
    jqdata._load()   # 触发懒加载（shim 数据包首次访问才读盘）
    dates = jqdata._dates
    close = jqdata._price['close']
    money = jqdata._price['money']
    member = jqdata._member
    mcap = jqdata._mcap

    # ---------- 2. 指数代理（成分股按市值加权，全部只用当期及过去信息） ----------
    ret = close.pct_change()
    w = mcap.where(member)
    w = w.div(w.sum(axis=1), axis=0)
    idx_ret = (ret * w).sum(axis=1)
    idx_close = (1 + idx_ret.fillna(0)).cumprod()

    proxies = {}
    proxies['趋势强度MA20'] = idx_close / idx_close.rolling(20).mean() - 1
    proxies['趋势强度MA60'] = idx_close / idx_close.rolling(60).mean() - 1
    proxies['指数5日动量'] = idx_close.pct_change(5)
    proxies['已实现波动率20'] = idx_ret.rolling(20).std()
    tot_money = money.where(member).sum(axis=1)
    tot_mcap = mcap.where(member).sum(axis=1)
    proxies['市场换手率'] = tot_money / tot_mcap
    proxies['上涨家数占比'] = (ret.where(member) > 0).sum(axis=1) / member.sum(axis=1)
    proxies['涨停家数'] = (ret.where(member) >= 0.095).sum(axis=1)
    proxies['截面离散度'] = ret.where(member).std(axis=1)

    # ---------- 3. 信号面板 ----------
    signals = {}
    signals['comp_w'] = pd.read_csv(os.path.join(BASE, 'results_ml_composite', 'pred_comp_w.csv'),
                                    index_col=0)
    for n in ['alpha003', 'alpha006', 'alpha015', 'alpha019', 'alpha021', 'alpha026']:
        signals[n] = pd.read_csv(os.path.join(BASE, 'results_corr_6factors', 'factor_%s.csv' % n),
                                 index_col=0)
    for k in signals:
        signals[k].index = pd.to_datetime(signals[k].index)

    # ---------- 4. 日度 IC（对 5 日前向收益，t+1 成交口径） ----------
    fwd5 = core.forward_returns(close, 5)
    fwd5_r = fwd5.rank(axis=1, pct=True).astype(np.float32)
    ic_series = {}
    for name, sig in signals.items():
        f5 = fwd5_r.reindex(columns=sig.columns)
        sr = sig.rank(axis=1, pct=True).astype(np.float32)
        ic_series[name] = core.rank_ic_from_ranked(sr, f5)

    # ---------- 5. 分档诊断 ----------
    rows = []
    for pname, p in proxies.items():
        p = p.reindex(ic_series['comp_w'].index)
        valid_p = p.notna()
        bins = pd.Series(np.nan, index=p.index)
        pr = p[valid_p].rank(method='first')
        bins.loc[valid_p] = pd.qcut(pr, 5, labels=False).values
        for b in range(5):
            mask = bins == b
            for sname, ic in ic_series.items():
                seg = ic[mask & ic.notna()].dropna()
                if len(seg) < 10:
                    continue
                std = seg.std()
                rows.append({'proxy': pname, 'bin': int(b) + 1, 'signal': sname,
                             'n': len(seg),
                             'IC均值': seg.mean(),
                             'ICIR': seg.mean() / std if std > 0 else np.nan,
                             'IC>0占比': (seg > 0).mean()})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'regime_diagnosis.csv'), index=False, encoding='utf-8-sig')

    # ---------- 6. 汇总：高档(5) - 低档(1) 的 IC 均值差 ----------
    piv = df.pivot_table(index=['proxy', 'signal'], columns='bin', values='IC均值')
    piv['高档-低档'] = piv[5] - piv[1]
    piv.to_csv(os.path.join(OUT, 'regime_summary.csv'), encoding='utf-8-sig')

    # ---------- 7. 信号强度(|IC|)与代理的秩相关 ----------
    corr_rows = []
    for pname, p in proxies.items():
        p = p.reindex(ic_series['comp_w'].index)
        for sname, ic in ic_series.items():
            abs_ic = ic.abs()
            m = p.notna() & abs_ic.notna()
            if m.sum() > 100:
                r = abs_ic[m].rank().corr(p[m].rank())
                corr_rows.append({'proxy': pname, 'signal': sname, '|IC|秩相关': r})
    cdf = pd.DataFrame(corr_rows)
    cdf.to_csv(os.path.join(OUT, 'proxy_ic_corr.csv'), index=False, encoding='utf-8-sig')

    # ---------- 8. 打印汇总 ----------
    pd.set_option('display.width', 200)
    print('===== 各代理高档-低档 IC 均值差（正=状态越极端因子越强）=====')
    print(piv[['高档-低档']].round(4).unstack('signal').to_string())
    print()
    print('===== comp_w 在各代理分档下的 ICIR =====')
    comp_bins = df[df.signal == 'comp_w'].pivot_table(index='proxy', columns='bin', values='ICIR')
    print(comp_bins.round(3).to_string())
    print()
    print('===== 信号强度与代理的秩相关（|IC| 越大信号越强）=====')
    print(cdf.pivot_table(index='proxy', columns='signal', values='|IC|秩相关').round(3).to_string())
    print()
    print('输出目录:', OUT)


if __name__ == '__main__':
    main()
