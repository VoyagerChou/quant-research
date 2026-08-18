# -*- coding: utf-8 -*-
"""
六因子"因子值"相关性计算（聚宽研究环境）
========================================
背景：合并表里的 ic_series 只能算"IC 序列相关性"（预测力时序同动性），
判断"两个因子是不是同一个信号"必须算因子值本身的横截面相关性——
即每天对两只股票的因子值排名做 Spearman 相关，再对时间取平均。

本脚本：
  1. 复用 alpha101_test_core 的数据获取与中性化逻辑（不重复造轮子）
  2. 计算 CORR_FACTORS 中每个因子的中性化因子值
  3. 输出：
     - results_corr_6factors/corr_matrix.csv        全样本 6×6 相关矩阵
     - results_corr_6factors/corr_matrix_train.csv 训练期矩阵
     - results_corr_6factors/corr_matrix_valid.csv 验证期矩阵
     - results_corr_6factors/factor_<name>.csv     各因子的截面z-score值
       （保存下来，后续做复合因子合成时直接用，不用再跑一遍）

运行：把本文件与 alpha101_operators.py / alpha101_factors.py /
      alpha101_test_core.py / run_alpha101_first10_test.py 一起上传，
      直接运行本文件即可（预计 5-8 分钟，主要是数据拉取）。
"""
import os
import numpy as np
import pandas as pd
from jqdata import *
import alpha101_test_core as core
from alpha101_factors import FACTOR_FUNCS

CORR_FACTORS = ['alpha003', 'alpha006', 'alpha019', 'alpha015', 'alpha021', 'alpha026']
OUT_DIR = 'results_corr_6factors'
PERIODS = [('train', '2015-01-01', '2023-12-31'),
           ('valid', '2024-01-01', '2025-12-31')]


def build_factor_panel():
    """拉数据 → 算 6 个中性化因子 → 返回 {因子名: DataFrame(日期×股票)}"""
    stocks, snap_list = core.fetch_index_members(core.INDEX_CODE, core.FETCH_START, core.VALID_END)
    core._log('[corr] 成员并集 %d 只' % len(stocks))
    price = core.fetch_price(stocks, core.FETCH_START, core.VALID_END)
    price['returns'] = price['close'].pct_change()
    mcap = core.fetch_market_cap(stocks, core.FETCH_START, core.VALID_END)
    log_mcap = np.log(mcap.replace(0, np.nan)).reindex(
        price['close'].index, columns=price['close'].columns)
    st = core.fetch_st(stocks, core.FETCH_START, core.VALID_END)
    member_mask = core.build_member_mask(snap_list, price['close'].index, stocks, core.VALID_END)
    raw_ind = get_industry(stocks)
    industry_map = {k: core._extract_industry_name(v) for k, v in raw_ind.items()}
    industry_l2 = {k: core._extract_industry_level(v, 'sw_l2') for k, v in raw_ind.items()}
    valid_mask = core.build_valid_mask(price, st, member_mask)

    base = {k: price[k] for k in ['open', 'close', 'high', 'low', 'volume', 'avg', 'returns']}
    money = price.pop('money')
    cap_df = mcap.reindex(price['close'].index, columns=price['close'].columns)
    data = core._LazyData(base, money, cap_df,
                          pd.Series(industry_map), pd.Series(industry_l2))

    panels = {}
    for name in CORR_FACTORS:
        f = FACTOR_FUNCS[name](data)
        if not isinstance(f, pd.DataFrame):
            f = pd.DataFrame(f, index=price['close'].index, columns=price['close'].columns)
        f = f.where(valid_mask)
        f_neu = core.neutralize(f, industry_map, log_mcap)
        panels[name] = f_neu
        core._log('[corr] %s 中性化后有效值 %d' % (name, int(f_neu.notna().sum().sum())))
    return panels


def pairwise_spearman(panels, start, end):
    """逐日横截面 Spearman 相关（=排名后的 Pearson），再取时间平均"""
    names = sorted(panels.keys())
    ranked = {}
    for n in names:
        seg = panels[n].loc[start:end]
        ranked[n] = seg.rank(axis=1, pct=True)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j < i:
                continue
            ra, rb = ranked[a], ranked[b]
            ca = ra.sub(ra.mean(axis=1), axis=0)
            cb = rb.sub(rb.mean(axis=1), axis=0)
            num = (ca * cb).sum(axis=1)
            den = ((ca ** 2).sum(axis=1) * (cb ** 2).sum(axis=1)) ** 0.5
            daily = (num / den).replace([np.inf, -np.inf], np.nan)
            c = daily.mean()
            M.loc[a, b] = c
            M.loc[b, a] = c
    return M.astype(float)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    core._log('===== [corr] RUN START =====')
    panels = build_factor_panel()

    # 保存各因子截面 z-score（供后续合成直接使用）
    for n, f in panels.items():
        sd = f.std(axis=1).replace(0, np.nan)
        z = f.sub(f.mean(axis=1), axis=0).div(sd, axis=0).clip(-5, 5)
        z.to_csv(os.path.join(OUT_DIR, 'factor_%s.csv' % n), encoding='utf-8-sig')
    core._log('[corr] 因子 z-score 已保存')

    M_full = pairwise_spearman(panels, core.FETCH_START, core.VALID_END)
    M_full.to_csv(os.path.join(OUT_DIR, 'corr_matrix.csv'), encoding='utf-8-sig')
    print('===== 六因子 因子值相关矩阵（全样本 2014-06~2025-12）=====')
    print(M_full.round(3).to_string())
    print()

    for label, ps, pe in PERIODS:
        M = pairwise_spearman(panels, ps, pe)
        M.to_csv(os.path.join(OUT_DIR, 'corr_matrix_%s.csv' % label), encoding='utf-8-sig')
        print('===== 六因子 因子值相关矩阵（%s %s~%s）=====' % (label, ps, pe))
        print(M.round(3).to_string())
        print()

    print('输出目录: %s/' % OUT_DIR)
    for fn in sorted(os.listdir(OUT_DIR)):
        print('  %s/%s' % (OUT_DIR, fn))
    core._log('===== [corr] RUN DONE =====')


if __name__ == '__main__':
    main()
