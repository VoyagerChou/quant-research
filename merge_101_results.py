# -*- coding: utf-8 -*-
"""
合并 4 批因子检验结果为 101 因子总表（本地运行）
================================================
把 results_10yrs_000300 下 4 个批次目录的
  ic_series.csv / ic_stats.csv / layered_stats.csv / ls_net_series.csv
分别合并成 4 张总表，输出到本目录：
  all_101_ic_series.csv / all_101_ic_stats.csv
  all_101_layered_stats.csv / all_101_ls_net_series.csv

同时基于合并数据计算：
  1) 六因子 IC 序列相关性（预测力时序同动性，训练/验证分开）
  2) 六因子多空净值相关性（策略收益同动性，训练/验证分开）
输出 corr_ic_series_6.csv / corr_ls_net_6.csv
（注意：这是"IC 序列相关"，不是"因子值相关"；因子值相关需研究环境补跑
  run_factor_corr.py —— 那才是判断"两个因子是否同一信号"的正确度量）
"""
import os
import numpy as np
import pandas as pd

BASE = r'D:\Quant\Quant_research\results_10yrs_000300'
BATCH_DIRS = ['results_10yrs_000300_fac001010', 'results_alpha011_020',
              'results_alpha021_030', 'results_alpha031_101']

TIER1 = ['alpha003', 'alpha006', 'alpha019', 'alpha015', 'alpha021', 'alpha026']
TRAIN = ('2015-01-01', '2023-12-31')
VALID = ('2024-01-01', '2025-12-31')


def merge_all():
    for fname in ['ic_series.csv', 'ic_stats.csv', 'layered_stats.csv', 'ls_net_series.csv']:
        parts = []
        for d in BATCH_DIRS:
            p = os.path.join(BASE, d, fname)
            if os.path.isfile(p):
                parts.append(pd.read_csv(p))
        if not parts:
            print('未找到 %s，跳过' % fname)
            continue
        out = pd.concat(parts, ignore_index=True)
        out_name = 'all_101_' + fname
        out.to_csv(os.path.join(BASE, out_name), index=False, encoding='utf-8-sig')
        print('合并 %-20s -> %-30s 行数 %d, 因子数 %d'
              % (fname, out_name, len(out), out['factor'].nunique() if 'factor' in out else 0))
    return True


def corr_matrix_from_panel(panel, period):
    """panel: 宽表（index=日期, columns=因子），返回相关系数矩阵"""
    seg = panel.loc[period[0]:period[1]].dropna(how='all')
    return seg.corr()


def compute_series_corrs():
    ic = pd.read_csv(os.path.join(BASE, 'all_101_ic_series.csv'))
    lsn = pd.read_csv(os.path.join(BASE, 'all_101_ls_net_series.csv'))
    ic['date'] = pd.to_datetime(ic['date'])
    lsn['date'] = pd.to_datetime(lsn['date'])

    # IC 序列：中性化版本
    ic_neu = ic[ic.version == 'neu'].pivot(index='date', columns='factor', values='ic')
    # 净值序列：中性化版本
    ls_neu = lsn[lsn.version == 'neu'].pivot(index='date', columns='factor', values='ls_net')

    for label, panel, out_name in [('IC序列', ic_neu, 'corr_ic_series_6.csv'),
                                   ('多空净值', ls_neu, 'corr_ls_net_6.csv')]:
        panel6 = panel.reindex(columns=[c for c in TIER1 if c in panel.columns])
        ct = corr_matrix_from_panel(panel6, TRAIN)
        cv = corr_matrix_from_panel(panel6, VALID)
        print()
        print('===== 六因子 %s 相关性矩阵（训练期 2015-2023）=====' % label)
        print(ct.round(3).to_string())
        print()
        print('===== 六因子 %s 相关性矩阵（验证期 2024-2025）=====' % label)
        print(cv.round(3).to_string())
        ct.to_csv(os.path.join(BASE, out_name.replace('.csv', '_train.csv')),
                  encoding='utf-8-sig')
        cv.to_csv(os.path.join(BASE, out_name.replace('.csv', '_valid.csv')),
                  encoding='utf-8-sig')


if __name__ == '__main__':
    merge_all()
    compute_series_corrs()
    print('\n完成。')
