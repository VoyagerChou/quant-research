# -*- coding: utf-8 -*-
"""
生成预计算信号文件 comp_eq_signal.npz（本地运行）
=================================================
信号：14因子等权 comp_eq（截面z-score，行业+市值中性化，固定符号+1）
时间：2014-06 ~ 数据终点（本地已拼接 2026-08-18）
预处理（回测不可交易性的过滤在此完成，回测只查表）：
  - 动态成分掩码（月度快照前向填充，消除幸存者偏差）
  - 非ST + 价格有效
  - 流动性：近60日日均成交额 ≥ 2000万
  - 高价股：收盘价 ≤ 单只资金/200（5万÷10÷200 = 25元，固定按规格书）

输出：D:\Quant\Quant_research\code\comp_eq_signal.npz
  dates:  全部交易日 str
  stocks: 信号有效股票代码
  signal: (T, N) float32，NaN = 不可交易，值 = comp_eq z-score
"""
import io
import sys
import numpy as np
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r'D:\Quant\Quant_research\code')

from jqdata import *
import jqdata
import alpha101_test_core as core
from alpha101_factors import FACTOR_FUNCS

INDEX = '000852.XSHG'
FACTORS = ['alpha003', 'alpha006', 'alpha019', 'alpha015', 'alpha021',
           'alpha026', 'alpha040', 'alpha012', 'alpha045', 'alpha018',
           'alpha024', 'alpha069', 'alpha088', 'alpha044']
SIGNS = {f: 1 for f in FACTORS}
MIN_ADV = 2e7           # 流动性：日均成交额≥2000万
MAX_PRICE = 50000 / 10 / 200.0   # 高价过滤：单只资金÷200 = 25元


def main():
    print('== 生成预计算信号文件 ==')
    jqdata._load()
    end = jqdata._dates[-1].strftime('%Y-%m-%d')
    start = '2014-06-01'
    print('数据范围: %s ~ %s (%d 交易日)' % (start, end, len(jqdata._dates)))

    # 1. 成分并集 + 月度快照
    stocks, snap_list = core.fetch_index_members(INDEX, start, end)
    print('历史成员并集: %d 只 | 快照 %d 个' % (len(stocks), len(snap_list)))

    # 2. 数据
    price = core.fetch_price(stocks, start, end)
    price['returns'] = price['close'].pct_change()
    mcap = core.fetch_market_cap(stocks, start, end)
    log_mcap = np.log(mcap.replace(0, np.nan)).reindex(
        price['close'].index, columns=price['close'].columns)
    st = core.fetch_st(stocks, start, end)
    raw_ind = get_industry(stocks)
    ind_map = {}
    for k, v in raw_ind.items():
        try:
            ind_map[k] = v.get('sw_l1', {}).get('industry_name', None)
        except Exception:
            ind_map[k] = None
    print('行业映射: %d/%d' % (sum(1 for v in ind_map.values() if v), len(stocks)))

    # 3. 动态成员掩码
    member_mask = core.build_member_mask(snap_list, price['close'].index, stocks, end)

    # 4. 因子面板（与研究环境同口径）
    money = price['money']
    cap_df = mcap.reindex(price['close'].index, columns=price['close'].columns)
    base = {k: price[k] for k in ['open', 'close', 'high', 'low', 'volume', 'avg', 'returns']}
    ind_l1 = pd.Series(ind_map)
    data = core._LazyData(base, money, cap_df, ind_l1, ind_l1)
    valid = member_mask & (~st.fillna(False)) & price['close'].notna()
    panels = {}
    for name in FACTORS:
        f = FACTOR_FUNCS[name](data)
        if not isinstance(f, pd.DataFrame):
            f = pd.DataFrame(f, index=price['close'].index, columns=price['close'].columns)
        f = f.where(valid)
        panels[name] = core.neutralize(f, ind_map, log_mcap)
        print('  %s 中性化有效 %d' % (name, int(panels[name].notna().sum().sum())))

    # 5. comp_eq（固定符号等权）
    zs = []
    cols0 = None
    for n in FACTORS:
        f = panels[n].mul(SIGNS[n], axis=0)
        sd = f.std(axis=1).replace(0, np.nan)
        z = f.sub(f.mean(axis=1), axis=0).div(sd, axis=0).clip(-5, 5)
        zs.append(z)
        if cols0 is None:
            cols0 = z.columns
    z_arr = np.nan_to_num(np.stack([z.reindex(columns=cols0).values for z in zs])
                           .astype(np.float32))
    comp_eq = pd.DataFrame(np.nanmean(z_arr, axis=0),
                           index=price['close'].index, columns=cols0)
    print('comp_eq 面板: %s' % str(comp_eq.shape))

    # 6. 可交易性过滤（写入信号文件）
    sig = comp_eq.where(valid.reindex(columns=cols0) & comp_eq.notna())
    # 流动性：近60日日均成交额
    adv = price['money'].rolling(60, min_periods=20).mean().reindex(columns=cols0)
    sig = sig.where(adv >= MIN_ADV)
    # 高价过滤
    close = price['close'].reindex(columns=cols0)
    sig = sig.where(close <= MAX_PRICE)
    print('过滤后非NaN: %d (%.1f%%)' % (int(sig.notna().sum().sum()),
                                       sig.notna().sum().sum() / sig.size * 100))

    # 7. 存盘（关键：dates/stocks 用 S 字节数组，allow_pickle=False，
    #    避免 npz 内嵌 pickle 引用 numpy._core.multiarray 导致旧版 numpy 加载失败）
    out = r'D:\Quant\Quant_research\code\comp_eq_signal.npz'
    np.savez_compressed(out, allow_pickle=False,
                        dates=np.array(sig.index.strftime('%Y-%m-%d'), dtype='S10'),
                        stocks=np.array(sig.columns, dtype='S16'),
                        signal=sig.values.astype(np.float32))
    import os
    print('已保存: %s (%.1f MB)' % (out, os.path.getsize(out) / 1048576))
    # 抽样验证
    d = '2025-06-30'
    row = sig.loc[d].dropna()
    print('抽样 %s: 有效 %d 只, top5: %s' % (d, len(row), list(row.sort_values(ascending=False).index[:5])))


if __name__ == '__main__':
    main()
