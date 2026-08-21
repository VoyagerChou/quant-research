# -*- coding: utf-8 -*-
"""
2026 增量数据导出脚本（聚宽研究环境）
====================================
用途：为最终验证导出 2026 年数据（含 3 个月重叠期用于本地价格校准），
不重新导出历史，避免前复权基准变化污染 2015-2025 已有数据。

区间：2025-10-01 ~ 2026-08-18（重叠期 2025-10~2025-12 供本地拼接校准）
股票池：该区间内 000852 的动态成分并集（fetch_index_members）

输出（export_data_000852_2026/）：
  - dates.csv / stocks.csv
  - price_batch_*.npz   OHLCV + money + avg（float32，分批 300 只）
  - mcap_batch_*.npz    总市值（float64）
  - st_batch_*.npz      ST 标记（bool）
  - member_mask.npz     指数动态成分掩码
  - industry_l1.csv / industry_l2.csv

本地拼接时：用重叠期旧包/新包价格比值校准缩放，历史部分完全不动。
"""
import os
import gc
import numpy as np
import pandas as pd
from jqdata import *
import alpha101_test_core as core

# ---- 增量区间配置（只导出 2026 + 重叠期）----
EXP_START = '2025-10-01'      # 含 3 个月重叠，供本地校准前复权基准
EXP_END = '2026-08-18'        # 导出到最近（可手动改到今天）
INDEX = '000852.XSHG'         # 中证1000

OUT = 'export_data_000852_2026'
EXPORT_BATCH = 300


def main():
    os.makedirs(OUT, exist_ok=True)
    print('== 增量导出 %s | %s ~ %s ==' % (INDEX, EXP_START, EXP_END))

    stocks, snap_list = core.fetch_index_members(INDEX, EXP_START, EXP_END)
    snap_list = [(d, ms) for d, ms in snap_list if ms]
    stocks = sorted({s for _, ms in snap_list for s in ms})
    print('快照 %d 个 | 成员并集 %d 只' % (len(snap_list), len(stocks)))
    if not stocks:
        print('成员为空，请检查区间/指数代码')
        return

    batches = [stocks[i:i + EXPORT_BATCH] for i in range(0, len(stocks), EXPORT_BATCH)]
    dates = None
    for bi, batch in enumerate(batches):
        print('-- 批次 %d/%d: %d 只 --' % (bi + 1, len(batches), len(batch)))
        price = core.fetch_price(batch, EXP_START, EXP_END)
        if dates is None:
            dates = price['close'].index
        np.savez_compressed(
            os.path.join(OUT, 'price_batch_%d.npz' % bi),
            open=price['open'].values.astype(np.float32),
            close=price['close'].values.astype(np.float32),
            high=price['high'].values.astype(np.float32),
            low=price['low'].values.astype(np.float32),
            volume=price['volume'].values.astype(np.float32),
            money=price['money'].values.astype(np.float32),
            avg=price['avg'].values.astype(np.float32))
        cols = price['close'].columns
        del price
        gc.collect()

        mcap = core.fetch_market_cap(batch, EXP_START, EXP_END)
        mcap_r = mcap.reindex(dates, columns=cols)
        np.savez_compressed(os.path.join(OUT, 'mcap_batch_%d.npz' % bi),
                            mcap=mcap_r.values.astype(np.float64))
        del mcap, mcap_r
        gc.collect()

        st = core.fetch_st(batch, EXP_START, EXP_END)
        st_r = st.reindex(dates, columns=cols).fillna(False)
        np.savez_compressed(os.path.join(OUT, 'st_batch_%d.npz' % bi),
                            st=st_r.values.astype(np.bool_))
        del st, st_r
        gc.collect()
        print('    已落盘')

    member_mask = core.build_member_mask(snap_list, dates, stocks, EXP_END)
    np.savez_compressed(os.path.join(OUT, 'member_mask.npz'),
                        mm=member_mask.values.astype(np.bool_))
    del member_mask
    gc.collect()

    raw_ind = get_industry(stocks)
    industry_l1 = {k: core._extract_industry_name(v) for k, v in raw_ind.items()}
    industry_l2 = {k: core._extract_industry_level(v, 'sw_l2') for k, v in raw_ind.items()}
    pd.Series(dates.strftime('%Y-%m-%d')).to_csv(os.path.join(OUT, 'dates.csv'),
                                                 index=False, header=False)
    pd.Series(stocks).to_csv(os.path.join(OUT, 'stocks.csv'), index=False, header=False)
    pd.Series(industry_l1).to_csv(os.path.join(OUT, 'industry_l1.csv'),
                                  header=False, encoding='utf-8-sig')
    pd.Series(industry_l2).to_csv(os.path.join(OUT, 'industry_l2.csv'),
                                  header=False, encoding='utf-8-sig')

    print('== 导出完成：%s/' % OUT)
    total = 0
    for fn in sorted(os.listdir(OUT)):
        total += os.path.getsize(os.path.join(OUT, fn))
    print('合计 %.1f MB' % (total / 1048576.0))
    print('请把 %s/ 下载到本地 D:/Quant/Quant_research/data/2026_000852/' % OUT)


if __name__ == '__main__':
    main()
