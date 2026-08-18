# -*- coding: utf-8 -*-
"""
数据导出脚本（聚宽研究环境，分批落盘版）
========================================
把指定指数（core.INDEX_CODE 配置）的原始数据导出为文件。

【内存安全设计】行情/市值/ST 按 EXPORT_BATCH 只股票为一批，
每批算完立刻存盘并释放内存——聚宽侧的内存占用恒定在"一批"的水平
（默认 300 只，与沪深300导出时的实测安全量级一致），
指数成员并集再大也不会爆内存。

导出内容（export_data_<指数代码>/ 目录）：
  - dates.csv / stocks.csv           交易日与股票清单
  - price_batch_*.npz                OHLCV + money + avg 面板（float32，分批）
  - mcap_batch_*.npz                 总市值宽表（float64，分批，已对齐）
  - st_batch_*.npz                   ST 标记（bool，分批，已对齐）
  - member_mask.npz                  指数动态成分掩码（bool，单文件）
  - industry_l1.csv / industry_l2.csv

运行：上传 4 个文件；改好 core.INDEX_CODE；运行本文件。
      跑完把 export_data_<指数代码>/ 下载到本地
      D:/Quant/Quant_research/data/ 并改名为纯数字代码（如 000905）。

【中证2000 特殊说明】2023-08 才发布：导出前把 core.FETCH_START 改为
  '2022-12-01'，大幅减小数据体积。
"""
import os
import gc
import numpy as np
import pandas as pd
from jqdata import *
import alpha101_test_core as core

OUT = 'export_data_%s' % core.INDEX_CODE.split('.')[0]
EXPORT_BATCH = 300          # 每批股票数（内存安全量级）


def main():
    os.makedirs(OUT, exist_ok=True)
    print('== 导出指数: %s | 区间 %s ~ %s | 批次大小 %d =='
          % (core.INDEX_CODE, core.FETCH_START, core.VALID_END, EXPORT_BATCH))

    # ---- 1. 成员快照（轻量：只有代码列表）----
    stocks, snap_list = core.fetch_index_members(core.INDEX_CODE, core.FETCH_START, core.VALID_END)
    snap_list = [(d, ms) for d, ms in snap_list if ms]
    stocks = sorted({s for _, ms in snap_list for s in ms})
    print('有效快照 %d 个 | 成员并集 %d 只' % (len(snap_list), len(stocks)))
    if not stocks:
        print('成员为空，请检查 INDEX_CODE / 快照日期范围')
        return

    # ---- 2. 分批导出行情/市值/ST，逐批落盘 ----
    batches = [stocks[i:i + EXPORT_BATCH] for i in range(0, len(stocks), EXPORT_BATCH)]
    dates = None
    for bi, batch in enumerate(batches):
        print('-- 批次 %d/%d: %d 只 --' % (bi + 1, len(batches), len(batch)))
        price = core.fetch_price(batch, core.FETCH_START, core.VALID_END)
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

        mcap = core.fetch_market_cap(batch, core.FETCH_START, core.VALID_END)
        mcap_r = mcap.reindex(dates, columns=cols)
        np.savez_compressed(os.path.join(OUT, 'mcap_batch_%d.npz' % bi),
                            mcap=mcap_r.values.astype(np.float64))
        del mcap, mcap_r
        gc.collect()

        st = core.fetch_st(batch, core.FETCH_START, core.VALID_END)
        st_r = st.reindex(dates, columns=cols).fillna(False)
        np.savez_compressed(os.path.join(OUT, 'st_batch_%d.npz' % bi),
                            st=st_r.values.astype(np.bool_))
        del st, st_r
        gc.collect()
        print('    已落盘')

    # ---- 3. 成员掩码（基于交易日历+快照，不依赖行情面板，内存极小）----
    member_mask = core.build_member_mask(snap_list, dates, stocks, core.VALID_END)
    np.savez_compressed(os.path.join(OUT, 'member_mask.npz'),
                        mm=member_mask.values.astype(np.bool_))
    del member_mask
    gc.collect()

    # ---- 4. 行业映射 + 清单 ----
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

    print('== 导出完成 ==')
    total = 0
    for fn in sorted(os.listdir(OUT)):
        total += os.path.getsize(os.path.join(OUT, fn))
    print('合计 %.1f MB' % (total / 1048576.0))
    print('请把 %s/ 下载到本地 D:/Quant/Quant_research/data/，改名为 %s'
          % (OUT, core.INDEX_CODE.split('.')[0]))


if __name__ == '__main__':
    main()
