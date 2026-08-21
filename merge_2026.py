# -*- coding: utf-8 -*-
"""
本地拼接：把 2026 增量数据包合并进现有 data/000852
====================================================
【核心问题】聚宽 get_price 前复权基准随导出时点变化：2025 导出的
000852 历史价格基准 = 2025-12 时点；2026 增量导出的价格基准 = 2026
时点。两者直接拼接会在 2026-01-01 处产生价格断档（除权因子不同）。

【校准方案】利用重叠期（2025-10-01 ~ 2025-12-31）：
  每只股票的校准系数 c = median(旧价/新价)（重叠期内应近似常数，
  除权日会出现跳变，用中位数抗噪）。
  新包价格 × c → 与旧包同基准，拼接连续。

【拼接规则】
  - 日期轴：旧 dates + 新 dates（去重，新部分只取 2026 及以后）
  - 股票轴：旧 stocks ∪ 新 stocks；旧股票历史行保留原值（不动！），
    新股票（2026 才出现的成分）历史行填 NaN
  - 2026 段：旧股票用校准后的新包值；新股票用新包值
  - member_mask：2026 段用新包掩码；历史段旧掩码（新股票列全 False）
  - ST：2026 段新包；历史段旧包（新股票 NaN→False）
  - 行业：旧 + 新（新股票取新包行业）

输出：直接写回 data/000852（先备份原目录为 data/000852_backup_20250818）
"""
import os
import shutil
import glob
import numpy as np
import pandas as pd

DATA = r'D:\Quant\Quant_research\data'
OLD = os.path.join(DATA, '000852')
NEW = os.path.join(DATA, '2026_000852')
BACKUP = os.path.join(DATA, '000852_backup_20250818')

FIELDS = ('open', 'close', 'high', 'low', 'volume', 'money', 'avg')


def load_pkg(d, fields=FIELDS):
    """加载数据包：返回 dates, stocks, {field: DataFrame}, member, st, ind"""
    dates = pd.DatetimeIndex(pd.read_csv(os.path.join(d, 'dates.csv'), header=None)[0])
    stocks = pd.read_csv(os.path.join(d, 'stocks.csv'), header=None)[0].tolist()
    p = {f: [] for f in fields}
    for pf in sorted(glob.glob(os.path.join(d, 'price_batch_*.npz'))):
        z = np.load(pf)
        for f in fields:
            p[f].append(z[f])
    price = {f: np.concatenate(p[f], axis=1).astype(np.float32) for f in fields}
    mcap = np.concatenate([np.load(mf)['mcap'] for mf in
                           sorted(glob.glob(os.path.join(d, 'mcap_batch_*.npz')))], axis=1)
    st = np.concatenate([np.load(sf)['st'] for sf in
                         sorted(glob.glob(os.path.join(d, 'st_batch_*.npz')))], axis=1)
    member = np.load(os.path.join(d, 'member_mask.npz'))['mm']
    ind1 = pd.read_csv(os.path.join(d, 'industry_l1.csv'), header=None,
                       index_col=0, encoding='utf-8-sig')[1].tolist()
    ind2 = pd.read_csv(os.path.join(d, 'industry_l2.csv'), header=None,
                       index_col=0, encoding='utf-8-sig')[1].tolist()
    return dates, stocks, price, mcap, st, member, ind1, ind2


def main():
    if not os.path.isdir(NEW):
        raise IOError('未找到增量数据包 %s，请先从聚宽下载' % NEW)
    old_dates, old_stocks, old_p, old_mcap, old_st, old_member, old_i1, old_i2 = load_pkg(OLD)
    new_dates, new_stocks, new_p, new_mcap, new_st, new_member, new_i1, new_i2 = load_pkg(NEW)
    print('旧包: %d 日 x %d 股 (%s ~ %s)' % (len(old_dates), len(old_stocks),
                                            old_dates[0].date(), old_dates[-1].date()))
    print('新包: %d 日 x %d 股 (%s ~ %s)' % (len(new_dates), len(new_stocks),
                                            new_dates[0].date(), new_dates[-1].date()))

    # ---- 1. 重叠期校准系数 ----
    ov = (new_dates >= '2025-10-01') & (new_dates <= '2025-12-31')
    if ov.sum() == 0:
        raise IOError('新包无重叠期数据（需包含 2025-10 ~ 2025-12）')
    common = sorted(set(old_stocks) & set(new_stocks))
    new_idx = {s: i for i, s in enumerate(new_stocks)}
    old_idx = {s: i for i, s in enumerate(old_stocks)}
    scale = {}
    for s in common:
        o = old_p['close'][old_dates >= '2025-10-01', old_idx[s]]
        # 旧包从 2025-10-01 起（若数据包从更早开始则取重叠段）
        oi = (old_dates >= '2025-10-01') & (old_dates <= '2025-12-31')
        a = old_p['close'][oi, old_idx[s]]
        b = new_p['close'][ov, new_idx[s]]
        m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        if m.sum() >= 20:
            ratio = a[m] / b[m]
            # 去掉极端值（除权日跳变），取中位数
            lo, hi = np.percentile(ratio, [5, 95])
            scale[s] = np.median(ratio[(ratio >= lo) & (ratio <= hi)])
        else:
            scale[s] = np.nan
    n_ok = sum(1 for v in scale.values() if np.isfinite(v))
    print('重叠期校准: %d/%d 只股票有系数' % (n_ok, len(common)))

    # ---- 2. 拼接日期轴 ----
    all_dates = old_dates.append(new_dates[new_dates > old_dates[-1]])
    all_dates = pd.DatetimeIndex(sorted(set(all_dates)))
    T_new = len(all_dates)
    old_T = len(old_dates)
    new_only_days = all_dates > old_dates[-1]
    print('拼接后: %d 日（历史 %d + 新增 %d）' % (T_new, old_T, new_only_days.sum()))

    # ---- 3. 拼接股票轴 ----
    all_stocks = old_stocks + [s for s in new_stocks if s not in old_idx]
    N = len(all_stocks)
    all_idx = {s: i for i, s in enumerate(all_stocks)}
    new_cols_only = [s for s in all_stocks if s not in old_idx]
    print('拼接后: %d 股（新增 %d 只）' % (N, len(new_cols_only)))

    # ---- 4. 拼接面板 ----
    def merge_panel(old_arr, new_arr, scale_cols, new_only):
        out = np.full((T_new, N), np.nan, dtype=np.float32)
        out[:old_T, :len(old_stocks)] = old_arr
        # 新日期行：旧股票
        for s in common:
            c = scale.get(s, np.nan)
            if not np.isfinite(c):
                continue
            j_old = all_idx[s]
            j_new = new_idx[s]
            out[old_T:, j_old] = new_arr[ov.sum():, j_new] * c
        # 新日期行：新股票（直接用新包值，历史段 NaN）
        for s in new_only:
            j_new = new_idx[s]
            out[old_T:, all_idx[s]] = new_arr[ov.sum():, j_new]
        return out

    # 新包中重叠期之后的日期行
    new_after = new_dates > old_dates[-1]
    # 确保 new_arr 行顺序 = new_dates 顺序（已是）

    print('拼接价格面板...')
    price = {}
    for f in FIELDS:
        price[f] = merge_panel(old_p[f], new_p[f], scale, new_cols_only)
    mcap = merge_panel(old_mcap, new_mcap, scale, new_cols_only).astype(np.float64)
    print('拼接 ST...')
    st = merge_panel(old_st, new_st, scale, new_cols_only).astype(np.bool_)

    # ---- 5. member_mask：历史段旧掩码（列对齐），2026 段新掩码 ----
    print('拼接 member_mask...')
    member = np.zeros((T_new, N), dtype=bool)
    member[:old_T, :len(old_stocks)] = old_member
    if len(new_cols_only):
        member[old_T:, len(old_stocks):] = new_member[ov.sum():, [new_idx[s] for s in new_cols_only]]
    for s in common:
        member[old_T:, all_idx[s]] = new_member[ov.sum():, new_idx[s]]

    # ---- 6. 行业：旧 + 新股票 ----
    ind1 = old_i1 + [new_i1[new_stocks.index(s)] for s in new_cols_only]
    ind2 = old_i2 + [new_i2[new_stocks.index(s)] for s in new_cols_only]
    # 行业文件是 Series（按 stocks 顺序），需对齐 all_stocks
    ind1_map = dict(zip(old_stocks, old_i1))
    ind1_map.update({s: new_i1[new_stocks.index(s)] for s in new_cols_only})
    ind2_map = dict(zip(old_stocks, old_i2))
    ind2_map.update({s: new_i2[new_stocks.index(s)] for s in new_cols_only})
    ind1_all = [ind1_map[s] for s in all_stocks]
    ind2_all = [ind2_map[s] for s in all_stocks]

    # ---- 7. 备份并写回 ----
    if not os.path.isdir(BACKUP):
        print('备份旧包 -> %s' % BACKUP)
        shutil.copytree(OLD, BACKUP)
    os.makedirs(OLD, exist_ok=True)
    pd.Series(all_dates.strftime('%Y-%m-%d')).to_csv(os.path.join(OLD, 'dates.csv'),
                                                     index=False, header=False)
    pd.Series(all_stocks).to_csv(os.path.join(OLD, 'stocks.csv'), index=False, header=False)
    # 分批写回（保持与 shim 兼容：price_batch_*.npz）
    B = 300
    for bi in range(0, N, B):
        sl = slice(bi, min(bi + B, N))
        np.savez_compressed(
            os.path.join(OLD, 'price_batch_%d.npz' % (bi // B)),
            open=price['open'][:, sl], close=price['close'][:, sl],
            high=price['high'][:, sl], low=price['low'][:, sl],
            volume=price['volume'][:, sl], money=price['money'][:, sl],
            avg=price['avg'][:, sl])
        np.savez_compressed(os.path.join(OLD, 'mcap_batch_%d.npz' % (bi // B)),
                            mcap=mcap[:, sl])
        np.savez_compressed(os.path.join(OLD, 'st_batch_%d.npz' % (bi // B)),
                            st=st[:, sl])
    np.savez_compressed(os.path.join(OLD, 'member_mask.npz'), mm=member)
    # 行业文件格式：两列"代码,行业"（与导出脚本一致）
    pd.DataFrame({'code': all_stocks, 'ind': ind1_all}).to_csv(
        os.path.join(OLD, 'industry_l1.csv'), index=False, header=False, encoding='utf-8-sig')
    pd.DataFrame({'code': all_stocks, 'ind': ind2_all}).to_csv(
        os.path.join(OLD, 'industry_l2.csv'), index=False, header=False, encoding='utf-8-sig')
    print('== 拼接完成 ==')
    print('新包: %s' % all_dates[-1].date())
    # 拼接点 = 旧包最后一日 → 其后第一个交易日
    j1 = np.where(all_dates == old_dates[-1])[0]
    j2 = old_T
    if len(j1) == 0 or j2 >= T_new:
        print('注意: 无法定位拼接点，请检查数据')
        return
    i1_ = j1[0]
    print('验证: %s 与 %s 拼接连续性（抽样5只）' % (all_dates[i1_].date(), all_dates[j2].date()))
    import random
    rng = random.Random(42)
    for s in rng.sample(common, 5):
        j = all_idx[s]
        v1 = price['close'][i1_, j]
        v2 = price['close'][j2, j]
        chg = v2 / v1 - 1 if v1 > 0 else np.nan
        print('  %s: %.2f -> %.2f (%.2f%%)' % (s, v1, v2, chg * 100))


if __name__ == '__main__':
    main()
