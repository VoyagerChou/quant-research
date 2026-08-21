# -*- coding: utf-8 -*-
"""
六因子复合合成与检验（聚宽研究环境）
====================================
对第一梯队的 6 个因子（alpha003/006/019/015/021/026，中性化口径）做复合：

  1. 等权合成（comp_eq）：6 个因子截面 z-score 的方向对齐平均
  2. 滚动 ICIR 加权合成（comp_w）：权重 = 过去 252 个交易日的 |ICIR|，
     逐日滚动、且只用 t-1 及以前的信息（严格避免未来函数）

关键防前视设计：
  - 每个因子每日先 z-score 化（横截面）
  - 方向对齐符号 = 滚动 12 个月 IC 均值的符号（仅用历史）
  - ICIR 权重同样来自滚动窗口并 shift(1)

输出（results_composite/）：
  - ic_stats_composite.csv    复合因子 + 6 个单因子的五段 IC 统计（同口径便于对比）
  - layered_stats_composite.csv 同上，分层回测
  - weights_history.csv       ICIR 加权合成的逐日权重历史
  - comp_eq_z.csv / comp_w_z.csv  复合因子日度截面 z-score（供后续策略回测用）
  - composite_compare.png     累计IC/多空净值对比图

运行：与其它 4 个文件一起上传后直接运行（预计 6-10 分钟，主要是数据拉取）。
"""
import os
import numpy as np
import pandas as pd
from jqdata import *
import alpha101_test_core as core
from alpha101_factors import FACTOR_FUNCS

CORR_FACTORS = ['alpha003', 'alpha006', 'alpha019', 'alpha015', 'alpha021', 'alpha026']
import os
import jqdata
# 支持环境变量覆盖：COMP_FACTORS=因子子集（逗号分隔），COMP_TAG=输出目录后缀
_env_factors = os.environ.get('COMP_FACTORS')
if _env_factors:
    CORR_FACTORS = [x.strip() for x in _env_factors.split(',')]
    print('[comp] 使用自定义因子子集: %s' % CORR_FACTORS)
COMP_TAG = os.environ.get('COMP_TAG', '')
OUT_DIR = os.path.join(r'D:\Quant\Quant_research', 'results_composite_%s%s' % (jqdata._get_active_index(), COMP_TAG))
IC_WINDOW = 252          # ICIR 滚动窗口（交易日，约 12 个月）
IC_MINP = 60             # 最小样本数（少于它则权重为 NaN）


def build_panels():
    """拉数据 → 算 6 个中性化因子 → 返回 (因子面板, 有效掩码)"""
    stocks, snap_list = core.fetch_index_members(core.INDEX_CODE, core.FETCH_START, core.VALID_END)
    core._log('[comp] 成员并集 %d 只' % len(stocks))
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
        panels[name] = core.neutralize(f, industry_map, log_mcap)
        core._log('[comp] %s 中性化后有效值 %d' % (name, int(panels[name].notna().sum().sum())))
    return panels, price, valid_mask


def build_composites(panels, price):
    """构造两个复合因子（严格时点信息）"""
    names = list(panels.keys())
    idx = panels[names[0]].index
    cols = panels[names[0]].columns

    # ---- 每个因子的截面 z-score ----
    z_scores = {n: f.sub(f.mean(axis=1), axis=0)
                .div(f.std(axis=1).replace(0, np.nan), axis=0).clip(-5, 5)
                for n, f in panels.items()}

    # ---- 逐日 IC（h=1）→ 滚动 IC 均值/ICIR（只用 t-1 及以前）----
    fwd1 = core.forward_returns(price['close'], 1)
    fwd1_r = fwd1.rank(axis=1, pct=True).astype(np.float32)
    ic_series = {}
    for n, f in panels.items():
        f_r = f.rank(axis=1, pct=True).astype(np.float32)
        ic_series[n] = core.rank_ic_from_ranked(f_r, fwd1_r)
    ic_df = pd.DataFrame(ic_series)
    ic_mean = ic_df.rolling(IC_WINDOW, min_periods=IC_MINP).mean().shift(1)
    ic_std = ic_df.rolling(IC_WINDOW, min_periods=IC_MINP).std().shift(1)
    icir = ic_mean / ic_std.replace(0, np.nan)

    signs = np.sign(ic_mean)                     # 方向对齐符号（历史 IC 的符号）
    w_abs = icir.abs()
    wsum = w_abs.sum(axis=1)
    w = w_abs.div(wsum.replace(0, np.nan), axis=0)   # 归一化权重

    core._log('[comp] 滚动权重样本（最近5日）:\n%s' % w.tail(5).round(3).to_string())
    w.to_csv(os.path.join(OUT_DIR, 'weights_history.csv'), encoding='utf-8-sig')

    # ---- 方向对齐后的 z-score ----
    aligned = {n: z_scores[n].mul(signs[n], axis=0) for n in names}
    z_arr = np.nan_to_num(np.stack([aligned[n].values for n in names])
                           .astype(np.float32))  # (6, T, N)

    # 等权：跳过 NaN 取平均
    comp_eq = pd.DataFrame(np.nanmean(z_arr, axis=0), index=idx, columns=cols)

    # ICIR 加权：逐日权重 × 对齐 z
    w2 = w.fillna(0)
    comp_w_vals = np.einsum('tn,nts->ts', w2.values, z_arr)
    comp_w_vals = np.where(np.isnan(wsum.values)[:, None], np.nan, comp_w_vals)
    comp_w = pd.DataFrame(comp_w_vals, index=idx, columns=cols)

    return comp_eq, comp_w, ic_series, w


def build_orth(panels, price, names, idx, cols):
    """正交化+ICIR 复合：逐日截面对称白化（Z_orth = Zc @ Sigma^{-1/2}，
    因子间相关归零），再按滚动 ICIR 加权（严格只用 t-1 及以前信息）。

    白化矩阵用当日截面协方差（同 z-score 一样是时点截面统计，不含未来收益），
    方向对齐与权重用滚动 IC（shift(1)），无前视。"""
    K = len(names)
    z_stack = np.stack([panels[n].values for n in names], axis=2).astype(np.float32)
    T, N, _ = z_stack.shape
    out = np.full_like(z_stack, np.nan, dtype=np.float32)
    for t in range(T):
        zt = z_stack[t]
        ok = np.isfinite(zt).all(axis=1)
        if ok.sum() <= K:
            continue
        Z = zt[ok]
        Zc = Z - Z.mean(axis=0)
        S = (Zc.T @ Zc) / (len(Zc) - 1)
        try:
            U, s, _ = np.linalg.svd(S, full_matrices=False)
            s = np.maximum(s, 1e-8)
            Sinv_half = (U / np.sqrt(s)) @ U.T
        except np.linalg.LinAlgError:
            continue
        out[t, ok] = Zc @ Sinv_half
    orth = {names[k]: pd.DataFrame(out[:, :, k], index=idx, columns=cols)
            for k in range(K)}
    # 正交化后方向不确定：用滚动 IC 符号重新对齐，滚动 ICIR 加权
    fwd1 = core.forward_returns(price['close'], 1)
    fwd1_r = fwd1.rank(axis=1, pct=True).astype(np.float32)
    ic_series = {}
    for n, f in orth.items():
        f_r = f.rank(axis=1, pct=True).astype(np.float32)
        ic_series[n] = core.rank_ic_from_ranked(f_r, fwd1_r)
    ic_df = pd.DataFrame(ic_series)
    ic_mean = ic_df.rolling(IC_WINDOW, min_periods=IC_MINP).mean().shift(1)
    ic_std = ic_df.rolling(IC_WINDOW, min_periods=IC_MINP).std().shift(1)
    icir = ic_mean / ic_std.replace(0, np.nan)
    signs = np.sign(ic_mean)
    w_abs = icir.abs()
    wsum = w_abs.sum(axis=1)
    w = w_abs.div(wsum.replace(0, np.nan), axis=0)
    aligned = {n: orth[n].mul(signs[n], axis=0) for n in names}
    z_arr = np.nan_to_num(np.stack([aligned[n].values for n in names])
                           .astype(np.float32))
    comp_orth_vals = np.einsum('tn,nts->ts', w.fillna(0).values, z_arr)
    comp_orth_vals = np.where(np.isnan(wsum.values)[:, None], np.nan, comp_orth_vals)
    comp_orth = pd.DataFrame(comp_orth_vals, index=idx, columns=cols)
    return comp_orth, ic_series, w


def test_composite(cname, c, fwd_ranked, ret1, ic_periods, ic_rows, ls_rows):
    """对复合因子做与单因子相同的检验"""
    c_r = c.rank(axis=1, pct=True).astype(np.float32)
    for h in core.HORIZONS:
        ic = core.rank_ic_from_ranked(c_r, fwd_ranked[h])
        for period, ps, pe in ic_periods:
            st_ = core.ic_stats(ic, ps, pe)
            ic_rows.append({'factor': cname, 'version': 'composite',
                            'period': period, 'horizon': h, **st_})
    nets, ls_net = core.layered_backtest(c_r, ret1)
    for period, (ps, pe) in [('train', (core.TRAIN_START, core.TRAIN_END)),
                             ('valid', (core.VALID_START, core.VALID_END))]:
        ann, sharpe, mdd, win = core.net_stats(ls_net, ps, pe)
        ls_rows.append({'factor': cname, 'version': 'composite', 'period': period,
                        '年化收益': ann, '夏普': sharpe, '最大回撤': mdd, '月胜率': win})
        for grp, net in nets.items():
            gann, _, _, _ = core.net_stats(net, ps, pe)
            ls_rows[-1]['G%d' % grp] = gann
    return ls_net


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    core._log('===== [comp] RUN START =====')
    panels, price, valid_mask = build_panels()
    comp_eq, comp_w, ic_series, w = build_composites(panels, price)
    comp_orth, orth_ic, orth_w = build_orth(panels, price, list(panels.keys()),
                                            panels[list(panels.keys())[0]].index,
                                            panels[list(panels.keys())[0]].columns)

    # 有效掩码对齐（复合因子列 = 中性化后列），写 CSV 前必须应用掩码，
    # 否则非成分股被 nan_to_num 填成 0，下游排名/分层会被污染
    for cname, c in [('comp_eq', comp_eq), ('comp_w', comp_w), ('comp_orth', comp_orth)]:
        c = c.where(valid_mask.reindex(columns=c.columns))
        c.to_csv(os.path.join(OUT_DIR, '%s_z.csv' % cname), encoding='utf-8-sig')

    # 前向收益排名（全部周期）
    fwd = {h: core.forward_returns(price['close'], h) for h in core.HORIZONS}
    ret1 = core.forward_returns(price['close'], 1)
    fwd_ranked = {h: fwd[h].rank(axis=1, pct=True).astype(np.float32) for h in core.HORIZONS}

    ic_periods = ([('train', core.TRAIN_START, core.TRAIN_END)] + core.TRAIN_SLICES
                  + [('valid', core.VALID_START, core.VALID_END)])
    ic_rows, ls_rows = [], []

    # 复合因子
    ls_nets = {}
    for cname, c in [('comp_eq', comp_eq), ('comp_w', comp_w), ('comp_orth', comp_orth)]:
        c = c.where(valid_mask.reindex(columns=c.columns))
        ls_nets[cname] = test_composite(cname, c, fwd_ranked, ret1,
                                        ic_periods, ic_rows, ls_rows)

    # 单因子基准（同口径，方便对比）
    for name, f in panels.items():
        f_r = f.rank(axis=1, pct=True).astype(np.float32)
        for h in core.HORIZONS:
            ic = core.rank_ic_from_ranked(f_r, fwd_ranked[h])
            for period, ps, pe in ic_periods:
                st_ = core.ic_stats(ic, ps, pe)
                ic_rows.append({'factor': name, 'version': 'neu',
                                'period': period, 'horizon': h, **st_})
        nets, ls_net = core.layered_backtest(f_r, ret1)
        ls_nets[name] = ls_net
        for period, (ps, pe) in [('train', (core.TRAIN_START, core.TRAIN_END)),
                                 ('valid', (core.VALID_START, core.VALID_END))]:
            ann, sharpe, mdd, win = core.net_stats(ls_net, ps, pe)
            ls_rows.append({'factor': name, 'version': 'neu', 'period': period,
                            '年化收益': ann, '夏普': sharpe, '最大回撤': mdd, '月胜率': win})
            for grp, net in nets.items():
                gann, _, _, _ = core.net_stats(net, ps, pe)
                ls_rows[-1]['G%d' % grp] = gann

    ic_df = pd.DataFrame(ic_rows)
    ic_df.to_csv(os.path.join(OUT_DIR, 'ic_stats_composite.csv'),
                 index=False, encoding='utf-8-sig')
    ls_df = pd.DataFrame(ls_rows)
    ls_df.to_csv(os.path.join(OUT_DIR, 'layered_stats_composite.csv'),
                 index=False, encoding='utf-8-sig')

    # ---- 汇总打印 ----
    print('\n===== 复合因子 vs 单因子：验证期 ICIR（5个周期）=====')
    piv = ic_df[ic_df.period == 'valid'].pivot_table(index='factor', columns='horizon',
                                                     values='ICIR').round(3)
    print(piv.to_string())
    print('\n===== 分层多空绩效（train/valid）=====')
    print(ls_df[ls_df.version.isin(['composite', 'neu'])].round(3).to_string(index=False))

    # ---- 对比图 ----
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        for label, net in ls_nets.items():
            seg = net.loc[core.VALID_START:core.VALID_END].dropna()
            if len(seg) > 1:
                ax.plot(range(len(seg)), seg.values, label=label, lw=1.2)
        ax.set_title('验证期多空净值（复合 vs 单因子）')
        ax.legend()
        ax = axes[1]
        for cname in ['comp_eq', 'comp_w']:
            wp = w[cname] if cname in w.columns else None
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'composite_compare.png'), dpi=100)
        plt.show()
    except Exception as e:
        print('[警告] 作图失败: %s' % e)

    print('\n输出目录: %s/' % OUT_DIR)
    for fn in sorted(os.listdir(OUT_DIR)):
        print('  %s/%s' % (OUT_DIR, fn))
    core._log('===== [comp] RUN DONE =====')


if __name__ == '__main__':
    main()
