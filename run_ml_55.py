# -*- coding: utf-8 -*-
"""
55 因子机器学习合成（本地运行）
================================
特征池：五段 ICIR 全正的 55 个 Alpha101 因子（中性化 z-score），
在 6 因子实验（结论：无非线性肉）基础上扩大特征池，回答
"特征多了之后 ML 能否超过线性合成"。

模型与标签（与 6 因子实验同构，便于对比）：
  - XGBoost / 随机森林 × 3日 / 5日前向收益标签
  - 线性基线：55 因子等权合成 comp_eq55 / 滚动ICIR加权 comp_w55

时间方案（严格防前视）：
  - 调参：训练 2015-2020，早停/评估 2021-2023（只碰 2015-2023）
  - 测试：2024-01 起每 6 个月 walk-forward 重训，2024-2025 只测一次

输出（results_ml_55/）：
  - ic_stats_ml.csv / layered_stats_ml.csv / turnover_ml.csv
  - pred_*.csv（各序列预测面板）
  - feature_importance.csv / dependence.csv（xgb 前20特征）/ interaction.csv（前10对）

本地运行：python run_ml_55.py（预计 40~90 分钟，取决于 CPU 核数）
"""
import os
import gc
import numpy as np
import pandas as pd
from jqdata import *
import alpha101_test_core as core
from alpha101_factors import FACTOR_FUNCS

# 五段 ICIR 全正的 55 个因子（中性化口径，由合并总表筛出）
POOL_FACTORS = [
    'alpha003', 'alpha004', 'alpha006', 'alpha008', 'alpha012', 'alpha013',
    'alpha014', 'alpha015', 'alpha016', 'alpha019', 'alpha021', 'alpha023',
    'alpha024', 'alpha026', 'alpha027', 'alpha029', 'alpha031', 'alpha033',
    'alpha035', 'alpha036', 'alpha037', 'alpha038', 'alpha039', 'alpha040',
    'alpha044', 'alpha045', 'alpha046', 'alpha047', 'alpha050', 'alpha053',
    'alpha055', 'alpha056', 'alpha058', 'alpha059', 'alpha060', 'alpha061',
    'alpha063', 'alpha066', 'alpha067', 'alpha068', 'alpha069', 'alpha070',
    'alpha072', 'alpha075', 'alpha080', 'alpha081', 'alpha083', 'alpha087',
    'alpha088', 'alpha089', 'alpha092', 'alpha094', 'alpha095', 'alpha096',
    'alpha097',
]

import jqdata
OUT_DIR = 'results_ml55_%s' % jqdata._get_active_index()
LABELS = [('ret3', 3), ('ret5', 5)]
TUNE_TRAIN = ('2015-01-01', '2020-12-31')
TUNE_VALID = ('2021-01-01', '2023-12-31')
TEST_START = '2024-01-01'
RF_SAMPLE = 150000
PD_SAMPLE = 100000
TOP_DEP = 20      # 偏依赖只输出重要性前 20 的因子
TOP_INTER = 10    # 交互只输出重要性前 10 对
MIN_FEATURES = 20 # 样本有效特征数下限（55 个因子中至少 20 个非 NaN 才纳入训练）


def _log(msg):
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, 'debug_log.txt'), 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')
    except Exception:
        pass


import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
_log('依赖自检: xgboost=%s sklearn=%s' % (xgb.__version__, 'OK'))


# ---------------- 数据与因子 ----------------
def build_features():
    stocks, snap_list = core.fetch_index_members(core.INDEX_CODE, core.FETCH_START, core.VALID_END)
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

    z_panels = {}
    for name in POOL_FACTORS:
        f = FACTOR_FUNCS[name](data)
        if not isinstance(f, pd.DataFrame):
            f = pd.DataFrame(f, index=price['close'].index, columns=price['close'].columns)
        f = f.where(valid_mask)
        f_neu = core.neutralize(f, industry_map, log_mcap)
        # std=0（截面退化的日子，如三态因子全同值）置 NaN，避免除零
        sd = f_neu.std(axis=1).replace(0, np.nan)
        z = f_neu.sub(f_neu.mean(axis=1), axis=0).div(sd, axis=0).clip(-5, 5)
        z_panels[name] = z.astype(np.float32)
        _log('[ml55] %s 有效值 %d' % (name, int(z.notna().sum().sum())))

    labels = {}
    for lname, h in LABELS:
        fwd = core.forward_returns(price['close'], h)
        labels[lname] = fwd.sub(fwd.mean(axis=1), axis=0).div(fwd.std(axis=1), axis=0).astype(np.float32)

    fwd_all = {h: core.forward_returns(price['close'], h) for h in core.HORIZONS}
    ret1 = core.forward_returns(price['close'], 1)
    fwd_ranked = {h: fwd_all[h].rank(axis=1, pct=True).astype(np.float32) for h in core.HORIZONS}
    return z_panels, labels, fwd_ranked, ret1, valid_mask, price['close']


def build_full_arrays(z_panels, labels):
    """建全期 (T,N,55) 特征数组。缺失 z-score 用 0 填充（=截面中性位置），
    样本有效性放宽为"至少 MIN_FEATURES 个特征非 NaN"——否则 55 个因子的
    NaN 交集会把覆盖率压到不足 1%（此前训练样本仅 3793 个的根因）"""
    names = list(z_panels.keys())
    idx = z_panels[names[0]].index
    cols = z_panels[names[0]].columns
    arr_raw = np.stack([z_panels[n].values for n in names], axis=2).astype(np.float32)
    arr = np.nan_to_num(arr_raw, nan=0.0)
    Ys = {ln: labels[ln].reindex(columns=cols).values.astype(np.float32) for ln in labels}
    valid = np.isfinite(arr_raw).sum(axis=2) >= MIN_FEATURES
    return arr, Ys, valid, idx, cols, names


def slice_window(arr, Y, valid, idx, t_start, t_end):
    mask = (idx >= t_start) & (idx <= t_end)
    m = mask.values if hasattr(mask, 'values') else mask
    a = arr[m]
    y = Y[m]
    v = valid[m] & np.isfinite(y)
    return a[v].astype(np.float32), y[v].astype(np.float32), v, idx[m]


def rebuild_panel(pred, valid_win, dates, cols):
    panel = np.full((len(dates), len(cols)), np.nan, dtype=np.float32)
    panel[valid_win] = pred
    return pd.DataFrame(panel, index=dates, columns=cols)


def panel_icir(pred_panel, fwd_ranked_h, t_start, t_end):
    f_h = fwd_ranked_h.reindex(columns=pred_panel.columns)
    p = pred_panel.rank(axis=1, pct=True).astype(np.float32)
    ic = core.rank_ic_from_ranked(p, f_h)
    seg = ic.loc[t_start:t_end].dropna()
    if len(seg) < 60 or seg.std() == 0:
        return np.nan
    return seg.mean() / seg.std()


def fit_xgb(X, Y, Xv, Yv, depth, lr, mcw, early):
    m = xgb.XGBRegressor(max_depth=depth, learning_rate=lr, min_child_weight=mcw,
                         n_estimators=400, subsample=0.8, colsample_bytree=0.6,
                         objective='reg:squarederror', n_jobs=-1, random_state=42,
                         early_stopping_rounds=early, eval_metric='rmse')
    if Xv is not None and early:
        m.fit(X, Y, eval_set=[(Xv, Yv)], verbose=False)
    else:
        m.fit(X, Y)
    return m


def fit_rf(X, Y, depth, leaf):
    m = RandomForestRegressor(max_depth=depth, min_samples_leaf=leaf,
                              n_estimators=80, n_jobs=-1, random_state=42)
    if len(X) > RF_SAMPLE:
        rng = np.random.RandomState(42)
        sel = rng.choice(len(X), RF_SAMPLE, replace=False)
        m.fit(X[sel], Y[sel])
    else:
        m.fit(X, Y)
    return m


def tune_models(arr, Ys, valid, idx, cols, fwd_ranked):
    results = {}
    lab_map = dict(LABELS)
    for lname in lab_map:
        h = lab_map[lname]
        X, Y, vt, dt = slice_window(arr, Ys[lname], valid, idx, TUNE_TRAIN[0], TUNE_TRAIN[1])
        Xv, Yv, vv, dv = slice_window(arr, Ys[lname], valid, idx, TUNE_VALID[0], TUNE_VALID[1])
        _log('[tune] %s 训练样本 %d | 调参样本 %d' % (lname, len(X), len(Xv)))
        best = (None, -np.inf)
        for depth in (2, 3):
            for lr in (0.03, 0.05):
                for mcw in (100, 300):
                    try:
                        m = fit_xgb(X, Y, Xv, Yv, depth, lr, mcw, early=25)
                        pred = m.predict(Xv)
                        pnl = rebuild_panel(pred, vv, dv, cols)
                        icir = panel_icir(pnl, fwd_ranked[h], TUNE_VALID[0], TUNE_VALID[1])
                        _log('[tune] xgb %s d=%d lr=%.2f mcw=%d ICIR=%.3f' % (lname, depth, lr, mcw, icir))
                        if icir > best[1]:
                            best = ({'max_depth': depth, 'learning_rate': lr,
                                     'min_child_weight': mcw}, icir)
                    except Exception as e:
                        _log('[tune] xgb %s 失败 d=%d lr=%.2f: %s' % (lname, depth, lr, e))
        results['xgb_%s' % lname] = ('xgb', best[0], best[1])
        gc.collect()
        best = (None, -np.inf)
        for depth in (3, 4):
            for leaf in (100, 300):
                try:
                    m = fit_rf(X, Y, depth, leaf)
                    pred = m.predict(Xv)
                    pnl = rebuild_panel(pred, vv, dv, cols)
                    icir = panel_icir(pnl, fwd_ranked[h], TUNE_VALID[0], TUNE_VALID[1])
                    _log('[tune] rf %s d=%d leaf=%d ICIR=%.3f' % (lname, depth, leaf, icir))
                    if icir > best[1]:
                        best = ({'max_depth': depth, 'min_samples_leaf': leaf}, icir)
                except Exception as e:
                    _log('[tune] rf %s 失败 d=%d leaf=%d: %s' % (lname, depth, leaf, e))
        results['rf_%s' % lname] = ('rf', best[0], best[1])
        gc.collect()
    return results


def walk_forward(arr, Ys, valid, idx, cols, fwd_ranked, tuned):
    preds = {}
    retrain_dates = pd.date_range(TEST_START, '2025-12-31', freq='6MS')
    lab_map = dict(LABELS)
    for key, (mtype, params, _icir) in tuned.items():
        if params is None:
            _log('[wf] %s 无有效超参，跳过' % key)
            continue
        lname = key.split('_', 1)[1]
        h = lab_map[lname]
        panel = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=np.float32)
        for r in retrain_dates:
            rs = r.strftime('%Y-%m-%d')
            r_end = (r + pd.DateOffset(months=6) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            if r_end > '2025-12-31':
                r_end = '2025-12-31'
            t_end = (pd.Timestamp(rs) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            X, Y, vt, dt = slice_window(arr, Ys[lname], valid, idx, core.TRAIN_START, t_end)
            ev_start = (pd.Timestamp(rs) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
            Xe, Ye, ve, de = slice_window(arr, Ys[lname], valid, idx, ev_start, t_end)
            if len(Xe) == 0:
                Xe, Ye = None, None
            Xp, Yp, vp, dp = slice_window(arr, Ys[lname], valid, idx, rs, r_end)
            try:
                if mtype == 'xgb':
                    m = fit_xgb(X, Y, Xe, Ye, params['max_depth'], params['learning_rate'],
                                params['min_child_weight'], early=25)
                else:
                    m = fit_rf(X, Y, params['max_depth'], params['min_samples_leaf'])
                pred = m.predict(Xp)
                pnl = rebuild_panel(pred, vp, dp, cols)
                panel.loc[pnl.index, pnl.columns] = pnl
                _log('[wf] %s 重训 %s 完成（训练样本 %d）' % (key, rs, len(X)))
            except Exception as e:
                _log('[wf] %s 重训 %s 失败: %s' % (key, rs, e))
            gc.collect()
        preds[key] = panel
        panel.to_csv(os.path.join(OUT_DIR, 'pred_%s.csv' % key), encoding='utf-8-sig')
    return preds


def linear_baselines(z_panels, fwd1_r):
    names = list(z_panels.keys())
    idx, cols = z_panels[names[0]].index, z_panels[names[0]].columns
    fwd1_r = fwd1_r.reindex(columns=cols)
    ic_series = {}
    for n, f in z_panels.items():
        f_r = f.rank(axis=1, pct=True).astype(np.float32)
        ic_series[n] = core.rank_ic_from_ranked(f_r, fwd1_r)
    ic_df = pd.DataFrame(ic_series)
    ic_mean = ic_df.rolling(252, min_periods=60).mean().shift(1)
    ic_std = ic_df.rolling(252, min_periods=60).std().shift(1)
    icir = ic_mean / ic_std.replace(0, np.nan)
    signs = np.sign(ic_mean)
    w = icir.abs().div(icir.abs().sum(axis=1).replace(0, np.nan), axis=0)
    aligned = {n: z_panels[n].mul(signs[n], axis=0) for n in names}
    z_arr = np.nan_to_num(np.stack([aligned[n].values for n in names])
                           .astype(np.float32), nan=0.0)
    comp_eq = pd.DataFrame(np.nanmean(z_arr, axis=0), index=idx, columns=cols)
    comp_w_vals = np.einsum('tn,nts->ts', w.fillna(0).values, z_arr)
    comp_w = pd.DataFrame(comp_w_vals, index=idx, columns=cols)
    return comp_eq, comp_w


def evaluate(preds, fwd_ranked, ret1):
    ic_rows, ls_rows, to_rows = [], [], []
    linear_periods = ([('train', core.TRAIN_START, core.TRAIN_END)] + core.TRAIN_SLICES
                      + [('valid', core.VALID_START, core.VALID_END)])
    for pname, panel in preds.items():
        is_ml = pname.startswith(('xgb_', 'rf_'))
        periods = ([('valid', core.VALID_START, core.VALID_END),
                    ('valid_p1_2024', '2024-01-01', '2024-12-31'),
                    ('valid_p2_2025', '2025-01-01', '2025-12-31')]
                   if is_ml else linear_periods)
        p_r = panel.rank(axis=1, pct=True).astype(np.float32)
        fr_aligned = {h: fwd_ranked[h].reindex(columns=panel.columns) for h in core.HORIZONS}
        ret1_aligned = ret1.reindex(columns=panel.columns)
        for h in core.HORIZONS:
            ic = core.rank_ic_from_ranked(p_r, fr_aligned[h])
            for period, ps, pe in periods:
                st_ = core.ic_stats(ic, ps, pe)
                ic_rows.append({'factor': pname, 'version': 'ml' if is_ml else 'linear',
                                'period': period, 'horizon': h, **st_})
        nets, ls_net = core.layered_backtest(p_r, ret1_aligned)
        for period, (ps, pe) in [('train', (core.TRAIN_START, core.TRAIN_END)),
                                 ('valid', (core.VALID_START, core.VALID_END))]:
            if is_ml and period == 'train':
                continue
            ann, sharpe, mdd, win = core.net_stats(ls_net, ps, pe)
            ls_rows.append({'factor': pname, 'version': 'ml' if is_ml else 'linear',
                            'period': period, '年化收益': ann, '夏普': sharpe,
                            '最大回撤': mdd, '月胜率': win})
            for grp, net in nets.items():
                gann, _, _, _ = core.net_stats(net, ps, pe)
                ls_rows[-1]['G%d' % grp] = gann
        to_rows.append({'factor': pname,
                        '日均单边换手': turnover_of(p_r, core.VALID_START, core.VALID_END)})
    ic_df = pd.DataFrame(ic_rows)
    ic_df.to_csv(os.path.join(OUT_DIR, 'ic_stats_ml.csv'), index=False, encoding='utf-8-sig')
    ls_df = pd.DataFrame(ls_rows)
    ls_df.to_csv(os.path.join(OUT_DIR, 'layered_stats_ml.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(to_rows).to_csv(os.path.join(OUT_DIR, 'turnover_ml.csv'),
                                 index=False, encoding='utf-8-sig')
    return ic_df, ls_df


def turnover_of(p_r, t_start, t_end):
    seg = p_r.loc[t_start:t_end]
    top = (seg >= 0.8).fillna(False).astype(bool)
    bot = (seg <= 0.2).fillna(False).astype(bool)
    to_top = (top & ~top.shift(1).fillna(False)).sum(axis=1) \
        / top.sum(axis=1).replace(0, np.nan)
    to_bot = (bot & ~bot.shift(1).fillna(False)).sum(axis=1) \
        / bot.sum(axis=1).replace(0, np.nan)
    return 0.5 * (to_top + to_bot).mean()


def interpretability(arr, Ys, valid, idx, cols, names, tuned):
    """特征重要性（全部）+ 偏依赖/交互（仅 XGBoost，按重要性取前 N）"""
    imp_rows, dep_rows, inter_rows = [], [], []
    t_end_full = (pd.Timestamp(TEST_START) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    for key, (mtype, params, _) in tuned.items():
        if params is None:
            continue
        lname = key.split('_', 1)[1]
        X, Y, v, dt = slice_window(arr, Ys[lname], valid, idx, core.TRAIN_START, t_end_full)
        if mtype == 'xgb':
            m = fit_xgb(X, Y, None, None, params['max_depth'], params['learning_rate'],
                        params['min_child_weight'], early=None)
        else:
            m = fit_rf(X, Y, params['max_depth'], params['min_samples_leaf'])
        imp = getattr(m, 'feature_importances_')
        for n, vv in zip(names, imp):
            imp_rows.append({'model': key, 'factor': n, 'importance': float(vv)})
        if mtype != 'xgb':
            gc.collect()
            continue
        # 偏依赖与交互只对 XGBoost 做：偏依赖取重要性前 TOP_DEP 因子，
        # 交互取前 12 因子的全部两两组合
        top_idx = np.argsort(imp)[::-1][:TOP_DEP]
        top_names = [names[i] for i in top_idx]
        inter_names = top_names[:12]
        n_s = min(PD_SAMPLE, len(X))
        rng = np.random.RandomState(42)
        sel = rng.choice(len(X), n_s, replace=False)
        Xs, pred = X[sel], m.predict(X[sel])
        for fi in top_idx:
            fname = names[fi]
            bins = pd.qcut(pd.Series(Xs[:, fi]), 20, labels=False, duplicates='drop')
            for b in sorted(set(bins)):
                if np.isnan(b):
                    continue
                dep_rows.append({'model': key, 'factor': fname, 'bin': int(b),
                                 'bin_mean_value': float(np.nanmean(Xs[:, fi][bins == b])),
                                 'mean_pred': float(np.nanmean(pred[bins == b]))})
        for a in range(len(inter_names)):
            for b in range(a + 1, len(inter_names)):
                ia = names.index(inter_names[a])
                ib = names.index(inter_names[b])
                bi = pd.qcut(pd.Series(Xs[:, ia]), 5, labels=False, duplicates='drop')
                bj = pd.qcut(pd.Series(Xs[:, ib]), 5, labels=False, duplicates='drop')
                for i in range(5):
                    for j in range(5):
                        msk = (bi == i) & (bj == j)
                        if msk.sum() > 5:
                            inter_rows.append({'model': key, 'f1': inter_names[a], 'f2': inter_names[b],
                                               'b1': int(i), 'b2': int(j),
                                               'mean_pred': float(np.nanmean(pred[msk]))})
        gc.collect()
    pd.DataFrame(imp_rows).to_csv(os.path.join(OUT_DIR, 'feature_importance.csv'),
                                  index=False, encoding='utf-8-sig')
    pd.DataFrame(dep_rows).to_csv(os.path.join(OUT_DIR, 'dependence.csv'),
                                  index=False, encoding='utf-8-sig')
    pd.DataFrame(inter_rows).to_csv(os.path.join(OUT_DIR, 'interaction.csv'),
                                    index=False, encoding='utf-8-sig')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _log('===== [ml55] RUN START | 特征池 %d 个因子 =====' % len(POOL_FACTORS))
    z_panels, labels, fwd_ranked, ret1, valid_mask, close = build_features()
    arr, Ys, valid, idx, cols, names = build_full_arrays(z_panels, labels)
    _log('[ml55] 特征矩阵 %s' % str(arr.shape))

    fwd1 = core.forward_returns(close, 1)
    fwd1_r = fwd1.rank(axis=1, pct=True).astype(np.float32)
    comp_eq, comp_w = linear_baselines(z_panels, fwd1_r)
    comp_eq.to_csv(os.path.join(OUT_DIR, 'pred_comp_eq55.csv'), encoding='utf-8-sig')
    comp_w.to_csv(os.path.join(OUT_DIR, 'pred_comp_w55.csv'), encoding='utf-8-sig')
    _log('[ml55] 线性基线完成')

    _log('[ml55] 开始调参（训练 %s ~ %s / 调参 %s ~ %s）'
         % (TUNE_TRAIN[0], TUNE_TRAIN[1], TUNE_VALID[0], TUNE_VALID[1]))
    tuned = tune_models(arr, Ys, valid, idx, cols, fwd_ranked)
    _log('[ml55] 调参结果: %s' % str(tuned))

    _log('[ml55] 开始 walk-forward 测试预测')
    preds_ml = walk_forward(arr, Ys, valid, idx, cols, fwd_ranked, tuned)

    preds_all = dict(preds_ml)
    preds_all['comp_eq55'] = comp_eq
    preds_all['comp_w55'] = comp_w
    ic_df, ls_df = evaluate(preds_all, fwd_ranked, ret1)

    interpretability(arr, Ys, valid, idx, cols, names, tuned)

    print('\n===== 测试段(2024-2025) ICIR 对比 =====')
    piv = ic_df[ic_df.period == 'valid'].pivot_table(index='factor', columns='horizon',
                                                     values='ICIR').round(3)
    print(piv.to_string())
    print('\n===== 分层多空（测试段）=====')
    print(ls_df[ls_df.period == 'valid'].round(3).to_string(index=False))
    print('\n输出目录: %s/' % OUT_DIR)
    for fn in sorted(os.listdir(OUT_DIR)):
        print('  %s/%s' % (OUT_DIR, fn))
    _log('===== [ml55] RUN DONE =====')


if __name__ == '__main__':
    main()
