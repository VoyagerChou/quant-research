# -*- coding: utf-8 -*-
"""
六因子机器学习合成（聚宽研究环境）
====================================
对第一梯队 6 因子（中性化 z-score）做非线性合成：
  - XGBoost（主攻）与 随机森林（无偏对照）
  - 标签：3 日前向收益 / 5 日前向收益（每日截面 z-score 化）
  - 线性基线：等权合成 / 滚动ICIR加权合成（同口径对比）

时间方案（严格防前视）：
  - 调参阶段：训练 2015-2020，早停/评估 2021-2023 → 选超参
    （只碰 2015-2023，2024-2025 完全不参与）
  - 测试阶段：2024-01 起每 6 个月 walk-forward 重训（扩展窗口、固定超参），
    逐日产出 2024-2025 预测——测试集只测一次

输出（results_ml_composite/）：见文件末尾清单。
运行：与其它文件一起上传后直接运行。预计 25~50 分钟。
"""
import os
import gc
import numpy as np
import pandas as pd
from jqdata import *
import alpha101_test_core as core
from alpha101_factors import FACTOR_FUNCS

CORR_FACTORS = ['alpha003', 'alpha006', 'alpha019', 'alpha015', 'alpha021', 'alpha026']
OUT_DIR = 'results_ml_composite'
LABELS = [('ret3', 3), ('ret5', 5)]
TUNE_TRAIN = ('2015-01-01', '2020-12-31')
TUNE_VALID = ('2021-01-01', '2023-12-31')
TEST_START = '2024-01-01'
RF_SAMPLE = 100000
PD_SAMPLE = 100000


def _log(msg):
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, 'debug_log.txt'), 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')
    except Exception:
        pass


# ---------------- 依赖自检 ----------------
XGB_ERR, SK_ERR = None, None
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError as e:
    HAS_XGB = False
    XGB_ERR = str(e)
try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_RF = True
except ImportError as e:
    HAS_RF = False
    SK_ERR = str(e)
_log('依赖自检: xgboost=%s (%s) | sklearn.RF=%s (%s)'
     % (HAS_XGB, XGB_ERR or 'OK', HAS_RF, SK_ERR or 'OK'))


# ---------------- 数据与因子 ----------------
def build_features():
    """拉数据 → 6 中性化因子 z-score + 前向收益标签 + 检验用排名"""
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
    for name in CORR_FACTORS:
        f = FACTOR_FUNCS[name](data)
        if not isinstance(f, pd.DataFrame):
            f = pd.DataFrame(f, index=price['close'].index, columns=price['close'].columns)
        f = f.where(valid_mask)
        f_neu = core.neutralize(f, industry_map, log_mcap)
        sd = f_neu.std(axis=1).replace(0, np.nan)
        z = f_neu.sub(f_neu.mean(axis=1), axis=0).div(sd, axis=0).clip(-5, 5)
        z_panels[name] = z.astype(np.float32)
        _log('[ml] %s z-score 有效值 %d' % (name, int(z.notna().sum().sum())))

    labels = {}
    for lname, h in LABELS:
        fwd = core.forward_returns(price['close'], h)
        labels[lname] = fwd.sub(fwd.mean(axis=1), axis=0).div(fwd.std(axis=1), axis=0).astype(np.float32)

    fwd_all = {h: core.forward_returns(price['close'], h) for h in core.HORIZONS}
    ret1 = core.forward_returns(price['close'], 1)
    fwd_ranked = {h: fwd_all[h].rank(axis=1, pct=True).astype(np.float32) for h in core.HORIZONS}
    return z_panels, labels, fwd_ranked, ret1, valid_mask, price['close']


# ---------------- 矩阵工具 ----------------
def build_full_arrays(z_panels, labels):
    """一次性建全期 (T,N,6) 特征数组与各标签数组。
    注意：标签须对齐到因子列（中性化后因子列 = 有行业映射的股票子集）"""
    names = list(z_panels.keys())
    idx = z_panels[names[0]].index
    cols = z_panels[names[0]].columns
    arr = np.stack([z_panels[n].values for n in names], axis=2).astype(np.float32)  # (T,N,6)
    Ys = {ln: labels[ln].reindex(columns=cols).values.astype(np.float32) for ln in labels}
    valid = np.isfinite(arr).all(axis=2)
    return arr, Ys, valid, idx, cols, names


def slice_window(arr, Y, valid, idx, t_start, t_end):
    """切出时间窗口并摊平为 (样本, 特征)；返回 X, Y, 窗口内有效性, 窗口日期, 占位"""
    mask = (idx >= t_start) & (idx <= t_end)
    m = mask.values if hasattr(mask, 'values') else mask   # 兼容 ndarray / Series
    a = arr[m]
    y = Y[m]
    v = valid[m] & np.isfinite(y)
    X = a[v].astype(np.float32)
    Yv = y[v].astype(np.float32)
    return X, Yv, v, idx[m], None


def rebuild_panel(pred, valid_win, dates, cols):
    """展平预测 → (日期×股票) 面板"""
    panel = np.full((len(dates), len(cols)), np.nan, dtype=np.float32)
    panel[valid_win] = pred
    return pd.DataFrame(panel, index=dates, columns=cols)


def panel_icir(pred_panel, fwd_ranked_h, t_start, t_end):
    """预测面板在区间上的 RankICIR（对 h 日前向收益）"""
    f_h = fwd_ranked_h.reindex(columns=pred_panel.columns)   # 对齐股票列
    p = pred_panel.rank(axis=1, pct=True).astype(np.float32)
    ic = core.rank_ic_from_ranked(p, f_h)
    seg = ic.loc[t_start:t_end].dropna()
    if len(seg) < 60 or seg.std() == 0:
        return np.nan
    return seg.mean() / seg.std()


# ---------------- 模型 ----------------
def fit_xgb(X, Y, Xv, Yv, depth, lr, mcw, early):
    m = xgb.XGBRegressor(max_depth=depth, learning_rate=lr, min_child_weight=mcw,
                         n_estimators=500, subsample=0.8, colsample_bytree=0.8,
                         objective='reg:squarederror', n_jobs=1, random_state=42,
                         early_stopping_rounds=early, eval_metric='rmse')
    if Xv is not None and early:
        m.fit(X, Y, eval_set=[(Xv, Yv)], verbose=False)
    else:
        m.fit(X, Y)
    return m


def fit_rf(X, Y, depth, leaf):
    m = RandomForestRegressor(max_depth=depth, min_samples_leaf=leaf,
                              n_estimators=100, n_jobs=-1, random_state=42)
    if len(X) > RF_SAMPLE:
        rng = np.random.RandomState(42)
        sel = rng.choice(len(X), RF_SAMPLE, replace=False)
        m.fit(X[sel], Y[sel])
    else:
        m.fit(X, Y)
    return m


def tune_models(arr, Ys, valid, idx, cols, fwd_ranked):
    """调参：训练窗拟合 + 调参窗 ICIR 选优。返回 {key: (类型, 超参, 调参窗ICIR)}"""
    results = {}
    lab_map = dict(LABELS)
    for lname in lab_map:
        h = lab_map[lname]
        X, Y, vt, dt, _ = slice_window(arr, Ys[lname], valid, idx, TUNE_TRAIN[0], TUNE_TRAIN[1])
        Xv, Yv, vv, dv, _ = slice_window(arr, Ys[lname], valid, idx, TUNE_VALID[0], TUNE_VALID[1])
        _log('[tune] %s 训练样本 %d | 调参样本 %d' % (lname, len(X), len(Xv)))
        if HAS_XGB:
            best = (None, -np.inf)
            for depth in (2, 3):
                for lr in (0.03, 0.05):
                    for mcw in (50, 200):
                        try:
                            m = fit_xgb(X, Y, Xv, Yv, depth, lr, mcw, early=25)
                            pred = m.predict(Xv)
                            pnl = rebuild_panel(pred, vv, dv, cols)
                            icir = panel_icir(pnl, fwd_ranked[h], TUNE_VALID[0], TUNE_VALID[1])
                            _log('[tune] xgb %s d=%d lr=%.2f mcw=%d ICIR=%.3f'
                                 % (lname, depth, lr, mcw, icir))
                            if icir > best[1]:
                                best = ({'max_depth': depth, 'learning_rate': lr,
                                         'min_child_weight': mcw}, icir)
                        except Exception as e:
                            _log('[tune] xgb %s 失败 d=%d lr=%.2f: %s' % (lname, depth, lr, e))
            results['xgb_%s' % lname] = ('xgb', best[0], best[1])
            gc.collect()
        if HAS_RF:
            best = (None, -np.inf)
            for depth in (3, 4):
                for leaf in (50, 200):
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
    """2024-2025 每 6 个月重训（扩展窗口+固定超参），输出各序列预测面板"""
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
            X, Y, vt, dt, _ = slice_window(arr, Ys[lname], valid, idx, core.TRAIN_START, t_end)
            # 早停集：训练窗最后 12 个月（单独切片，维度与 X 对齐）
            ev_start = (pd.Timestamp(rs) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
            Xe, Ye, ve, de, _ = slice_window(arr, Ys[lname], valid, idx, ev_start, t_end)
            if len(Xe) == 0:
                Xe, Ye = None, None
            Xp, Yp, vp, dp, _ = slice_window(arr, Ys[lname], valid, idx, rs, r_end)
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


# ---------------- 线性基线 ----------------
def linear_baselines(z_panels, fwd1_r):
    """等权与滚动ICIR加权合成（时点权重，与 run_composite 一致）"""
    names = list(z_panels.keys())
    idx, cols = z_panels[names[0]].index, z_panels[names[0]].columns
    fwd1_r = fwd1_r.reindex(columns=cols)   # 对齐股票列
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


# ---------------- 评估 ----------------
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
        # 对齐前向收益排名到面板股票列（面板=中性化后的 659 列）
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
    """多空组合（前20%/后20%）的日均单边换手"""
    seg = p_r.loc[t_start:t_end]
    # pandas 3.x 下 NaN 与标量比较返回 NaN（而非 False），须显式转 bool
    top = (seg >= 0.8).fillna(False).astype(bool)
    bot = (seg <= 0.2).fillna(False).astype(bool)
    to_top = (top & ~top.shift(1).fillna(False)).sum(axis=1) \
        / top.sum(axis=1).replace(0, np.nan)
    to_bot = (bot & ~bot.shift(1).fillna(False)).sum(axis=1) \
        / bot.sum(axis=1).replace(0, np.nan)
    return 0.5 * (to_top + to_bot).mean()


# ---------------- 可解释性 ----------------
def interpretability(arr, Ys, valid, idx, cols, names, tuned):
    """特征重要性 + 偏依赖表 + 交互表（全部表化，供读表分析）"""
    imp_rows, dep_rows, inter_rows = [], [], []
    t_end_full = (pd.Timestamp(TEST_START) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    for key, (mtype, params, _) in tuned.items():
        if params is None or not (HAS_XGB if mtype == 'xgb' else HAS_RF):
            continue
        lname = key.split('_', 1)[1]
        X, Y, v, dt, _ = slice_window(arr, Ys[lname], valid, idx, core.TRAIN_START, t_end_full)
        if mtype == 'xgb':
            m = fit_xgb(X, Y, None, None, params['max_depth'], params['learning_rate'],
                        params['min_child_weight'], early=None)
        else:
            m = fit_rf(X, Y, params['max_depth'], params['min_samples_leaf'])
        for n, vv in zip(names, getattr(m, 'feature_importances_')):
            imp_rows.append({'model': key, 'factor': n, 'importance': float(vv)})
        n_s = min(PD_SAMPLE, len(X))
        rng = np.random.RandomState(42)
        sel = rng.choice(len(X), n_s, replace=False)
        Xs, pred = X[sel], m.predict(X[sel])
        for fi, fname in enumerate(names):
            bins = pd.qcut(pd.Series(Xs[:, fi]), 20, labels=False, duplicates='drop')
            for b in sorted(set(bins)):
                if np.isnan(b):
                    continue
                dep_rows.append({'model': key, 'factor': fname, 'bin': int(b),
                                 'bin_mean_value': float(np.nanmean(Xs[:, fi][bins == b])),
                                 'mean_pred': float(np.nanmean(pred[bins == b]))})
        if mtype == 'xgb':
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    bi = pd.qcut(pd.Series(Xs[:, i]), 5, labels=False, duplicates='drop')
                    bj = pd.qcut(pd.Series(Xs[:, j]), 5, labels=False, duplicates='drop')
                    for a in range(5):
                        for b in range(5):
                            msk = (bi == a) & (bj == b)
                            if msk.sum() > 5:
                                inter_rows.append({'model': key, 'f1': names[i], 'f2': names[j],
                                                   'b1': int(a), 'b2': int(b),
                                                   'mean_pred': float(np.nanmean(pred[msk]))})
        gc.collect()
    pd.DataFrame(imp_rows).to_csv(os.path.join(OUT_DIR, 'feature_importance.csv'),
                                  index=False, encoding='utf-8-sig')
    pd.DataFrame(dep_rows).to_csv(os.path.join(OUT_DIR, 'dependence.csv'),
                                  index=False, encoding='utf-8-sig')
    pd.DataFrame(inter_rows).to_csv(os.path.join(OUT_DIR, 'interaction.csv'),
                                    index=False, encoding='utf-8-sig')


# ---------------- 主流程 ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _log('===== [ml] RUN START =====')
    if not (HAS_XGB or HAS_RF):
        _log('[ml] 致命：xgboost 与 sklearn 均不可用，退出')
        print('依赖缺失：请确认研究环境是否安装了 xgboost / scikit-learn')
        return

    z_panels, labels, fwd_ranked, ret1, valid_mask, close = build_features()
    arr, Ys, valid, idx, cols, names = build_full_arrays(z_panels, labels)
    _log('[ml] 特征矩阵 %s | 标签 %s' % (str(arr.shape), str(list(Ys.keys()))))

    fwd1 = core.forward_returns(close, 1)
    fwd1_r = fwd1.rank(axis=1, pct=True).astype(np.float32)
    comp_eq, comp_w = linear_baselines(z_panels, fwd1_r)
    comp_eq.to_csv(os.path.join(OUT_DIR, 'pred_comp_eq.csv'), encoding='utf-8-sig')
    comp_w.to_csv(os.path.join(OUT_DIR, 'pred_comp_w.csv'), encoding='utf-8-sig')

    _log('[ml] 开始调参（训练 %s ~ %s / 调参 %s ~ %s）'
         % (TUNE_TRAIN[0], TUNE_TRAIN[1], TUNE_VALID[0], TUNE_VALID[1]))
    tuned = tune_models(arr, Ys, valid, idx, cols, fwd_ranked)
    _log('[ml] 调参结果: %s' % str(tuned))

    _log('[ml] 开始 walk-forward 测试预测（%s 起）' % TEST_START)
    preds_ml = walk_forward(arr, Ys, valid, idx, cols, fwd_ranked, tuned)

    preds_all = dict(preds_ml)
    preds_all['comp_eq'] = comp_eq
    preds_all['comp_w'] = comp_w
    ic_df, ls_df = evaluate(preds_all, fwd_ranked, ret1)

    interpretability(arr, Ys, valid, idx, cols, names, tuned)

    print('\n===== 测试段(2024-2025) ICIR 对比 =====')
    piv = ic_df[ic_df.period == 'valid'].pivot_table(index='factor', columns='horizon',
                                                     values='ICIR').round(3)
    print(piv.to_string())
    print('\n===== 分层多空（测试段）=====')
    print(ls_df[ls_df.period == 'valid'].round(3).to_string(index=False))
    print('\n===== 换手率 =====')
    print(pd.read_csv(os.path.join(OUT_DIR, 'turnover_ml.csv')).round(4).to_string(index=False))
    print('\n输出目录: %s/' % OUT_DIR)
    for fn in sorted(os.listdir(OUT_DIR)):
        print('  %s/%s' % (OUT_DIR, fn))
    _log('===== [ml] RUN DONE =====')


if __name__ == '__main__':
    main()
