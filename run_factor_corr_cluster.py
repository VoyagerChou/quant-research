# -*- coding: utf-8 -*-
"""纯多头稳定候选因子的相关性分析与聚类。

两个口径（均基于本地数据，2025-12-31 截止，2026 不加载）：
  1. 全截面 IC 相关：每个因子逐日 RankIC（vs 未来1日收益）序列，
     计算因子间 IC 序列的 Pearson 相关（多空时代同款口径）。
  2. Top20% 重合度：每个因子每日排名前 20% 的股票集合，
     两两集合的 Jaccard 重合率（纯多头真正关心的口径）。

聚类：基于 (1 - IC相关) 做层次聚类（Ward），输出类标签与每类代表
（类内与其余因子平均相关最高的因子）。

输出：research_notes/long_only_quintile_1000/corr_cluster/
  - ic_corr_matrix.csv          全截面IC相关矩阵
  - top20_overlap_matrix.csv    top20%重合度矩阵
  - cluster_assignment.csv      聚类结果（因子、类、代表）
"""
from __future__ import annotations

import os
import sys
from typing import Final

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\Quant\Quant_research\code")

import jqdata
import alpha101_test_core as core
from alpha101_factors import FACTOR_FUNCS
from jqdata import get_industry

INDEX_CODE: Final[str] = "000852.XSHG"
FETCH_START: Final[str] = "2014-06-01"
TRAIN_START: Final[str] = "2015-01-01"
TRAIN_END: Final[str] = "2023-12-31"
VALID_START: Final[str] = "2024-01-01"
VALID_END: Final[str] = "2025-12-31"
OUT_DIR: Final[str] = r"D:\Quant\Quant_research\research_notes\long_only_quintile_1000\corr_cluster"
N_GROUPS_TOP: Final[int] = 5   # top20% 分组（五分位最高组）

stable = pd.read_csv(
    r"D:\Quant\Quant_research\research_notes\long_only_quintile_1000\long_only_quintile_stable.csv"
)
FACTORS: Final[tuple[str, ...]] = tuple(sorted(stable["factor"].unique()))


def _load_panels() -> dict[str, pd.DataFrame]:
    """构建稳定候选因子的中性化 z-score 面板（与纯多头研究同源）。"""
    jqdata._load()
    stocks, snapshots = core.fetch_index_members(INDEX_CODE, FETCH_START, VALID_END)
    price = core.fetch_price(stocks, FETCH_START, VALID_END)
    price["returns"] = price["close"].pct_change()
    market_cap = core.fetch_market_cap(stocks, FETCH_START, VALID_END)
    log_market_cap = np.log(market_cap.replace(0, np.nan)).reindex(
        price["close"].index, columns=price["close"].columns
    )
    st = core.fetch_st(stocks, FETCH_START, VALID_END)
    member_mask = core.build_member_mask(
        snapshots, price["close"].index, stocks, VALID_END
    )
    valid_mask = core.build_valid_mask(price, st, member_mask)
    raw_industry = get_industry(stocks)
    industry = {
        code: core._extract_industry_name(value)
        for code, value in raw_industry.items()
    }
    money = price["money"]
    base = {
        key: price[key]
        for key in ("open", "close", "high", "low", "volume", "avg", "returns")
    }
    data = core._LazyData(base, money, market_cap, pd.Series(industry), pd.Series(industry))
    panels: dict[str, pd.DataFrame] = {}
    for factor in FACTORS:
        raw = FACTOR_FUNCS[factor](data)
        factor_frame = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(
            raw, index=price["close"].index, columns=price["close"].columns
        )
        factor_frame = factor_frame.where(valid_mask)
        neutral = core.neutralize(factor_frame, industry, log_market_cap)
        std = neutral.std(axis=1).replace(0, np.nan)
        panels[factor] = neutral.sub(neutral.mean(axis=1), axis=0).div(
            std, axis=0
        ).clip(-5, 5).astype(np.float32)
        print(f"{factor}: {int(panels[factor].notna().sum().sum())} values")
    # 统一列集：取所有因子共有列，避免 neutralize 剔除列不一致导致矩阵错位
    common_cols = sorted(set.intersection(*[set(p.columns) for p in panels.values()]))
    for factor in panels:
        panels[factor] = panels[factor].reindex(columns=common_cols)
    print(f"统一列集: {len(common_cols)} 只")
    return panels


def _ic_series(panel: pd.DataFrame, close: pd.DataFrame) -> pd.Series:
    """因子逐日 RankIC（vs 未来1日收益）。"""
    forward = core.forward_returns(close, 1)
    factor_rank = panel.rank(axis=1, pct=True).astype(np.float32)
    return_rank = forward.rank(axis=1, pct=True).reindex(columns=panel.columns)
    return core.rank_ic_from_ranked(factor_rank, return_rank)


def _ic_corr_matrix(panels: dict[str, pd.DataFrame], close: pd.DataFrame) -> pd.DataFrame:
    """全截面 IC 序列相关矩阵（验证期为主，训练期另存）。"""
    ics = {f: _ic_series(p, close) for f, p in panels.items()}
    ic_frame = pd.DataFrame(ics)
    matrices = {}
    for name, start, end in (
        ("train", TRAIN_START, TRAIN_END),
        ("valid", VALID_START, VALID_END),
        ("all", FETCH_START, VALID_END),
    ):
        seg = ic_frame.loc[start:end].dropna()
        matrices[name] = seg.corr(method="pearson")
    return matrices


def _top20_overlap_matrix(
    panels: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Top20% 股票集合的 Jaccard 重合率（逐日平均，矩阵乘法加速）。

    Jaccard = |A∩B| / |A∪B|；|A∪B| = |A| + |B| - |A∩B|。
    rank 产生的 NaN（非有效截面）视为不在集合且不计入并集。
    """
    names = list(panels.keys())
    n = len(names)
    # (n, T, N) 布尔矩阵：是否进入当日 top20%
    arr = np.stack([
        (p.rank(axis=1, pct=True) >= 1 - 1 / N_GROUPS_TOP).fillna(False).values
        for p in panels.values()
    ], axis=0).astype(np.float32)
    inter = np.einsum('itk,jtk->tij', arr, arr)            # (T, n, n) 交集大小
    size = arr.sum(axis=2)                                  # (n, T) 各因子集合大小
    # |A∪B| = |A| + |B| - |A∩B|，广播: size_a[t,i] + size_b[t,j] - inter[t,i,j]
    union = size.T[:, :, None] + size.T[:, None, :] - inter
    union_safe = np.where(union == 0, np.nan, union)
    jac = np.nanmean(inter / union_safe, axis=0)             # (n, n)，跳过空集合日
    return pd.DataFrame(jac, index=names, columns=names)


def _cluster(ic_corr: pd.DataFrame) -> pd.DataFrame:
    """基于 (1 - IC相关) 的层次聚类，输出类标签与代表因子。"""
    from scipy.cluster.hierarchy import fcluster, linkage

    names = list(ic_corr.columns)
    dist = np.clip(1 - ic_corr.values, 0, None)
    np.fill_diagonal(dist, 0)
    condensed = dist[np.triu_indices(len(names), k=1)]
    z = linkage(condensed, method="ward")
    # 类数：距离阈值 1.0（相关 > 0 即认为可归并的宽松标准）
    labels = fcluster(z, t=1.0, criterion="distance")
    rows = []
    for i, name in enumerate(names):
        rows.append({"factor": name, "cluster": int(labels[i])})
    assign = pd.DataFrame(rows)
    # 每类代表：类内与其余成员平均 IC 相关最高者
    reps: list[dict[str, object]] = []
    for cid in sorted(assign["cluster"].unique()):
        members = assign[assign["cluster"] == cid]["factor"].tolist()
        if len(members) == 1:
            reps.append({"cluster": int(cid), "n_members": 1, "representative": members[0]})
            continue
        sub = ic_corr.loc[members, members]
        avg_corr = sub.mean(axis=1)
        reps.append({
            "cluster": int(cid),
            "n_members": len(members),
            "representative": avg_corr.idxmax(),
        })
    rep_frame = pd.DataFrame(reps)
    assign = assign.merge(rep_frame, on="cluster", how="left")
    return assign


def main() -> None:
    """计算相关性矩阵并聚类。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    panels = _load_panels()
    close = jqdata._price["close"]

    print("== 计算全截面 IC 相关矩阵 ==")
    ic_matrices = _ic_corr_matrix(panels, close)
    for name, mat in ic_matrices.items():
        mat.to_csv(os.path.join(OUT_DIR, f"ic_corr_{name}.csv"), encoding="utf-8-sig")
        print(f"ic_corr_{name}: {mat.shape}")

    print("== 计算 Top20% 重合度矩阵 ==")
    overlap = _top20_overlap_matrix(panels)
    overlap.to_csv(os.path.join(OUT_DIR, "top20_overlap.csv"), encoding="utf-8-sig")
    print("top20_overlap:", overlap.shape)

    print("== 层次聚类（基于验证期 IC 相关）==")
    assign = _cluster(ic_matrices["valid"])
    assign.to_csv(os.path.join(OUT_DIR, "cluster_assignment.csv"), index=False, encoding="utf-8-sig")
    print(assign.to_string(index=False))

    print("\n== 高重合度因子对（Top20口径，验证期）==")
    pairs = (
        overlap.where(np.triu(np.ones(overlap.shape, dtype=bool), k=1))
        .stack()
        .sort_values(ascending=False)
    )
    print(pairs.head(15).round(3).to_string())


if __name__ == "__main__":
    main()
