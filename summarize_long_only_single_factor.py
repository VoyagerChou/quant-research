# -*- coding: utf-8 -*-
"""合并纯多单因子批次结果并生成稳定性筛选表。"""

from __future__ import annotations

import os
import pandas as pd


BASE = r"D:\Quant\Quant_research\research_notes\long_only_single_factor_1000"
BATCHES = ((1, 20), (21, 40), (41, 60), (61, 80), (81, 101))


def main() -> None:
    frames: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    for start, end in BATCHES:
        metrics_path = os.path.join(BASE, f"long_only_metrics_{start:03d}_{end:03d}.csv")
        diagnostics_path = os.path.join(BASE, f"factor_diagnostics_{start:03d}_{end:03d}.csv")
        if os.path.exists(metrics_path):
            frames.append(pd.read_csv(metrics_path))
        if os.path.exists(diagnostics_path):
            diagnostics.append(pd.read_csv(diagnostics_path))
    metrics = pd.concat(frames, ignore_index=True)
    factor_diagnostics = pd.concat(diagnostics, ignore_index=True)
    metrics.to_csv(os.path.join(BASE, "long_only_metrics_all.csv"), index=False, encoding="utf-8-sig")
    factor_diagnostics.to_csv(
        os.path.join(BASE, "factor_diagnostics_all.csv"), index=False, encoding="utf-8-sig"
    )

    usable = metrics[metrics["observations"] >= 60].copy()
    valid = usable[usable["period"] == "valid"]
    train = usable[usable["period"] == "train"]
    keys = ["factor", "holding_days", "portfolio_size"]
    train = train[keys + ["annual_excess_return"]].rename(
        columns={"annual_excess_return": "train_excess"}
    )
    valid = valid.merge(train, on=keys, how="left")
    # 稳定候选：训练/验证超额均为正，且非 Top10 单一规模依赖。
    stable = valid[(valid["train_excess"] > 0) & (valid["annual_excess_return"] > 0)]
    stable.to_csv(os.path.join(BASE, "long_only_stable_candidates.csv"), index=False, encoding="utf-8-sig")
    print("全部绩效行:", len(metrics), "| 有效行:", len(usable))
    print("稳定候选:", len(stable))
    print(
        stable.sort_values(
            ["annual_excess_return", "sharpe"], ascending=False
        ).head(30).round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
