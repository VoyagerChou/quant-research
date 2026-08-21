# -*- coding: utf-8 -*-
"""合并纯多头五分位结果并生成稳定性筛选表。"""
from __future__ import annotations

import os
import pandas as pd

BASE = r"D:\Quant\Quant_research\research_notes\long_only_quintile_1000"
BATCHES = ((1, 20), (21, 40), (41, 60), (61, 80), (81, 101))


def main() -> None:
    frames: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    for start, end in BATCHES:
        metrics_path = os.path.join(BASE, f"long_only_quintile_{start:03d}_{end:03d}.csv")
        diagnostics_path = os.path.join(BASE, f"factor_diagnostics_{start:03d}_{end:03d}.csv")
        if os.path.exists(metrics_path):
            frames.append(pd.read_csv(metrics_path))
        if os.path.exists(diagnostics_path):
            diagnostics.append(pd.read_csv(diagnostics_path))
    metrics = pd.concat(frames, ignore_index=True)
    diag = pd.concat(diagnostics, ignore_index=True)
    metrics.to_csv(os.path.join(BASE, "long_only_quintile_all.csv"), index=False, encoding="utf-8-sig")
    diag.to_csv(os.path.join(BASE, "factor_diagnostics_all.csv"), index=False, encoding="utf-8-sig")

    valid = metrics[metrics["period"] == "valid"]
    train = metrics[metrics["period"] == "train"]
    merged = valid.merge(
        train[["factor", "annual_return"]].rename(columns={"annual_return": "train_annual"}),
        on="factor", how="left",
    ).merge(
        diag[["factor", "train_icir", "valid_icir"]], on="factor", how="left",
    )
    merged["train_positive"] = merged["train_annual"] > 0
    merged["valid_positive"] = merged["annual_return"] > 0
    stable = merged[
        merged["train_positive"] & merged["valid_positive"]
        & (merged["valid_icir"] > 0) & (merged["train_icir"] > 0)
    ]
    stable.to_csv(os.path.join(BASE, "long_only_quintile_stable.csv"), index=False, encoding="utf-8-sig")
    print("有效因子:", len(valid), "| 稳定候选:", len(stable))
    print()
    print(stable.sort_values(["annual_return", "sharpe"], ascending=False).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
