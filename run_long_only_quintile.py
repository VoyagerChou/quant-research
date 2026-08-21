# -*- coding: utf-8 -*-
"""纯多头 Alpha101 单因子研究管线（研究环境同源口径）。

与多空研究的唯一差异：只看最高分组（第5组，前20%）的净值表现，
不做任何策略层假设（无开盘建仓、无成本、无基准对齐）。

口径（与 alpha101_test_core 完全一致）：
* 因子 t 日收盘可得，t+1 日收益（close-to-close）
* 动态成分掩码、ST/价格过滤、行业+市值中性化、截面 z-score
* 分组：五分位，第5组 = 纯多头
* 方向：训练期(2015-2023) IC 均值定符号；验证期(2024-2025) 固定只测一次
* 2026 不加载

输出：research_notes/long_only_quintile_1000/
"""
from __future__ import annotations

import io
import os
import sys
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
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
FACTOR_START: Final[int] = int(os.environ.get("LONG_ONLY_FACTOR_START", "1"))
FACTOR_END: Final[int] = int(os.environ.get("LONG_ONLY_FACTOR_END", "101"))
ACTIVE_FACTORS: Final[tuple[str, ...]] = tuple(
    f"alpha{i:03d}" for i in range(FACTOR_START, FACTOR_END + 1)
)
TOP_GROUP: Final[int] = 5   # 纯多头 = 最高分组


@dataclass(frozen=True, slots=True)
class LongOnlyResult:
    """一个因子的纯多头绩效。"""

    factor: str
    period: str
    direction: int
    annual_return: float
    sharpe: float
    max_drawdown: float
    monthly_win_rate: float
    observations: int


def _load_and_build() -> tuple[
    dict[str, pd.DataFrame], pd.DataFrame, pd.Series, pd.DataFrame
]:
    """加载数据并计算中性化 z-score 因子面板。"""
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
    for factor in ACTIVE_FACTORS:
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
    return panels, price["close"], pd.Series(industry), valid_mask


def _direction_from_train(panel: pd.DataFrame, close: pd.DataFrame) -> int:
    """只用训练期一日 RankIC 均值决定方向。"""
    forward = core.forward_returns(close, 1)
    factor_rank = panel.rank(axis=1, pct=True).astype(np.float32)
    return_rank = forward.rank(axis=1, pct=True).reindex(columns=panel.columns)
    ic = core.rank_ic_from_ranked(factor_rank, return_rank)
    mean_ic = ic.loc[TRAIN_START:TRAIN_END].mean()
    return 1 if mean_ic >= 0 else -1


def _top_group_net(
    panel: pd.DataFrame, close: pd.DataFrame, direction: int
) -> pd.Series:
    """最高分组（前20%）的累计净值，t+1 日收益口径。"""
    score = panel * direction
    pct = score.rank(axis=1, pct=True).astype(np.float32)
    ret1 = core.forward_returns(close, 1)
    group = np.ceil(pct * core.N_GROUPS).where(pct.notna())
    top_return = ret1.where(group == TOP_GROUP).mean(axis=1)
    return (1 + top_return.fillna(0)).cumprod()


def _result(
    net: pd.Series,
    factor: str,
    direction: int,
    period: str,
) -> LongOnlyResult:
    """训练/验证分段绩效。"""
    if period == "train":
        start, end = TRAIN_START, TRAIN_END
    else:
        start, end = VALID_START, VALID_END
    seg = net.loc[start:end].dropna()
    if len(seg) < 60:
        return LongOnlyResult(factor, period, direction, np.nan, np.nan, np.nan, np.nan, len(seg))
    returns = seg.pct_change().dropna()
    # 区间内年化：用区间首尾净值（net 是 2014 年起累计净值，不能拿全局末值当区间末值）
    annual = (seg.iloc[-1] / seg.iloc[0]) ** (252 / len(seg)) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    mdd = (seg / seg.cummax() - 1).min()
    monthly = (1 + returns).resample("ME").prod() - 1
    return LongOnlyResult(
        factor=factor,
        period=period,
        direction=direction,
        annual_return=float(annual),
        sharpe=float(sharpe),
        max_drawdown=float(mdd),
        monthly_win_rate=float((monthly > 0).mean()),
        observations=len(seg),
    )


def main() -> None:
    """运行纯多头单因子研究并保存 CSV。"""
    panels, close, _, _ = _load_and_build()
    diagnostics: list[dict[str, float | str]] = []
    results: list[LongOnlyResult] = []
    for factor, panel in panels.items():
        direction = _direction_from_train(panel, close)
        forward = core.forward_returns(close, 1)
        ic = core.rank_ic_from_ranked(
            panel.rank(axis=1, pct=True).astype(np.float32),
            forward.rank(axis=1, pct=True).reindex(columns=panel.columns),
        )
        diagnostics.append({
            "factor": factor,
            "direction": direction,
            "train_ic_mean": float(ic.loc[TRAIN_START:TRAIN_END].mean()),
            "train_icir": float(ic.loc[TRAIN_START:TRAIN_END].mean() / ic.loc[TRAIN_START:TRAIN_END].std()),
            "valid_ic_mean": float(ic.loc[VALID_START:VALID_END].mean()),
            "valid_icir": float(ic.loc[VALID_START:VALID_END].mean() / ic.loc[VALID_START:VALID_END].std()),
        })
        net = _top_group_net(panel, close, direction)
        for period in ("train", "valid"):
            results.append(_result(net, factor, direction, period))
    out_dir = r"D:\Quant\Quant_research\research_notes\long_only_quintile_1000"
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{FACTOR_START:03d}_{FACTOR_END:03d}"
    pd.DataFrame(diagnostics).to_csv(
        os.path.join(out_dir, f"factor_diagnostics{suffix}.csv"), index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([asdict(r) for r in results]).to_csv(
        os.path.join(out_dir, f"long_only_quintile{suffix}.csv"), index=False, encoding="utf-8-sig"
    )
    print("写入:", out_dir)
    valid_results = [r for r in results if r.period == "valid"]
    ranked = sorted(valid_results, key=lambda r: r.annual_return, reverse=True)
    print(pd.DataFrame([asdict(r) for r in ranked]).head(25).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
