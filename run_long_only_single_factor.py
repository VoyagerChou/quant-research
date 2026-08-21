# -*- coding: utf-8 -*-
"""纯多头 Alpha101 单因子研究管线。

研究口径：
* 只使用 CSI1000 动态成分、ST/价格有效股票；2026 不加载。
* 训练期 2015-2023 只用于确定因子方向；验证期 2024-2025 固定方向后只测一次。
* 非重叠持有队列：信号日收盘排序，下一交易日开盘建仓，固定持有 3/5 个交易日。
* 组合规模 Top10/20/50/100；输出纯多净收益、基准超额、风险和换手。

输出到 D:\\Quant\\Quant_research\\research_notes\\long_only_single_factor_1000\\。
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
FACTORS: Final[tuple[str, ...]] = tuple(
    f"alpha{i:03d}" for i in range(1, 102)
)
FACTOR_START: Final[int] = int(os.environ.get("LONG_ONLY_FACTOR_START", "1"))
FACTOR_END: Final[int] = int(os.environ.get("LONG_ONLY_FACTOR_END", "101"))
ACTIVE_FACTORS: Final[tuple[str, ...]] = tuple(
    f"alpha{i:03d}" for i in range(FACTOR_START, FACTOR_END + 1)
)
HOLDING_DAYS: Final[tuple[int, ...]] = (3, 5)
PORTFOLIO_SIZES: Final[tuple[int, ...]] = (10, 20, 50, 100)

# 研究成本：买卖佣金各万2.5，卖出印花税千0.5，单边滑点0.1%。
OPEN_COMMISSION: Final[float] = 0.00025
CLOSE_COMMISSION: Final[float] = 0.00025
CLOSE_TAX: Final[float] = 0.0005
SLIPPAGE: Final[float] = 0.001
INITIAL_CAPITAL: Final[float] = 100_000.0


@dataclass(frozen=True, slots=True)
class PortfolioResult:
    """一个因子、一个持有期和一个组合规模的绩效结果。"""

    factor: str
    holding_days: int
    portfolio_size: int
    period: str
    direction: int
    annual_return: float
    annual_excess_return: float
    sharpe: float
    max_drawdown: float
    monthly_win_rate: float
    daily_turnover: float
    observations: int


def _load_market_data() -> tuple[
    dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame
]:
    """加载 2025 年底前的行情、市值、有效掩码和行业映射。"""
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
    return price, market_cap, log_market_cap, pd.Series(industry), valid_mask


def _build_factor_panels(
    price: dict[str, pd.DataFrame],
    market_cap: pd.DataFrame,
    log_market_cap: pd.DataFrame,
    valid_mask: pd.DataFrame,
    industry: pd.Series,
) -> dict[str, pd.DataFrame]:
    """逐因子计算中性化 z-score 面板，严格截至 2025-12-31。"""
    close = price["close"]
    money = price["money"]
    base = {key: price[key] for key in (
        "open", "close", "high", "low", "volume", "avg", "returns"
    )}
    data = core._LazyData(base, money, market_cap, industry, industry)
    panels: dict[str, pd.DataFrame] = {}
    for factor in ACTIVE_FACTORS:
        raw = FACTOR_FUNCS[factor](data)
        factor_frame = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(
            raw, index=close.index, columns=close.columns
        )
        factor_frame = factor_frame.where(valid_mask)
        neutral = core.neutralize(factor_frame, industry.to_dict(), log_market_cap)
        std = neutral.std(axis=1).replace(0, np.nan)
        panels[factor] = neutral.sub(neutral.mean(axis=1), axis=0).div(
            std, axis=0
        ).clip(-5, 5).astype(np.float32)
        print(f"{factor}: {int(panels[factor].notna().sum().sum())} values")
    return panels


def _direction_from_train(panel: pd.DataFrame, close: pd.DataFrame) -> int:
    """只用训练期一日 RankIC 均值决定方向，验证期不反转。"""
    forward = core.forward_returns(close, 1)
    factor_rank = panel.rank(axis=1, pct=True).astype(np.float32)
    return_rank = forward.rank(axis=1, pct=True).reindex(columns=panel.columns)
    ic = core.rank_ic_from_ranked(factor_rank, return_rank)
    mean_ic = ic.loc[TRAIN_START:TRAIN_END].mean()
    return 1 if mean_ic >= 0 else -1


def _rebalance_dates(index: pd.DatetimeIndex, holding_days: int) -> pd.DatetimeIndex:
    """按固定持有期构造不重叠队列，信号日为可观测收盘日。"""
    usable = index[index >= FETCH_START]
    return usable[::holding_days]


def _portfolio_returns(
    signal: pd.DataFrame,
    open_price: pd.DataFrame,
    close: pd.DataFrame,
    factor: str,
    direction: int,
    holding_days: int,
    portfolio_size: int,
    benchmark_daily: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """生成非重叠纯多组合收益与单边换手序列。

    每个队列在信号日 t 后的下一个交易日开盘建仓，持有 holding_days 日，
    下一队列从到期后的下一个交易日开始，避免重叠持仓和隐含日内换手。
    """
    score = signal * direction
    trade_dates = _rebalance_dates(signal.index, holding_days)
    returns: list[float] = []
    benchmark_returns: list[float] = []
    turnovers: list[float] = []
    result_dates: list[pd.Timestamp] = []
    previous: set[str] = set()
    for signal_date in trade_dates:
        signal_pos = signal.index.get_loc(signal_date)
        entry_pos = signal_pos + 1
        exit_pos = entry_pos + holding_days
        if exit_pos > len(signal.index):
            continue
        entry_date = signal.index[entry_pos]
        exit_date = signal.index[exit_pos]
        row = score.loc[signal_date].dropna().sort_values(ascending=False)
        selected = set(row.head(portfolio_size).index)
        available = selected & set(open_price.columns) & set(close.columns)
        available = {
            stock for stock in available
            if np.isfinite(open_price.loc[entry_date, stock])
            and open_price.loc[entry_date, stock] > 0
            and np.isfinite(close.loc[exit_date, stock])
            and close.loc[exit_date, stock] > 0
        }
        if not available:
            continue
        period_returns = close.loc[exit_date, list(available)].values / open_price.loc[
            entry_date, list(available)
        ].values - 1
        gross = float(np.nanmean(period_returns))
        benchmark_segment = benchmark_daily.loc[entry_date:exit_date]
        benchmark_returns.append(float((1 + benchmark_segment.iloc[1:]).prod() - 1))
        buys = len(available - previous)
        sells = len(previous - available)
        denominator = max(len(previous), len(available), 1)
        turnover = (buys + sells) / (2 * denominator)
        cost = turnover * (OPEN_COMMISSION + CLOSE_COMMISSION + CLOSE_TAX + 2 * SLIPPAGE)
        returns.append(gross - cost)
        turnovers.append(turnover)
        result_dates.append(exit_date)
        previous = available
    result_index = pd.DatetimeIndex(result_dates)
    return (
        pd.Series(returns, index=result_index),
        pd.Series(turnovers, index=result_index),
        pd.Series(benchmark_returns, index=result_index),
    )


def _benchmark_returns(
    close: pd.DataFrame,
    market_cap: pd.DataFrame,
    valid_mask: pd.DataFrame,
) -> pd.Series:
    """构造动态有效成分的市值加权基准日收益。"""
    weights = market_cap.where(valid_mask.reindex(columns=market_cap.columns))
    weights = weights.div(weights.sum(axis=1), axis=0)
    return (close.pct_change() * weights).sum(axis=1)


def _metrics(
    returns: pd.Series,
    benchmark: pd.Series,
    period: str,
    factor: str,
    direction: int,
    holding_days: int,
    portfolio_size: int,
    turnover: pd.Series,
) -> PortfolioResult:
    """计算纯多组合及相对基准绩效。"""
    if period == "train":
        start, end = TRAIN_START, TRAIN_END
    else:
        start, end = VALID_START, VALID_END
    segment = returns.loc[start:end].dropna()
    base = benchmark.reindex(segment.index).fillna(0)
    if len(segment) == 0:
        values = (np.nan,) * 5
        return PortfolioResult(factor, holding_days, portfolio_size, period, direction,
                               values[0], values[1], values[2], values[3], values[4],
                               np.nan, 0)
    wealth = (1 + segment).cumprod()
    base_wealth = (1 + base).cumprod()
    annual_return = wealth.iloc[-1] ** (252 / (len(segment) * holding_days)) - 1
    benchmark_return = base_wealth.iloc[-1] ** (252 / (len(segment) * holding_days)) - 1
    daily_std = segment.std()
    sharpe = (
        segment.mean() / daily_std * np.sqrt(252 / holding_days)
        if daily_std > 0 else np.nan
    )
    max_drawdown = (wealth / wealth.cummax() - 1).min()
    monthly = (1 + segment).resample("ME").prod() - 1
    return PortfolioResult(
        factor=factor,
        holding_days=holding_days,
        portfolio_size=portfolio_size,
        period=period,
        direction=direction,
        annual_return=float(annual_return),
        annual_excess_return=float(annual_return - benchmark_return),
        sharpe=float(sharpe),
        max_drawdown=float(max_drawdown),
        monthly_win_rate=float((monthly > 0).mean()),
        daily_turnover=float(turnover.loc[start:end].mean()) if len(turnover) else np.nan,
        observations=len(segment),
    )


def main() -> None:
    """运行训练/验证纯多单因子研究并保存 CSV。"""
    price, market_cap, log_market_cap, industry, valid_mask = _load_market_data()
    close = price["close"]
    panels = _build_factor_panels(price, market_cap, log_market_cap, valid_mask, industry)
    open_price = jqdata._price["open"].reindex(index=close.index, columns=close.columns)
    benchmark = _benchmark_returns(close, market_cap, valid_mask)
    diagnostics: list[dict[str, float | str]] = []
    results: list[PortfolioResult] = []
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
        for holding_days in HOLDING_DAYS:
            for portfolio_size in PORTFOLIO_SIZES:
                returns, turnover, period_benchmark = _portfolio_returns(
                    panel, open_price, close, factor, direction, holding_days,
                    portfolio_size, benchmark
                )
                for period in ("train", "valid"):
                    results.append(_metrics(
                        returns, period_benchmark, period, factor, direction,
                        holding_days, portfolio_size, turnover
                    ))
    out_dir = r"D:\Quant\Quant_research\research_notes\long_only_single_factor_1000"
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{FACTOR_START:03d}_{FACTOR_END:03d}"
    pd.DataFrame(diagnostics).to_csv(
        os.path.join(out_dir, f"factor_diagnostics{suffix}.csv"), index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([asdict(r) for r in results]).to_csv(
        os.path.join(out_dir, f"long_only_metrics{suffix}.csv"), index=False, encoding="utf-8-sig"
    )
    print("写入:", out_dir)
    print(pd.DataFrame([asdict(r) for r in results if r.period == "valid"])
          .sort_values("annual_excess_return", ascending=False)
          .head(20).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
