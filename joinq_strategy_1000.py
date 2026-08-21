# -*- coding: utf-8 -*-
"""
中证1000 · 14因子等权 · 纯多10只 · 5日调仓（聚宽回测策略 v3）
============================================================
规格书：《策略规格书_1000池14因子等权_v1.md》

v3 重大变更：预计算信号文件
  - 信号（comp_eq）已本地预计算为 comp_eq_signal.npz（含动态成分掩码、
    流动性≥2000万、高价≤25元等全部可交易性过滤），回测只查表
  - 回测不再拉取行情/市值/ST 数据，性能从"每5天算14因子"变为"查一行"
  - 信号是截面排名，对复权基准不敏感；价格用 use_real_price 实时获取，
    无未来函数

策略要点：
  - 股票池：中证1000 动态成分（每月 get_index_stocks）
  - 持仓：前10名等权（每只约10%），纯多满仓
  - 调仓：每5个交易日；缓冲带（持仓排名>15才卖、新股排名<10才买）
  - 约束：同行业最多2只（get_industry）
  - 风控：个股止损 -12%；组合回撤≥10%清仓、创新高恢复满仓
  - 成本：佣金万2.5 + 印花税0.05%（卖出）+ 滑点0.2%

上传聚宽文件：
  comp_eq_signal.npz  （信号文件，必须与本策略同目录）
  本文件（策略页粘贴本文件全部代码）

回测纪律：研究回测 ≤ 2025-12-31；2026 仅模拟盘最终验证。
"""
import sys
import types as _types
import numpy as np
import pandas as pd
try:
    from jqdata import *
except Exception:
    pass  # 回测环境 API 全局注入
import comp_eq_signal_data   # 预计算信号数据模块（base64 嵌入 npz）

# 防御性获取 numpy（聚宽环境污染 np 名字绑定的兜底）
_np_mod = sys.modules.get('numpy')
if _np_mod is None or not isinstance(_np_mod, _types.ModuleType):
    _np_mod = __import__('numpy')
np = _np_mod

# ==================== 配置区（按规格书定稿，禁止再调） ====================
INDEX_CODE = '000852.XSHG'      # 中证1000
N_HOLD = 10                     # 持仓数量
HOLD_DAYS = 5                   # 固定持有期（交易日）
SELL_RANK = 15                  # 缓冲带：持仓排名跌出前15名才卖
BUY_RANK = 10                   # 缓冲带：新股进入前10名才买
MAX_SAME_IND = 2                # 同行业最多2只
STOP_LOSS = 0.12                # 个股止损 -12%
DRAWDOWN_LIMIT = 0.10           # 组合回撤熔断线 10%
SLIPPAGE = 0.002                # 滑点 0.2%
COST_COMMISSION = 0.00025       # 佣金万2.5
COST_STAMP = 0.0005             # 印花税 0.05%（卖出）
MIN_COMMISSION = 5.0            # 最低佣金 5元


def initialize(context):
    set_benchmark(INDEX_CODE)
    # 标准设置：防未来函数 + 真实价格
    set_option('avoid_future_data', True)
    set_option('use_real_price', True)
    # 成本：佣金万2.5（最低5元）+ 印花税 0.05%（卖出）
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=COST_STAMP,
        open_commission=COST_COMMISSION,
        close_commission=COST_COMMISSION,
        close_today_commission=0,
        min_commission=MIN_COMMISSION,
    ), type='stock')
    # 滑点 0.2%（百分比滑点，与小盘股保守设定一致）
    set_slippage(PriceRelatedSlippage(SLIPPAGE), type='stock')
    # 日志级别：订单/系统仅错误，策略保留信息
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    log.set_level('strategy', 'info')

    # ---- 全局状态 ----
    g.sig = None                # 信号 DataFrame（日期×股票，NaN=不可交易）
    g.members_month = None      # 当月成分对应的月份
    g.members_now = None        # 当月成分（调仓选股用）
    g.ind_map = {}              # 股票→行业名
    g.rebalance_count = 0       # 已调仓次数（每5个交易日）
    g.dd_triggered = False      # 熔断状态
    g.peak_value = 0.0          # 净值峰值

    # 加载预计算信号
    _load_signal()

    run_daily(rebalance, time='every_bar')


# ==================== 信号与数据 ====================
def _load_signal():
    """从数据模块加载预计算信号（import 方式，回测环境同目录可用）。
    信号已含：动态成分掩码、ST/价格过滤、流动性≥2000万、高价≤25元。"""
    g.sig = comp_eq_signal_data.load_dataframe()[1]
    log.info('信号加载: %d 日 × %d 股 (%s ~ %s)'
             % (len(g.sig), g.sig.shape[1], g.sig.index[0].date(), g.sig.index[-1].date()))


def _update_members(context):
    """每月更新当日成分 + 行业映射（行业只需一次）"""
    dt = context.current_dt
    ym = (dt.year, dt.month)
    if g.members_month != ym:
        g.members_month = ym
        g.members_now = get_index_stocks(INDEX_CODE, dt.strftime('%Y-%m-%d'))
        if not g.ind_map:
            raw_ind = get_industry(g.members_now)
            for k, v in raw_ind.items():
                try:
                    g.ind_map[k] = v.get('sw_l1', {}).get('industry_name', None)
                except Exception:
                    g.ind_map[k] = None
            log.info('行业映射: %d 只' % len(g.ind_map))
    return g.members_now


# ==================== 调仓主逻辑 ====================
def rebalance(context):
    dt = context.current_dt
    members = _update_members(context)

    # ---- 熔断检查（每日）----
    total = context.portfolio.total_value
    if total > g.peak_value:
        g.peak_value = total
    dd = (g.peak_value - total) / g.peak_value if g.peak_value > 0 else 0
    if g.dd_triggered:
        if dd <= 0.001:  # 创新高 → 恢复满仓
            g.dd_triggered = False
            log.info('熔断恢复: 净值创新高，恢复满仓')
    elif dd >= DRAWDOWN_LIMIT:
        g.dd_triggered = True
        log.warn('熔断触发: 回撤 %.1f%% ≥ 10%%，清仓等待恢复' % (dd * 100))
        for stock in list(context.portfolio.positions):
            order_target(stock, 0)
        return

    # ---- 个股止损（每日，-12%）----
    cd = get_current_data()
    for stock in list(context.portfolio.positions):
        p = context.portfolio.positions[stock]
        if p.total_amount > 0 and stock in cd:
            cur = cd[stock].last_price
            if cur > 0 and cur / p.avg_cost - 1 <= -STOP_LOSS:
                order_target(stock, 0)
                log.info('止损: %s 跌幅 %.1f%%' % (stock, (cur / p.avg_cost - 1) * 100))

    # ---- 调仓日：每5个交易日 ----
    g.rebalance_count += 1
    if (g.rebalance_count - 1) % HOLD_DAYS != 0:
        return
    _do_rebalance(context, dt, cd, members)


def _signal_on_date(dstr):
    """取指定日期的信号行；无该日数据时取最近的前一有效日（防未来函数）。"""
    ts = pd.Timestamp(dstr)
    valid = g.sig.index[g.sig.index <= ts]
    if len(valid) == 0:
        return None
    return g.sig.loc[valid[-1]]


def _do_rebalance(context, dt, cd, members):
    """调仓：信号查表 + 排名选股 + 缓冲带 + 约束"""
    # 调仓日开盘前，用上一交易日收盘信号（信号文件日期即收盘日）
    prev = _prev_trade_date(dt.strftime('%Y-%m-%d'))
    sig = _signal_on_date(prev)
    if sig is None:
        log.warn('调仓日无信号（%s），跳过' % prev)
        return
    sig = sig.dropna()
    if len(sig) == 0:
        log.warn('调仓日信号全空，跳过')
        return

    # 仅从当日成分中选（信号文件是全历史并集，这里用当月成分）
    members_set = set(members)
    sig = sig[[s for s in sig.index if s in members_set]]

    # 排名（1=最强）
    ranks = sig.rank(ascending=False)
    ranked = ranks.sort_values()

    # 缓冲带：持仓且排名 ≤ SELL_RANK 的继续持有
    positions = {s for s in context.portfolio.positions
                 if context.portfolio.positions[s].total_amount > 0}
    keep = {s for s in positions if s in ranked.index and ranked[s] <= SELL_RANK}

    # 新买入：排名 ≤ BUY_RANK，行业约束（同行业最多2只）
    to_buy = []
    ind_count = {}
    for s in ranked.index:
        if len(keep) + len(to_buy) >= N_HOLD:
            break
        if s in keep or s in to_buy:
            continue
        ind = g.ind_map.get(s)
        if ind is not None and ind_count.get(ind, 0) >= MAX_SAME_IND:
            continue
        to_buy.append(s)
        if ind is not None:
            ind_count[ind] = ind_count.get(ind, 0) + 1

    target = list(keep) + to_buy
    # 卖出不在目标中的
    for s in positions - set(target):
        order_target(s, 0)
        log.info('调出: %s' % s)
    # 买入目标（等权；已持仓的补齐差额）
    if target:
        per = context.portfolio.total_value / len(target)
        for s in target:
            order_target_value(s, per)
    log.info('调仓 %s: 目标 %d 只（保留 %d / 新买 %d）'
             % (dt.strftime('%Y-%m-%d'), len(target), len(keep), len(to_buy)))


def _prev_trade_date(dstr):
    """上一交易日。get_trade_days(start,end) 含首尾：
    若 dstr 是交易日，days[-1]=dstr，需取 days[-2]；
    若 dstr 非交易日（周末/假期），days[-1] 即最后交易日。"""
    import datetime
    d = pd.Timestamp(dstr)
    days = get_trade_days(start_date=(d - datetime.timedelta(days=10)).strftime('%Y-%m-%d'),
                          end_date=dstr)
    if len(days) < 2:
        return dstr
    return days[-2].strftime('%Y-%m-%d') if days[-1] == pd.Timestamp(dstr) \
        else days[-1].strftime('%Y-%m-%d')
