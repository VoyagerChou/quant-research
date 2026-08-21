# -*- coding: utf-8 -*-
"""
Alpha101 前 10 因子单因子检验 —— 核心逻辑模块
=============================================
【为什么逻辑放在模块里 + 为什么要清理 sys.modules】
聚宽研究环境（旧版，Python3.6）启动时会把 sys.modules['numpy'] 替换成
字符串，导致 import numpy 拿到 str（报 "'str' object has no attribute 'log'"）。
本模块在导入区先清理被污染的非模块条目再重新导入（见下方代码），
并把全部逻辑放在模块命名空间内，入口脚本 run_alpha101_first10_test.py
只负责调用，最大程度隔离平台环境问题。

【功能】
  1) 多周期 RankIC 检验（持有周期 1/2/3/4/5 个交易日）
  2) 五分组分层回测 + 多空组合（因子 t 日收盘可得，t+1 日成交口径）
  3) 两个口径：原始因子 / 行业+市值中性化因子
  4) 训练期(2015-2023) + 三段切片(分时段稳定性) / 验证期(2024-2025) 分开展示
  5) 动态指数成分（逐月快照 + 前向填充），消除静态成分股的幸存者偏差

【使用方法（分批跑 101 个因子，防内存爆）】
  1. 将 alpha101_operators.py、alpha101_factors.py、本文件、
     run_alpha101_first10_test.py 四个文件上传到聚宽研究环境（同一目录）
  2. 修改本文件【配置】区的 FACTOR_START / FACTOR_END（如 11, 20），运行入口脚本
  3. 每批结果存到独立目录 results_alpha011_020 / results_alpha021_030 / ...
     互不覆盖，全部跑完后一起打包下载
  4. 批次建议：11~20, 21~30, ..., 91~101（前 10 个已完成）

【当前简化处理（第一版，先看信号强度）】
  - 分层回测未扣交易成本、未做涨停/停牌可交易性过滤（IC 检验为第一参考）
  - 行业映射使用当前申万行业（历史上行业会变更，影响较小）
  - 因子在"历史成员并集"上计算，横截面按当期成员掩码筛选（排名口径
    近似于成员内排名，差异很小）
"""
import os
import gc
import sys
import types as _types
import pandas as pd
import matplotlib.pyplot as plt
# ==== 绕过旧版聚宽研究环境对 `import numpy` 语句的改写 ====
# 该环境会把用户代码（单元格和上传的 .py 文件）中的 `import numpy` 语句
# 改写成绑定到字符串，导致 "'str' object has no attribute 'log'"；
# 而 sys.modules['numpy'] 本身仍是真实模块（已用诊断脚本验证）。
# 因此不用 import 语句，直接从 sys.modules 取真实模块对象。
_np_mod = sys.modules.get('numpy')
if _np_mod is None or not isinstance(_np_mod, _types.ModuleType):
    _np_mod = __import__('numpy')          # 兜底：真正加载一次
np = _np_mod
try:
    from jqdata import *
except Exception:
    pass
# ==== 回测环境 API 兜底 ====
# 聚宽回测（jqboson）把 API 注入主策略模块（user_code）的全局命名空间，
# 上传的辅助模块经 import 加载时看不到这些全局函数。兜底策略：
#   1) 从 __main__ 模块取（若用户代码以 __main__ 运行）
#   2) 扫描 sys.modules，找任意持有该 API 的模块（user_code 等）
# 研究环境 `from jqdata import *` 已提供 API，本兜底不会覆盖。
_API_NAMES = ('get_price', 'get_index_stocks', 'get_valuation', 'get_extras',
              'get_industry', 'get_trade_days')
import __main__ as _main_mod
for _api in _API_NAMES:
    if _api in globals():
        continue
    _f = getattr(_main_mod, _api, None)
    if _f is None:
        for _mod in list(sys.modules.values()):
            try:
                _f = getattr(_mod, _api, None)
            except Exception:
                _f = None
            if _f is not None:
                break
    if _f is not None:
        globals()[_api] = _f
from alpha101_factors import FACTOR_FUNCS

plt.rcParams['font.sans-serif'] = ['SimHei']      # 中文显示
plt.rcParams['axes.unicode_minus'] = False

# ------------------------- 配置 -------------------------
INDEX_CODE  = '000300.XSHG'                 # 股票池指数：沪深300（动态成分，见 fetch_index_members）
FETCH_START = '2014-06-01'                  # 数据起点（预热：60日窗口 + 120日次新过滤）
TRAIN_START, TRAIN_END = '2015-01-01', '2023-12-31'   # 训练期（9年，覆盖多轮风格切换）
VALID_START, VALID_END = '2024-01-01', '2025-12-31'   # 验证期
HORIZONS   = [1, 2, 3, 4, 5]                # IC 检验的持有周期（交易日）
N_GROUPS   = 5                              # 分层回测分组数
SEC_BATCH  = 150                            # 拉行情时分批次（防单次请求过大）
MIN_HIST   = 120                            # 剔除上市不满 120 个交易日的次新

# ===== 本次运行的因子范围（手动分批跑，防内存爆）=====
# 本地运行时内存无压力，直接设 1, 101 一批跑完；聚宽上跑才需要分批
FACTOR_START, FACTOR_END = 1, 101
# 结果目录带指数代码，避免不同指数的结果互相覆盖
RESULT_DIR = 'results_alpha%03d_%03d_%s' % (FACTOR_START, FACTOR_END, INDEX_CODE.split('.')[0])
SKIP_PLOTS = True      # 跳过出图（本地大批量跑时提速；需要看图再设 False）

# 训练期切片（分时段稳定性检验：因子需在每段 ICIR 均为正）
TRAIN_SLICES = [
    ('train_p1_2015_17', '2015-01-01', '2017-12-31'),
    ('train_p2_2018_20', '2018-01-01', '2020-12-31'),
    ('train_p3_2021_23', '2021-01-01', '2023-12-31'),
]


# ------------------------- 数据获取 -------------------------
def _log(msg):
    """把诊断信息追加写入 results/debug_log.txt（用于远程排错）"""
    try:
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(os.path.join(RESULT_DIR, 'debug_log.txt'), 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')
    except Exception:
        pass


def fetch_price(stocks, start, end):
    """分股票批次拉日线行情。返回 dict: 字段 -> DataFrame(日期×股票)，停牌日为 NaN。

    兼容性说明：聚宽研究环境（旧版，Python3.6 / pandas 0.23）中
    get_price(panel=False) 对多标的的返回结构有问题（日期混入数据区，
    dtype 变成 datetime64，导致 pct_change 报错）。
    因此改用 panel=True（旧环境原生 Panel 格式），fill_paused=False 让停牌日为 NaN。
    """
    fields = ['open', 'close', 'high', 'low', 'volume', 'money', 'avg']
    out = {f: [] for f in fields}
    # 回测环境注意事项：get_price(fq='pre') 必须显式传前复权基准日期
    # pre_factor_ref_date，否则报"请设置前复权基准日期"。研究环境自动
    # 默认。用 __code__.co_varnames 探测参数是否存在（jqfactor_analyzer
    # 同款做法），避免老环境传参报 TypeError。
    _kw = {}
    try:
        if 'pre_factor_ref_date' in get_price.__code__.co_varnames:
            _kw['pre_factor_ref_date'] = end
    except Exception:
        pass
    for i in range(0, len(stocks), SEC_BATCH):
        batch = stocks[i:i + SEC_BATCH]
        try:
            p = get_price(batch, start_date=start, end_date=end,
                          frequency='daily', fields=fields,
                          skip_paused=False, fq='pre', panel=True,
                          fill_paused=False, **_kw)
        except TypeError:
            # 极老版本没有 fill_paused 参数时的兜底
            p = get_price(batch, start_date=start, end_date=end,
                          frequency='daily', fields=fields,
                          skip_paused=False, fq='pre', panel=True, **_kw)
        # 返回结构自适应：
        #   - 旧版 panel=True → Panel（p[f] 是 日期×股票 DataFrame）
        #   - 新版宽表 dict[str, DataFrame]（p[f] 同样是宽表）
        #   - 新版 MultiIndex DataFrame（index 含 time,code）→ 转宽表
        if isinstance(p, pd.DataFrame) and 'code' in p.index.names:
            p = {f: p[f].unstack('code').sort_index() for f in fields}
        for f in fields:
            # float32 存储：内存减半，OHLCV 精度完全足够
            out[f].append(p[f].astype(np.float32))
    for f in fields:
        out[f] = pd.concat(out[f], axis=1).sort_index()
    return out


def fetch_market_cap(stocks, start, end):
    """按月 × 股票批次分块拉市值表，返回 DataFrame(日期×股票)。

    兼容新旧两种返回格式：
    - 新版：宽表（index=日期, columns=股票代码）
    - 旧版：长表（3 列：日期 + 代码 + 市值，需 pivot 成宽表）

    分块原因：旧版研究环境 get_valuation 单次返回行数有上限
    （实测整段只回来约 1 万行、仅 41 天），因此按月×每批150只切碎。
    """
    parts = []
    batches = [stocks[i:i + 150] for i in range(0, len(stocks), 150)]
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    n_calls = 0
    while cur <= end_ts:
        ms = cur.strftime('%Y-%m-%d')
        me = (cur + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
        if pd.Timestamp(me) > end_ts:
            me = end
        for bi, batch in enumerate(batches):
            n_calls += 1
            try:
                df = get_valuation(batch, ms, me, fields=['market_cap'])
            except Exception as e:
                _log('get_valuation 异常 %s~%s: %s' % (ms, me, e))
                continue
            if df is not None and len(df):
                parts.append(df)
                if bi == 0 and 'day' in df.columns:
                    _log('  月块 %s~%s 返回 %d 行, day 范围 %s ~ %s'
                         % (ms, me, len(df), df['day'].min(), df['day'].max()))
        cur = pd.Timestamp(me) + pd.Timedelta(days=1)
    _log('get_valuation 共调用 %d 次' % n_calls)
    if not parts:
        raise ValueError('get_valuation 未返回任何数据')

    # 依据第一块判断返回格式：宽表(列=股票代码) 还是 长表(3列)
    first_wide = (len(parts[0].columns) >= 10
                  and set(parts[0].columns).issubset(set(stocks)))
    if first_wide:
        # 宽表：索引是日期，按索引去重
        mcap = pd.concat(parts, axis=0)
        mcap = mcap[~mcap.index.duplicated(keep='first')].sort_index()
    else:
        # 长表：每块索引是行号（重复无意义），必须 ignore_index；
        # 日期+代码的重复在下面 pivot 前用 drop_duplicates 处理
        mcap = pd.concat(parts, axis=0, ignore_index=True)
    _log('市值表原始 shape: %s | 列名: %s' % (str(mcap.shape), list(mcap.columns)))
    if 'day' in mcap.columns:
        _log('市值表 day 列范围: %s ~ %s' % (mcap['day'].min(), mcap['day'].max()))

    # 宽表判定：列全部是股票代码（新版返回格式）
    wide = first_wide
    if not wide:
        # ---- 旧版长表：定位 代码列 / 市值列 / 日期 ----
        code_col = None
        for c in mcap.columns:
            smp = mcap[c].dropna()
            if len(smp) and isinstance(smp.iloc[0], str) \
                    and ('.XSH' in smp.iloc[0] or '.XSHE' in smp.iloc[0]):
                code_col = c
                break
        val_col = None
        for c in mcap.columns:
            if str(c).lower() == 'market_cap':
                val_col = c
                break
        if val_col is None:
            for c in mcap.columns:
                if c != code_col and mcap[c].dtype == 'float64':
                    val_col = c
                    break
        if code_col is None or val_col is None:
            raise ValueError('get_valuation 返回格式无法识别: %s' % list(mcap.columns))
        # 日期：优先从剩余列里找能解析成日期的列，否则认为日期在 index 上
        date_col = None
        for c in mcap.columns:
            if c == code_col or c == val_col:
                continue
            try:
                pd.to_datetime(mcap[c])
                date_col = c
                break
            except Exception:
                continue
        if date_col is not None:
            mcap[date_col] = pd.to_datetime(mcap[date_col])
            mcap = mcap.drop_duplicates([date_col, code_col], keep='last')
            mcap = mcap.pivot(index=date_col, columns=code_col, values=val_col)
        else:
            mcap = mcap.reset_index()
            date_col = mcap.columns[0]
            mcap = mcap.drop_duplicates([date_col, code_col], keep='last')
            mcap = mcap.pivot(index=date_col, columns=code_col, values=val_col)
        mcap.index = pd.to_datetime(mcap.index)
    # 数值化：旧版数据可能有字符串脏值
    try:
        mcap = mcap.astype(float)
    except (TypeError, ValueError):
        mcap = mcap.apply(pd.to_numeric, errors='coerce')
    _log('市值表清洗后 shape: %s | 日期范围: %s ~ %s'
         % (str(mcap.shape), str(mcap.index.min()), str(mcap.index.max())))
    return mcap


def fetch_st(stocks, start, end):
    """ST 状态表，返回 DataFrame(日期×股票)，True=ST。
    兼容新旧返回结构：宽表(日期×股票) / MultiIndex(time,code) 长表。"""
    st = get_extras('is_st', stocks, start_date=start, end_date=end)
    if isinstance(st, pd.DataFrame) and 'code' in st.index.names:
        st = st['is_st'].unstack('code').sort_index() if 'is_st' in st.columns \
            else st.iloc[:, 0].unstack('code').sort_index()
    print('  ST 表 dtype:', st.dtypes.value_counts().to_dict())
    return st


def fetch_index_members(index_code, start, end):
    """按月取指数成分快照，返回 (历史成员并集列表, 快照列表)。
    解决静态成分股的幸存者偏差：只把"当期成员"纳入对应日期的横截面。
    旧版 get_index_stocks 不支持 date 参数时退化为静态成分（日志注明）。"""
    snaps = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=5), end, freq='MS')
    snap_list = []
    for d in snaps:
        try:
            members = get_index_stocks(index_code, d.strftime('%Y-%m-%d'))
        except TypeError:
            _log('get_index_stocks 不支持 date 参数，退化为静态成分（仍有幸存者偏差）')
            members = get_index_stocks(index_code)
            snap_list.append((pd.Timestamp(start), members))
            break
        snap_list.append((d, members))
    union = sorted({s for _, ms in snap_list for s in ms})
    _log('成分快照 %d 个 | 历史成员并集 %d 只' % (len(snap_list), len(union)))
    return union, snap_list


def build_member_mask(snap_list, price_index, union, end):
    """把成分快照前向填充成 交易日×股票 的布尔成员表。
    每个交易日只允许该时点的指数成员进入横截面。"""
    member_mask = pd.DataFrame(False, index=price_index, columns=union)
    n = len(snap_list)
    for i, (d, members) in enumerate(snap_list):
        next_d = snap_list[i + 1][0] if i + 1 < n else pd.Timestamp(end) + pd.Timedelta(days=1)
        seg = price_index[(price_index >= d) & (price_index < next_d)]
        cols = [s for s in members if s in member_mask.columns]
        if len(seg) and cols:
            member_mask.loc[seg, cols] = True
    _log('成员掩码: %d 交易日 × %d 只 | 日均成员数 %.0f'
         % (member_mask.shape[0], member_mask.shape[1], member_mask.sum(axis=1).mean()))
    return member_mask


def build_valid_mask(price, st, member_mask=None):
    """可交易性过滤：上市满 MIN_HIST 个交易日 且 非 ST 且 当期指数成员"""
    close = price['close']
    volume = price['volume']
    listed = volume.notna().cumsum() >= MIN_HIST
    st_flag = st.reindex(close.index, columns=close.columns).fillna(False)
    valid = listed & (~st_flag)
    if member_mask is not None:
        member = member_mask.reindex(close.index, columns=close.columns).fillna(False)
        valid = valid & member
    return valid


# ------------------------- 因子与中性化 -------------------------
def _extract_industry_name(v):
    """兼容新旧版 get_industry 返回格式：
    新版返回字符串（如 '银行'）；
    旧版返回嵌套 dict（如 {'sw_l1': {'industry_name': '银行', 'industry_code': '801780'}, ...}）。
    统一提取成字符串，取不到返回 None。"""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        # 优先申万一级，其次聚宽一级、证监会行业
        for key in ('sw_l1', 'jq_l1', 'zjw'):
            sub = v.get(key)
            if isinstance(sub, dict):
                name = sub.get('industry_name')
                if isinstance(name, str) and name:
                    return name
        # 兜底：取 dict 里第一个非空字符串值
        for sub in v.values():
            if isinstance(sub, str) and sub:
                return sub
    return None


def _extract_industry_level(v, level_key):
    """从 get_industry 嵌套 dict 中提取指定层级（如 sw_l1/sw_l2）的行业名"""
    if not isinstance(v, dict):
        return None
    sub = v.get(level_key)
    if isinstance(sub, dict):
        name = sub.get('industry_name')
        if isinstance(name, str) and name:
            return name
    return None


def neutralize(factor, industry_map, log_mcap):
    """行业去均值 + 对数市值回归残差（两步法，按天横截面）。
    返回与 factor 相同 shape 的 DataFrame（无行业映射的股票被剔除）。"""
    ind = pd.Series(industry_map).reindex(factor.columns)
    valid_cols = ind.dropna().index
    f = factor[valid_cols].copy()
    ind_v = ind[valid_cols]
    # 第一步：行业内横截面去均值（显式循环，兼容所有 pandas 版本）
    for label in ind_v.unique():
        cols = ind_v[ind_v == label].index
        if len(cols) == 1:
            f[cols] = 0.0
        else:
            f[cols] = f[cols].sub(f[cols].mean(axis=1), axis=0)
    # 第二步：对 ln 市值做逐日线性回归取残差（先中心化，闭式解）
    lm = log_mcap[valid_cols]
    f_c = f.sub(f.mean(axis=1), axis=0)
    lm_c = lm.sub(lm.mean(axis=1), axis=0)
    beta = (f_c * lm_c).sum(axis=1) / (lm_c ** 2).sum(axis=1)
    resid = f_c - beta.values[:, None] * lm_c
    return resid


def _monthly_return(ret_series):
    """日收益序列按月聚合为月度收益（兼容 pandas 1.x 'M' 与 2.x 'ME'）"""
    try:
        m = (1 + ret_series).resample('ME').prod() - 1
    except (ValueError, KeyError):
        m = (1 + ret_series).resample('M').prod() - 1
    return m


def forward_returns(close, h):
    """持有 h 日的前向收益（t 日收盘可得因子，t+1 日成交，持有到 t+1+h）：
    在 t 日位置的值 = close[t+1+h] / close[t+1] - 1"""
    return close.shift(-(h + 1)).div(close.shift(-1)) - 1


class _LazyData(dict):
    """惰性数据字典：adv{d} 与 cap 在因子真正用到时才计算并缓存。
    避免一次性为 12 个 adv 字段 + 市值分配约 110MB 内存——
    分批跑 10 个因子时只算实际用到的几个字段。"""
    def __init__(self, base, money, mcap, ind_l1, ind_l2):
        dict.__init__(self, base)
        self._money = money
        self._mcap = mcap
        self._cache = {}
        self['ind_l1'] = ind_l1
        self['ind_l2'] = ind_l2

    def __missing__(self, key):
        """仅当 key 不在字典里时触发（dict 协议）"""
        if key.startswith('adv') and key[3:].isdigit():
            d = int(key[3:])
            if d not in self._cache:
                self._cache[d] = self._money.rolling(d).mean().astype(np.float32)
            val = self._cache[d]
        elif key == 'cap':
            if 'cap' not in self._cache:
                self._cache['cap'] = self._mcap
            val = self._cache['cap']
        else:
            raise KeyError(key)
        self[key] = val          # 存入字典，后续直接命中
        return val


# ------------------------- 检验统计 -------------------------
def rank_ic_from_ranked(fr, rr):
    """由已排名数据（0~1 分位）计算逐日 Spearman RankIC，避免重复排名"""
    f_c = fr.sub(fr.mean(axis=1), axis=0)
    r_c = rr.sub(rr.mean(axis=1), axis=0)
    num = (f_c * r_c).sum(axis=1)
    den = ((f_c ** 2).sum(axis=1) * (r_c ** 2).sum(axis=1)) ** 0.5
    return num / den


def ic_stats(ic, start, end):
    """某区间内 IC 序列的统计指标"""
    seg = ic.loc[start:end].dropna()
    if len(seg) == 0:
        return {'IC均值': np.nan, 'ICIR': np.nan, 'IC>0占比': np.nan, 't值': np.nan}
    mean, std = seg.mean(), seg.std()
    return {
        'IC均值': mean,
        'ICIR': mean / std if std > 0 else np.nan,
        'IC>0占比': (seg > 0).mean(),
        't值': mean / std * np.sqrt(len(seg)) if std > 0 else np.nan,
    }


def layered_backtest(pct, ret1, n_groups=N_GROUPS):
    """每日按因子分位（pct 为已排名的 0~1 分位）分组，计算组内 t+1 日收益。
    输出：nets: dict 组号 -> 累计净值；ls_net: 多空组合(第5组-第1组)累计净值"""
    g = np.ceil(pct * n_groups).where(pct.notna())
    nets = {}
    for grp in range(1, n_groups + 1):
        r = ret1.where(g == grp).mean(axis=1)          # 组内等权，mean 自动跳过 NaN
        nets[grp] = (1 + r.fillna(0)).cumprod()
    ls_ret = ret1.where(g == n_groups).mean(axis=1) - ret1.where(g == 1).mean(axis=1)
    ls_net = (1 + ls_ret.fillna(0)).cumprod()
    return nets, ls_net


def net_stats(net, start, end):
    """净值曲线在区间内的绩效指标"""
    seg = net.loc[start:end].dropna()
    if len(seg) < 30:
        return (np.nan,) * 4
    ret = seg.pct_change().dropna()
    ann = (seg.iloc[-1] / seg.iloc[0]) ** (252 / len(seg)) - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    mdd = (seg / seg.cummax() - 1).min()
    monthly = _monthly_return(ret)
    win = (monthly > 0).mean()
    return ann, sharpe, mdd, win


# ------------------------- 主流程 -------------------------
def main():
    _log('===== RUN START: %s, 股票池%s, %s~%s =====' % (pd.Timestamp.now(), INDEX_CODE, FETCH_START, VALID_END))
    print('== 拉取成分股（%s，动态成分，消除幸存者偏差）==' % INDEX_CODE)
    stocks, snap_list = fetch_index_members(INDEX_CODE, FETCH_START, VALID_END)
    print('历史成员并集: %d 只' % len(stocks))

    print('== 拉取行情（%s ~ %s）==' % (FETCH_START, VALID_END))
    price = fetch_price(stocks, FETCH_START, VALID_END)
    price['returns'] = price['close'].pct_change()
    print('行情面板: %s' % str(price['close'].shape))

    print('== 拉取市值 / ST / 行业 / 成员掩码 ==')
    mcap = fetch_market_cap(stocks, FETCH_START, VALID_END)
    log_mcap = np.log(mcap.replace(0, np.nan)).reindex(
        price['close'].index, columns=price['close'].columns)
    st = fetch_st(stocks, FETCH_START, VALID_END)
    member_mask = build_member_mask(snap_list, price['close'].index, stocks, VALID_END)
    raw_ind = get_industry(stocks)   # 旧版返回嵌套 dict，新版返回字符串
    _log('行业映射样本(原始): %s' % str(dict(list(raw_ind.items())[:1]))[:200])
    industry_map = {k: _extract_industry_name(v) for k, v in raw_ind.items()}
    industry_l2 = {k: _extract_industry_level(v, 'sw_l2') for k, v in raw_ind.items()}
    _log('行业映射 有效数: 一级 %d/%d, 二级 %d/%d | 样本: %s'
         % (sum(1 for v in industry_map.values() if v), len(industry_map),
            sum(1 for v in industry_l2.values() if v), len(industry_l2),
            str(dict(list(industry_map.items())[:3]))))
    valid_mask = build_valid_mask(price, st, member_mask)

    print('== 准备因子数据（adv/cap 惰性计算，按需加载）==')
    base = {k: price[k] for k in
            ['open', 'close', 'high', 'low', 'volume', 'avg', 'returns']}
    money = price.pop('money')          # 从 price 移出，交给惰性字典按需算 adv
    cap_df = mcap.reindex(price['close'].index, columns=price['close'].columns)
    data = _LazyData(base, money, cap_df,
                     pd.Series(industry_map), pd.Series(industry_l2))

    print('== 计算前向收益 ==')
    fwd = {h: forward_returns(price['close'], h) for h in HORIZONS}
    ret1 = forward_returns(price['close'], 1)        # t+1 对 t 的收益（分层回测用）
    # 前向收益的横截面排名只算一次（5 个周期），全部因子复用；float32 省一半内存
    fwd_ranked = {h: fwd[h].rank(axis=1, pct=True).astype(np.float32) for h in HORIZONS}

    # 检验区间：训练全段 + 三个切片（分时段稳定性） + 验证段
    ic_periods = [('train', TRAIN_START, TRAIN_END)] + TRAIN_SLICES \
        + [('valid', VALID_START, VALID_END)]

    os.makedirs(RESULT_DIR, exist_ok=True)
    ic_rows, ls_rows = [], []
    ic_series_rows, ls_net_series_rows = [], []   # 日度序列（图的数据表化，供后续分析）

    total = sum(1 for n in FACTOR_FUNCS if FACTOR_START <= int(n[5:]) <= FACTOR_END)
    for idx, (name, func) in enumerate(FACTOR_FUNCS.items(), 1):
        fnum = int(name[5:])
        if not (FACTOR_START <= fnum <= FACTOR_END):
            continue
        try:
            f = func(data)
            if not isinstance(f, pd.DataFrame):
                f = pd.DataFrame(f, index=price['close'].index,
                                 columns=price['close'].columns)
            f = f.where(valid_mask)                # 不可交易处置 NaN
            f_r = f.rank(axis=1, pct=True).astype(np.float32)   # 排名一次，IC/分层复用

            # ---- 中性化版本 ----
            fac_neu = neutralize(f, industry_map, log_mcap)
            fac_neu = fac_neu.where(valid_mask.reindex(fac_neu.columns, axis=1))
            fac_neu_r = fac_neu.rank(axis=1, pct=True).astype(np.float32)
            _log('%s (%d/%d) 覆盖率 %.1f%%'
                 % (name, idx, total, f.notna().mean().mean() * 100))

            # ---- IC 检验（两个口径 × 5个区间 × 多周期）----
            for h in HORIZONS:
                ic_raw = rank_ic_from_ranked(f_r, fwd_ranked[h])
                ic_neu = rank_ic_from_ranked(fac_neu_r, fwd_ranked[h])
                for ver, ic in [('raw', ic_raw), ('neu', ic_neu)]:
                    if h == HORIZONS[-1]:
                        # 记录作图周期的日度 IC 序列（图的数据表化）
                        ic_series_rows.append(pd.DataFrame({
                            'factor': name, 'version': ver,
                            'date': ic.index, 'ic': ic.values}))
                    for period, ps, pe in ic_periods:
                        st_ = ic_stats(ic, ps, pe)
                        ic_rows.append({'factor': name, 'version': ver, 'period': period,
                                        'horizon': h, **st_})
            ic_df = pd.DataFrame(ic_rows)
            ic_df.to_csv(os.path.join(RESULT_DIR, 'ic_stats.csv'),
                         index=False, encoding='utf-8-sig')

            # ---- 分层回测（1 日持有，t+1 成交口径）----
            for ver, pct in [('raw', f_r), ('neu', fac_neu_r)]:
                nets, ls_net = layered_backtest(pct, ret1)
                # 记录日度多空净值序列（图的数据表化）
                ls_net_series_rows.append(pd.DataFrame({
                    'factor': name, 'version': ver,
                    'date': ls_net.index, 'ls_net': ls_net.values}))
                for period, (ps, pe) in [('train', (TRAIN_START, TRAIN_END)),
                                         ('valid', (VALID_START, VALID_END))]:
                    ann, sharpe, mdd, win = net_stats(ls_net, ps, pe)
                    ls_rows.append({'factor': name, 'version': ver, 'period': period,
                                    '年化收益': ann, '夏普': sharpe,
                                    '最大回撤': mdd, '月胜率': win})
                    # 分组单调性：各组年化收益
                    for grp, net in nets.items():
                        gann, _, _, _ = net_stats(net, ps, pe)
                        ls_rows[-1]['G%d' % grp] = gann
            ls_df = pd.DataFrame(ls_rows)
            ls_df.to_csv(os.path.join(RESULT_DIR, 'layered_stats.csv'),
                         index=False, encoding='utf-8-sig')

            # ---- 作图：累计 IC + 多空净值（训练/验证分线）----
            if SKIP_PLOTS:
                pass
            else:
                try:
                    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
                    ic_plot = rank_ic_from_ranked(fac_neu_r, fwd_ranked[HORIZONS[-1]]).cumsum()
                    ax = axes[0]
                    for pname, (ps, pe) in [('训练期', (TRAIN_START, TRAIN_END)),
                                            ('验证期', (VALID_START, VALID_END))]:
                        seg = ic_plot.loc[ps:pe].dropna()
                        if len(seg) > 1:
                            # 用整数横轴绘图，避开旧版 matplotlib 日期轴转换的坑
                            ax.plot(range(len(seg)), seg.values, label=pname)
                    ax.axhline(0, color='grey', lw=0.6)
                    ax.set_title('%s 累计RankIC(%d日,中性化)' % (name, HORIZONS[-1]))
                    ax.legend()
                    ax = axes[1]
                    _, ls_net = layered_backtest(fac_neu_r, ret1)
                    for pname, (ps, pe) in [('训练期', (TRAIN_START, TRAIN_END)),
                                            ('验证期', (VALID_START, VALID_END))]:
                        seg = ls_net.loc[ps:pe].dropna()
                        if len(seg) > 1:
                            ax.plot(range(len(seg)), seg.values, label=pname)
                    ax.set_title('%s 多空净值(中性化,日调仓未扣费)' % name)
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(RESULT_DIR, '%s.png' % name), dpi=100)
                    plt.show()
                except Exception as e:
                    print('  [警告] %s 作图失败: %s' % (name, e))
            # 释放本因子临时对象，防内存累积
            del f, f_r, fac_neu, fac_neu_r
            gc.collect()
        except Exception as e:
            _log('%s 计算失败，跳过: %s' % (name, e))
            print('[警告] %s 计算失败: %s' % (name, e))
            continue

    # ---- 保存日度序列文件与汇总报告 ----
    ic_series_df = pd.concat(ic_series_rows, ignore_index=True)
    ic_series_df.to_csv(os.path.join(RESULT_DIR, 'ic_series.csv'),
                        index=False, encoding='utf-8-sig')
    ls_net_series_df = pd.concat(ls_net_series_rows, ignore_index=True)
    ls_net_series_df.to_csv(os.path.join(RESULT_DIR, 'ls_net_series.csv'),
                            index=False, encoding='utf-8-sig')
    with open(os.path.join(RESULT_DIR, 'report.md'), 'w', encoding='utf-8') as rf:
        rf.write('# Alpha101 全 101 因子检验报告\n\n')
        rf.write('- 股票池: %s（动态成分）\n' % INDEX_CODE)
        rf.write('- 训练期: %s ~ %s（含三段切片）\n' % (TRAIN_START, TRAIN_END))
        rf.write('- 验证期: %s ~ %s\n' % (VALID_START, VALID_END))
        rf.write('- IC 周期（交易日）: %s\n' % HORIZONS)
        rf.write('- 分层回测: 每日按因子分 %d 组, t+1 日成交, 未扣费, 多空=第%d组-第1组\n\n'
                 % (N_GROUPS, N_GROUPS))
        rf.write('## RankIC 的 ICIR 透视表（因子 × 版本 × 周期 × 区间）\n\n')
        piv_ic = ic_df.pivot_table(index=['factor', 'version', 'horizon'],
                                   columns='period', values='ICIR').round(3)
        rf.write('```\n%s\n```\n\n' % piv_ic.to_string())
        rf.write('## 分层回测绩效（1 日持有）\n\n')
        rf.write('```\n%s\n```\n' % ls_df.round(3).to_string(index=False))
        rf.write('\n> 说明：ic_series.csv / ls_net_series.csv 为图中曲线的日度数据；\n')
        rf.write('> *.png 为对应图。\n')

    # ---- 汇总打印 ----
    print('\n================ RankIC 统计（多周期） ================')
    pivot = ic_df.pivot_table(index=['factor', 'version', 'horizon'],
                              columns='period', values=['IC均值', 'ICIR', 'IC>0占比'])
    print(pivot.round(4))
    print('\n================ 分层回测多空绩效（1日持有） ================')
    print(ls_df.round(4))
    print('\n================ 输出文件清单（%s/ 目录）================' % RESULT_DIR)
    for fn in sorted(os.listdir(RESULT_DIR)):
        print('  %s/%s' % (RESULT_DIR, fn))
    _log('===== RUN DONE =====')
