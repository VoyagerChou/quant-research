# -*- coding: utf-8 -*-
"""
本地 jqdata 假模块（仅供本地运行研究脚本使用，不要上传到聚宽！）
================================================================
功能：把 export_data.py 导出的数据包包装成聚宽研究环境的同名 API，
使 alpha101_test_core.py / run_composite.py / run_ml_composite.py 等
脚本在本地无需任何改动即可运行（脚本里都是 `from jqdata import *`）。

数据目录：本文件上级目录下的 data/<指数代码>/（由 data/active_index.txt 指定激活指数）。
注意：数据是导出时点的快照，不会自动更新；要更新就重新导出覆盖。
"""
import os
import numpy as np
import pandas as pd

# 星导入白名单：只暴露与聚宽 API 同名的 5 个函数，
# 防止本模块的 np/pd/os 经 `from jqdata import *` 泄漏进调用方命名空间
__all__ = ['get_price', 'get_index_stocks', 'get_valuation', 'get_extras', 'get_industry']

# 数据根目录（本文件的上级目录下的 data/），每个指数的数据放在独立子目录：
#   data/000300/  data/000905/  data/000852/  data/932000/
# 当前激活的指数由 data/active_index.txt 指定（一行一个代码，如 000905），
# 或环境变量 JQ_INDEX 覆盖。跑不同指数前改这个文件即可，代码不用动。
_DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def _get_active_index():
    env = os.environ.get('JQ_INDEX')
    if env and env.strip():
        return env.strip()
    f = os.path.join(_DATA_ROOT, 'active_index.txt')
    if os.path.isfile(f):
        v = open(f, encoding='utf-8-sig').read().strip()   # utf-8-sig 兼容 BOM
        if v:
            return v
    return '000300'


def _find_data_dir():
    idx = _get_active_index()
    cands = [
        os.path.join(_DATA_ROOT, idx),          # 标准布局：data/<指数代码>/
        _DATA_ROOT,                             # 兼容旧平铺布局
    ]
    import glob
    for c in cands:
        if os.path.isdir(c) and (
                os.path.isfile(os.path.join(c, 'price.npz'))
                or glob.glob(os.path.join(c, 'price_batch_*.npz'))):
            return c
    raise IOError('找不到指数 %s 的数据包（price.npz 或 price_batch_*.npz）。'
                  '请确认数据在 data/%s/ 下。\n可用索引：%s' % (idx, idx, os.listdir(_DATA_ROOT)))

_dates = None
_stocks = None
_price = {}
_mcap = None
_st = None
_member = None
_ind_l1 = None
_ind_l2 = None


def _load():
    """懒加载数据包（首次调用时读盘）"""
    global _dates, _stocks, _price, _mcap, _st, _member, _ind_l1, _ind_l2
    if _dates is not None:
        return
    import glob
    _d = _find_data_dir()
    _dates = pd.DatetimeIndex(pd.read_csv(os.path.join(_d, 'dates.csv'),
                                          header=None)[0])
    _stocks = pd.read_csv(os.path.join(_d, 'stocks.csv'), header=None)[0].tolist()
    # 行情：优先读取分批文件（price_batch_*.npz），否则退回单文件布局
    pfiles = sorted(glob.glob(os.path.join(_d, 'price_batch_*.npz')))
    fields = ('open', 'close', 'high', 'low', 'volume', 'money', 'avg')
    if pfiles:
        acc = {f: [] for f in fields}
        for pf in pfiles:
            z = np.load(pf)
            for f in fields:
                acc[f].append(z[f])
        for f in fields:
            _price[f] = pd.DataFrame(np.concatenate(acc[f], axis=1).astype(np.float32),
                                     index=_dates, columns=_stocks)
    else:
        z = np.load(os.path.join(_d, 'price.npz'))
        for f in fields:
            _price[f] = pd.DataFrame(z[f].astype(np.float32), index=_dates, columns=_stocks)
    # 市值 / ST：同样兼容分批与单文件
    mfiles = sorted(glob.glob(os.path.join(_d, 'mcap_batch_*.npz')))
    if mfiles:
        parts = [np.load(mf)['mcap'] for mf in mfiles]
        _mcap = pd.DataFrame(np.concatenate(parts, axis=1).astype(np.float64),
                             index=_dates, columns=_stocks)
    else:
        _mcap = pd.DataFrame(np.load(os.path.join(_d, 'mcap.npz'))['mcap'].astype(np.float64),
                             index=_dates, columns=_stocks)
    sfiles = sorted(glob.glob(os.path.join(_d, 'st_batch_*.npz')))
    if sfiles:
        parts = [np.load(sf)['st'] for sf in sfiles]
        _st = pd.DataFrame(np.concatenate(parts, axis=1), index=_dates,
                           columns=_stocks).astype(bool)
    else:
        _st = pd.DataFrame(np.load(os.path.join(_d, 'st_bool.npz'))['st'],
                           index=_dates, columns=_stocks).astype(bool)
    _member = pd.DataFrame(np.load(os.path.join(_d, 'member_mask.npz'))['mm'],
                           index=_dates, columns=_stocks).astype(bool)
    _ind_l1 = pd.read_csv(os.path.join(_d, 'industry_l1.csv'), header=None,
                          index_col=0, encoding='utf-8-sig')[1].to_dict()
    _ind_l2 = pd.read_csv(os.path.join(_d, 'industry_l2.csv'), header=None,
                          index_col=0, encoding='utf-8-sig')[1].to_dict()


class _FakePanel(object):
    """模仿旧版 pandas.Panel 的字段访问（脚本只用 p[field]）"""
    def __init__(self, data):
        self._data = data

    def __getitem__(self, f):
        return self._data[f]


def get_price(security, start_date=None, end_date=None, frequency='daily',
              fields=None, skip_paused=False, fq='pre', count=None,
              panel=True, fill_paused=True):
    """返回 Panel-like：p[field] → DataFrame(日期×股票)"""
    _load()
    if isinstance(security, str):
        security = [security]
    if fields is None:
        fields = ['open', 'close', 'high', 'low', 'volume', 'money', 'avg']
    start = pd.Timestamp(start_date) if start_date else _dates[0]
    end = pd.Timestamp(end_date) if end_date else _dates[-1]
    out = {f: _price[f].loc[start:end, security] for f in fields}
    if panel:
        return _FakePanel(out)
    return out


def get_index_stocks(index_code, date=None):
    """返回指定日期（≤date 的最近交易日）的指数成分股列表"""
    _load()
    if date is None:
        d = _dates[-1]
    else:
        d = pd.Timestamp(date)
        d = min(max(d, _dates[0]), _dates[-1])
        d = _dates[_dates <= d][-1]
    row = _member.loc[d]
    return row[row].index.tolist()


def get_valuation(securities, start_date=None, end_date=None, fields=None, count=None):
    """返回长表 DataFrame（列: day / code / market_cap），与线上旧版格式一致"""
    _load()
    if isinstance(securities, str):
        securities = [securities]
    start = pd.Timestamp(start_date) if start_date else _dates[0]
    end = pd.Timestamp(end_date) if end_date else _dates[-1]
    seg = _mcap.loc[start:end, securities]
    # 手工构造长表（避免 stack/reset_index 在不同 pandas 版本下的行为差异）
    t, k = seg.shape
    long = pd.DataFrame({
        'day': pd.to_datetime(np.repeat(seg.index.values, k)),
        'code': np.tile(seg.columns.values, t),
        'market_cap': seg.values.ravel(order='C').astype(np.float64),
    })
    return long


def get_extras(info, security_list, start_date=None, end_date=None, df=True, count=None):
    """ST 标记表：DataFrame(日期×股票) bool"""
    _load()
    if isinstance(security_list, str):
        security_list = [security_list]
    start = pd.Timestamp(start_date) if start_date else _dates[0]
    end = pd.Timestamp(end_date) if end_date else _dates[-1]
    return _st.loc[start:end, security_list].astype(bool)


def get_industry(security_list):
    """返回 {代码: {'sw_l1': {'industry_name': ...}, 'sw_l2': {...}}}，与线上旧版一致"""
    _load()
    if isinstance(security_list, str):
        security_list = [security_list]
    out = {}
    for s in security_list:
        entry = {}
        l1 = _ind_l1.get(s)
        l2 = _ind_l2.get(s)
        if isinstance(l1, str) and l1 and str(l1) != 'nan':
            entry['sw_l1'] = {'industry_name': l1}
        if isinstance(l2, str) and l2 and str(l2) != 'nan':
            entry['sw_l2'] = {'industry_name': l2}
        out[s] = entry
    return out
