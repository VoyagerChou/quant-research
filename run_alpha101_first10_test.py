# -*- coding: utf-8 -*-
"""
Alpha101 前 10 因子单因子检验 —— 入口脚本（聚宽研究环境）
=============================================
【直接运行本文件（或整段复制到单元格执行）即可】

实际逻辑在 alpha101_test_core.py 中。
为什么拆成模块：旧版聚宽研究环境会改写 notebook 单元格里对
numpy/pandas 的名字绑定（np/_np 都曾变成字符串导致报错），
而模块命名空间不受单元格污染影响。

需要上传到研究环境（同一目录）的四个文件：
  1. alpha101_operators.py   算子库
  2. alpha101_factors.py     Alpha#1~10 因子公式
  3. alpha101_test_core.py   检验逻辑与配置（改参数改这个文件顶部配置区）
  4. 本文件                  入口
"""
import alpha101_test_core

alpha101_test_core.main()
