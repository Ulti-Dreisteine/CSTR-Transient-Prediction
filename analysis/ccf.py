# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

不同变量间的互相关函数计算
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import json
import sys

sys.path.append('../')

from mods.config_loader import config


def cross_correlation(series_a, series_b, d, eps = 1e-6):
	"""
	在延迟为d上的互相关分析, series_a, 移动series_b: d > 0向右移，d < 0向左移
	:param series_a: np.ndarray, 目标变量, shape = (-1,)
	:param series_b: np.ndarray, 外生变量, shape = (-1,)
	:param d: time delay, 延迟阶数
	:return:
	"""
	try:
		assert type(series_a) == type(series_b) == np.ndarray
		assert len(series_a.shape) == len(series_b.shape) == 1
		assert len(series_a) == len(series_b)
	except Exception:
		raise ValueError('输入数据格式不正确')
	
	series_x = series_a.copy()
	series_y = series_b.copy()
	
	len_y = len(series_y)
	series_y = np.hstack((series_y[-(d % len_y):], series_y[: -(d % len_y)]))
	
	mean_x, mean_y = np.mean(series_x), np.mean(series_y)
	numerator = np.sum((series_x - mean_x) * (series_y - mean_y))
	denominator = np.sqrt(np.sum(np.power((series_x - mean_x), 2))) * np.sqrt(
		np.sum(np.power((series_y - mean_y), 2)))
	
	return numerator / (denominator + eps)


def mean_ccf(series_a, series_b, d, seg_len, start_locs):
	""""""
	series_list = []
	try:
		assert start_locs[0] + d >= 0
	except Exception:
		raise ValueError('start locs起始位置不能被分析')
	
	try:
		assert start_locs[-1] + d <= len(series_a) - 1
	except Exception:
		raise ValueError('start locs末尾位置不能被分析')
	
	for start_loc in start_locs:
		series_x = erase_long_term_trend(series_a.copy()[start_loc: start_loc + seg_len])
		series_y = erase_long_term_trend(series_b.copy()[(start_loc + d): (start_loc + seg_len + d)])
		series_list.append([series_x, series_y])
		# series_list.append(
		# 	[
		# 		series_a.copy()[start_loc: start_loc + seg_len],
		# 		series_b.copy()[(start_loc + d): (start_loc + seg_len + d)]])
	
	series_list = pd.DataFrame(series_list, columns = ['series_a', 'series_b'])
	series_list['ccf_value'] = series_list.apply(lambda x: cross_correlation(x[0], x[1], 0), axis = 1)
	ccf = np.mean(list(series_list['ccf_value']))
	
	return ccf


def time_delay_ccf_func(series_a, series_b, d_list, seg_len, start_locs):
	"""
	含时滞计算的ccf
	:param series_a: array
	:param series_b: array
	:param d_list: list or array
	:param seg_len: int
	:param start_locs: list or array
	:return: time_delay_ccf, dict
	"""
	time_delay_ccf = {}
	for d in d_list:
		time_delay_ccf[int(d)] = mean_ccf(series_a, series_b, d, seg_len, start_locs)
	
	return time_delay_ccf


def erase_long_term_trend(series):
	"""消除数据长期趋势"""
	series = series.copy().reshape(-1, 1)
	lr = LinearRegression().fit(np.arange(len(series)).reshape(-1, 1), series)
	series -= lr.predict(np.arange(len(series)).reshape(-1, 1))
	series = series.flatten()
	return series


if __name__ == '__main__':
	# 读取数据
	results = np.load('../files/results.npy')
	op_param_records = np.load('../files/op_param_records.npy')
	
	source_columns = config.conf['source_columns']
	target_columns = config.conf['target_columns']
	data = [
		results[:, 0],
		results[:, 1],
		op_param_records[:, 0],
		op_param_records[:, 1],
		op_param_records[:, 2]]
	
	# ccf分析
	d_list = np.arange(-2500, 2505, 20)
	seg_len = 500  # 单次ccf计算选取的序列长度
	start_locs = np.arange(2500, 50000, 200)
	
	plt.figure(figsize = [10, 14])
	total_time_delay_ccf = {}
	for i in range(len(source_columns)):
		total_time_delay_ccf[source_columns[i]] = {}
		for j in range(len(target_columns)):
			print('processing col_x: {}, col_y: {}'.format(source_columns[i], target_columns[j]))
			series_a, series_b = data[i], data[j]
			time_delay_ccf = time_delay_ccf_func(series_a, series_b, d_list, seg_len, start_locs)
			total_time_delay_ccf[source_columns[i]][target_columns[j]] = time_delay_ccf

			plt.subplot(
				len(source_columns), len(target_columns), len(target_columns) * i + j + 1)
			plt.plot(time_delay_ccf.keys(), np.abs(list(time_delay_ccf.values())))
			plt.fill_between(time_delay_ccf.keys(), np.abs(list(time_delay_ccf.values())))
			plt.plot([d_list[0], d_list[-1]], [0, 0], 'k--', linewidth = 0.3)
			plt.plot([0, 0], [-1.0, 1.0], 'k-', linewidth = 0.3)
			plt.xlim([d_list[0], d_list[-1]])
			plt.ylim([-1, 1])
			plt.xticks(fontsize = 6)
			plt.yticks(fontsize = 6)
			if i == 0:
				plt.title(target_columns[j], fontsize = 8)

			if j == 0:
				plt.ylabel(source_columns[i], fontsize = 8)

			plt.tight_layout()
			plt.show()
			plt.pause(1.0)
	
	plt.savefig('../graphs/ccf_analysis_on_continuous.png')
	
	with open('../files/total_time_delay_ccf_results.json', 'w') as f:
		json.dump(total_time_delay_ccf, f)
