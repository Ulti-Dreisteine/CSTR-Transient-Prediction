# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

运行cstr模型
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import pandas as pd
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.reaction_funcs import ode_funcs


def run_cstr(seq_len_range, show = False, save = True):
	"""运行cstr模型"""
	# 设定初始值
	ca = config.conf['ca_init']
	T = config.conf['T_init']
	
	# 操作参数
	ca_0 = config.conf['ca_0']
	T_0 = config.conf['T_0']
	q = config.conf['q']
	
	# 进行求解
	params = [ca_0, T_0, q]
	
	op_params_bounds = config.conf['operate_params_bounds']
	op_param_names = list(op_params_bounds.keys())
	
	steps = config.conf['steps']
	dt = config.conf['dt']
	while True:
		time_seg_len = np.random.randint(seq_len_range[0], seq_len_range[1])  # 每次平稳操作参数运行时间长度
		t = np.arange(0, time_seg_len * dt, dt)
		outputs = odeint(ode_funcs, [ca, T], t, (params,))
		
		# 更新变量初始值
		[ca, T] = list(outputs[-1, :])
		
		# 更新op_params
		selected_op_param_num = np.random.randint(0, len(op_param_names))
		selected_op_param_bounds = op_params_bounds[op_param_names[selected_op_param_num]]
		params[selected_op_param_num] = np.random.uniform(
			selected_op_param_bounds[0], selected_op_param_bounds[1])
		
		# 保存操作参数记录
		op_params = np.array(params * outputs.shape[0]).reshape(-1, len(op_param_names))
		if 'op_param_records' in locals().keys():
			op_param_records = np.vstack((op_param_records, op_params))
		else:
			op_param_records = op_params
		
		# 保存变量结果
		if 'results' in locals().keys():
			results = np.vstack((results, outputs))
		else:
			results = outputs
		
		# 终止条件
		if results.shape[0] > steps:
			results = results[: steps, :]
			op_param_records = op_param_records[: steps, :]
			break
	
	if save:
		np.save('../files/results.npy', results)
		np.save('../files/op_param_records.npy', op_param_records)
	
	if show:
		plt.figure(figsize = [6, 6])
		plt.subplot(5, 1, 1)
		plt.plot(results[:, 0], label = 'ca')
		plt.legend(fontsize = 6, loc = 'upper right')
		plt.subplot(5, 1, 2)
		plt.plot(results[:, 1], label = 'T')
		plt.legend(fontsize = 6, loc = 'upper right')
		plt.subplot(5, 1, 3)
		plt.plot(op_param_records[:, 0], 'r', label = 'ca_0')
		plt.legend(fontsize = 6, loc = 'upper right')
		plt.subplot(5, 1, 4)
		plt.plot(op_param_records[:, 1], 'r', label = 'T_0')
		plt.legend(fontsize = 6, loc = 'upper right')
		plt.subplot(5, 1, 5)
		plt.plot(op_param_records[:, 2], 'r', label = 'q')
		plt.legend(fontsize = 6, loc = 'upper right')
		plt.tight_layout()
	
	return results, op_param_records


def normalize(data, columns):
	"""
	对数据指定列进行归一化
	:param data:
	:return:
		data_copy: pd.DataFrame, 归一化后的数据
	"""
	data_copy = data.copy()
	bounds = config.conf['variable_bounds']

	for col in columns:
		normalize = lambda x: (x - bounds[col][0]) / (bounds[col][1] - bounds[col][0])
		data_copy.loc[:, col] = data_copy.loc[:, col].apply(normalize)

	return data_copy


if __name__ == '__main__':
	columns = config.conf['columns']
	
	seq_len_range = [60, 120]
	results, op_param_records = run_cstr(seq_len_range, show = True)
	
	data = pd.DataFrame(np.hstack((results, op_param_records)), columns = columns)
	data['ptime'] = data.index
	
	# 归一化
	print('\n')
	for column in columns:
		print('min {}: {}, max {}: {}'.format(column, np.min(data[column]), column, np.max(data[column])))
	
	data = normalize(data, columns)
	
	data.to_csv('../files/normalized_process_data.csv', index = False)