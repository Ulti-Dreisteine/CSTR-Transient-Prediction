# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

单个样本在72小时上的预测效果和与真实值的对比
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.build_train_and_test_samples import build_test_samples_and_targets
from mods.nn_models import load_models
from analysis.model_evaluation import model_prediction


if __name__ == '__main__':
	# 参数设置
	pred_dim = config.conf['pred_dim']
	target_columns = config.conf['target_columns']
	
	# 载入模型
	nn_model = load_models()
	
	# 构造测试数据
	X_test, y_test, continuous_columns_num = build_test_samples_and_targets()
	
	sample_num = 4500  #
	x_test, y_test_raw = X_test[sample_num - 1: sample_num, :], y_test[sample_num - 1: sample_num, :]
	y_test_raw = y_test_raw[-1:, :]
	y_test_model = model_prediction(x_test, nn_model)
	
	# 各污染物数据切分
	y_test_raw_dict, y_test_model_dict = {}, {}
	for i in range(len(target_columns)):
		y_test_raw_dict[target_columns[i]] = y_test_raw[:, i * pred_dim: (i + 1) * pred_dim]
		y_test_model_dict[target_columns[i]] = y_test_model[:, i * pred_dim: (i + 1) * pred_dim]

	# 还原为真实值
	for column in target_columns:
		bounds = config.conf['variable_bounds'][column]
		y_test_raw_dict[column] = y_test_raw_dict[column] * (bounds[1] - bounds[0]) + bounds[0]
		y_test_model_dict[column] = y_test_model_dict[column] * (bounds[1] - bounds[0]) + bounds[0]

	# 载入计算得到的各时间预测步loss数据
	eval_records = pd.read_csv('../files/nn_evaluation_results.csv')
	loss_dict = {}
	for column in target_columns:
		loss_dict[column] = 3.0 * np.array(eval_records.loc[:, column + '_rmse']).flatten()

	# 计算预测结果以及上下界数据
	pred_curves = {}
	for column in target_columns:
		pred_curves[column] = {}
		pred_curves[column]['middle'] = y_test_model_dict[column].flatten()
		pred_curves[column]['upper'] = pred_curves[column]['middle'] + loss_dict[column]
		pred_curves[column]['lower'] = pred_curves[column]['middle'] - loss_dict[column]
		pred_curves[column]['lower'][pred_curves[column]['lower'] < 0] = 0

	plt.figure('pred results', figsize = [4, 2 * len(target_columns)])
	for column in target_columns:
		plt.subplot(len(target_columns), 1, target_columns.index(column) + 1)
		plt.plot(y_test_raw_dict[column].flatten(), 'r')
		plt.plot(pred_curves[column]['middle'], 'b')
		plt.plot(pred_curves[column]['upper'], 'b--', linewidth = 0.5)
		plt.plot(pred_curves[column]['lower'], 'b--', linewidth = 0.5)
		plt.ylabel(column + ' value', fontsize = 10)
		plt.xticks(fontsize = 6)
		plt.yticks(fontsize = 6)
		plt.legend(['real', 'pred', 'upper_bound', 'lower_bound'], fontsize = 6, loc = 'upper right')
	plt.xlabel('pred time step', fontsize = 10)
	plt.tight_layout()

