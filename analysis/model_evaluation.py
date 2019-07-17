# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

模型预测
"""
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch
from torch.autograd import Variable
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.model_eval_funcs import rmse, smape, mae, r2
from mods.build_train_and_test_samples import build_test_samples_and_targets, build_targets_data_frame
from mods.nn_models import load_models
from mods.project_graph import callgraph


def get_real_targets():
	"""获得真实目标值"""
	exist_record_time = config.conf['exist_record_time']
	pred_horizon_len = config.conf['pred_horizon_len']
	data = pd.read_csv('../files/normalized_process_data.csv')
	y_test_raw = build_targets_data_frame(data)
	y_test_raw = y_test_raw[
		(y_test_raw.ptime >= exist_record_time + 1) & (y_test_raw.ptime < exist_record_time + pred_horizon_len + 1)]
	y_test_raw = np.array(y_test_raw.iloc[:, 1:])
	return y_test_raw


def model_prediction(X_test, nn_model):
	"""模型预测"""
	var_x_test = Variable(torch.from_numpy(X_test.astype(np.float32)))
	y_test_model = nn_model(var_x_test).detach().cpu().numpy()
	return y_test_model


def model_evaluation(y_test_raw, y_test_model, column, show_fitting = True, show_scatter = True, show_evals = True,
					 show_heatmap = True):
	"""模型效果评估"""
	steps = [0, 3, 7, 11, 23, 50, 100, 200, 300, 400, 499]
	rmse_results, smape_results, mae_results, r2_results = [], [], [], []
	for i in range(y_test_raw.shape[1]):
		rmse_results.append(rmse(y_test_raw[:, i], y_test_model[:, i]))
		smape_results.append(smape(y_test_raw[:, i], y_test_model[:, i]))
		mae_results.append(mae(y_test_raw[:, i], y_test_model[:, i]))
		r2_results.append(r2(y_test_raw[:, i], y_test_model[:, i]))
	
	print('\n========== {} PREDICTION EFFECTS ==========='.format(column))
	for step in [0, 3, 7, 11, 23, 50, 100, 200, 300, 400, 499]:
		print('{} hr: rmse {:4f}, smape {:4f}, mae {:4f}, r2 {:4f}'.format(
			step, rmse_results[step], smape_results[step], mae_results[step], r2_results[step])
		)
	print('============================================')
	
	if show_fitting:
		plt.figure('{} fitting results'.format(column), figsize = [5, 10])
		for step in steps:
			plt.subplot(len(steps), 1, steps.index(step) + 1)
			if steps.index(step) == 0:
				plt.title('fitting results at different pred steps')
			plt.plot(y_test_raw[:, step])
			plt.plot(y_test_model[:, step], 'r')
			plt.ylabel(column)
			plt.legend(['step = {}'.format(step + 1)], loc = 'upper right')
			if steps.index(step) == len(steps) - 1:
				plt.xlabel('time step')
				plt.tight_layout()
	
	if show_scatter:
		bounds = config.conf['variable_bounds'][column]
		plt.figure('{} scatter plot'.format(column), figsize = [8, 10])
		plt.suptitle('comparison of true and pred values at different predicting time steps')
		for step in steps:
			plt.subplot(2, len(steps) / 2, steps.index(step) + 1)
			plt.scatter(y_test_raw[:, step], y_test_model[:, step], s = 1)
			plt.plot(bounds, bounds, 'k--')
			plt.xlim(bounds)
			plt.ylim(bounds)
			plt.xlabel('true value')
			plt.ylabel('pred value')
			plt.legend(['step = {}, r2_score: {:.2f}'.format(step, r2_results[step])], loc = 'upper right')
			if steps.index(step) == len(steps) - 1:
				plt.xlabel('time step')
				plt.tight_layout()
	
	if show_evals:
		eval_methods = ['mae', 'smape', 'rmse', 'r2']
		plt.figure('{} evaluation results at different time steps'.format(column), figsize = [5, 6])
		for method in eval_methods:
			plt.subplot(len(eval_methods), 1, eval_methods.index(method) + 1)
			if eval_methods.index(method) == 0:
				plt.title('model evaluations with different methods')
			plt.plot(eval(method + '_results'))
			plt.xlim([0, 23])
			plt.ylabel(method)
			if eval_methods.index(method) == len(eval_methods) - 1:
				plt.ylim([-0.2, 1.0])
				plt.xlabel('time step')
				plt.tight_layout()
	
	if show_heatmap:
		plt.figure(figsize = [12, 6])
		plt.subplot(1, 2, 1)
		sns.heatmap(y_test_raw, cmap = 'Blues')
		plt.subplot(1, 2, 2)
		sns.heatmap(y_test_model, cmap = 'Blues')
		plt.tight_layout()
	
	return mae_results, r2_results, rmse_results, smape_results


@callgraph
def nn_evaluation_results():
	"""模型评估"""
	# 参数设置
	pred_dim = config.conf['pred_dim']
	target_columns = config.conf['target_columns']

	# 载入训练好的模型
	nn_model = load_models()

	# 构造测试数据
	X_test, y_test, continuous_columns_num = build_test_samples_and_targets()

	# 真实目标值
	y_test_raw = get_real_targets()

	# 模型预测
	y_test_model = model_prediction(X_test, nn_model)

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

	# 模型效果评估
	evaluations_results = pd.DataFrame(list(range(1, pred_dim + 1)), columns = ['pred_time_step'])
	for column in target_columns:
		column_evaluation = {}
		mae_results, r2_results, rmse_results, smape_results = model_evaluation(
			y_test_raw_dict[column],
			y_test_model_dict[column],
			column,
			show_fitting = False,
			show_scatter = False,
			show_evals = False,
			show_heatmap = False
		)
		column_evaluation[column + '_mae'] = mae_results
		column_evaluation[column + '_rmse'] = rmse_results
		column_evaluation[column + '_r2'] = r2_results
		column_evaluation[column + '_smape'] = smape_results

		column_evaluation = pd.DataFrame(column_evaluation)

		evaluations_results = pd.concat([evaluations_results, column_evaluation], axis = 1)

	evaluations_results.to_csv('../files/nn_evaluation_results.csv', index = False)

	# 打印loss曲线
	with open('../files/nn_train_loss.json', 'r') as f:
		train_loss_list = json.load(f)
	with open('../files/nn_verify_loss.json', 'r') as f:
		verify_loss_list = json.load(f)

	plt.figure('loss curve', figsize = [4, 3])
	plt.plot(train_loss_list)
	plt.plot(verify_loss_list, 'r')
	plt.legend(['train set', 'verify set'])
	plt.xlabel('epoch')
	plt.ylabel('loss value')
	plt.tight_layout()
	
	plt.savefig('../graphs/nn_train_verify_loss_curves.png')


if __name__ == '__main__':
	nn_evaluation_results()
