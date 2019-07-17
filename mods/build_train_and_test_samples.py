# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

构建训练和测试样本
"""
import copy
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
import sys
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
import sys

sys.path.append('../')

from mods.config_loader import config


def build_single_dim_manifold(time_series, embed_dim, lag, direc = 1):
	"""
	构建一维时间序列嵌入流形样本
	:param direc: int, 平移方向，1为向下，-1为向上
	:param time_series: np.ndarray or pd.DataFrame, 一维时间序列, shape = (-1,)
	:param embed_dim: int, 嵌入维数
	:param lag: int, 嵌入延迟
	:return: manifold: np.ndarray, 嵌入流形数组, shape = (-1, embed_dim)
	"""
	time_series_copy = copy.deepcopy(time_series)
	manifold = []
	for dim in range(embed_dim):
		manifold.append(shift(time_series_copy, direc * dim * lag))
	manifold = np.array(manifold).T
	return manifold


def build_samples_data_frame(data, save = False):
	"""
	构建目标数据集
	:param data: pd.DataFrame, 数据表
	:return:
	"""
	columns = config.conf['columns']
	embed_lags = config.conf['embed_lags']
	acf_lags = config.conf['acf_lags']
	forecast_lags = config.conf['forecast_lags']
	
	data = data.copy()
	
	# 求得连续数值型变量各自对应的维数
	embed_dims = dict()
	for column in columns:
		if column in forecast_lags.keys():
			embed_dims[column] = (acf_lags[column] + forecast_lags[column]) // embed_lags[column]
		else:
			embed_dims[column] = acf_lags[column] // embed_lags[column]
		print('embed_dim for {} is {}'.format(column, embed_dims[column]))
	continuous_columns_num = sum(embed_dims.values())
	
	# 对包含预测的数据向上平移
	for column in forecast_lags.keys():
		data[column] = shift(data[column], -1 * forecast_lags[column])
	
	# 连续数值变量流形样本构建
	data_new = data[['ptime']]
	print('\n')
	for column in columns:
		print('building samples for column {}'.format(column))
		samples = build_single_dim_manifold(data.loc[:, column], embed_dims[column], embed_lags[column])
		columns = [column + '_{}'.format(i) for i in range(samples.shape[1])]
		samples = pd.DataFrame(samples, columns = columns)
		data_new = pd.concat([data_new, samples], axis = 1, sort = True)
	
	if save:
		data_new.to_csv('../files/samples_df.csv', index = False)
	
	return data_new, continuous_columns_num, list(data_new.columns)


def build_targets_data_frame(data, save = False):
	"""
	构建目标数据集
	:param data: pd.DataFrame, 数据表
	:return:
	"""
	target_columns = config.conf['target_columns']
	embed_lag = 1
	pred_dim = config.conf['pred_dim']
	
	data = data.copy()
	data_new = data[['ptime']]
	for target_column in target_columns:
		samples = build_single_dim_manifold(data.loc[:, target_column], pred_dim, embed_lag, direc = -1)
		columns = [target_column + '_{}'.format(i) for i in range(samples.shape[1])]
		samples = pd.DataFrame(samples, columns = columns)
		data_new = pd.concat([data_new, samples], axis = 1, sort = True)
	
	if save:
		data_new.to_csv('../files/targets_df.csv', index = False)
	
	return data_new


def build_train_samples_and_targets(use_local = False, **kwargs):
	"""
	获取训练集的样本和目标数据集
	:param samples_df: pd.DataFrame
	:return:
		X_train: np.ndarray, 训练集样本
		y_train: np.ndarray, 训练集目标
	"""
	# 载入数据
	if use_local:
		samples_df = pd.read_csv('../files/samples_df.csv')
		targets_df = pd.read_csv('../files/targets_df.csv')
	else:
		data = pd.read_csv('../files/normalized_process_data.csv')
		samples_df, continuous_columns_num, _ = build_samples_data_frame(data, **kwargs)
		targets_df = build_targets_data_frame(data, **kwargs)
	
	exist_record_time = config.conf['exist_record_time']
	samples_len = config.conf['samples_len']
	
	X_train_df = samples_df[
		(samples_df.ptime >= exist_record_time - samples_len) & (samples_df.ptime < exist_record_time)]
	y_train_df = targets_df[
		(targets_df.ptime >= exist_record_time - samples_len + 1) & (targets_df.ptime < exist_record_time + 1)]
	
	X_train = np.array(X_train_df.iloc[:, 1:])
	y_train = np.array(y_train_df.iloc[:, 1:])
	
	return X_train, y_train, continuous_columns_num


def build_test_samples_and_targets(use_local = False, **kwargs):
	"""
	获取训练集的样本和目标数据集
	:param samples_df: pd.DataFrame
	:return:
		X_test: np.ndarray, 测试集样本
		y_test: np.ndarray, 测试集目标
	"""
	# 载入数据
	if use_local:
		samples_df = pd.read_csv('../files/samples_df.csv')
		targets_df = pd.read_csv('../files/targets_df.csv')
	else:
		data = pd.read_csv('../files/normalized_process_data.csv')
		samples_df, continuous_columns_num, _ = build_samples_data_frame(data, **kwargs)
		targets_df = build_targets_data_frame(data, **kwargs)
	
	# 数据集构建
	exist_record_time = config.conf['exist_record_time']
	pred_horizon_len = config.conf['pred_horizon_len']
	
	X_test_df = samples_df[
		(samples_df.ptime >= exist_record_time) & (samples_df.ptime < exist_record_time + pred_horizon_len)]
	y_test_df = targets_df[
		(targets_df.ptime >= exist_record_time + 1) & (targets_df.ptime < exist_record_time + pred_horizon_len + 1)]
	
	X_test = np.array(X_test_df.iloc[:, 1:])
	y_test = np.array(y_test_df.iloc[:, 1:])
	
	return X_test, y_test, continuous_columns_num


def build_train_and_verify_datasets():
	"""构建训练和验证样本集"""
	batch_size = config.conf['batch_size']
	
	X, y, continuous_columns_num = build_train_samples_and_targets()
	
	# shuffle操作
	id_list = np.random.permutation(range(X.shape[0]))
	X, y = X[list(id_list), :], y[list(id_list), :]
	
	# 划分训练集和验证集
	split_num = int(0.8 * X.shape[0])
	X_train, y_train = X[:split_num, :], y[:split_num, :]
	X_verify, y_verify = X[split_num:, :], y[split_num:, :]
	
	train_dataset = Data.TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32)))
	trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
	verify_dataset = Data.TensorDataset(
		torch.from_numpy(X_verify.astype(np.float32)),
		torch.from_numpy(y_verify.astype(np.float32))
	)
	verifyloader = DataLoader(verify_dataset, batch_size = X_verify.shape[0])
	
	return trainloader, verifyloader, X_train, y_train, X_verify, y_verify, continuous_columns_num
	

if __name__ == '__main__':
	# 载入数据
	data = pd.read_csv('../files/normalized_process_data.csv')
	
	X_train, y_train, continuous_columns_num = build_train_samples_and_targets(use_local = False, save = False)
	X_test, y_test, continuous_columns_num = build_test_samples_and_targets(use_local = False, save = False)
	
	
	


