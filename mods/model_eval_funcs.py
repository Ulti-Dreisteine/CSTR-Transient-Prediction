# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

模型结果评估
"""
from sklearn.metrics import r2_score
import numpy as np


def rmse(time_series_0, time_series_1):
	"""
	计算rmse
	:param time_series_0:
	:param time_series_1:
	:return:
	"""
	mse = np.sum(pow((time_series_0 - time_series_1), 2)) / len(time_series_0)
	mse = np.power(mse, 0.5)
	return mse


def smape(time_series_0, time_series_1, eps = 1e-12):
	"""
	计算smape
	:param eps:
	:param time_series_0:
	:param time_series_1:
	:return:
	"""
	smape = 0
	for i in range(len(time_series_0)):
		smape += np.abs(time_series_0[i] - time_series_1[i]) / ((np.abs(time_series_0[i]) + np.abs(time_series_1[i])) / 2 + eps)
	smape = smape / len(time_series_0)
	return smape


def mae(time_series_0, time_series_1):
	"""
	平均绝对误差
	:param time_series_0:
	:param time_series_1:
	:return:
	"""
	mae = 0
	for i in range(len(time_series_0)):
		mae += np.abs(time_series_0[i] - time_series_1[i])
	mae = mae / len(time_series_0)
	return mae


def r2(time_series_0, time_series_1):
	"""
	R2评估
	:param time_series_0:
	:param time_series_1:
	:return:
	"""
	return r2_score(time_series_0, time_series_1)


