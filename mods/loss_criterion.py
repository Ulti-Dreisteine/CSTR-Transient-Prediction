# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

定义损失函数
"""
import numpy as np
import torch
from torch import nn


def smape(y_true, y_pred):
	"""
	平均百分比误差
	:param y_true: torch.Tensor, 真实值
	:param y_pred: torch.Tensor, 预测值
	:return: torch.tensor, smape结果
	"""
	numerator = torch.abs(y_pred - y_true)
	denominator = torch.div(torch.add(torch.abs(y_pred), torch.abs(y_true)), 2.0)
	return torch.mean(torch.abs(torch.div(numerator, denominator)))


def criterion(y_true, y_pred):
	"""
	损失函数
	:param y_true: torch.Tensor, 真实值
	:param y_pred: torch.Tensor, 预测值
	:return: torch.tensor, loss函数结果
	"""
	l1 = nn.L1Loss()
	loss = torch.add(10 * l1(y_true, y_pred), smape(y_true, y_pred))
	return loss


def mask_criterion(y_true, y_pred):
	"""
	损失函数 给近期时间添加惩罚权重
	:param y_true: torch.Tensor, 真实值
	:param y_pred: torch.Tensor, 预测值
	:return: torch.tensor, loss函数结果
	"""
	dim = 6
	
	mask = [1 + 50 * (15 - i // dim) / 15 for i in range(dim * 15)] + [1 for _ in range(dim * (72 - 15))]
	mask = np.array(mask).astype(np.float32)
	mask_var = torch.from_numpy(mask)
	if torch.cuda.is_available():
		mask_var = mask_var.cuda()
	
	l1 = nn.L1Loss()
	loss = torch.add(10 * l1(y_true * mask_var, y_pred * mask_var), smape(y_true, y_pred))
	return loss