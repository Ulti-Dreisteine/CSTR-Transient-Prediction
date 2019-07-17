# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

pytorch的神经网络模型
"""
import json
import sys
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as f

sys.path.append('../')

from mods.config_loader import config


class NN(nn.Module):
	def __init__(self, input_size, hidden_sizes, output_size):
		super(NN, self).__init__()
		self.input_size = input_size
		self.hidden_sizes = hidden_sizes
		self.output_size = output_size
		
		self.bn_in = nn.BatchNorm1d(self.input_size)
		
		self.fc_0 = nn.Linear(self.input_size, self.hidden_sizes[0])
		self._init_layer(self.fc_0)
		self.bn_0 = nn.BatchNorm1d(self.hidden_sizes[0])
		
		self.fcs = []
		self.bns = []
		for i in range(len(hidden_sizes) - 1):
			fc_i = nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1])
			setattr(self, 'fc_{}'.format(i + 1), fc_i)
			self._init_layer(fc_i)
			bn_i = nn.BatchNorm1d(self.hidden_sizes[i + 1])
			setattr(self, 'bn_{}'.format(i + 1), bn_i)
			self.fcs.append(fc_i)
			self.bns.append(bn_i)
		
		self.fc_out = nn.Linear(self.hidden_sizes[-1], self.output_size)
		self._init_layer(self.fc_out)
		self.bn_out = nn.BatchNorm1d(self.output_size)
		
	def _init_layer(self, layer):
		init.normal_(layer.weight)  # 使用这种初始化方式能降低过拟合
		# init.constant_(layer.bias, 0.5)
		init.normal_(layer.bias)
		
	def forward(self, x):
		x = self.bn_in(x)
		x = self.fc_0(x)
		x = self.bn_0(x)
		
		for i in range(len(self.fcs)):
			x = self.fcs[i](x)
			x = self.bns[i](x)
			x = f.elu(x)
		
		x = self.fc_out(x)
		x = self.bn_out(x)
		x = f.softplus(x)
		
		return x


def load_models():
	"""载入已经训练好的模型"""
	target_columns = config.conf['target_columns']
	
	with open('../files/nn_struc_params.json', 'r') as f:
		model_struc_params = json.load(f)
	
	model_path = '../files/nn_state_dict_{}.pth'.format(target_columns)
	pretrained_model_dict = torch.load(model_path, map_location = 'cpu')
	
	input_size = model_struc_params['nn_model']['input_size']
	hidden_sizes = model_struc_params['nn_model']['hidden_sizes']
	output_size = model_struc_params['nn_model']['output_size']
	nn_model = NN(input_size, hidden_sizes, output_size)
	nn_model.load_state_dict(pretrained_model_dict, strict = False)
	
	nn_model.eval()
	
	return nn_model


if __name__ == '__main__':
	nn_model = NN(2, [2, 3], 2)
