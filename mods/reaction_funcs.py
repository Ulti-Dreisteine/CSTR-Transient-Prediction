# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

反应方程
"""
import numpy as np
import sys

sys.path.append('../')

from mods.config_loader import config


def ode_funcs(X, t, params):
	"""
	ode方程组
	:param X: list, 自变量: [ca, T]
	:param t: time
	:param params: list, 参数: [ca_0, T_0, q]
	:return:
	"""
	V = config.conf['V']
	k_0 = config.conf['k_0']
	E_div_R = config.conf['E_div_R']
	delta_H = config.conf['delta_H']
	ro = config.conf['ro']
	roc = config.conf['roc']
	Cp = config.conf['Cp']
	Cpc = config.conf['Cpc']
	qc = config.conf['qc']
	hA = config.conf['hA']
	Tc_0 = config.conf['Tc_0']
	
	ca, T = X[0], X[1]
	ca_0, T_0, q = params[0], params[1], params[2]
	
	const = np.exp(-E_div_R / T)
	dca_dt = q / V * (ca_0 - ca) - k_0 * ca * const
	dT_dt = q / V * (T_0 - T) - (-delta_H / (ro * Cp)) * k_0 * ca * const + \
			roc * Cpc / (ro * Cp * V) * qc * (1 - np.exp(-hA / (qc * roc * Cpc))) * (Tc_0 - T)
	
	return np.array([dca_dt, dT_dt])
	
	
	



