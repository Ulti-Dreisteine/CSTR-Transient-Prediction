# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

根据ccf计算结构构建图网络
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import networkx as nx
import sys

sys.path.append('../')

from mods.config_loader import config


def load_ccf_results():
	with open('../files/total_time_delay_ccf_results.json', 'r') as f:
		total_ccf_results = json.load(f)
	return total_ccf_results


def peak_and_significant_locs(name_a, name_b, total_ccf_results, detect_sep_loc):
	"""
	寻找两个变量有显著作用的时间范围
	:param name_a: str
	:param name_b: str
	:param detect_sep_loc: 设置背景分界位置, 用于检测的时滞范围在[-diag_sep_loc, diag_sep_loc + 1]
	:return:
	"""
	ccf = total_ccf_results[name_a][name_b]
	
	bg_ccf = {k: np.abs(v) for k, v in ccf.items() if (int(k) < -detect_sep_loc) | (int(k) > detect_sep_loc)}
	
	bg_values = list(bg_ccf.values())
	std_var = np.std(bg_values)
	mean_value = np.mean(bg_values)
	upper_bound = mean_value + 3.0 * std_var
	
	main_ccf = {k: np.abs(v) for k, v in ccf.items() if (-detect_sep_loc <= int(k) <= detect_sep_loc)}
	
	# 首先检测有没有明显峰
	peak_value = np.max(list(main_ccf.values()))
	if peak_value >= upper_bound:
		peak = {int(k): v for k, v in main_ccf.items() if v == peak_value}
		significant_locs = {int(k): v for k, v in main_ccf.items() if (int(k) >= 0) & (v >= upper_bound)}
	else:
		peak = {}
		significant_locs = {}
	
	return peak, significant_locs


def time_delays_and_significant_time_ranges(x_columns, y_columns, total_ccf_results, detect_sep_loc):
	""""""
	significant_time_ranges = {}
	time_delay_peaks = {}
	for col_x in x_columns:
		significant_time_ranges[col_x], time_delay_peaks[col_x] = {}, {}
		for col_y in y_columns:
			peak, significant_locs = peak_and_significant_locs(col_x, col_y, total_ccf_results, detect_sep_loc)
			if peak == {}:
				time_delay_peaks[col_x][col_y] = {}
				significant_time_ranges[col_x][col_y] = 0
			else:
				time_delay_peaks[col_x][col_y] = peak
				if significant_locs == {}:
					significant_time_ranges[col_x][col_y] = 0
				else:
					significant_time_ranges[col_x][col_y] = np.max(list(significant_locs.keys()))
	return significant_time_ranges, time_delay_peaks
	

if __name__ == '__main__':
	# 参数设定
	source_columns = config.conf['source_columns']
	target_columns = config.conf['target_columns']
	detect_sep_loc = 1500

	# 计算显著作用时间范围
	total_ccf_results = load_ccf_results()
	significant_time_ranges, time_delay_peaks = time_delays_and_significant_time_ranges(
		source_columns, target_columns, total_ccf_results, detect_sep_loc)

	# 生成边表
	edges = []
	for col_x in source_columns:
		for col_y in target_columns:
			if significant_time_ranges[col_x][col_y] != 0:
				edges.append([col_x, col_y, list(time_delay_peaks[col_x][col_y].values())[0]])
	g = nx.DiGraph()
	for edge in edges:
		g.add_edge(edge[0], edge[1], weight = edge[2])

	edges = list(g.edges())
	weights = []
	for edge in edges:
		weights.append(g.get_edge_data(edge[0], edge[1])['weight'])

	pos = nx.spring_layout(g, iterations = 5000)
	plt.figure('network', figsize = [6, 6])
	nx.draw_networkx_nodes(g, pos, node_color = 'w', node_size = 200)
	nx.draw_networkx_edges(
		g, pos, edge_width = 10 * weights, edge_color = weights, edge_cmap = plt.cm.Blues, arrowsize = 6)
	nx.draw_networkx_labels(g, pos, font_size = 6, font_weight = 'bold')
	plt.savefig('../graphs/graph_net.png')
	


