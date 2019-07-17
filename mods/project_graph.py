# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

绘制项目函数调用图
"""
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pycallgraph.config import Config
from pycallgraph import GlobbingFilter


def callgraph(func):
	def wrapper(*args, **kwargs):
		# 隐藏其他多余函数信息
		config = Config()
		config.trace_filter = GlobbingFilter(
			exclude = [
				'_*',
				'*SourceFileLoader*',
				'*cache_from_source*',
				'*ModuleSpec*',
				'*spec_from_file_location*',
				'*path_hook_for_FileFinder*',
				'cb',
				'*FileFinder*',
				'pydev*',
				'*VFModule*',
				'__new__',
				'*spec',
				'*<*',
				'*>*',
				'pycallgraph.*',
				'*.secret_function',
		])
		
		graphviz = GraphvizOutput()
		graphviz.output_file = '../graphs/{func}.png'.format(func = func.__name__)
		with PyCallGraph(output = graphviz, config = config):
			result = func(*args, **kwargs)
		return result
	return wrapper
		
	
	
	


