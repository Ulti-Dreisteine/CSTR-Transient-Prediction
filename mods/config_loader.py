# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

配置器
"""
import os
import yaml
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))


def _absolute_conf_path(path):
	return os.path.join(os.path.dirname(__file__), path)


class ConfigLoader(object):
	def __init__(self, config_path = None):
		self._config_path = config_path or _absolute_conf_path('../config/config.yml')
		self._load_config()
	
	def _load_config(self):
		with open(self._config_path, 'r') as f:
			self._conf = yaml.load(f, Loader = yaml.Loader)
	
	@property
	def conf(self):
		return self._conf


config = ConfigLoader()
	

