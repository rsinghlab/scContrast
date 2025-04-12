import os

import numpy as np
import pandas as pd


def read_chromsizes_file(path):
    data = open(path).read().split('\n')
    data = list(map(lambda x: x.split(' '), data))
    data = list(map(lambda x: [x[0], int(x[1])], data))

    return dict(data)


def create_directory(path):
	if not os.path.exists(path):
		try:
			# Create the directory
			os.makedirs(path)
		except OSError as e:
			print(f"Error: {e}")