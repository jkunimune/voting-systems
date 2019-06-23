import shapefile
import math
import matplotlib.path as plt_path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from election import elect


NUM_CANDIDATES = 5

N_ROWS = 1320
N_COLS = 8688
STEP = 0.04166666666670
X_0 = -180.97916666666666
Y_0 = 72.97916666671064
REDUCTION = 2


def encloses(geometry, x, y):
	for i in range(len(geometry.parts)):
		try:
			vertices = geometry.points[geometry.parts[i]:(geometry.parts[i+1])]
		except IndexError:
			vertices = geometry.points[geometry.parts[i]:]
		path = plt_path.Path(vertices)
		return path.contains_points([[x, y]])


def y_to_i(y):
	return (Y_0 - y)/(STEP*REDUCTION)

def x_to_j(x):
	return (x - X_0)/(STEP*REDUCTION)


if __name__ == '__main__':
	pop_table_raw = np.fromfile('data/pop/usap00g.bil', dtype=np.dtype('>u4'))
	pop_table_raw = pop_table_raw.reshape((N_ROWS, N_COLS))
	pop_Z = np.zeros((N_ROWS//REDUCTION, N_COLS//REDUCTION))
	for i in range(REDUCTION):
		for j in range(REDUCTION):
			pop_Z += pop_table_raw[i::REDUCTION, j::REDUCTION]
	pop_X, pop_Y = np.meshgrid(
		np.arange(X_0, X_0+N_COLS*STEP,  STEP*REDUCTION),
		np.arange(Y_0, Y_0-N_ROWS*STEP, -STEP*REDUCTION))
	pop_Z, pop_X, pop_Y = pop_Z[:,:pop_Z.shape[1]//3], pop_X[:,:pop_Z.shape[1]//3], pop_Y[:,:pop_Z.shape[1]//3]

	shpf = shapefile.Reader('data/states/states')

	cities = pd.read_csv('data/cities.csv', sep=';')
	cities['y_str'], cities['x_str'] = zip(*cities['Coordinates'].str.split(', '))
	cities['y'], cities['x'] = cities['y_str'].astype(float), cities['x_str'].astype(float)

	for record, state_border in zip(shpf.records(), shpf.shapes()):
		state_name = record[0]
		state_cities = cities[[encloses(state_border, city.x, city.y) for _,city in cities.iterrows()]]
		if len(state_cities) <= 1:	continue
		state_cities = state_cities.nlargest(NUM_CANDIDATES, 'Population')

		i = np.expand_dims(np.arange(math.floor(y_to_i(state_border.bbox[3])), math.ceil(y_to_i(state_border.bbox[1]))+1), axis=1)
		j = np.expand_dims(np.arange(math.floor(x_to_j(state_border.bbox[0])), math.ceil(x_to_j(state_border.bbox[2]))+1), axis=0)
		state_pop_x, state_pop_y, state_pop_z = pop_X[i,j].ravel(), pop_Y[i,j].ravel(), pop_Z[i,j].ravel()
		valid = [k for k in range(len(state_pop_z)) if encloses(state_border, state_pop_x[k], state_pop_y[k])]
		state_pop_x, state_pop_y, state_pop_z = state_pop_x[valid], state_pop_y[valid], state_pop_z[valid]

		winner_idx = elect(candidates=state_cities.loc[:,['x','y']].values, voters=np.stack((state_pop_x, state_pop_y, state_pop_z), axis=1),
			system='primary')
		winner = state_cities.iloc[winner_idx]
		print("{} wins {}".format(winner.City, state_name))
