import shapefile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


N_ROWS = 1320
N_COLS = 8688
STEP = 0.04166666666670
X_0 = -180.97916666666666
Y_0 = 72.97916666671064
REDUCTION = 6
pop_table_raw = np.fromfile('data/pop/usap00g.bil', dtype=np.dtype('>u4'))
pop_table_raw = pop_table_raw.reshape((N_ROWS, N_COLS))
pop_table = np.zeros((N_ROWS//REDUCTION, N_COLS//REDUCTION))
for i in range(REDUCTION):
	for j in range(REDUCTION):
		pop_table += pop_table_raw[i::REDUCTION, j::REDUCTION]
pop_x, pop_y = np.arange(X_0, X_0+N_COLS*STEP, STEP*REDUCTION), np.arange(Y_0, Y_0-N_ROWS*STEP, -STEP*REDUCTION)
pop_table, pop_x = pop_table[:,:len(pop_x)//3], pop_x[:len(pop_x)//3]
plt.pcolor(pop_x, pop_y, pop_table)

shpf = shapefile.Reader('data/states/states')
for record, geom in zip(shpf.records(), shpf.shapes()):
	print(record[0])
	x, y = zip(*geom.points)
	plt.plot(x, y)

cities = pd.read_csv('data/cities.csv', sep=';')
print(cities.Coordinates.str.split(', '))
cities['y_str'], cities['x_str'] = zip(*cities['Coordinates'].str.split(', '))
cities['y'], cities['x'] = cities['y_str'].astype(float), cities['x_str'].astype(float)
print(cities)
plt.scatter(cities.x, cities.y, c=cities.Rank)

plt.show()