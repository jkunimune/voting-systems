import shapefile
import math
import matplotlib.path as plt_path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from election import elect

# sns.set_style('white')
sns.set(style="white", color_codes=True)


NUM_CANDIDATES = 5

N_ROWS = 1320
N_COLS = 8688
STEP = 0.04166666666670
X_0 = -180.97916666666666
Y_0 = 72.97916666671064
REDUCTION = 1

ELECTORAL_SYSTEMS = ['Plurality','Primary','Runoff','Instant-runoff','Condorcet','Approval','Score']


def encloses(geometry, x, y):
	for i in range(len(geometry.parts)):
		try:
			vertices = geometry.points[geometry.parts[i]:(geometry.parts[i+1])]
		except IndexError:
			vertices = geometry.points[geometry.parts[i]:]
		path = plt_path.Path(vertices)
		if path.contains_points([[x, y]]):
			return True
	return False


def y_to_i(y):
	return (Y_0 - y)/(STEP*REDUCTION)

def x_to_j(x):
	return (x - X_0)/(STEP*REDUCTION)


def mercator_forward(x):
    return np.degrees(np.arcsinh(np.tan(np.radians(x))))


def mercator_inverse(x):
    return np.degrees(np.arctan(np.sinh(np.radians(x))))


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

	columns = ['state','system','winner_idx','winner_name','winner_distance'] + ['candidate_{}'.format(i) for i in range(NUM_CANDIDATES)]
	results = pd.DataFrame(columns=columns)
	for idx, (record, state_border) in enumerate(zip(shpf.records(), shpf.shapes())):
		state_name = record[0]
		# if state_name != 'Texas':	continue
		state_cities = cities[cities.State==state_name]
		if len(state_cities) <= 2:	continue
		state_cities = state_cities.nlargest(NUM_CANDIDATES, 'Population')
		print(state_name)

		i = np.expand_dims(np.arange(math.floor(y_to_i(state_border.bbox[3])), math.ceil(y_to_i(state_border.bbox[1]))+1), axis=1)
		j = np.expand_dims(np.arange(math.floor(x_to_j(state_border.bbox[0])), math.ceil(x_to_j(state_border.bbox[2]))+1), axis=0)
		state_pop_x, state_pop_y, state_pop_z = pop_X[i,j].ravel(), pop_Y[i,j].ravel(), pop_Z[i,j].ravel()
		valid = [k for k in range(len(state_pop_z)) if encloses(state_border, state_pop_x[k], state_pop_y[k])]
		state_pop_x, state_pop_y, state_pop_z = state_pop_x[valid], state_pop_y[valid], state_pop_z[valid]

		candidate_array = state_cities.loc[:,['x','y']].values
		voter_array = np.stack((state_pop_x, state_pop_y, state_pop_z), axis=1)
		candidate_array[:,1] = mercator_forward(candidate_array[:,1])
		voter_array[:,1] = mercator_forward(voter_array[:,1])

		distances = np.average(np.hypot(
			np.expand_dims(voter_array[:,0], 0) - np.expand_dims(candidate_array[:,0], 1),
			np.expand_dims(voter_array[:,1], 0) - np.expand_dims(candidate_array[:,1], 1)), weights=voter_array[:,2], axis=1)
		distances /= distances.min()

		for i in range(len(state_border.parts)):
			part_start, part_end = state_border.parts[i], (state_border.parts[i+1] if i+1 < len(state_border.parts) else len(state_border.points))
			border_array = np.array(state_border.points[part_start:part_end])
			border_array[:,1] = mercator_forward(border_array[:,1])
			plt.plot(border_array[:,0], border_array[:,1], 'k-', linewidth=1, zorder=0)
		# plt.plot([voter_array[:,0].min(), voter_array[:,0].max()], [32.5, 32.5], 'k-', linewidth=1, zorder=0)
		plt.scatter(voter_array[:,0], voter_array[:,1], s=voter_array[:,2]/voter_array[:,2].max()*20, c='b', marker='.', zorder=-1)
		plt.scatter(candidate_array[:,0], candidate_array[:,1], s=50, c='w', marker='o', zorder=1)
		plt.scatter(candidate_array[:,0], candidate_array[:,1], s=50, c='k', marker='.', zorder=2)
		for city in state_cities.itertuples():
			other_cities = state_cities[state_cities.City!=city.City]
			nearest_city = other_cities.iloc[np.hypot(
				other_cities.x - city.x, other_cities.y - city.y).idxmin()]
			if nearest_city.x > city.x:
				alignment = 'right'
			else:
				alignment = 'left'
			valignment = 'top' if city.City != 'Houston' else 'bottom'
			plt.text(city.x, mercator_forward(city.y), ' '+city.City+' ', horizontalalignment=alignment, verticalalignment=valignment, family="Times New Roman", size=10, weight='normal')
		# plt.gca().set_yscale('function', functions=(mercator_forward, mercator_inverse)) # not supported with custom axes. Cry.
		plt.xticks([])
		plt.yticks([])
		plt.axis('equal')
		plt.show()

		for system in ELECTORAL_SYSTEMS:
			winner_idx = elect(candidates=candidate_array, voters=voter_array, system=system.lower(), verbose=False)
			winner = state_cities.iloc[winner_idx]
			result = dict(
				state=state_name, system=system, winner_idx=winner_idx, winner_name=winner.City, winner_distance=distances[winner_idx])
			for i in range(state_cities.shape[0]):
				result['candidate_{}'.format(i)] = state_cities.iloc[i].City
			results = results.append(result, ignore_index=True)

	results['usefulness'] = [results[results.state==state].winner_name.nunique() for state in results.state]
	results = results.sort_values(by='usefulness', ascending=False, kind='mergesort')

	print(results.to_string())
	for system in ELECTORAL_SYSTEMS:
		print("{} scores {} on average".format(system, results[results.system==system].winner_distance.mean()))

	# sns.violinplot(x='system', y='winner_distance', inner=None, data=results)
	# plt.xlabel("Electoral system")
	# plt.ylabel("Mean distance to capital (normalised)")
	# plt.show()

	results.to_csv('results.csv')