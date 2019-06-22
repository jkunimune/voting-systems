import matplotlib.pyplot as plt
import numpy as np


def elect(candidates, voters, system='plurality', verbose=True):
	""" candidates:	[[x, y]]
		voters:		[[x, y, weight]
	"""
	candidates[:,1] = np.degrees(np.arcsinh(np.tan(np.radians(candidates[:,1]))))
	voters[:,1] = np.degrees(np.arcsinh(np.tan(np.radians(voters[:,1]))))

	if system == 'plurality':
		return plurality(candidates, voters, verbose)
	elif system == 'primaries':
		return plurality(candidates, voters, verbose)
	elif system == 'runoff':
		return runoff(candidates, voters, verbose)
	elif system == 'instant-runoff':
		return instant_runoff(candidates, voters, verbose)
	elif system == 'condorcet':
		return condorcet(candidates, voters, verbose)
	elif system == 'score':
		return score(candidates, voters, 10, verbose)
	elif system == 'aproval':
		return score(candidates, voters, 1, verbose)
	else:
		raise ValueError(system)


def plurality(candidates, voters, verbose):
	votes = np.argmin(np.hypot(
		np.expand_dims(voters[:,0], 0) - np.expand_dims(candidates[:,0], 1),
		np.expand_dims(voters[:,1], 0) - np.expand_dims(candidates[:,1], 1)), axis=0)
	tallies = np.histogram(votes, bins=candidates.shape[0], weights=voters[:,2])[0]
	if verbose:
		print(tallies)
	return np.argmax(tallies)
