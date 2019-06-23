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
	elif system == 'primary':
		return primary(candidates, voters, verbose)
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
	votes = np.argmax(1/np.hypot(
		np.expand_dims(voters[:,0], 0) - np.expand_dims(candidates[:,0], 1),
		np.expand_dims(voters[:,1], 0) - np.expand_dims(candidates[:,1], 1)), axis=0)
	tallies = np.sum(voters[:,2]*(votes==np.expand_dims(np.arange(candidates.shape[0]), 1)), axis=1)
	if verbose:
		print("Tallies: {}".format(tallies/tallies.sum()))
	return np.argmax(tallies)


def primary(candidates, voters, verbose):
	covariance = np.cov(voters[:,0:2].transpose(), fweights=voters[:,2])
	w, v = np.linalg.eig(covariance) # Give me that eigenvalue!
	party_axis = v[:,np.argmax(w)]
	party_boundary = np.average(np.sum(party_axis*voters[:,0:2], axis=1), weights=voters[:,2])

	voter_aff = np.sum(party_axis*voters[:,0:2], axis=1) >= party_boundary
	candidate_aff = np.sum(party_axis*candidates[:,0:2], axis=1) >= party_boundary

	primary_tallies = [None, None]
	qualified = np.zeros((candidates.shape[0], 1))
	for i in range(2):
		affiliated = np.expand_dims(candidate_aff==i, 1)
		votes = np.argmax(1/np.hypot(
			np.expand_dims(voters[voter_aff==i,0], 0) - np.expand_dims(candidates[:,0], 1),
			np.expand_dims(voters[voter_aff==i,1], 0) - np.expand_dims(candidates[:,1], 1))*affiliated, axis=0)
		primary_tallies[i] = np.sum(voters[voter_aff==i,2]*(votes==np.expand_dims(np.arange(candidates.shape[0]), 1)), axis=1)
		qualified[np.argmax(primary_tallies[i])] = 1

	votes = np.argmax(1/np.hypot(
		np.expand_dims(voters[:,0], 0) - np.expand_dims(candidates[:,0], 1),
		np.expand_dims(voters[:,1], 0) - np.expand_dims(candidates[:,1], 1))*qualified, axis=0)
	final_tallies = np.sum(voters[:,2]*(votes==np.expand_dims(np.arange(candidates.shape[0]), 1)), axis=1)

	if verbose:
		for i, party in [(0, "S"), (1, "N")]:
			print("Primary {} tallies: {}".format(party, primary_tallies[i]/primary_tallies[i].sum()))
		print("Final tallies:   {}".format(final_tallies/final_tallies.sum()))

	return np.argmax(final_tallies)


def runoff(candidates, voters, verbose):
	votes = np.argmax(1/np.hypot(
		np.expand_dims(voters[:,0], 0) - np.expand_dims(candidates[:,0], 1),
		np.expand_dims(voters[:,1], 0) - np.expand_dims(candidates[:,1], 1)), axis=0)
	initial_tallies = np.sum(voters[:,2]*(votes==np.expand_dims(np.arange(candidates.shape[0]), 1)), axis=1)

	if initial_tallies.max() > initial_tallies.sum()/2:
		if verbose:
			print("Initial tallies: {}".format(initial_tallies/initial_tallies.sum()))
		return np.argmax(initial_tallies)

	qualified = np.array([[np.sum(initial_tallies > t) <= 1] for t in initial_tallies])
	votes = np.argmax(1/np.hypot(
		np.expand_dims(voters[:,0], 0) - np.expand_dims(candidates[:,0], 1),
		np.expand_dims(voters[:,1], 0) - np.expand_dims(candidates[:,1], 1))*qualified, axis=0)
	final_tallies = np.sum(voters[:,2]*(votes==np.expand_dims(np.arange(candidates.shape[0]), 1)), axis=1)

	if verbose:
		print("Initial tallies: {}".format(initial_tallies/initial_tallies.sum()))
		print("Final tallies:   {}".format(final_tallies/final_tallies.sum()))

	return np.argmax(final_tallies)


def instant_runoff(candidates, voters, verbose):
	votes = rankdata(np.hypot(
		np.expand_dims(voters[:,0], 0) - np.expand_dims(candidates[:,0], 1),
		np.expand_dims(voters[:,1], 0) - np.expand_dims(candidates[:,1], 1)), axis=0)
	qualified = list(range(candidates.shape[0]))
	disqualified = []

	while True:
		tallies = np.sum(voters[:,2]*(votes==0), axis=1)
		if verbose:
			print("Tallies: {}".format(tallies/tallies.sum()))

		if tallies.max() > tallies.sum()/2:
			return tallies.argmax()
		else:
			loser = qualified[tallies[qualified].argmin()]
			qualified.remove(loser)
			disqualified.append(loser)
			while np.any(votes[disqualified,:]==0):
				for loser in disqualified[::-1]: # make sure no one's vote goes to a disqualified candidate, ever.
					votes[:,votes[loser,:]==0] -= 1


def rankdata(data, axis=None):
	return np.argsort(np.argsort(data, axis=axis), axis=axis) # yeah, it's inefficient, but I'm only using it for, like, 5 things at a time.
