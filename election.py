import matplotlib.pyplot as plt


def elect(candidates, voters):
	""" candidates:	[[x, y]]
		voters:		[[x, y, weight]
	"""
	plt.figure()
	plt.scatter(voters[:,0], voters[:,1], c=voters[:,2], s=1)
	plt.scatter(candidates[:,0], candidates[:,1], c='k', marker='x')
	plt.show()
	return 0