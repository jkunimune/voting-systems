import matplotlib.pyplot as plt
import numpy as np


def elect(candidates, voters):
	""" candidates:	[[x, y]]
		voters:		[[x, y, weight]
	"""
	candidates[:,1] = np.degrees(np.arcsinh(np.tan(np.radians(candidates[:,1]))))
	voters[:,1] = np.degrees(np.arcsinh(np.tan(np.radians(voters[:,1]))))
	plt.figure()
	plt.scatter(voters[:,0], voters[:,1], c=voters[:,2], s=4)
	plt.scatter(candidates[:,0], candidates[:,1], c='k', marker='x')
	plt.axis('equal')
	plt.show()
	return 0
