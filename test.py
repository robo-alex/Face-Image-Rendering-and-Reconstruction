import numpy as np

from itertools import product
from numpy.linalg import inv

def concat(L1, L2):
	L = L1.copy()
	L.extend(L2)
	return L

def column_vector(x):
	return np.array([x], dtype='float64').transpose()

def poly(b, m, n):
	r, _ = b.shape
	A, Y = [], []
	for i, j in product(range(np.sqrt(r)),range(np.sqrt(r))):
		x, y = np.float64(1 + i), np.float64(1 + j)
		z1, z2, z3 = b[i,j]
		A.append([i * x ** (i - 1) * y ** j if (i != 0) else 0 \
			for i in range(m + 1) for j in range(n + 1) if (i != 0 or j != 0)])
		Y.append(-z1 / z3)
		A.append([j * x ** i * y ** (j - 1) if (j != 0) else 0\
			for i in range(m + 1) for j in range(n + 1) if (i != 0 or j != 0)])
		Y.append(-z2 / z3)

	A = np.matrix(A)
	Y = column_vector(Y)
	X = np.dot(np.dot(inv(np.dot(A.transpose(),A)),A.transpose()),Y)

	return np.reshape(concat([0],X.flatten().tolist()[0]),(m + 1,n + 1))


m, n = 4, 5
b = np.zeros((400,3))
k = lambda x,y: 1
for i in range(20): 	 	
	for j in range(20):
		x, y = 1 + i, 1 + j
		b[i, j] = (2*(3*x**2 + 5*y**2), 2*(4*y**3 + 10*x*y + 3), -2)

np.set_printoptions(suppress=True)

print(poly(b, m, n))