import io
import numpy as np

from math import sqrt
from itertools import product
from numpy.linalg import inv
from decimal import *


getcontext().prec = 50

def poly(b, m, n):
	def concat(L1, L2):

		L = L1.copy()
		L.extend(L2)
		return L

	def column_vector(x):

		return np.array([x]).transpose()

	def eval_polynomial(coeff,x,y):
		S=0
		Infinity=range(10**10)
		for i,d in zip(Infinity,coeff):
			for j,c in zip(Infinity,d):
				S += c*x**i*y**j
		return S

	def eliminate(r1, r2, col, target=0):
		fac = (r2[col]-target) / r1[col]
		for i in range(len(r2)):
			r2[i] -= fac * r1[i]

	def gauss(a):
		for i in range(len(a)):
			if a[i][i] == 0:
				for j in range(i+1, len(a)):
					if a[i][j] != 0:
						a[i], a[j] = a[j], a[i]
						break
				else:
					print("MATRIX NOT INVERTIBLE")
					return -1
			for j in range(i+1, len(a)):
				eliminate(a[i], a[j], i)
		for i in range(len(a)-1, -1, -1):
			for j in range(i-1, -1, -1):
				eliminate(a[i], a[j], i)
		for i in range(len(a)):
			eliminate(a[i], a[i], i, target=1)
		return a

	def inverse(a):
		a = a.tolist()
		tmp = [[] for _ in a]
		for i,row in enumerate(a):
			assert len(row) == len(a)
			tmp[i].extend(row + [0]*i + [1] + [0]*(len(a)-i-1))
		gauss(tmp)
		ret = []
		for i in range(len(tmp)):
			ret.append(tmp[i][len(tmp[i])//2:])
		return np.mat(ret)

	def chazhi(b, m, n, x0, y0, dw, dh):
		#w, h为矩阵b的维数
		w, h, _ = b.shape
		A, Y = [], []
		#product是迭代器的笛卡尔积
		for i0, j0 in product(range(w),range(h)):
			x, y = x0 + i0 * dw, y0 + j0 * dh
			z1, z2, z3 = b[i0,j0]
			if abs(z3) > 0:
				A.append([(i*x**(i-1)*y**j if (i!=0) else 0) \
					for i in range(m+1) for j in range(n+1)])
				Y.append(-z1/z3)
				A.append([(j*x**i*y**(j-1) if (j!=0) else 0) \
					for i in range(m+1) for j in range(n+1)])
				Y.append(-z2/z3)
		A = np.array(A, dtype = "float64")
		Y = np.array(column_vector(Y), dtype = "float64")
		X = np.dot(np.dot(np.linalg.pinv(np.dot(A.transpose(),A)),A.transpose()),Y)
		return np.reshape(X.transpose().tolist()[0],(m+1,n+1))

	def sgn(x):
		return 1 if (x>=0) else -1

	def get_k(b):
		M, N = b.shape[0],b.shape[1]
		k = np.zeros((M, N))
		for i,j in product(range(M),range(N)):
			k[i,j]=-sgn(b[i,j,2])*sqrt(b[i,j,0]**2+b[i,j,1]**2+b[i,j,2]**2)
		return k

	Minx = - 168 / 100
	Miny = - 168 / 100
	dx = 1 / 50
	dy = 1 / 50

	N=int(sqrt(len(b))+0.5)
	b=b.reshape((N,N,3)).transpose(1,0,2)
	coeff=chazhi(b,m,n,Minx,Miny,dx,dy)
	S=np.zeros((N,N))
	for i,x in zip(range(N),range(1,N+1)):
		for j,y in zip(range(N),range(1,N+1)):
			if b[i, j, 0] != 0 or b[i, j, 1] != 0 or b[i, j, 2] != 0 or min(i, N - i) **2 + min(j, N - j) **2 >= 400:
				S[i,j] = 100 * eval_polynomial(coeff,x * dx + Minx,y * dy + Miny) + 200
			else:
				S[i,j] = 0
	return S, get_k(b)

if __name__ == "__main__":

	#z = y^4 + x^3 + 5*x*y^2 + 3*y + 1
	m, n = 4, 5
	b = []
	k = lambda x,y: x+y
	for y in range(1,169):
		for x in range(1,169):
			b.append(k(x,y)*(np.array([3*x**2+5*y**2, 4*y**3+10*x*y+3, -1])/ \
				sqrt((3*x**2+5*y**2)**2+(4*y**3+10*x*y+3)**2+1)))
	b=np.array(b)


	np.set_printoptions(suppress=True)
	value,k = poly(b,m,n)
	print(value)
	print(k)
