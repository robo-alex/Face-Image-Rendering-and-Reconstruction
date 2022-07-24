import numpy as np
from math import pi, sin, cos


def inter(b):
    h = w = 168

    b = np.array([[b[i, j] if b[i, j] != 0 or j != 3 else 1e-10 for j in range(3)] for i in range(w * h)])

    b = np.array([[b[i, j] if b[i, j] != 0 or j == 3 else 1e-30 for j in range(3)] for i in range(w * h)])

    z_x = np.reshape(-b[:, 0] / b[:, 2] , (h, w))
    z_y = np.reshape(-b[:, 1] / b[:, 2], (h, w))
    
    max = 1
    lam = 1
    u = 3

    for k in range(1, w - 1):
        for l in range(1, h - 1):
            if abs(z_x[l, k])>max or abs(z_y[l, k]) > max:
                z_x[l, k] = (z_x[l - 1, k] + z_x[l + 1, k] + z_x[l, k + 1] + z_x[l, k - 1]) / 4
                z_y[l, k] = (z_y[l - 1, k] + z_y[l + 1, k] + z_y[l, k + 1] + z_y[l, k - 1]) / 4

    zz_x = np.zeros((h*2, w*2))
    zz_x[0 : h, 0 : w] = z_x[:, :]
    zz_x[h : 2 * h, 0 : w] = z_x[h - 1 : : -1]
    zz_x[:, w : w * 2] = zz_x[:, w - 1 : : -1]
    zz_y = np.zeros((h*2,w*2))
    zz_y[0 : h, 0 : w] = z_y[:, :]
    zz_y[h : 2 * w, 0 : w] = z_y[h - 1 : : -1]
    zz_y[:, w : w * 2] = zz_y[:, w - 1 : : -1]
    z_x = zz_x
    z_y = zz_y

    h = h * 2
    w = w * 2

    for i in range(1, w - 1):
        for j in range(1, h - 1):
            if abs(z_x[j, i]) > max or abs(z_y[j, i]) > max:
                z_x[j, i] = (z_x[j - 1, i] + z_x[j + 1, i] + z_x[j, i + 1] + z_x[j, i - 1]) / 4
                z_y[j, i] = (z_y[j - 1, i] + z_y[j + 1, i] + z_y[j, i + 1] + z_y[j, i - 1]) / 4
    
    C_x = np.fft.fft2(z_x)
    C_y = np.fft.fft2(z_y)

    C = np.zeros((h, w)).astype('complex')
    C_xx = np.zeros((h, w)).astype('complex')
    C_yy = np.zeros((h, w)).astype('complex')

    for m in range(w):
        for n in range(h):
            wx = 2 * pi * (m - 1) / w
            wy = 2 * pi * (n - 1) / h
            if sin(wx) == 0 and sin(wy) == 0:
                C[n, m] = 0
            else:
                cons=(1 + lam) * (sin(wx) ** 2 + sin(wy) ** 2) + u * (sin(wx) ** 2 + sin(wy) **2) ** 2
                C[n, m]=(C_x[n, m] * (complex(0, -1) * sin(wx)) + C_y[n, m] * (complex(0, -1) * sin(wy))) / cons
            C_xx[n, m] = complex(0, 1) * sin(wx) * C[n, m]
            C_yy[n, m] = complex(0, 1) * sin(wy) * C[n, m]
    
    h = h // 2
    w = w // 2
    Z = np.fft.ifft2(C).real
    Z = Z[0 : h, 0 : w]

    Z_xx = np.fft.ifft2(C_xx).real
    Z_yy = np.fft.ifft2(C_yy).real

    Z_xx = Z_xx[0 : h, 0 : w]
    Z_yy = Z_yy[0 : h, 0 : w]

    for i in range(1, w - 1):
        for j in range(1, h - 1):
            if abs(Z_xx[j, i]) > max or abs(Z_yy[j, i]) > max:
                Z_xx[j, i] = (Z_xx[j - 1, i] + Z_xx[j + 1, i] + Z_xx[j, i + 1] + Z_xx[j, i - 1]) / 4
                Z_yy[j, i] = (Z_yy[j - 1, i] + Z_yy[j + 1, i] + Z_yy[j, i + 1] + Z_yy[j, i - 1]) / 4
                Z[j, i] = (Z[j - 1, i] + Z[j + 1, i] + Z[j, i + 1] + Z[j, i - 1]) / 4


    return Z
