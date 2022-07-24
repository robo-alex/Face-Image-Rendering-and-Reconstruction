import cv2
import numpy as np
import os
from rendering import rendering

dirname = 'dataset_offline'
dirs = os.listdir(dirname)
for dir in dirs:
    if dir[0] == 'P':
        z, imgs = rendering(os.path.join(dirname, dir))
        np.save(os.path.join(dirname, dir, 'z.npy'), z)
        for i in range(len(imgs)):
            cv2.imwrite(os.path.join(dirname, dir, str(i)+'.bmp'), imgs[i])

