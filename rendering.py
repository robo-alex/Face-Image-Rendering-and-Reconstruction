import numpy as np
import cv2
import os
import msvcrt
import math
from matplotlib import pyplot as plt
import calZ
from mpl_toolkits.mplot3d import Axes3D

filename = 'data.txt'
dataset = []
file = open(filename,mode='r')
for line in file:
    line = line.split()
    dataset.append(line)
file.close()
i1=0
btest=np.zeros([168*168,3]).astype(np.float)
for item in dataset:
    btest[i1,0]=float(item[0])
    btest[i1,1]=float(item[1])
    btest[i1,2]=float(item[2])
    i1+=1


def rendering(dir):

    z = np.zeros([168, 168])

    img_S = ReadS(dir)
    imgx, average = ReadX(dir)

    img_valid = CheckValid(imgx, average)
    img_bg, img_validcount = CheckBG(img_valid)
    img_b = Step1(imgx, img_S, img_valid, img_bg)
    ''' 
    with open("b.txt", "w") as f:
        s = ""
        for i in range(168 ** 2):
            s += str(int(img_b[i][0] != 0))
            if i % 168 == 167:
                f.write(s)
                f.write('\n')
                s = ""
    '''
    imgs = render(dir, img_b, img_bg)
    
    z=calZ.inter(btest)
    
    #z=OptimizeZ(z)
    '''
    # print(z)
    with open("z.txt", "w") as f:
        for i in range(0,168):
            for j in range(0,168):
                s = ""
                s += str(z[i,j])
                f.write(s)
                if (j+1)%168==0:
                    f.write('\n')
                else:
                    f.write(' ')
    '''
    show3d(z)
    #sh = shadow(z, testS,img_bg)
    # ShowBGMap(img_bg)
    # ShowValidMap(img_valid)
    # msvcrt.getch()
    return z, imgs

def ReadS(dir):
    tempS = np.zeros([3, 7]).astype(np.float)
    with open(dir+'/train.txt', 'r') as file:
        for i in range(0, 7):
            lines = file.readline()
            _, a, b = (float(x) for x in lines.split(','))
            tempS[0][i] = math.sin(a*3.1416/180)*(math.cos(b*3.1416/180))
            tempS[1][i] = math.sin(b*3.1416/180)
            tempS[2][i] = (-1)*math.cos(b*3.1416/180)*(math.cos(a*3.1416/180))
    return tempS


def ReadX(dir):
    train_img_read = np.zeros([168, 168, 3, 7]).astype(np.uint8)
    imgx = np.zeros([168*168, 7]).astype(np.uint8)
    imgmean = np.zeros([7]).astype(np.float)
    for i in range(0, 7):
        train_img_read[:, :, :, i] = cv2.imread(
            dir+'/train/'+str(i+1)+'.bmp')
        imgx[:, i] = np.mean(train_img_read[:, :, :, i],
                             axis=2).flatten()
        imgmean[i] = np.mean(imgx[i])
    average = np.mean(imgmean)
    return imgx, average


def CheckValid(imgx, average):
    img_valid = np.zeros([168*168, 7]).astype(np.bool)
    for i in range(0, 168*168):
        for j in range(0, 7):
            if (imgx[i, j] > average):
                img_valid[i, j] = 1
            else:
                img_valid[i, j] = 0
    return img_valid


def ShowValidMap(img_valid):
    img_validmap = img_valid.reshape((168, 168, 7))
    for i in range(0, 7):
        plt.figure("Image")
        plt.imshow(img_validmap[:, :, 1], cmap='gray')
        plt.axis('on')
        plt.title('image')
        plt.show()
    msvcrt.getch()


def CheckBG(img_valid):
    img_bg = np.zeros([168*168]).astype(np.uint8)
    img_validcount = np.zeros([168*168]).astype(np.uint8)
    for i in range(0, 168*168):
        count = 0
        for j in range(0, 7):
            if img_valid[i, j] > 0:
                count += 1
        if (count >= 4):
            img_bg[i] = 1
            img_validcount = count
    return img_bg, img_validcount


def ShowBGMap(img_bg):
    temp = img_bg.reshape((168, 168))
    plt.figure("Image")
    plt.imshow(temp, cmap='gray')
    plt.axis('on')
    plt.title('image')
    plt.show()


def Step1(imgx, img_S, img_valid, img_bg):
    img_b = np.zeros([168*168, 3]).astype(np.float)

    for i in range(0, 168*168):
        if img_bg[i] > 0:
            temp_s = img_S[:, img_valid[i]]
            temp_x = imgx[i, img_valid[i]]
            temp_st = temp_s.transpose(1, 0)
            temp_inv = np.linalg.pinv(np.dot(temp_s, temp_st))
            img_b[i] = np.dot(np.dot(temp_x, temp_st), temp_inv)
            # print(img_S.transpose(1,0))
            # print(temp_st)
            # print(img_valid[i])
            # print(img_b[i])
            # msvcrt.getch()

    return img_b


def render(dir, img_b, img_bg):
    imgs = np.zeros([10, 168, 168]).astype(np.uint8)
    imgtemp = np.zeros([10, 168*168]).astype(np.uint8)
    testS = np.zeros([3, 10]).astype(np.float)
    with open(dir+'/test.txt', 'r') as file:
        for i in range(0, 10):
            lines = file.readline()
            _, a, b = (float(x) for x in lines.split(','))
            testS[0][i] = math.sin(a*3.1416/180)*(math.cos(b*3.1416/180))
            testS[1][i] = math.sin(b*3.1416/180)
            testS[2][i] = (-1)*math.cos(b*3.1416/180)*(math.cos(a*3.1416/180))

    
    for i in range(0, 10):
        for j in range(0, 168*168):
            if (img_bg[j] > 0):
                imgtemp[i][j] = np.dot(img_b[j], testS[:, i])
            else:
                imgtemp[i][j] = 0
            if(imgtemp[i][j] < 0):
                imgtemp[i][j] = 0
            if (j % 100 == 0):
                # print('Processing')
                # print(j)
                pass
        imgs[i] = imgtemp[i].reshape((168, 168))
    return imgs


# def shadow(Z, s,img_bg):
#     sh = np.zeros([10,168*168]).astype(np.uint8)
#     for im in range(0,10):
#         a=s[0,im]
#         b=s[1,im]
#         c=s[2,im]
#         for i in range(0, 168):
#             for j in range(0, 168):
#                 if (img_bg[i*168+j]):
#                     x0 = i
#                     y0 = 167-j
#                     if a > 0 and b < 0:
#                         for m in range(x0-1, 0):
#                             n = b*(m-x0)/a+y0
#                             if(math.ceil(n) <= 167):
#                                 if(Z[im,m, 167-math.ceil(n)] > c*(m-x0)/a+Z[im,x0, y0]):
#                                     sh[im,i, j] = 1
#                                     break;
#                                 if(Z[im,m, 167-math.floor(n)] > c*(m-x0)/a+Z[im,x0, y0]):
#                                     sh[im,i, j] = 1
#                                     break  
#                             elif(math.floor(n))<=167: 
                    
#                                 if(Z(im,m,167-math.floor(n))>c*(m-x0)/a+Z(im,x0,y0)):
#                                     sh[im,i,j]=1
#                                     break
#                             else:
#                                 break

#                         for n in range(y0+1,167):
#                             m=a*(n-y0)/b+x0
#                             if(math.floor(m)>=0):
#                                 if(Z(im,math.floor(m),167-n)>c*(n-y0)/b+Z(im,x0,y0)):
#                                     sh[im,i,j]=1
#                                     break
#                                 if(Z(im,math.ceil(m),167-n)>c*(n-y0)/b+Z(im,x0,y0)):
#                                     sh[im,i,j]=1
#                                     break
                    
#                             elif(math.ceil(n)>=0):
                    
#                                 if(Z(im,math.ceil(m),167-n)>c*(n-y0)/b+Z(im,x0,y0)):
#                                     sh[im,i,j]=1
#                                     break
                    
#                             else:
#                                 break

#                     elif a>0 and b>0:
            
#                         for m in range(x0-1,0):
#                             n=b*(m-x0)/a+y0
#                             if(math.floor(n)>=0):
                    
#                                 if(Z[im,m,167-math.ceil(n)]>c*(m-x0)/a+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
#                                 if(Z[im,167-math.floor(n)]>c*(m-x0)/a+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
                    
#                             elif(math.ceil(n)>=0):
#                                 if(Z[im,m,167-math.ceil(n)]>c*(m-x0)/a+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
#                             else:
#                                 break
            
#                         for n in range(y0-1,0):
#                             m=a*(n-y0)/b+x0
#                             if(math.floor(m)>=0):
                    
#                                 if(Z[im,math.floor(m),167-n]>c*(n-y0)/b+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
#                                 if(Z[im,math.ceil(m),167-n]>c*(n-y0)/b+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
                    
#                             elif(math.ceil(n)>=0):
                    
#                                 if(Z[im,math.ceil(m),167-n]>c*(n-y0)/b+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
                    
#                             else:
#                                 break

#                     elif a<0 and b<0:
            
#                         for m in range(x0+1,167):
#                             n=b*(m-x0)/a+y0
#                             if(math.ceil(n)<=167):
                    
#                                 if(Z[im,m,167-math.ceil(n)]>c*(m-x0)/a+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
#                                 if(Z[im,m,167-math.floor(n)]>c*(m-x0)/a+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
                    
#                             elif(math.floor(n)<=167):
                    
#                                 if(Z[im,m,167-math.floor(n)]>c*(m-x0)/a+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break 
                    
#                             else:
#                                 break

#                         for n in range(y0+1,167):
#                             m=a*(n-y0)/b+x0
#                             if(math.ceil(m)<=167):
                    
#                                 if(Z[im,math.ceil(m),167-n]>c*(n-y0)/b+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
#                                 if(Z[im,math.floor(m),167-n]>c*(n-y0)/b+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
                    
#                             elif(math.floor(m)<=167):
                    
#                                 if(Z[im,math.floor(m),167-n]>c*(n-y0)/b+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
                    
#                             else:
#                                 break

#                     else:
            
#                         for m in range(x0+1,167):
#                             n=b*(m-x0)/a+y0
#                             if(math.floor(n)>=0):
                    
#                                 if(Z[im,m,167-math.ceil(n)]>c*(m-x0)/a+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
#                                 if(Z[im,m,167-math.floor(n)]>c*(m-x0)/a+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
                    
#                             elif(math.ceil(n)<=167):
                    
#                                 if(Z[im,m,167-math.ceil(n)]>c*(m-x0)/a+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
                    
#                             else:
#                                 break
            
#                         for n in range(y0-1,0):
#                             m=a*(n-y0)/b+x0
#                             if(math.ceil(m)<=167):
#                                 if(Z[im,math.ceil(m),167-n]>c*(n-y0)/b+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
#                                 if(Z[im,math.floor(m),167-n]>c*(n-y0)/b+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
                    
#                             elif(math.floor(n)<=167):
                    
#                                 if(Z[im,math.floor(m),167-n]>c*(n-y0)/b+Z[im,x0,y0]):
#                                     sh[im,i,j]=1
#                                     break
                    
#                             else:
#                                 break 
#     for im in range(0,10):
#         for i in range(1,167):
#             for j in range(1,167):
#                 if(sh[im,i,j]==1) and (sh[im,i-1,j]==0 or sh[im,i+1,j]==0 or sh[im,i,j-1]==0 or sh[im,i,j+1]==0):
#                     sh[im,i,j]=0.5
#     return sh


def show3d(z):
    
    x = np.linspace(0,167,168)
    y = np.linspace(0,167,168)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(200, 200))
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, z,
                       rstride=1, 
                       cstride=1, 
                       cmap=plt.get_cmap('rainbow'))  
    ax.set_zlim(-50, 300)
    plt.title("3D")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def OptimizeZ(Z):
    ZZ=Z.copy()
    for i in range(0,84):
        ZZ[:,i]=Z[:,167-i]
        ZZ[:,167-i]=Z[:,i]
    Z=(Z+ZZ)/2
    # Z=Z*1.2
    frontpoint=np.max(Z)
    backpoint=np.min(Z)
    depth=168
    scale=depth/(frontpoint-backpoint)
    Z=Z*scale
    return Z

if __name__ == '__main__':
    z, imgs = rendering('./dataset/dataset_offline/P1')

