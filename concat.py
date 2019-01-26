import cv2
import os
import numpy as np

path = '../../results/style_monet_pretrained/test_latest/images'
files= os.listdir(path) #得到文件夹下的所有文件名称

a, b = 0, 0
for file in files:
    if file[1] == '_':
        x = int(file[0])
        y = int(file[2])
        if x > a:
            a = x
        if y > b:
            b = y

img = np.zeros((256*(a+1),256*(b+1),3), np.uint8)

img2 = np.zeros((256*(a+1),256*(b+1),3), np.uint8)
for file in files:
    if file[-10:-4] == 'fake_B':
        image = cv2.imread(path+'/'+file)
        if file[1] != '_':
            i = int(file[:2])
            j = int(file[3:5])
            print("i,j",i,j)
            try:
                img2[(128+256*(i-10)):(128+256*(i-9)), (128+256*(j-10)):(128+256*(j-9))] = image
            except:
                continue

        else:
            i = int(file[0])
            j = int(file[2])
            img[256*i:256*(i+1), 256*j:256*(j+1)] = image
#img=cv2.GaussianBlur(img, (7, 7), 0)
#img=cv2.medianBlur(img,11)
img = (0.8*img+0.2*img2)
cv2.imwrite('result.jpg', img)


