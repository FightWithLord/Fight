import cv2

path = '1.jpg'
img = cv2.imread(path)
row = img.shape[0]
col = img.shape[1]
print(int(row/256))
for i in range(int(row/256)):
    for j in range(int(col/256)):
        try:
            image = img[256*i:256*(i+1), 256*j:256*(j+1)]
            image1 = img[(128+256*i):(128+256*(i+1)), (128+256*j):(128+256*(j+1))]
            print("img")
            cv2.imwrite('./testA/'+str(10+i)+'_'+str(10+j)+'.jpg',image1)
            cv2.imwrite('./testA/'+ str(i) + '_' + str(j) + '.jpg', image)
        except:
            continue
