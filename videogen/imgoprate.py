import os
import cv2
import numpy as np

namefmt = "%05d_synthesized_image.jpg"
img_root = '/Users/kayc/Downloads/123/'  # 这里写你的文件夹路径，比如：/home/youname/data/img/,注意最后一个文件夹要有斜杠


def genvideo():
    fps = 24  # 保存视频的FPS，可以适当调整

    # 可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter('saveVideo.avi', fourcc, fps, (512, 512))  # 最后一个是保存图片的尺寸

    for i in range(1000):
        frame = cv2.imread(img_root + namefmt % (i + 1))
        # print(img_root + namefmt % (i + 1) + '.jpg')
        videoWriter.write(frame)
    videoWriter.release()


# genvideo()


def scale():
    # img = cv2.imread(img_root + namefmt % 1)
    # res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('res', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    img = cv2.imread(img_root + namefmt % 1, 1)
    img_info = img.shape
    print(img_info)
    image_height = img_info[0]
    image_weight = img_info[1]
    image_mode = img_info[2]

    mat_scale = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
    dst = cv2.warpAffine(img, mat_scale, (int(image_height * 2), int(image_weight * 2)))
    cv2.imshow('image_dst', dst)
    cv2.waitKey(0)


scale()


def move():
    img = cv2.imread('messi5.jpg', 0)
    rows, cols = img.shape
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow('img', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotation():
    img = cv2.imread('messi5.jpg', 0)
    rows, cols = img.shape

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))


def gray():
    img = cv2.imread('img/image.png')
    b, g, r = cv2.split(img)
    img = cv2.merge((b, g, r))


def border():
    BLUE = [255, 0, 0]

    img1 = cv2.imread('opencv_logo.png')

    replicate = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
    wrap = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_WRAP)
    constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=BLUE)


def overlap():
    # Load two images
    img1 = cv2.imread('227351.jpg')  # 背景
    img2 = cv2.imread('logo.png')  # logo

    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cv2tColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)  # 这个254很重要
    mask_inv = cv2.bitwise_not(mask)

    cv2.imshow('mask', mask_inv)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)  # 这里是mask,我参考的博文写反了,我改正了,费了不小劲

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)  # 这里才是mask_inv

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst

    cv2.imshow('res', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
