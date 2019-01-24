import os
import cv2
import numpy as np

namefmt = "%05d_synthesized_image.jpg"


# namefmt = "%d.jpg"
# img_root = '/Users/kayc/Downloads/123/'  # 这里写你的文件夹路径，比如：/home/youname/data/img/,注意最后一个文件夹要有斜杠


def genvideo():
    img_root = '/Users/kayc/Downloads/images/'
    fps = 24  # 保存视频的FPS，可以适当调整

    # 可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter('saveVideo.avi', fourcc, fps, (512, 512))  # 最后一个是保存图片的尺寸

    for i in range(1000):
        frame = cv2.imread(img_root + namefmt % (i + 1))
        # print(img_root + namefmt % (i + 1) + '.jpg')
        videoWriter.write(frame)
    videoWriter.release()


def genvideo_fromimgs(imgs):
    fps = 24  # 保存视频的FPS，可以适当调整

    # 可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter('saveVideo.avi', fourcc, fps, (512, 512))  # 最后一个是保存图片的尺寸

    for i in range(len(imgs)):
        # print(img_root + namefmt % (i + 1) + '.jpg')
        videoWriter.write(imgs[i])
    videoWriter.release()


genvideo()


#
# def scale():
#     img = cv2.imread(img_root + namefmt % 1)
#     res = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
#     cv2.imshow('res', res)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def scale(img, factor):
    # img = cv2.imread(img_root + namefmt % 1)
    res = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    return res
    # cv2.imshow('res', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # img = cv2.imread(img_root + namefmt % 1, 1)
    # img_info = img.shape
    # print(img_info)
    # image_height = img_info[0]
    # image_weight = img_info[1]
    # image_mode = img_info[2]
    #
    # mat_scale = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
    # dst = cv2.warpAffine(img, mat_scale, (int(image_height * 2), int(image_weight * 2)))
    # cv2.imshow('image_dst', dst)
    # cv2.waitKey(0)


# scale()


def move():
    img = cv2.imread(img_root + namefmt % 1, 1)
    print(img.shape)
    rows, cols, _ = img.shape
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow('img', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# move()


def rotation():
    img = cv2.imread('messi5.jpg', 0)
    rows, cols = img.shape

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))


def gray(img):
    # img = cv2.imread('img/image.png')
    b, g, r = cv2.split(img)
    img = cv2.merge((b, g, r))
    return img


def border(img):
    BLUE = [0, 0, 0]

    # img1 = cv2.imread('opencv_logo.png')

    # replicate = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    # reflect = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT)
    # reflect101 = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
    # wrap = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_WRAP)
    # ratio = img.shape[0] / img.shape[1]
    scale_factor = 1
    if img.shape[0] < img.shape[1]:
        scale_factor = 512 / img.shape[0]
    else:
        scale_factor = 512 / img.shape[1]

    # print("border1", img.shape)
    img = scale(img, scale_factor)
    # print("border2", img.shape)

    width = int(max(0, (512 - img.shape[0]) / 2))
    height = int(max(0, (512 - img.shape[1]) / 2))
    img = cv2.copyMakeBorder(img, width, width, height, height, cv2.BORDER_CONSTANT, value=BLUE)

    # print("border4", img.shape)
    # cv2.imshow("1", img)
    # cv2.waitKey(0)
    return img


def crop(img, x, y, scale):
    window = 512
    # img = cv2.imread(img_root + namefmt % 1)
    print(scale)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # print(img.shape)
    # print(x, x + window)
    # print(y, y + window)
    img = img[x:x + window, y:y + window]
    return img

    # cv2.imshow('res', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# crop(10, 200, 1.5)


def h_forpng1(img_back, img_mask):
    # img_mask = np.zeros_like(img_back)
    # img_mask = img_back[:, :, :3]
    img_mask = img_mask.reshape((img_mask.shape[0], img_mask.shape[1], 1))
    img_mask_stack = np.concatenate((img_mask, img_mask, img_mask), axis=2)
    img_back_zero = img_back.copy()
    img_back_zero[:, :, 3] = 0
    img_ret = np.where(img_mask_stack == 1, img_back, img_back_zero)

    return img_ret


def h_forpng(img_back, img_fore, x, y):
    # img_mask = np.zeros_like(img_back)
    # img_mask = img_back[:, :, :3]
    img_crop = img_back[x:x + img_fore.shape[0], y:y + img_fore.shape[1]]  # 截取部分
    img_condition = img_fore[:, :, 3].reshape((img_fore.shape[0], img_fore.shape[1], 1))  # alpha值
    img_condition = np.concatenate((img_condition, img_condition, img_condition), axis=2)
    img_fore_rgb = img_fore[:, :, :3]  # rgb值
    # print()
    img_mask = np.where(img_condition <= 200, img_crop, img_fore_rgb)
    img_back[x:x + img_fore.shape[0], y:y + img_fore.shape[1]] = img_mask
    # img_mask[x:x + img_png.shape[0], y:y + img_png.shape[1]] = img_png[:, :, :3]

    return img_back


def overlap():
    name = ['1.png', '2.png', '3.jpg']
    # Load two images
    img1 = cv2.imread('/Volumes/KHD/Proj/2019年01月16日GoogleHackathon/6矢量化素材/img_10.jpg')  # logo
    img2 = cv2.imread("/Volumes/KHD/Proj/2019年01月16日GoogleHackathon/6矢量化素材/img_70.png", -1)  # 背景
    # print(img2[:10, :10, 3])
    h_forpng(img1, img2, 0, 0)
    # I want to put logo on top-left corner, So I create a ROI
    # rows, cols, channels = img2.shape
    # roi = img1[0:rows, 0:cols]
    #
    # # Now create a mask of logo and create its inverse mask also
    # img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)  # 这个254很重要
    # mask_inv = cv2.bitwise_not(mask)
    #
    # # cv2.imshow('mask', mask_inv)
    # # Now black-out the area of logo in ROI
    # img1_bg = cv2.bitwise_and(roi, roi, mask=mask)  # 这里是mask,我参考的博文写反了,我改正了,费了不小劲
    #
    # # Take only region of logo from logo image.
    # img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)  # 这里才是mask_inv
    #
    # # Put logo in ROI and modify the main image
    # dst = cv2.add(img1_bg, img2_fg)
    # img1[0:rows, 0:cols] = dst
    #
    # cv2.imwrite("./1.jpg", img1)
    cv2.imshow('res', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# overlap()


def overlap_img(img1, img2):
    np.zeros((512, 512))
    # Load two images
    # img1 = cv2.imread('227351.jpg')  # 背景
    # img2 = cv2.imread('logo.png')  # logo

    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cv2tColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)  # 这个254很重要
    mask_inv = cv2.bitwise_not(mask)

    # cv2.imshow('mask', mask_inv)
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


def genvideo_fromfile_p1():
    img_root = '/Volumes/KHD/Proj/2019年01月16日GoogleHackathon/9p1/'
    fps = 24  # 保存视频的FPS，可以适当调整
    name = ['1.png', '2.png', '3.jpg']
    # name = ['1.png']

    # 可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter('saveVideo.avi', fourcc, fps, (512, 512), isColor=False)  # 最后一个是保存图片的尺寸
    # scale, (x_src,x_target)
    script = [[[1.5, 1.1], [1, 0], [300, 0]],
              [[1.1, 1.1], [0, 60], [0, 0]],
              [[1.5, 1.1], [200, 20], [0, 30]],
              [[1.1, 1.1], [20, 20], [30, 60]],
              [[1, 1], [0, 1], [10, 250]],
              [[1, 1], [0, 1], [250, 450]]
              ]
    periods = [40, 50, 40]
    for i in range(2 * len(name)):
        frame = cv2.imread(img_root + name[i // 2])
        print("1")
        frame = border(frame)
        period = periods[i // 2]
        s_scale = script[i][0][0]
        t_scale = script[i][0][1]
        s_x = script[i][1][0]
        t_x = script[i][1][1]
        s_y = script[i][2][0]
        t_y = script[i][2][1]
        print("2")
        for j in range(period):
            # print((t_x - s_x) / period * j + s_x, (t_y - s_y) / period * j + t_y,
            #       (t_scale - s_scale) / period * j + s_scale)
            person = cv2.imread("/Users/kayc/Downloads/123/%05d_synthesized_image.jpg" % (j + 300))
            moon = cv2.imread("/Volumes/KHD/Proj/2019年01月16日GoogleHackathon/6矢量化素材/img_70.png", -1)
            moon = scale(moon, 0.15)

            # person = cv2.imread("/Volumes/KHD/Proj/2019年01月16日GoogleHackathon/6矢量化素材/img_70.png", -1)
            # print("/Users/kayc/Downloads/123/%05d_synthesized_image.jpg" % (j + 1))
            # print(person.shape)
            person = scale(person, 0.35)
            pp = np.ones_like(person) * 255
            person = np.concatenate((person, pp), axis=2)
            if i in [0, 1]:
                h_forpng(frame, moon, 50, 400)
            if i in [2, 3]:
                h_forpng(frame, moon, 50, 450)
            if i in [2, 3]:
                h_forpng(frame, person, 300, 100)

            # h_forpng(frame, person, 50, 400)
            f = crop(frame,
                     int((t_x - s_x) / period * j + s_x),
                     int((t_y - s_y) / period * j + s_y),
                     (t_scale - s_scale) / period * j + s_scale
                     )

            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            # f = np.hstack([f, f, f])

            # print(img_root + namefmt % (i + 1) + '.jpg')

            videoWriter.write(f)
    print("3")
    videoWriter.release()

# genvideo_fromfile()
