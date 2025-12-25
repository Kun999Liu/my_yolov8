import glob
import os
import time
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils.tif import writeTif, readTiff, readTif, readTiff_uint16


def plot_one_box(x, im, color=(255, 255, 255), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    # assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return im


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.". format(device))
    # 权重文件
    model = YOLO(r'D:\code\yolov8-4b\runs\detect\train\weights\best.pt')
    model.to(device)
    clipSize = 416

    imgDir = r"D:\code\detect\detectimages\1201\*.tiff"
    resultPath = r"D:\code\detect"

    for imgPath in glob.glob(imgDir):
        t0 = time.time()
        im_width, im_height, im_bands, projection, geotrans, im_data = readTiff(imgPath)
        im_ori = readTiff_uint16(imgPath)
        heightI = int(im_height / clipSize)
        widthI = int(im_width / clipSize)
        imgName = os.path.basename(imgPath)[:-5]
        BigImg = im_data
        OriImg = im_ori
        print(imgName)
        fristV = True
        for ih in range(heightI):
            imgH = None
            fristH = True
            print("行块：" + str(ih))
            for iw in range(widthI):
                img = BigImg[ih * clipSize:(ih + 1) * clipSize, iw * clipSize:(iw + 1) * clipSize, :]
                ori_img = OriImg[ih * clipSize:(ih + 1) * clipSize, iw * clipSize:(iw + 1) * clipSize, :]
                ori_img = np.ascontiguousarray(ori_img)
                s, imgClip, conf_list, coord_list = model.predict(source=img, save=True, conf=0.6)
                if coord_list != []:
                    for co in range(0, len(coord_list)):
                        coord_list_co = coord_list[co]
                        conf_list_co = conf_list[co]

                        c1, c2 = (int(coord_list_co[0]), int(coord_list_co[1])), (int(coord_list_co[2]), int(coord_list_co[3]))
                        retval, baseLine = cv2.getTextSize(conf_list_co, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
                        topleft = (c1[0], c1[1] - retval[1] - 5)
                        bottomright = (topleft[0] + retval[0], topleft[1] + retval[1] + 5)
                        cv2.rectangle(ori_img, (topleft[0], topleft[1] - baseLine), bottomright, thickness=-1, color=(255, 255, 255))
                        cv2.rectangle(ori_img, c1, c2, color=(255, 255,  255), thickness=3)
                        cv2.putText(ori_img, conf_list_co, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255),
                                    2, cv2.LINE_AA)
                if fristH == True:
                    imgH = ori_img
                    fristH = False
                else:
                    imgH = np.concatenate((imgH, ori_img), axis=1)

            if fristV == True:
                imgV = imgH
                fristV = False
            else:
                imgV = np.concatenate((imgV, imgH), axis=0)

        result = np.transpose(imgV, (2, 0, 1))
        writeTif(os.path.join(resultPath, imgName + '.tif'), projection, geotrans, result)
        t1 = time.time()
        print(f' ({t1 - t0:.3f}s)')


if __name__ == '__main__':
    main()
