import glob
import os
import time
import numpy as np
import math
import torch
from ultralytics import YOLO
from ultralytics.utils.tif import writeTif, readTiff, readTif


def TifCroppingArray(img, SideLength):
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目
    ColumnNum = int((img.shape[0] - SideLength * 2) / (550 - SideLength * 2))
    #  行上图像块数目
    RowNum = int((img.shape[1] - SideLength * 2) / (550 - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (550 - SideLength * 2): i * (550 - SideLength * 2) + 550,
                      j * (550 - SideLength * 2): j * (550 - SideLength * 2) + 550]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[i * (550 - SideLength * 2): i * (550 - SideLength * 2) + 550,
                  (img.shape[1] - 550): img.shape[1]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - 550): img.shape[0],
                  j * (550 - SideLength * 2): j * (550 - SideLength * 2) + 550]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[(img.shape[0] - 550): img.shape[0],
              (img.shape[1] - 550): img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[0] - SideLength * 2) % (550 - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[1] - SideLength * 2) % (550 - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver


#  获得结果矩阵
def Result(shape, TifArray, npyfile, RepetitiveLength, RowOver, ColumnOver):
    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0
    for i, img in enumerate(npyfile):
        img = img.astype(np.uint8)

        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if (i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: 550 - RepetitiveLength, 0: 550 - RepetitiveLength] = img[0: 550 - RepetitiveLength,
                                                                               0: 550 - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                #  原来错误的
                # result[shape[0] - ColumnOver : shape[0], 0 : 550 - RepetitiveLength] = img[0 : ColumnOver, 0 : 550 - RepetitiveLength]
                #  后来修改的
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0: 550 - RepetitiveLength] = img[
                                                                                                        550 - ColumnOver - RepetitiveLength: 550,
                                                                                                        0: 550 - RepetitiveLength]
            else:
                result[j * (550 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                            550 - 2 * RepetitiveLength) + RepetitiveLength,
                0:550 - RepetitiveLength] = img[RepetitiveLength: 550 - RepetitiveLength, 0: 550 - RepetitiveLength]
                #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif (i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: 550 - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0: 550 - RepetitiveLength,
                                                                                  550 - RowOver: 550]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0], shape[1] - RowOver: shape[1]] = img[550 - ColumnOver: 550,
                                                                                        550 - RowOver: 550]
            else:
                result[j * (550 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                            550 - 2 * RepetitiveLength) + RepetitiveLength,
                shape[1] - RowOver: shape[1]] = img[RepetitiveLength: 550 - RepetitiveLength, 550 - RowOver: 550]
                #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: 550 - RepetitiveLength,
                (i - j * len(TifArray[0])) * (550 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (550 - 2 * RepetitiveLength) + RepetitiveLength
                ] = img[0: 550 - RepetitiveLength, RepetitiveLength: 550 - RepetitiveLength]
                #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if (j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0],
                (i - j * len(TifArray[0])) * (550 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (550 - 2 * RepetitiveLength) + RepetitiveLength
                ] = img[550 - ColumnOver: 550, RepetitiveLength: 550 - RepetitiveLength]
            else:
                result[j * (550 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                            550 - 2 * RepetitiveLength) + RepetitiveLength,
                (i - j * len(TifArray[0])) * (550 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (550 - 2 * RepetitiveLength) + RepetitiveLength,
                ] = img[RepetitiveLength: 550 - RepetitiveLength, RepetitiveLength: 550 - RepetitiveLength]
    return result


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.". format(device))
    # 权重文件
    model = YOLO(r'D:\code\yolov8-4b\runs\detect\train\weights\best.pt')
    model.to(device)

    area_perc = 0.5
    RepetitiveLength = int((1 - math.sqrt(area_perc)) * 550 / 2)

    imgDir = r"D:\code\detect\detectimages\1201\*.tiff"
    resultPath = r"D:\code"
    for imgPath in glob.glob(imgDir):
        t0 = time.time()
        imgName = os.path.basename(imgPath)[:-5]
        print(imgName)
        im_width, im_height, im_bands, projection, geotrans, big_image = readTiff(imgPath)
        TifArray, RowOver, ColumnOver = TifCroppingArray(big_image, RepetitiveLength)
        predicts = []
        for i in range(len(TifArray)):
            print("i:" + str(i))
            for j in range(len(TifArray[0])):
                image = TifArray[i][j]
                s, plot_img, coord_list, conf_list = model.predict(source=image, save=True, conf=0.7)
                predicts.append(plot_img)
        # 保存结果
        result_shape = (big_image.shape[0], big_image.shape[1], 3)
        result_data = Result(result_shape, TifArray, predicts, RepetitiveLength, RowOver, ColumnOver)

        result = np.transpose(result_data, (2, 0, 1))
        writeTif(os.path.join(resultPath, imgName + '.tif'), projection, geotrans, result)
        t1 = time.time()
        print(f' ({t1 - t0:.3f}s)')


if __name__ == '__main__':
    main()

