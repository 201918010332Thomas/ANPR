# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFont
import os


provincelist = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"]

wordlist = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"]

# --- 绘制边界框


def DrawBox(im, box):
    draw = ImageDraw.Draw(im)
    draw.rectangle([tuple(box[0]), tuple(box[1])],  outline="#FFFFFF", width=3)

# --- 绘制四个关键点


def DrawPoint(im, points):

    draw = ImageDraw.Draw(im)

    for p in points:
        center = (p[0], p[1])
        radius = 5
        right = (center[0]+radius, center[1]+radius)
        left = (center[0]-radius, center[1]-radius)
        draw.ellipse((left, right), fill="#FF0000")

# --- 绘制车牌


def DrawLabel(im, label):
    draw = ImageDraw.Draw(im)
   # draw.multiline_text((30,30), label.encode("utf-8"), fill="#FFFFFF")
    font = ImageFont.truetype('simsun.ttc', 64)
    draw.text((30, 30), label, font=font)

# --- 图片可视化


def ImgShow(imgpath, box, points, label):
    # 打开图片
    im = Image.open(imgpath)
    DrawBox(im, box)
    DrawPoint(im, points)
    DrawLabel(im, label)
    # 显示图片
    im.show()
    im.save('result.jpg')


def main(imgpath):
    # 图像路径
    #imgpath = 'D:/CCPD2019/ccpd_base/03-1_1-269&435_530&531-530&524_269&531_269&442_530&435-0_0_26_24_25_23_30-135-104.jpg'

    # 图像名
    imgname = os.path.basename(imgpath).split('.')[0]

    # 根据图像名分割标注
    _, _, box, points, label, brightness, blurriness = imgname.split('-')

    # --- 边界框信息
    box = box.split('_')
    box = [list(map(int, i.split('&'))) for i in box]

    # --- 关键点信息
    points = points.split('_')
    points = [list(map(int, i.split('&'))) for i in points]
    # 将关键点的顺序变为从左上顺时针开始
    points = points[-2:]+points[:2]

    # --- 读取车牌号
    label = label.split('_')
    # 省份缩写
    province = provincelist[int(label[0])]
    # 车牌信息
    words = [wordlist[int(i)] for i in label[1:]]
    # 车牌号
    label = province+''.join(words)
    #print(label)
    # --- 图片可视化
    #ImgShow(imgpath, box, points, label)
    return label

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            if temp.endswith(".jpg"):
                allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)

if __name__ == '__main__':
   #file_root = r"d:/ccpd/val"
   file_root = r"D:/CCPD2019/train/"
   file_list=[]
   count=0
   textFile=open('D:/CCPD2019/train/test.txt','w')
   allFilePath(file_root,file_list)
   print(len(file_list))
   for img_path in file_list:
        if count>=1000:
            break
        count+=1
        # img_path = r"ccpd_yolo_test/02-90_85-173&466_452&541-452&553_176&556_178&463_454&460-0_0_6_26_15_26_32-68-53.jpg"
        #text_path= img_path.replace(".jpg",".txt")
        #img =cv2.imread(img_path)
        #rect,landmarks,landmarks_sort=get_rect_and_landmarks(img_path)
        # annotation=x1x2y1y2_yolo(rect,landmarks,img)
        #annotation=xywh2yolo(rect,landmarks_sort,img)
        #str_label = "0 "
        #for i in range(len(annotation[0])):
        ##        str_label = str_label + " " + str(annotation[0][i])
        #str_label = str_label.replace('[', '').replace(']', '')
        #str_label = str_label.replace(',', '') + '\n'
        #with open(text_path,"w") as f:
        #        f.write(str_label)
        #print(count,img_path)
        #print(img_path,img_path[img_path.index('\\')+1:len(img_path)])
        #print(count,main(img_path))
        textFile.write(main(img_path)+'\n')
   textFile.close()

#main()

