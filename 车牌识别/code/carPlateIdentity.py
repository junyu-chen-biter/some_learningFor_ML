import cv2
import os
import sys
import numpy as np
import torch
from torchvision import transforms
from plateNeuralNet import *
from charNeuralNet import *
import random

torch.cuda.set_device(0)

# 设置随机种子，使这个每次训练效果差不多
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(233)

char_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵',
              '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '晋',
              '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']



"""
对灰度图像进行直方图均衡化，通过调整像素的累积概率分布，增强图像对比度，便于后续边缘检测。
这些工具都可以当作板子，用于特征工程
"""
def hist_image(img): # 灰度级转灰度图
    assert img.ndim==2     #确保传入的图像 img 是一个二维数组，即图像。
    hist = [0 for i in range(256)]
    img_h,img_w = img.shape[0],img.shape[1] #获取图像的高 img_h 和宽 img_w

    for row in range(img_h):
        for col in range(img_w): #遍历图像的每个像素，根据像素值（灰度级）在 hist 列表中对应的位置加一
            hist[img[row,col]] += 1
    p = [hist[n]/(img_w*img_h) for n in range(256)] #遍历图像的每个像素，根据像素值（灰度级）在 hist 列表中对应的位置加一
    p1 = np.cumsum(p) #使用 NumPy 的 cumsum 函数计算累积和，得到每个灰度级的累积概率分布
    for row in range(img_h):
        for col in range(img_w):
            v = img[row,col] #将每个像素的灰度级 v 替换为其在累积概率分布 p1 中对应的值
            img[row,col] = p1[v]*255 #乘以 255，以将值范围从 [0, 1] 转换回 [0, 255]
    return img


"""
对原始图像进行多步预处理，目的是突出车牌区域特征
返回的内容就是车牌部分的。
这个里面的亮点就是白色的点，有点神奇啊
"""
def find_board_area(img): # 通过检测行和列的亮点数目来提取矩形
    assert img.ndim==2 #确保传入的图像 img 是一个二维数组
    img_h,img_w = img.shape[0],img.shape[1] #获取图像的高度 img_h 和宽度 img_w
    top,bottom,left,right = 0,img_h,0,img_w #初始化边界变量 top、bottom、left 和 right，分别指向车牌区域的上边界、下边界、左边界和右边界
    flag = False

    #初始化水平直方图h_proj和垂直直方图v_proj，它们用于记录图像中每一行和每一列的亮点数目
    h_proj = [0 for i in range(img_h)]
    v_proj = [0 for i in range(img_w)]

    for row in range(round(img_h*0.5),round(img_h*0.8),3): #从图像中间偏上的位置开始，每隔 3 行向下遍历，直到图像的 80% 高度位置
        #遍历当前行的每个像素，如果像素值为 255（白色），则在水平直方图中对应的行位置加一
        for col in range(img_w):
            if img[row,col]==255:
                h_proj[row] += 1
        if flag==False and h_proj[row]>12: #如果当前行的亮点数目大于 12 并且 flag 为 False，则认为找到了车牌区域的上边界
            flag = True
            top = row
        if flag==True and row>top+8 and h_proj[row]<12: #如果已经找到上边界，且当前行距上边界超过 8 行，且当前行的亮点数目小于 12，则认为找到了车牌区域的下边界
            bottom = row
            flag = False

    for col in range(round(img_w*0.3),img_w,1): #从图像宽度的 30% 位置开始，逐列向右遍历，直到图像的最右端
        #遍历从上边界到下边界的每个像素，如果像素值为255，相应的在垂直直方图中的列位置加一
        for row in range(top,bottom,1):
            if img[row,col]==255:
                v_proj[col] += 1
        if flag==False and (v_proj[col]>10 or v_proj[col]-v_proj[col-1]>5): #如果当前列的亮点数目大于 10，或者当前列与前一列的亮点数目差大于 5，并且 flag 为 False，则认为找到了车牌区域的左边界
            left = col
            break
    return left,top,120,bottom-top-10 #left，top等等是一个位置


"""
这里本质还是用了面积大小的判断
返回的是布尔值
"""
def verify_scale(rotate_rect): #验证检测到的矩形区域是否符合车牌的尺寸比例和面积特征
    #定义了两个变量，error 表示允许的尺寸误差比例，aspect 表示车牌的期望宽高比
   error = 0.4
   aspect = 4#4.7272

    #根据期望的宽高比 aspect，计算车牌区域的最小和最大面积。这些值用于过滤掉不符合车牌尺寸的区域
   min_area = 10*(10*aspect)
   max_area = 150*(150*aspect)
    #计算车牌区域的宽高比的最小值和最大值，考虑到了误差范围
   min_aspect = aspect*(1-error)
   max_aspect = aspect*(1+error)
    #定义了 theta，表示车牌倾斜角度的最大容忍度
   theta = 30

   # 宽或高为0，不满足矩形直接返回False
   if rotate_rect[1][0]==0 or rotate_rect[1][1]==0:
       return False

   r = rotate_rect[1][0]/rotate_rect[1][1] #计算旋转矩形的宽度与高度的比值 r
   r = max(r,1/r)
   area = rotate_rect[1][0]*rotate_rect[1][1] #计算旋转矩形的面积

    #检查矩形的面积是否在最小和最大面积之间，以及宽高比是否在最小和最大宽高比之间
   if area>min_area and area<max_area and r>min_aspect and r<max_aspect:
       # 矩形的倾斜角度在不超过theta
       if ((rotate_rect[1][0] < rotate_rect[1][1] and rotate_rect[2] >= -90 and rotate_rect[2] < -(90 - theta)) or
               (rotate_rect[1][1] < rotate_rect[1][0] and rotate_rect[2] > -theta and rotate_rect[2] <= 0)):
           return True
   return False


"""
这里的car_rect说实话是个比较复杂的量，具体是在后面的函数里面的返回值
对已确定的候选车牌区域（如通过轮廓检测找到的旋转矩形）进行角度矫正，
将倾斜的车牌区域转换为水平矩形，方便后续字符分割
"""
def img_Transform(car_rect,image): #将检测到的车牌区域从原始图像中正确地裁剪出来，并进行必要的变换以满足后续处理的需要，如车牌矫正和大小调整
    img_h,img_w = image.shape[:2]
    rect_w,rect_h = car_rect[1][0],car_rect[1][1] #从 car_rect 中获取车牌区域的宽度 rect_w 和高度 rect_h
    angle = car_rect[2] #获取车牌区域的旋转角度 angle

    """
    这里是处理这个车牌的旋转的问题
    """
    return_flag = False
    if car_rect[2]==0:
        return_flag = True
    if car_rect[2]==-90 and rect_w<rect_h:
        rect_w, rect_h = rect_h, rect_w
        return_flag = True

    # 如果是无需旋转的,那么直接把车牌截取出来
    if return_flag:
        car_img = image[int(car_rect[0][1]-rect_h/2):int(car_rect[0][1]+rect_h/2),
                  int(car_rect[0][0]-rect_w/2):int(car_rect[0][0]+rect_w/2)]
        return car_img # 根据车牌区域的位置和尺寸直接裁剪原始图像，将车牌从图片中切割出来


    """
    下面这里的函数是在cv领域十分重要的，涉及到的计算角度，然后给他回正
    不过说实话挺难的，因为全是open cv的
    我就不会了现在，让ai写
    """

    car_rect = (car_rect[0],(rect_w,rect_h),angle)
    box = cv2.boxPoints(car_rect) # 调用函数获取矩形边框的四个点

    heigth_point = right_point = [0,0]
    left_point = low_point = [car_rect[0][0], car_rect[0][1]]
    for point in box:
        if left_point[0] > point[0]:
            left_point = point
        if low_point[1] > point[1]:
            low_point = point
        if heigth_point[1] < point[1]:
            heigth_point = point
        if right_point[0] < point[0]:
            right_point = point

    if left_point[1] <= right_point[1]:  # 正角度
        new_right_point = [right_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, M, (round(img_w*2), round(img_h*2)))
        car_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]

    elif left_point[1] > right_point[1]:  # 负角度
        new_left_point = [left_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, M, (round(img_w*2), round(img_h*2)))
        car_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]

    return car_img

# 图像预处理，处理成什么灰度图之类的东西
# 同时还处理的颜色，唉唉这里还是太难了
def pre_process(orig_img):

    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY) #从 BGR 色彩空间转换为灰度图像
    cv2.imwrite('gray_img.jpg', gray_img)

    blur_img = cv2.blur(gray_img, (3, 3)) #对灰度图像应用均值模糊，使用 3x3 的内核来减少图像噪声
    cv2.imwrite('blur.jpg', blur_img)

    sobel_img = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0, ksize=3) #使用 Sobel 算子计算图像的水平梯度幅度
    sobel_img = cv2.convertScaleAbs(sobel_img)
    cv2.imwrite('sobel.jpg', sobel_img)

    hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV) #将原始图像从 RGB 色彩空间转换为 HSV 色彩空间

    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    # 黄色色调区间[26，34],蓝色色调区间:[100,124]
    blue_img = (((h > 26) & (h < 34)) | ((h > 100) & (h < 124))) & (s > 70) & (v > 70)
    blue_img = blue_img.astype('float32')
    cv2.imwrite('hsv.jpg', blue_img)

    mix_img = np.multiply(sobel_img, blue_img) #将 Sobel 图像与 HSV 处理后的蓝色图像相乘，以突出可能的车牌区域
    cv2.imwrite('mix.jpg', mix_img)

    mix_img = mix_img.astype(np.uint8)

    ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite('binary.jpg',binary_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,5))
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel) #对二值图像应用闭运算，以填充可能的断裂并去除小的物体
    cv2.imwrite('close.jpg', close_img)

    return close_img


# 给候选车牌区域做漫水填充算法，一方面补全上一步求轮廓可能存在轮廓歪曲的问题，
# 另一方面也可以将非车牌区排除掉
"""
特征验证工具：它基于车牌的颜色特征（蓝底或黄底），判断候选区域是否符合真实车牌的颜色属性
同时可能修正轮廓检测的误差，排除非车牌区域（如颜色不符的广告牌、贴纸等）。
以我现在实力，只能是知道大题在干什么，因为这里都是涉及到很多的open cv
"""
def verify_color(rotate_rect,src_image):
    img_h,img_w = src_image.shape[:2]
    mask = np.zeros(shape=[img_h+2,img_w+2],dtype=np.uint8) #创建一个比原始图像每个维度大2的零矩阵 mask，用于存放漫水填充的结果
    connectivity = 4 #设置漫水填充的连通性，4 表示考虑上下左右四个方向
    loDiff,upDiff = 30,30 #定义漫水填充中的低阈值 loDiff 和高阈值 upDiff，用于确定填充的范围
    new_value = 255 #设置漫水填充的新值，这里是将填充的区域设置为白色（255）
    flags = connectivity
    flags |= cv2.FLOODFILL_FIXED_RANGE
    flags |= new_value << 8
    flags |= cv2.FLOODFILL_MASK_ONLY

    rand_seed_num = 5000
    valid_seed_num = 200
    adjust_param = 0.1
    box_points = cv2.boxPoints(rotate_rect)
    box_points_x = [n[0] for n in box_points]
    box_points_x.sort(reverse=False)
    adjust_x = int((box_points_x[2]-box_points_x[1])*adjust_param)
    col_range = [box_points_x[1]+adjust_x,box_points_x[2]-adjust_x]
    box_points_y = [n[1] for n in box_points]
    box_points_y.sort(reverse=False)
    adjust_y = int((box_points_y[2]-box_points_y[1])*adjust_param)
    row_range = [box_points_y[1]+adjust_y, box_points_y[2]-adjust_y]

    #使用 cv2.boxPoints 函数根据 rotate_rect 获取旋转矩形的四个角点。
    if (col_range[1]-col_range[0])/(box_points_x[3]-box_points_x[0])<0.4\
        or (row_range[1]-row_range[0])/(box_points_y[3]-box_points_y[0])<0.4:
        points_row = []
        points_col = []
        for i in range(2):
            pt1,pt2 = box_points[i],box_points[i+2]
            x_adjust,y_adjust = int(adjust_param*(abs(pt1[0]-pt2[0]))),int(adjust_param*(abs(pt1[1]-pt2[1])))
            if (pt1[0] <= pt2[0]):
                pt1[0], pt2[0] = pt1[0] + x_adjust, pt2[0] - x_adjust
            else:
                pt1[0], pt2[0] = pt1[0] - x_adjust, pt2[0] + x_adjust
            if (pt1[1] <= pt2[1]):
                pt1[1], pt2[1] = pt1[1] + adjust_y, pt2[1] - adjust_y
            else:
                pt1[1], pt2[1] = pt1[1] - y_adjust, pt2[1] + y_adjust
            temp_list_x = [int(x) for x in np.linspace(pt1[0],pt2[0],int(rand_seed_num /2))]
            temp_list_y = [int(y) for y in np.linspace(pt1[1],pt2[1],int(rand_seed_num /2))]
            points_col.extend(temp_list_x)
            points_row.extend(temp_list_y)
    else:
        points_row = np.random.randint(row_range[0],row_range[1],size=rand_seed_num)
        points_col = np.linspace(col_range[0],col_range[1],num=rand_seed_num).astype(np.int)

    points_row = np.array(points_row)
    points_col = np.array(points_col)
    hsv_img = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    h,s,v = hsv_img[:,:,0],hsv_img[:,:,1],hsv_img[:,:,2]

    flood_img = src_image.copy()
    seed_cnt = 0
    #循环 rand_seed_num 次
    for i in range(rand_seed_num):
        rand_index = np.random.choice(rand_seed_num,1,replace=False)
        row,col = points_row[rand_index],points_col[rand_index]

        if (((h[row,col]>26)&(h[row,col]<34))|((h[row,col]>100)&(h[row,col]<124)))&(s[row,col]>70)&(v[row,col]>70):
            cv2.floodFill(src_image, mask, (col,row), (255, 255, 255), (loDiff,) * 3, (upDiff,) * 3, flags)
            cv2.circle(flood_img,center=(col,row),radius=2,color=(0,0,255),thickness=2)
            seed_cnt += 1
            if seed_cnt >= valid_seed_num:
                break

    show_seed = np.random.uniform(1, 100, 1).astype(np.uint16)
    cv2.imwrite('floodfill.jpg',flood_img)
    cv2.imwrite('flood_mask.jpg',mask)

    mask_points = []
    for row in range(1,img_h+1):
        for col in range(1,img_w+1):
            if mask[row,col] != 0:
                mask_points.append((col-1,row-1))
    mask_rotateRect = cv2.minAreaRect(np.array(mask_points))
    if verify_scale(mask_rotateRect):
        return True,mask_rotateRect
    else:
        return False,mask_rotateRect

# 车牌定位
def locate_carPlate(orig_img,pred_image):
    carPlate_list = [] #初始化一个空列表 carPlate_list，用于存储检测到的车牌区域
    temp1_orig_img = orig_img.copy() #调试用
    temp2_orig_img = orig_img.copy() #调试用
    contours, heriachy = cv2.findContours(pred_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #使用 OpenCV 的 findContours 函数检测 pred_image 中的轮廓
    for i,contour in enumerate(contours):
        cv2.drawContours(temp1_orig_img, contours, i, (0, 255, 255), 2)
        # 获取轮廓最小外接矩形，返回值rotate_rect
        rotate_rect = cv2.minAreaRect(contour)
        # 根据矩形面积大小和长宽比判断是否是车牌
        if verify_scale(rotate_rect):
            ret,rotate_rect2 = verify_color(rotate_rect,temp2_orig_img) #如果尺寸比例验证通过，调用 verify_color 函数进一步验证车牌的颜色特征
            if ret == False:
                continue
            # 车牌位置矫正
            car_plate = img_Transform(rotate_rect2, temp2_orig_img) #如果颜色验证通过，调用 img_Transform 函数裁剪出车牌区域
            car_plate = cv2.resize(car_plate,(car_plate_w,car_plate_h)) #调整尺寸为后面CNN车牌识别做准备
            #========================调试看效果========================#
            box = cv2.boxPoints(rotate_rect2)
            for k in range(4):
                n1,n2 = k%4,(k+1)%4
                cv2.line(temp1_orig_img, (box[n1][0], box[n1][1]),(box[n2][0], box[n2][1]), (255,0,0), 2)
            cv2.imwrite('opencv.jpg', car_plate)
            #========================调试看效果========================#
            carPlate_list.append(car_plate)

    cv2.imwrite('contour.jpg', temp1_orig_img)
    return carPlate_list

# 左右切割
"""
这里是对图片里面的字符进行切割
这里传入的就是只有车牌的图像了
"""
# 对矫正后的车牌图像进行水平方向的字符分割，提取每个字符的左右边界
def horizontal_cut_chars(plate):
    char_addr_list = []  # 存储每个字符的左边界、右边界和宽度，格式为[(left, right, width), ...]
    area_left, area_right, char_left, char_right = 0, 0, 0, 0  # 初始化区域和字符的边界变量
    img_w = plate.shape[1]  # 获取车牌图像的宽度（水平方向像素数）

    # 定义内部函数：统计指定列中白色像素（255）的数量（归一化到0-1后求和，等效于计数）
    def getColSum(img, col):
        sum = 0  # 初始化该列白色像素总数
        for i in range(img.shape[0]):  # 遍历该列的每一行像素
            sum += round(img[i, col] / 255)  # 像素值归一化后四舍五入（0或1），累加得到白色像素数
        return sum;  # 返回该列的白色像素总数

    sum = 0  # 初始化所有列的白色像素总和
    for col in range(img_w):  # 遍历车牌图像的每一列
        sum += getColSum(plate, col)  # 累加每列的白色像素数
    col_limit = 0  # 设置判断“字符列”的阈值（此处为0，实际可根据需求调整，如均值的比例）
    # 定义单个字符宽度的合理范围（车牌总宽度的1/12到1/5，过滤过宽或过窄的区域）
    charWid_limit = [round(img_w / 12), round(img_w / 5)]
    is_char_flag = False  # 标记当前是否处于字符区域（False表示非字符区域，True表示字符区域）

    # 遍历每一列，定位字符的左右边界
    for i in range(img_w):
        colValue = getColSum(plate, i)  # 获取当前列的白色像素数
        # 如果当前列的白色像素数超过阈值，判断为字符区域
        if colValue > col_limit:
            # 若之前处于非字符区域，说明刚进入新的字符区域
            if is_char_flag == False:
                area_right = round((i + char_right) / 2)  # 计算上一个字符区域的右边界（取中间值）
                area_width = area_right - area_left  # 计算上一个字符区域的宽度
                char_width = char_right - char_left  # 计算上一个字符的实际宽度
                # 若上一个字符区域的宽度在合理范围内，则视为有效字符，记录边界
                if (area_width > charWid_limit[0]) and (area_width < charWid_limit[1]):
                    char_addr_list.append((area_left, area_right, char_width))
                char_left = i  # 更新当前字符的左边界为当前列
                area_left = round((char_left + char_right) / 2)  # 更新当前字符区域的左边界（取中间值）
                is_char_flag = True  # 标记为进入字符区域
        # 若当前列的白色像素数未超过阈值，判断为非字符区域
        else:
            # 若之前处于字符区域，说明刚离开字符区域
            if is_char_flag == True:
                char_right = i - 1  # 更新当前字符的右边界为前一列
                is_char_flag = False  # 标记为离开字符区域
    # 处理遍历结束后仍未闭合的字符区域（即最后一个字符未被完整分割）
    if area_right < char_left:
        area_right, char_right = img_w, img_w  # 将右边界设为图像最右侧
        area_width = area_right - area_left  # 计算最后一个字符区域的宽度
        char_width = char_right - char_left  # 计算最后一个字符的实际宽度
        # 若宽度在合理范围内，则视为有效字符，记录边界
        if (area_width > charWid_limit[0]) and (area_width < charWid_limit[1]):
            char_addr_list.append((area_left, area_right, char_width))
    return char_addr_list  # 返回所有有效字符的边界信息列表


"""
看不懂这个啊，这个就是和上一个函数配合，然后上一个函数是左右边界，而这个函数是一个垂直方向的
两个函数相互配合之后，就得到了单个单个的字符
"""
def get_chars(car_plate):
    # 获取车牌图像的高度和宽度（car_plate为二值化图像，shape[:2]返回(高,宽)）
    img_h, img_w = car_plate.shape[:2]
    # 存储水平投影区域的列表，每个元素为(起始行索引, 结束行索引)
    h_proj_list = []
    # 临时变量：h_temp_len记录当前水平投影区域的长度，v_temp_len未使用（预留）
    h_temp_len, v_temp_len = 0, 0
    # 记录水平投影区域的起始行和结束行索引
    h_startIndex, h_end_index = 0, 0
    # 水平投影的长度阈值范围（占图像宽度的比例），过滤不符合车牌字符特征的区域
    h_proj_limit = [0.2, 0.8]
    # 存储提取到的字符图像列表
    char_imgs = []

    # 初始化水平投影计数列表，记录每一行的白色像素（255）数量
    h_count = [0 for i in range(img_h)]
    # 遍历图像的每一行，统计每行白色像素数量
    for row in range(img_h):
        temp_cnt = 0  # 临时变量，记录当前行的白色像素数
        # 遍历当前行的每一列，统计白色像素
        for col in range(img_w):
            if car_plate[row, col] == 255:  # 若为白色像素
                temp_cnt += 1  # 计数加1
        h_count[row] = temp_cnt  # 存储当前行的白色像素数

        # 判断当前行是否为有效字符行：白色像素占比需在20%-80%之间
        if temp_cnt / img_w < h_proj_limit[0] or temp_cnt / img_w > h_proj_limit[1]:
            # 若当前处于一个水平投影区域中（h_temp_len != 0），则结束该区域
            if h_temp_len != 0:
                h_end_index = row - 1  # 结束行为当前行的上一行
                h_proj_list.append((h_startIndex, h_end_index))  # 记录该投影区域
                h_temp_len = 0  # 重置临时长度
            continue  # 跳过无效行

        # 若当前行有白色像素（属于有效区域）
        if temp_cnt > 0:
            # 若之前未处于投影区域（h_temp_len == 0），则标记起始行
            if h_temp_len == 0:
                h_startIndex = row  # 起始行为当前行
                h_temp_len = 1  # 初始化投影长度为1
            else:
                h_temp_len += 1  # 投影长度累加
        else:
            # 若当前行无白色像素，但之前处于投影区域，则结束该区域
            if h_temp_len > 0:
                h_end_index = row - 1  # 结束行为当前行的上一行
                h_proj_list.append((h_startIndex, h_end_index))  # 记录该投影区域
                h_temp_len = 0  # 重置临时长度

    # 遍历结束后，若仍有未闭合的水平投影区域（h_temp_len != 0），则补充记录
    if h_temp_len != 0:
        h_end_index = img_h - 1  # 结束行为图像最后一行
        h_proj_list.append((h_startIndex, h_end_index))  # 记录该投影区域

    # 筛选最长的水平投影区域（字符主要集中区域），需满足长度占图像高度的50%以上
    h_maxIndex, h_maxHeight = 0, 0  # 最长投影区域的索引和高度
    for i, (start, end) in enumerate(h_proj_list):
        current_height = end - start  # 计算当前投影区域的高度
        # 更新最长投影区域信息
        if h_maxHeight < current_height:
            h_maxHeight = current_height
            h_maxIndex = i
    # 若最长投影区域占比不足50%，则认为无有效字符，返回空列表
    if h_maxHeight / img_h < 0.5:
        return char_imgs

    # 确定字符区域的上下边界（最长投影区域的起止行）
    chars_top, chars_bottom = h_proj_list[h_maxIndex][0], h_proj_list[h_maxIndex][1]

    # 截取车牌图像中字符所在的水平区域（上下边界之间）
    plates = car_plate[chars_top:chars_bottom + 1, :]
    # 保存中间结果用于调试（原始车牌和截取的字符水平区域）
    cv2.imwrite('car.jpg', car_plate)
    cv2.imwrite('plate.jpg', plates)

    # 对截取的水平区域进行水平方向字符分割，获取每个字符的左右边界
    char_addr_list = horizontal_cut_chars(plates)

    # 根据字符边界提取每个字符图像，并调整为统一尺寸
    for i, addr in enumerate(char_addr_list):
        # 截取字符区域（addr[0]为左边界，addr[1]为右边界）
        char_img = car_plate[chars_top:chars_bottom + 1, addr[0]:addr[1]]
        # 将字符图像调整为预设的统一尺寸（char_w, char_h）
        char_img = cv2.resize(char_img, (char_w, char_h))
        char_imgs.append(char_img)  # 添加到字符列表

    # 返回提取到的所有字符图像
    return char_imgs

"""
对已定位的车牌图像进行预处理，并提取取出其中的字符图像，为后续的字符识别做准备
"""
def extract_char(car_plate):
    gray_plate = cv2.cvtColor(car_plate, cv2.COLOR_BGR2GRAY)
    ret,binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    char_img_list = get_chars(binary_plate) #调用 get_chars 函数处理二值化的车牌图像，以提取可能的字符区域，并将结果存储在列表
    return char_img_list


"""
这是最重要的部分了
"""
def cnn_select_carPlate(plate_list, model_path):
    #cnn训练
    if len(plate_list) == 0:
        return False, plate_list
    #pytorch可以直接load模型，不同在这里训练了
    model = torch.load(model_path)
    model = model.cuda()

    idx, val = 0, -1
    tf = transforms.ToTensor()
    for i in range(len(plate_list)):
        #禁用梯度计算，因为在推理阶段不需要进行梯度更新
        with torch.no_grad():
            input = tf(np.array(plate_list[i])).unsqueeze(0).cuda()
            output = torch.sigmoid(model(input))

            if output > val:
                idx, val = i, output
    return True, plate_list[idx]

def cnn_recongnize_char(img_list, model_path):
    model = torch.load(model_path)
    model = model.cuda()
    text_list = []

    if len(img_list) == 0:
        return text_list

    tf = transforms.ToTensor()
    for img in img_list:
        input = tf(np.array(img)).unsqueeze(0).cuda()
    # 数字、字母、汉字，从67维向量找到概率最大的作为预测结果
        with torch.no_grad():
            output = model(input)
            _, preds = torch.topk(output, 1)
            text_list.append(char_table[preds])

    return text_list

def list_all_files(root):
    files = []
    list = os.listdir(root)
    for i in range(len(list)):
        element = os.path.join(root, list[i])
        if os.path.isdir(element):
            files.extend(list_all_files(element))
        elif os.path.isfile(element):
            files.append(element)
    return files

if __name__ == '__main__':
    car_plate_w,car_plate_h = 136,36
    char_w,char_h = 20, 20
    plate_model_path = "plate.pth"
    char_model_path = "char.pth"
    root = '../images/test/' # 测试图片路径
    files = list_all_files(root)
    files.sort()

    for file in files:
        print(file)
        img = cv2.imread(file)
        if len(img) < 2:
            continue

        pred_img = pre_process(img)
        car_plate_list = locate_carPlate(img, pred_img)
        ret, car_plate = cnn_select_carPlate(car_plate_list,plate_model_path)
        if ret == False:
            print("未检测到车牌")
            continue
        cv2.imwrite('cnn_plate.jpg',car_plate)

        char_img_list = extract_char(car_plate)
        for idx in range(len(char_img_list)):
            img_name = 'char-' + str(idx) + '.jpg'
            cv2.imwrite(img_name, char_img_list[idx])

        text = cnn_recongnize_char(char_img_list,char_model_path)
        print(text)

    cv2.waitKey(0)