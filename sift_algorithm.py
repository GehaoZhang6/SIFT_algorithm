import cv2
import math
import numpy as np
import logging
from functools import cmp_to_key
from numpy.linalg import det, lstsq, norm

float_tolerance = 1e-7
# 1.适当的模糊和加倍图像以生成图像金字塔的基础图像
# 由于相机已对图片进行sigma=0.5的模糊，所以我们要将这0.5减去才可以计算出我们要的sigma变换
# 可以相减的原理可由概率密度公式推得，乘以2的原因是因为图片放大了两倍从而使像素距离放大了两倍
def generateBaseImage(image,sigma,assumed_blur):
    image=cv2.resize(image,(0,0),fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
    sigma_diff=math.sqrt(max((sigma**2)-((2*assumed_blur)**2),0.01))
    return cv2.GaussianBlur(image,(0,0),sigmaX=sigma_diff,sigmaY=sigma_diff)

#2.计算差分图像金字塔层数
#如果高斯金字塔有五层，差分金字塔只能有四层（-1），然而去除不可比较的顶层和底层还要再-2
def computeNumberOfOctaves(image_shape):
    return int(round(math.log2(min(image_shape[0:2]))-1))

#3.创建高斯核大小
#为什么要有不同的高斯核：
#多尺度表示：高斯金字塔是一种多尺度图像表示方法，通过在不同尺度下对图像进行模糊和采样来捕捉图像中的不同细节
#平滑过渡：由于每个组内的层索引 r 变化范围为 [0, s+2]，因此在同一个组内，相邻层之间的模糊系数会逐渐增加。
#这样做的好处是能够实现平滑的尺度过渡，确保每个层都能捕捉到合适的细节信息，而不会出现明显的间隔或断裂。
#σ（o,r)=σ0*2**（o+r/s）
#o为组索引序号，r为层索引序号，s为高斯差分金字塔每组层数(层内含有组）
#每层内每组的σ(s)=σ0*sqrt((k**s)**2-(k**(s-1))**2)
#k是什么？ 答：k使金字塔同层不同组的增值，例如一组内有s组那么其σ范围使σ-2σ，第一组为1,第二组为k，第五组为k**4,则k=2**(1/s)
#为什么不是k**(1/s-1)，为什么范围是σ-2σ？ 答：这两个其实是同一个问题，当图像尺寸缩小两倍时，σ也缩小两倍，2**（1/s）位于下一层，由此可得不同层之间其实是来连续的，缩小的好处是提高速度
def generateGaussianKernels(sigma,num_intervals):
    num_images_per_octave=num_intervals+3
    k=2**(1./num_intervals)
    gaussian_kernels=np.zeros(num_images_per_octave)
    gaussian_kernels[0]=sigma

    for image_index in range(1,num_images_per_octave):
        sigma_previous=(k**(image_index-1))*sigma
        sigma_current=k*sigma_previous
        gaussian_kernels[image_index]=math.sqrt(sigma_current**2-sigma_previous**2)
    return gaussian_kernels

#4.保证高斯差分金字塔的尺度空间（高斯模糊系数）的连续性，下一个Octave(i+1)的第一层由上一个Octave(i)的倒数第三层直接降采样不需要模糊产生。
def generateGaussianImages(image,num_octaves,gaussian_kernels):
    gaussian_images=[]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for octave_index in range(num_octaves):
        #将一张图片分为层
        gaussian_images_in_octaves=[]
        gaussian_images_in_octaves.append(image)
        for gaussian_kernel in gaussian_kernels[1:]:
            #将层内图片进行模糊处理
            image=cv2.GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octaves.append(image)
        gaussian_images.append(gaussian_images_in_octaves)
        octave_base=gaussian_images_in_octaves[-3]
        image=cv2.resize(octave_base,(int(octave_base.shape[1]/2),int(octave_base.shape[0]/2)),interpolation=cv2.INTER_NEAREST)
    return np.array(gaussian_images)
#([[]])

#5.减去相邻的高斯图像形成差分金字塔(极值检测)
def generateDoGImages(gaussian_images):
    dog_images=[]
    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave=[]
        for first_image,second_image in zip(gaussian_images_in_octave,gaussian_images_in_octave[1:]):
            image = cv2.subtract(second_image, first_image)
            dog_images_in_octave.append(image)
        dog_images.append(dog_images_in_octave)
    return np.array(dog_images)

#6.找极值
#6.1判断该点是否为极值点
def is_pixel_an_extrem_item(first_subimage,second_subimage,third_subimage,threshold):
    center_pixel_value = second_subimage[1, 1]
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return np.all(center_pixel_value >= first_subimage[:,:]) and\
                np.all(center_pixel_value >= third_subimage[:,:]) and\
                np.all(center_pixel_value >= second_subimage[0, :]) and\
                np.all(center_pixel_value >= second_subimage[2, :]) and\
                np.all(center_pixel_value >= second_subimage[1, 0]) and\
                np.all(center_pixel_value >= second_subimage[1, 2])

        elif center_pixel_value < 0:
            return  np.all(center_pixel_value <= first_subimage[:, :]) and\
                    np.all(center_pixel_value <= third_subimage[:, :]) and\
                    np.all(center_pixel_value <= second_subimage[0, :]) and\
                    np.all(center_pixel_value <= second_subimage[2, :]) and\
                    np.all(center_pixel_value <= second_subimage[1, 0]) and\
                    np.all(center_pixel_value <= second_subimage[1, 2])
    return False

#6.2计算一阶导数
def compute_gradient_at_center_pixel(pixel_array):
    dx=0.5*(pixel_array[1,1,2]-pixel_array[1,1,0])
    dy=0.5*(pixel_array[1,2,1]-pixel_array[1,0,1])
    ds=0.5*(pixel_array[2,1,1]-pixel_array[0,1,1])
    return np.array([dx,dy,ds])

#6.3计算二阶导数
def compute_hessian_at_center_pixel(pixel_array):
    center_pixel_value=pixel_array[1,1,1]
    dxx=pixel_array[1,1,2]-2*center_pixel_value+pixel_array[1,1,0]
    dyy=pixel_array[1,2,1]-2*center_pixel_value+pixel_array[1,0,1]
    dss=pixel_array[2,1,1]-2*center_pixel_value+pixel_array[0,1,1]
    dxy=0.25*((pixel_array[1,2,2]-pixel_array[1,2,0])-(pixel_array[1,0,2]-pixel_array[1,0,0]))
    dxs=0.25*((pixel_array[2,1,2]-pixel_array[2,1,0])-(pixel_array[0,1,2]-pixel_array[0,1,0]))
    dys=0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return np.array([[dxx, dxy, dxs],
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])

#6.4将离散极值点变为连续极值点
def localize_extrem_via_quadratic_fit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=10):
    extremum_is_outside_image = False  # 初始化标记变量，用于检查极值点是否在图像范围内
    image_shape = dog_images_in_octave[0].shape  # 获取图像形状（高度和宽度）

    # 迭代尝试多次，以使离散极值点变为连续极值点
    for attempt_index in range(num_attempts_until_convergence):
        first_image, second_image, third_image = dog_images_in_octave[image_index - 1:image_index + 2]

        # 提取像素块，用于后续计算梯度和Hessian矩阵
        pixel_cube = np.stack([first_image[i - 1:i + 2, j - 1:j + 2],
                             second_image[i - 1:i + 2, j - 1:j + 2],
                             third_image[i - 1:i + 2, j - 1:j + 2]]).astype('float32') / 255

        gradient = compute_gradient_at_center_pixel(pixel_cube)  # 计算中心像素点处的梯度
        hessian = compute_hessian_at_center_pixel(pixel_cube)  # 计算中心像素点处的Hessian矩阵

        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]  # 使用最小二乘法计算极值点的更新量

        # 判断极值点是否已经收敛，即更新量是否足够小
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break

        # 将极值点的位置和图像索引根据更新量进行更新
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))

        # 检查极值点是否超出图像边界或尺度空间范围
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break

    # 如果极值点在图像范围外，则返回None
    if extremum_is_outside_image:
        return None

    # 如果迭代达到最大尝试次数仍未收敛，则返回None
    if attempt_index >= num_attempts_until_convergence - 1:
        return None

    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)

    # 判断极值点的对比度是否满足阈值要求
    if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]  # 提取 Hessian 矩阵的前两行前两列部分
        xy_hessian_trace = np.trace(xy_hessian)  # 计算 xy_hessian 的迹
        xy_hessian_det = np.linalg.det(xy_hessian)  # 计算 xy_hessian 的行列式

        # 判断极值点的稳定性是否满足阈值要求
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            # 通过对比特征值比率检查，确保极值点具有足够的对比度和稳定性

            # 构建并返回 OpenCV KeyPoint 对象
            keypoint = cv2.KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))  # 极值点的位置(还原到初始图像位置）
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)  # 极值点所在的尺度空间和图像索引，keypoint.octave 的高位表示尺度空间信息，用于表示关键点所在的尺度空间，而低位表示图像索引信息，用于表示关键点所在尺度空间内的图像索引
            #2 ** 8 在这里是用作位移运算的位数，用于将图像索引乘以 256，即左移8位。这个位移运算的目的是将图像索引映射到一个合适的范围内，
            #以便在构建 keypoint.octave 的过程中将尺度空间和图像索引信息合并到一个整数值中。在 keypoint.octave 的计算过程中，我们
            #需要将尺度空间和图像索引信息合并成一个整数值，以便后续在特征匹配等处理中使用。为了实现这一点，我们将尺度空间和图像索引信息分别
            #左移 16 和 8 位，然后进行按位或运算来合并这两部分信息。
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * (2 ** (octave_index + 1))  # 极值点对应的尺度
            keypoint.response = abs(functionValueAtUpdatedExtremum)  # 极值点的响应值
            return keypoint, image_index  # 返回构建的关键点对象和图像索引

    return None


#6.5计算关键点的方向
def computeKeypointsWithOrientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    keypoints_with_orientations = []  # 存储带有方向信息的关键点列表
    image_shape = gaussian_image.shape  # 获取高斯图像的形状（高度和宽度）

    scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))  # 计算关键点的尺度大小
    radius = int(round(radius_factor * scale))  # 计算关键点邻域的半径
    weight_factor = -0.5 / (scale ** 2)  # 计算加权因子
    raw_histogram = np.zeros(num_bins)  # 存储未经平滑处理的方向直方图
    smooth_histogram = np.zeros(num_bins)  # 存储平滑处理后的方向直方图

    # 在关键点邻域内计算梯度信息，并将其分配到方向直方图中
    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i# keypoint圆的y坐标
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j# keypoint圆的x坐标
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)  # 计算梯度幅值
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))  # 计算梯度方向
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))  # 计算加权值
                    histogram_index = int(np.round(gradient_orientation * num_bins / 360.))  # 计算方向直方图索引
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    # 对方向直方图进行平滑处理，找到主方向的峰值
    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    orientation_max = max(smooth_histogram)
    orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
#EXAMPLE BEGIN
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.rcParams['font.family'] = 'sans-serif'
    # matplotlib.rcParams['font.sans-serif'] = ['Arial']
    #
    # num_bins = 36  # 方向直方图的 bin 数量
    # raw_histogram = np.array(
    #     [10, 20, 15, 12, 18, 25, 22, 30, 40, 35, 28, 15, 10, 5, 8, 12, 20, 28, 35, 42, 50, 40, 35, 30, 25, 20, 15, 10,
    #      5, 3, 6, 8, 12, 18, 25, 20])  # 未经平滑处理的方向直方图
    #
    # # 初始化平滑后的方向直方图
    # smooth_histogram = np.zeros(num_bins)
    #
    # # 平滑处理方向直方图
    # for n in range(num_bins):
    #     smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) +
    #                            raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    #
    # # 找到主方向的峰值位置
    # orientation_max = max(smooth_histogram)
    # orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1),
    #                                             smooth_histogram > np.roll(smooth_histogram, -1)))[0]
    #
    # # 可视化结果
    # plt.figure(figsize=(10, 6))
    #
    # # 绘制原始方向直方图
    # plt.subplot(2, 1, 1)
    # plt.bar(np.arange(num_bins), raw_histogram, align='center', color='b')
    # plt.title('Raw Histogram')
    # plt.xlabel('Bin')
    # plt.ylabel('Frequency')
    #
    # # 绘制平滑后的方向直方图
    # plt.subplot(2, 1, 2)
    # plt.plot(np.arange(num_bins), smooth_histogram, color='r')
    # plt.scatter(orientation_peaks, smooth_histogram[orientation_peaks], color='g', label='Peaks')
    # plt.title('Smooth Histogram with Peaks')
    # plt.xlabel('Bin')
    # plt.ylabel('Smoothed Frequency')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()
#EXAMPLE END

    # 将符合条件的关键点方向添加到 keypoints_with_orientations 列表中
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            # 最高的那个柱体所代表的方向就是该特征点处邻域范围内图像梯度的主方向，也就是该特征点的主方向。由于柱体所代表的角度只是一个范围，如第1柱的角度为
            # 0～9，因此还需要对离散的梯度方向直方图进行插值拟合处理，以得到更精确的方向角度值。例如我们已经得到了第i柱所代表的方向为特征点的主方向
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < float_tolerance:
                orientation = 0

            # 构建带有方向信息的新关键点，并将其添加到 keypoints_with_orientations 列表中
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)

    return keypoints_with_orientations  # 返回带有方向信息的关键点列表

#6.6返回关键点
def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width,contrast_threshold=0.04):
    # 计算用于确定极值点的阈值
    threshold = math.floor(0.5 * contrast_threshold / num_intervals * 255)  # 向下舍
    keypoints = []  # 存储找到的关键点列表

    # 遍历不同的金字塔组（octave）
    for octave_index, dog_images_in_octave in enumerate(dog_images):
        # 遍历当前组（octave）中的DOG图像（Difference of Gaussian）
        for image_index, (first_image, second_image, third_image) in enumerate(
                zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):

            # 遍历图像像素点，排除边缘像素（根据给定的边缘宽度）
            for i in range(image_border_width, first_image.shape[0] - image_border_width):  # 图像高度
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    # 检查当前像素是否是极值点
                    if is_pixel_an_extrem_item(first_image[i - 1:i + 2, j - 1:j + 2],
                                               second_image[i - 1:i + 2, j - 1:j + 2],
                                               third_image[i - 1:i + 2, j - 1:j + 2],
                                               threshold):
                        # 使用二次拟合对极值点进行精确定位
                        localization_result = localize_extrem_via_quadratic_fit(i, j, image_index + 1, octave_index,
                                                                                num_intervals, dog_images_in_octave,
                                                                                sigma, contrast_threshold,
                                                                                image_border_width)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            # 计算关键点的方向
                            keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index,
                                                                                             gaussian_images[
                                                                                                 octave_index][
                                                                                                 localized_image_index])
                            # 将计算出的关键点及其方向添加到关键点列表中
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints

# 比较两个关键点的各个属性，按照一定顺序进行排序
# 用于在removeDuplicateKeypoints函数中去除重复关键点
def compareKeypoints(keypoint1, keypoint2):
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def removeDuplicateKeypoints(keypoints):

    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints

def convertKeypointsToInputImageSize(keypoints):

    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints

def unpackOctave(keypoint):
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
    return octave, layer, scale

def generateDescriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    descriptors = []  # 存储生成的特征描述子

    for keypoint in keypoints:
        octave, layer, scale = unpackOctave(keypoint)  # 获取关键点所在的尺度空间索引、层级索引和尺度值
        gaussian_image = gaussian_images[octave + 1, layer]  # 获取当前关键点所在的高斯图像
        num_rows, num_cols = gaussian_image.shape  # 获取高斯图像的行数和列数
        point = np.round(scale * np.array(keypoint.pt)).astype('int')  # 计算关键点位置在原始图像中的坐标，并四舍五入为整数
        bins_per_degree = num_bins / 360.  # 每度的角度bin数
        angle = 360. - keypoint.angle  # 计算关键点的方向角度（考虑到坐标系不同）
        cos_angle = np.cos(np.deg2rad(angle))  # 计算方向角度的余弦值
        sin_angle = np.sin(np.deg2rad(angle))  # 计算方向角度的正弦值
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)  # 权重乘子，用于高斯加权

        # 存储关键点周围窗口内每个像素点的bin值、梯度幅值和梯度方向
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))  # 存储生成的直方图

        hist_width = scale_multiplier * 0.5 * scale * keypoint.size  # 计算用于计算梯度直方图的窗口宽度
        half_width = int(round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))  # 计算窗口宽度的一半
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))  # 确保窗口宽度不超过高斯图像的边界

        # 遍历窗口内的像素点，并计算梯度幅值和方向
        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle  # 计算了考虑方向角度的坐标旋转，将窗口内的像素点根据关键点的方向角度进行旋转。row_rot 和 col_rot 表示旋转后的坐标。
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5  # 计算像素点在直方图中的行bin位置
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5  # 计算像素点在直方图中的列bin位置
                #(row_rot / hist_width)：这一部分将像素点的旋转后的行坐标 row_rot 根据 hist_width 缩放到一个适当的尺度，以便能够映射到直方图的bin。
                # 0.5 * window_width：这是为了将缩放后的坐标移到直方图的中心，以确保计算得到的 row_bin 和 col_bin 落在合适的范围内。
                # 0.5：这是为了将计算得到的行bin和列bin值从中心位置偏移，以便将其映射到直方图中合适的范围。
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))  # 窗口内像素点的行索引
                    window_col = int(round(point[0] + col))  # 窗口内像素点的列索引
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]  # 计算x方向梯度
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]  # 计算y方向梯度
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)  # 计算梯度幅值
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360  # 计算梯度方向，保证角度在0到360度之间
                        weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))  # 计算高斯加权权重
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        # 将每个像素点的梯度信息投影到直方图中
        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:  # 表示梯度方向值在负数范围，那么就加上 num_bins，使其变为正数范围内的值。
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # 将直方图转换为一维数组形式作为特征向量

        # 对特征向量进行归一化处理
        threshold = norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(norm(descriptor_vector), float_tolerance)

        # 将特征向量的值映射到0到255之间，并转换为整型
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255

        descriptors.append(descriptor_vector)

    return np.array(descriptors, dtype='float32')



def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):

    image = image.astype('float32')
    base_image = generateBaseImage(image, sigma, assumed_blur)
    num_octaves = computeNumberOfOctaves(base_image.shape)
    gaussian_kernels = generateGaussianKernels(sigma, num_intervals)
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
    dog_images = generateDoGImages(gaussian_images)
    keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    keypoints = removeDuplicateKeypoints(keypoints)
    keypoints = convertKeypointsToInputImageSize(keypoints)
    descriptors = generateDescriptors(keypoints, gaussian_images)
    return keypoints, descriptors

# 加载图像
import time

# 开始计时
start_time = time.time()

image1 = cv2.imread('C:\\Users\\86152\\Desktop\\image_match\\computer1.jpg')
image2 = cv2.imread('C:\\Users\\86152\\Desktop\\image_match\\computer2.jpg')

sift = cv2.SIFT_create()
# 检测特征点和描述符
keypoints1, descriptors1 = computeKeypointsAndDescriptors(image1)
keypoints2, descriptors2 = computeKeypointsAndDescriptors(image2)
# print(keypoints1)
# < cv2.KeyPoint 0000020582EB7FC0>
#=========================================================
# 关键点：
# 创建一个关键点对象
# keypoint = cv2.KeyPoint(x=100, y=200, _size=2.5, _angle=45)
# # 访问关键点的属性
# x = keypoint.pt[0]  # 关键点的 x 坐标
# y = keypoint.pt[1]  # 关键点的 y 坐标
# size = keypoint.size  # 关键点的尺度
# angle = keypoint.angle  # 关键点的方向
#描述子：
# 例如，假设某个SIFT描述子的前8个维度的值如下：
# [0.1, 0.3, 0.2, 0.5, 0.8, 0.4, 0.6, 0.9]
# 这意味着该描述子的第一个维度对应于第一个柱状条目的计数为0.1，第二个维度对应于第二个柱状条目的计数为0.3，依此类推。
# 因此，SIFT描述子的每个维度可以解释为特定梯度方向范围内的特征计数或权重。这些数值反映了关键点周围区域中梯度方向的分布和强度。
#打印观察描述子：
# print(descriptors2.shape)
# (2455, 128)
#===========================================================

# 创建匹配器：选择合适的匹配器，如暴力匹配器（Brute-Force Matcher）或基于kd树的匹配器（KD-Tree Matcher）。
# 进行K邻近匹配：对于查询图像中的每个特征描述子，找到目标图像中K个最相似的特征描述子。可以使用匹配器的knnMatch方法实现，设置参数K为所需的最近邻数。
# 匹配结果处理：根据匹配的距离或其他准确度指标进行筛选和验证，排除一些不准确的匹配。常见的方法是根据最佳匹配和次佳匹配的距离比例进行筛选，如只保留比例小于某个阈值的匹配对

matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
#matches是一个元组
# print(matches)
# (< cv2.DMatch 000002B99674F930>, < cv2.DMatch 000002B99674F950>)
# ===========================================================
#匹配：
# cv2.DMatch是OpenCV库中用于表示特征匹配的数据结构。它存储了两个特征点的索引以及它们之间的距离。
# cv2.DMatch包含以下三个属性：
# queryIdx：查询图像中特征点的索引。
# trainIdx：训练图像中特征点的索引。
# distance：查询特征点和训练特征点之间的距离。
# ===============================================================
#查看matches属性方法：
# for i in matches:
#     for match in i:
#         query_idx = match.queryIdx
#         train_idx = match.trainIdx
#         distance = match.distance
#         print(f"Query Index: {query_idx}, Train Index: {train_idx}, Distance: {distance}")
#====================================================================================
# 应用筛选条件，例如低于一定阈值的匹配点对
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

#分别找到两幅图匹配点的坐标
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
# print(dst_pts)
# [[[577.4535    814.2693]]
# [[1561.197     397.63675]]
# [[1024.0914    106.56335]]
# [[1645.099     190.55672]]]

# 使用RANSAC算法估计单应矩阵
homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

#变换单应矩阵防止负坐标出现
#==============================================================
#变换原理：
#找到负坐标move_x,move_y,我们要让坐标平移，则可以对H矩阵进行处理
#以x为例子：
#x=(m00*x0+m01*y0+m02)/(m20*x0+m21*y0+m22)
#则：x-move_x=(m00*x0+m01*y0+m02)/(m20*x0+m21*y0+m22)-move_x
#x-move_x=x'=((m00-move_x*m20)*x0+(m01-move_x*m21)*y0+m02-move_x*m22)/(m20*x0+m21*y0+m22)
#H变为：
#m00-move_x*m20,m01-move_x*m21,m02-move_x*m22
#m10, m11, m12
#m20, m21, m22
#y同理可得
#================================================================
# 解析变换矩阵 M
m00, m01, m02 = homography[0, 0], homography[0, 1], homography[0, 2]
m10, m11, m12 = homography[1, 0], homography[1, 1], homography[1, 2]
m20, m21, m22 = homography[2, 0], homography[2, 1], homography[2, 2]
#[3*3][3*1]
# src_x = (m00 * dst_x + m01 * dst_y + m02) / (m20 * dst_x + m21 * dst_y + m22)
# src_y = (m10 * dst_x + m11 * dst_y + m12) / (m20 * dst_x + m21 * dst_y + m22)
#找最小的move_x和move_y
#左上角
top_left_x = (m00 * 0 + m01 * 0 + m02) / (m20 * 0 + m21 * 0 + m22)
top_left_y = (m10 * 0 + m11 * 0 + m12) / (m20 * 0 + m21 * 0 + m22)
print('top_left:',top_left_x,top_left_y)
#左下角
down_left_x = (m00 * 0 + m01 * image1.shape[0] + m02) / (m20 * 0 + m21 * image1.shape[0] + m22)
down_left_y = (m10 * 0 + m11 * image1.shape[0] + m12) / (m20 * 0 + m21 * image1.shape[0] + m22)
print('down_left:',down_left_x,down_left_y)
#右上角
top_right_x = (m00 * image1.shape[1] + m01 * 0 + m02) / (m20 * image1.shape[1] + m21 * 0 + m22)
top_right_y = (m10 * image1.shape[1] + m11 * 0 + m12) / (m20 * image1.shape[1] + m21 * 0 + m22)
print('top_right:',top_right_x,top_right_y)
#右下角
down_right_x=(m00 * image1.shape[1] + m01 * image1.shape[0] + m02) / (m20 * image1.shape[1] + m21 * image1.shape[0] + m22)
down_right_y=(m10 * image1.shape[1] + m11 * image1.shape[0] + m12) / (m20 * image1.shape[1] + m21 * image1.shape[0] + m22)
print('down_right:',down_right_x,down_right_y)
move_x=min(top_left_x,down_left_x,top_right_x,down_right_x)
move_y=min(top_left_y,down_left_y,top_right_y,down_right_y)
move_x=int(move_x)
move_y=int(move_y)

#对x变换
if move_x<0:
    m00-=move_x*m20
    m01-=move_x*m21
    m02-=move_x*m22
#对y变换
if move_y<0:
    m10-=move_y*m20
    m11-=move_y*m21
    m12-=move_y*m22
#重新给单应矩阵赋值
homography[0, 0], homography[0, 1], homography[0, 2]=m00, m01, m02
homography[1, 0], homography[1, 1], homography[1, 2]=m10, m11, m12
homography[2, 0], homography[2, 1], homography[2, 2]=m20, m21, m22
# 使用单应矩阵进行图像拼接
result = cv2.warpPerspective(image1, homography, (image1.shape[1]+ image2.shape[1]+3000, (image1.shape[0] + image2.shape[0]+3000)))
if move_x<0 and move_y<0:
    print('move_x<0 and move_y<0')
    result[-move_y:image2.shape[0] - move_y, -move_x:image2.shape[1] - move_x] = image2
elif move_x<0 and move_y>0:
    print('move_x<0 and move_y>0')
    result[0:image2.shape[0], -move_x:image2.shape[1] - move_x] = image2
elif move_x>0 and move_y<0:
    print('move_x>0 and move_y<0')
    result[-move_y:image2.shape[0] - move_y, 0:image2.shape[1]] = image2
else:
    print('move_x>0 and move_y>0')
    result[0:image2.shape[0], 0:image2.shape[1]] = image2
#删除黑边
# 使用阈值化操作将黑色区域转换为白色
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
#1代表阈值，255代表大于阈值的设置为255
#=====================================================================
#'_'的用法：
#不感兴趣的返回值用占位符_表示，此处_表示阈值
#=====================================================================
# 查找非零像素点的坐标范围
coords = cv2.findNonZero(threshold)
x, y, w, h = cv2.boundingRect(coords)
#cv2.boundingRect()：这是OpenCV中的一个函数，它以一组坐标作为输入，并计算包围所有点的最小边界矩形。它返回边界矩形的 x 坐标、y 坐标、宽度和高度

# 根据坐标范围裁剪图像
cropped_result = result[y:y+h, x:x+w]
cv2.imwrite("C:\\Users\\86152\\Desktop\\image_match\\bath_result.jpg",cropped_result)
# 结束计时
end_time = time.time()

# 计算时间差
elapsed_time = end_time - start_time

# 输出执行时间
print(f"执行时间：{elapsed_time} 秒")
