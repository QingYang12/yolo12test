# 红色方块检测示例
# 功能说明：使用 OpenCV 的颜色检测识别图片中红色方块的位置
# 原理：通过 HSV 颜色空间识别红色区域，然后找出方块的轮廓和位置

import cv2
import numpy as np
import os

def detect_red_square(image_path, output_path="red_square_detected.png"):
    """
    检测图片中的红色方块位置
    
    参数:
        image_path: 输入图片路径
        output_path: 输出结果图片路径
    
    功能:
        1. 读取图片并转换为 HSV 颜色空间
        2. 通过颜色范围筛选出红色区域
        3. 查找红色区域的轮廓
        4. 绘制检测框并显示位置信息
        5. 保存结果图片
    """
    print("🎯 红色方块检测程序")
    print("=" * 50)
    
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误：找不到图片文件 '{image_path}'")
        return
    
    # 读取图片
    print(f"\n📷 正在读取图片 '{image_path}'...")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ 错误：无法读取图片 '{image_path}'")
        return
    
    # 转换为 HSV 颜色空间
    # HSV 比 RGB 更适合做颜色检测
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 定义红色的 HSV 范围
    # 注意：红色在 HSV 色环上跨越 0 度，需要定义两个范围
    # 范围1：0-10（偏橙红）
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    
    # 范围2：170-180（偏紫红）
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # 创建红色掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 形态学操作：去除噪点和填充空洞
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 查找轮廓
    print("\n🔍 正在检测红色方块...")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建输出图片（在原图上绘制）
    result_image = image.copy()
    
    square_count = 0
    print("\n" + "=" * 50)
    print("🟥 检测到的红色方块位置信息：")
    print("=" * 50)
    
    # 遍历所有轮廓
    for i, contour in enumerate(contours):
        # 计算轮廓面积，过滤掉太小的区域（可能是噪点）
        area = cv2.contourArea(contour)
        
        # 设置最小面积阈值（根据实际情况调整）
        if area < 500:  # 面积小于 500 像素的忽略
            continue
        
        square_count += 1
        
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 计算中心点
        center_x = x + w // 2
        center_y = y + h // 2
        
        # 计算近似多边形（判断是否为方形）
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 打印检测信息
        print(f"\n红色方块 #{square_count}:")
        print(f"  📍 位置: ({x}, {y}) 到 ({x+w}, {y+h})")
        print(f"  🎯 中心点: ({center_x}, {center_y})")
        print(f"  📏 尺寸: 宽={w}px, 高={h}px")
        print(f"  📐 面积: {int(area)} 平方像素")
        print(f"  🔷 顶点数: {len(approx)}")
        
        # 在图片上绘制检测框（红色，线宽3）
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 0, 255), 3)
        
        # 绘制中心点
        cv2.circle(result_image, (center_x, center_y), 5, (255, 0, 0), -1)
        
        # 绘制标签
        label = f"Red Square #{square_count}"
        
        # 获取文本尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # 绘制标签背景（红色矩形）
        cv2.rectangle(result_image,
                     (x, y - text_height - 10),
                     (x + text_width, y),
                     (0, 0, 255), -1)
        
        # 绘制标签文本（白色）
        cv2.putText(result_image, label,
                   (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 2)
        
        # 绘制轮廓（绿色）
        cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
    
    # 输出检测总结
    print("\n" + "=" * 50)
    if square_count > 0:
        print(f"✅ 检测完成！共找到 {square_count} 个红色方块")
        
        # 保存结果图片
        cv2.imwrite(output_path, result_image)
        print(f"\n💾 检测结果已保存到: {output_path}")
        print(f"📊 图片尺寸: {image.shape[1]}x{image.shape[0]} 像素")
    else:
        print("❌ 未检测到红色方块")
        print("\n💡 提示：")
        print("   - 请确保图片中有明显的红色物体")
        print("   - 可以调整 HSV 颜色范围参数")
        print("   - 可以调整最小面积阈值")
    
    print("=" * 50)
    print("🚀 检测完成！")

def main():
    """
    主函数：设置图片路径并执行检测
    """
    # 设置输入图片路径（使用刚生成的测试图片）
    image_path = "test_images/test_image_1.jpg"  # 使用生成的测试图片
    
    # 执行检测
    detect_red_square(image_path)

if __name__ == "__main__":
    main()