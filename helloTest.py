# YOLOv12 目标检测示例
# 功能说明：使用 YOLOv12 模型识别图片中狗的位置
#  ttt_detected.png 是识别结果图片
#  COCO 数据集中狗的类别 ID 是 16   （COCO是微软数据集 “日常场景中常见物体” 都在这里有）
#  coco 数据集官方网站 https://cocodataset.org/

# 导入 Ultralytics 库中的 YOLO 类
# Ultralytics 是一个流行的计算机视觉库，提供了 YOLO 系列模型的实现
from ultralytics import YOLO
import cv2  # 用于图像处理和绘制检测框
import os  # 用于文件路径操作

def main():
    """
    主函数：使用 YOLOv12 检测图片中的狗
    
    功能：
    1. 加载预训练的 YOLOv12 模型
    2. 读取 ttt.png 图片
    3. 检测图片中的所有对象
    4. 筛选出狗的检测结果
    5. 在图片上绘制检测框和标签
    6. 保存并显示结果
    """
    # 打印欢迎信息
    print("🎯 YOLOv12 目标检测 - 识别狗的位置")
    print("=" * 50)
    
    # 图片文件路径
    image_path = "ttt.png"
    
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误：找不到图片文件 '{image_path}'")
        return
    
    # 提示用户模型正在加载
    print("\n正在加载 YOLOv12 模型...")
    
    # 创建 YOLO 模型实例
    # 参数说明：
    #   - "yolo12n.pt": 模型权重文件名
    #   - 'n' 代表 nano 版本，是最轻量级的模型
    #   - 首次运行时会自动从网络下载预训练权重文件
    #   - 模型文件会缓存到本地，后续运行会直接加载
    model = YOLO("yolo12n.pt")
    
    # 模型加载成功后的提示信息
    print(f"✅ 模型加载成功！")
    print(f"📋 模型信息: {model.model_name}")
    
    # 【查询功能】显示 COCO 数据集中所有类别
    # 通过 model.names 可以获取所有类别的 ID 和名称
    print(f"\n📚 COCO 数据集包含 {len(model.names)} 个类别")
    print("💡 提示：通过 model.names 查询类别 ID")
    print(f"   例如：狗(dog)的 ID = {list(model.names.keys())[list(model.names.values()).index('dog')]}")
    
    # 开始检测图片
    print(f"\n🔍 正在分析图片 '{image_path}'...")
    
    # 使用模型进行预测
    # results 是一个列表，包含检测到的所有对象
    results = model(image_path)
    
    # 获取第一个结果（因为我们只输入了一张图片）
    result = results[0]
    
    # 读取原始图片用于绘制检测框
    image = cv2.imread(image_path)
    
    # COCO 数据集中狗的类别 ID 是 16
    # YOLO 模型使用 COCO 数据集预训练，包含 80 个类别
    # 查询方法：使用 model.names 字典，格式为 {id: 'name'}
    # 示例：model.names = {0: 'person', 1: 'bicycle', ..., 16: 'dog', ...}
    dog_class_id = 16
    dog_count = 0
    
    print("\n" + "=" * 50)
    print("🐕 检测到的狗的位置信息：")
    print("=" * 50)
    
    # 遍历所有检测结果
    for box in result.boxes:
        # 获取类别 ID
        class_id = int(box.cls[0])
        
        # 只处理狗的检测结果
        if class_id == dog_class_id:
            dog_count += 1
            
            # 获取边界框坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # 获取置信度分数
            confidence = float(box.conf[0])
            
            # 计算中心点坐标
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 计算宽度和高度
            width = x2 - x1
            height = y2 - y1
            
            # 打印检测信息
            print(f"\n狗 #{dog_count}:")
            print(f"  📍 位置: ({int(x1)}, {int(y1)}) 到 ({int(x2)}, {int(y2)})")
            print(f"  🎯 中心点: ({int(center_x)}, {int(center_y)})")
            print(f"  📏 尺寸: 宽={int(width)}px, 高={int(height)}px")
            print(f"  💯 置信度: {confidence:.2%}")
            
            # 在图片上绘制检测框（绿色，线宽3）
            cv2.rectangle(image, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 255, 0), 3)
            
            # 准备标签文本
            label = f"Dog {confidence:.2%}"
            
            # 获取文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            # 绘制标签背景（绿色矩形）
            cv2.rectangle(image,
                         (int(x1), int(y1) - text_height - 10),
                         (int(x1) + text_width, int(y1)),
                         (0, 255, 0), -1)
            
            # 绘制标签文本（白色）
            cv2.putText(image, label,
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (255, 255, 255), 2)
    
    # 输出检测总结
    print("\n" + "=" * 50)
    if dog_count > 0:
        print(f"✅ 检测完成！共找到 {dog_count} 只狗")
        
        # 保存结果图片
        output_path = "ttt_detected.png"
        cv2.imwrite(output_path, image)
        print(f"\n💾 检测结果已保存到: {output_path}")
        print(f"📊 图片尺寸: {image.shape[1]}x{image.shape[0]} 像素")
    else:
        print("❌ 未检测到狗")
    
    print("=" * 50)
    print("🚀 检测完成！")
    
# Python 标准入口点判断
# 当脚本被直接运行时（而非被导入），执行 main() 函数
if __name__ == "__main__":
    main()