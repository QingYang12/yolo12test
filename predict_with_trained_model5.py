# 使用训练好的模型进行红色方块检测
# 功能：加载训练完成的 YOLO 模型，对新图片进行预测

from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def predict_single_image(model_path, image_path, save_dir="predictions"):
    """
    使用训练好的模型预测单张图片
    
    参数:
        model_path: 模型文件路径
        image_path: 输入图片路径
        save_dir: 结果保存目录
    """
    print(f"🎯 使用模型: {model_path}")
    print(f"📷 检测图片: {image_path}")
    print("=" * 60)
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到模型文件 '{model_path}'")
        print("💡 请先运行 train_custom_yolo4.py 完成训练")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ 错误：找不到图片文件 '{image_path}'")
        return
    
    # 加载模型
    print("\n🔄 正在加载模型...")
    model = YOLO(model_path)
    print("✅ 模型加载成功！")
    
    # 预测
    print("\n🔍 正在检测...")
    results = model(image_path, conf=0.25)  # 置信度阈值 25%
    
    # 创建保存目录
    Path(save_dir).mkdir(exist_ok=True)
    
    # 处理结果
    for i, result in enumerate(results):
        # 保存结果图片
        img_name = Path(image_path).stem
        save_path = f"{save_dir}/{img_name}_detected.jpg"
        result.save(filename=save_path)
        
        # 输出检测结果
        print("\n" + "=" * 60)
        print(f"🟥 检测结果:")
        print("=" * 60)
        
        if len(result.boxes) == 0:
            print("❌ 未检测到红色方块")
        else:
            print(f"✅ 检测到 {len(result.boxes)} 个红色方块\n")
            
            for j, box in enumerate(result.boxes, 1):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                # 计算中心点
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                width = int(x2 - x1)
                height = int(y2 - y1)
                
                print(f"方块 #{j}:")
                print(f"  📍 位置: ({int(x1)}, {int(y1)}) 到 ({int(x2)}, {int(y2)})")
                print(f"  🎯 中心点: ({center_x}, {center_y})")
                print(f"  📏 尺寸: 宽={width}px, 高={height}px")
                print(f"  💯 置信度: {conf:.2%}")
                print(f"  🏷️  类别: {class_name}\n")
        
        print(f"💾 结果已保存: {save_path}")
        print("=" * 60)

def predict_batch_images(model_path, images_dir, save_dir="predictions"):
    """
    批量预测多张图片
    
    参数:
        model_path: 模型文件路径
        images_dir: 图片目录
        save_dir: 结果保存目录
    """
    print(f"🎯 使用模型: {model_path}")
    print(f"📁 图片目录: {images_dir}")
    print("=" * 60)
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到模型文件 '{model_path}'")
        return
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
        image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ 错误：在 '{images_dir}' 中未找到图片文件")
        return
    
    print(f"\n📦 找到 {len(image_files)} 张图片")
    
    # 加载模型
    print("\n🔄 正在加载模型...")
    model = YOLO(model_path)
    print("✅ 模型加载成功！")
    
    # 创建保存目录
    Path(save_dir).mkdir(exist_ok=True)
    
    # 批量预测
    print("\n🔍 开始批量检测...")
    total_detections = 0
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 处理: {image_file.name}")
        
        # 预测
        results = model(str(image_file), conf=0.25)
        
        # 保存结果
        for result in results:
            save_path = f"{save_dir}/{image_file.stem}_detected{image_file.suffix}"
            result.save(filename=save_path)
            
            num_detections = len(result.boxes)
            total_detections += num_detections
            
            if num_detections > 0:
                print(f"  ✅ 检测到 {num_detections} 个方块")
            else:
                print(f"  ⚠️  未检测到方块")
    
    # 输出统计
    print("\n" + "=" * 60)
    print("📊 批量检测完成！")
    print(f"  📷 处理图片: {len(image_files)} 张")
    print(f"  🟥 总检测数: {total_detections} 个方块")
    print(f"  💾 结果保存: {save_dir}/")
    print("=" * 60)

def predict_with_onnx(onnx_path, image_path):
    """
    使用 ONNX 模型进行预测（演示用）
    
    参数:
        onnx_path: ONNX 模型路径
        image_path: 输入图片路径
    """
    print(f"🎯 使用 ONNX 模型: {onnx_path}")
    print(f"📷 检测图片: {image_path}")
    
    if not os.path.exists(onnx_path):
        print(f"❌ 错误：找不到 ONNX 模型 '{onnx_path}'")
        return
    
    # YOLO 也可以直接加载 ONNX 模型
    model = YOLO(onnx_path)
    results = model(image_path)
    
    print(f"✅ ONNX 模型预测完成！检测到 {len(results[0].boxes)} 个方块")

def main():
    """
    主函数：演示不同的预测方式
    """
    print("🚀 红色方块检测器 - 使用训练好的模型")
    print("=" * 60)
    
    # 模型路径（优先使用 .pt 模型）
    model_path = "runs/train/square_detector/weights/best.pt"
    
    # 方式 1: 预测单张图片
    print("\n【方式 1】预测单张图片")
    predict_single_image(
        model_path=model_path,
        image_path="test_images/test_image_1.jpg",
        save_dir="predictions"
    )
    
    # 方式 2: 批量预测（可选）
    # print("\n【方式 2】批量预测")
    # predict_batch_images(
    #     model_path=model_path,
    #     images_dir="test_images",
    #     save_dir="predictions"
    # )
    
    # 方式 3: 使用 ONNX 模型（可选）
    # onnx_path = "runs/train/square_detector/weights/best.onnx"
    # if os.path.exists(onnx_path):
    #     print("\n【方式 3】使用 ONNX 模型")
    #     predict_with_onnx(onnx_path, "test_images/test_image_1.jpg")
    
    print("\n🎉 预测完成！")
    print("\n💡 提示：")
    print("  - 修改 main() 函数中的图片路径来预测其他图片")
    print("  - 取消注释可以启用批量预测或 ONNX 模型预测")

if __name__ == "__main__":
    main()
