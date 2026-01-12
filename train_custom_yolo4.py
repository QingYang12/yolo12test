# 训练自定义 YOLO 模型 - 识别方块
# 使用说明：
#   1. 准备好标注数据集（images/ 和 labels/ 文件夹）
#   2. 修改 data.yaml 配置文件路径
#   3. 运行此脚本开始训练

from ultralytics import YOLO

def train_square_detector():
    """
    训练识别方块的 YOLO 模型
    """
    print("🚀 开始训练自定义 YOLO 模型")
    print("=" * 50)
    
    # 加载预训练模型（迁移学习，更快收敛）
    model = YOLO("yolo12n.pt")  # 使用 nano 版本，轻量快速
    
    # 开始训练
    results = model.train(
        data="data.yaml",           # 数据集配置文件
        epochs=20,                  # 训练轮数（减少到20轮，更快完成）
        imgsz=640,                  # 图片尺寸
        batch=16,                   # 批次大小（根据显存调整）
        name="square_detector",     # 项目名称
        device='cpu',               # 使用 CPU 训练
        patience=50,                # 早停耐心值
        save=True,                  # 保存模型
        plots=True,                 # 生成训练图表
        
        # 可选参数
        workers=8,                  # 数据加载线程数
        project="runs/train",       # 保存路径
        exist_ok=True,              # 允许覆盖
        pretrained=True,            # 使用预训练权重
        optimizer="auto",           # 优化器
        verbose=True,               # 详细输出
        seed=0,                     # 随机种子
        deterministic=True,         # 确定性训练
        
        # 数据增强参数
        hsv_h=0.015,               # 色调增强
        hsv_s=0.7,                 # 饱和度增强
        hsv_v=0.4,                 # 明度增强
        degrees=0.0,               # 旋转角度
        translate=0.1,             # 平移
        scale=0.5,                 # 缩放
        shear=0.0,                 # 剪切
        perspective=0.0,           # 透视变换
        flipud=0.0,                # 上下翻转
        fliplr=0.5,                # 左右翻转
        mosaic=1.0,                # 马赛克增强
        mixup=0.0,                 # mixup增强
    )
    
    print("\n" + "=" * 50)
    print("✅ 训练完成！")
    print(f"📁 模型保存在: runs/train/square_detector/weights/best.pt")
    print(f"📊 训练结果: runs/train/square_detector/")
    
    # 自动导出模型为多种格式
    print("\n📦 正在导出模型...")
    model_path = "runs/train/square_detector/weights/best.pt"
    export_model = YOLO(model_path)
    
    # 导出为 ONNX 格式（通用格式，可用于多种部署场景）
    try:
        onnx_path = export_model.export(format='onnx')
        print(f"✅ ONNX 模型: {onnx_path}")
    except Exception as e:
        print(f"⚠️ ONNX 导出失败: {e}")
    
    # 导出为 TorchScript 格式
    try:
        torchscript_path = export_model.export(format='torchscript')
        print(f"✅ TorchScript 模型: {torchscript_path}")
    except Exception as e:
        print(f"⚠️ TorchScript 导出失败: {e}")
    
    print("\n💾 模型文件说明：")
    print("  - best.pt: PyTorch 模型（推荐，Python 中使用）")
    print("  - best.onnx: ONNX 模型（通用格式，跨平台）")
    print("  - best.torchscript: TorchScript 模型（C++ 部署）")
    print("=" * 50)
    
    return results

def validate_model():
    """
    验证训练好的模型
    """
    print("\n🔍 验证模型性能...")
    
    # 加载训练好的模型
    model = YOLO("runs/train/square_detector/weights/best.pt")
    
    # 在验证集上评估
    metrics = model.val(data="data.yaml")
    
    print(f"\n📊 验证结果:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics

def test_detection(image_path):
    """
    使用训练好的模型进行测试
    
    参数:
        image_path: 测试图片路径
    """
    print(f"\n🎯 测试检测效果: {image_path}")
    
    # 加载训练好的模型
    model = YOLO("runs/train/square_detector/weights/best.pt")
    
    # 预测
    results = model(image_path)
    
    # 显示结果
    for result in results:
        # 保存结果到 test_images 目录
        save_path = "test_images/yolo_test_result.jpg"
        result.save(filename=save_path)
        print(f"\n💾 检测结果已保存: {save_path}")
        
        print(f"\n检测到 {len(result.boxes)} 个方块:")
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            print(f"  - {class_name}: 置信度={conf:.2%}, 位置=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")

if __name__ == "__main__":
    # 训练模型
    train_square_detector()
    
    # 验证模型
    validate_model()
    
    # 测试单张图片
    print("\n" + "=" * 50)
    print("🧪 正在测试模型...")
    test_detection("test_images/test_image_1.jpg")
    
    print("\n" + "=" * 50)
    print("🎉 全部完成！")
    print("\n📚 使用说明：")
    print("  1. 加载模型：model = YOLO('runs/train/square_detector/weights/best.pt')")
    print("  2. 检测图片：results = model('your_image.jpg')")
    print("  3. 查看结果：results[0].show()")
    print("=" * 50)
