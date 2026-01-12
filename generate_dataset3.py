# 自动生成红色方块训练数据集
# 功能：生成图片和对应的 YOLO 格式标注文件

import cv2
import numpy as np
import os
import random
from pathlib import Path

def generate_red_square_image(img_size=640, num_squares=None):
    """
    生成包含红色方块的图片
    
    参数:
        img_size: 图片尺寸
        num_squares: 方块数量（None表示随机1-3个）
    
    返回:
        image: 生成的图片
        labels: YOLO格式标注 [(class_id, center_x, center_y, width, height), ...]
    """
    # 创建随机背景
    # 随机背景颜色（避免红色）
    bg_color = [
        random.randint(100, 255),  # B
        random.randint(100, 255),  # G
        random.randint(50, 150),   # R (避免红色背景)
    ]
    image = np.full((img_size, img_size, 3), bg_color, dtype=np.uint8)
    
    # 添加一些噪点和纹理
    noise = np.random.randint(-30, 30, (img_size, img_size, 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 随机方块数量
    if num_squares is None:
        num_squares = random.randint(1, 3)
    
    labels = []
    
    for _ in range(num_squares):
        # 随机方块尺寸（50-200像素）
        square_size = random.randint(50, 200)
        
        # 随机位置（确保不超出边界）
        x = random.randint(0, img_size - square_size)
        y = random.randint(0, img_size - square_size)
        
        # 随机红色色调（纯红色到偏橙红）
        red_variations = [
            [30, 30, 255],      # 纯红
            [50, 50, 230],      # 深红
            [40, 80, 255],      # 偏橙红
            [20, 20, 200],      # 暗红
            [60, 60, 255],      # 亮红
        ]
        red_color = random.choice(red_variations)
        
        # 绘制方块
        cv2.rectangle(image, (x, y), (x + square_size, y + square_size), red_color, -1)
        
        # 添加随机效果
        if random.random() > 0.5:
            # 添加边框
            border_color = [c - 50 for c in red_color]
            cv2.rectangle(image, (x, y), (x + square_size, y + square_size), border_color, 2)
        
        # 计算 YOLO 格式标注（归一化）
        center_x = (x + square_size / 2) / img_size
        center_y = (y + square_size / 2) / img_size
        width = square_size / img_size
        height = square_size / img_size
        
        labels.append([0, center_x, center_y, width, height])  # class_id=0 表示 red_square
    
    return image, labels

def save_dataset(num_train=200, num_val=50, img_size=640):
    """
    生成并保存完整数据集
    
    参数:
        num_train: 训练集图片数量
        num_val: 验证集图片数量
        img_size: 图片尺寸
    """
    print("🎨 开始生成红色方块数据集")
    print("=" * 60)
    
    # 创建目录结构
    base_dir = Path("dataset")
    for split in ['train', 'val']:
        (base_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (base_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 生成训练集
    print(f"\n📦 生成训练集：{num_train} 张图片...")
    for i in range(num_train):
        # 生成图片和标注
        image, labels = generate_red_square_image(img_size)
        
        # 保存图片
        img_path = base_dir / 'images' / 'train' / f'train_{i:04d}.jpg'
        cv2.imwrite(str(img_path), image)
        
        # 保存标注
        label_path = base_dir / 'labels' / 'train' / f'train_{i:04d}.txt'
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
        
        if (i + 1) % 50 == 0:
            print(f"  ✓ 已生成 {i + 1}/{num_train} 张")
    
    print(f"✅ 训练集完成！")
    
    # 生成验证集
    print(f"\n📦 生成验证集：{num_val} 张图片...")
    for i in range(num_val):
        # 生成图片和标注
        image, labels = generate_red_square_image(img_size)
        
        # 保存图片
        img_path = base_dir / 'images' / 'val' / f'val_{i:04d}.jpg'
        cv2.imwrite(str(img_path), image)
        
        # 保存标注
        label_path = base_dir / 'labels' / 'val' / f'val_{i:04d}.txt'
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
        
        if (i + 1) % 20 == 0:
            print(f"  ✓ 已生成 {i + 1}/{num_val} 张")
    
    print(f"✅ 验证集完成！")
    
    # 生成一些测试图片
    print(f"\n📦 生成测试图片：5 张...")
    test_dir = Path('test_images')
    test_dir.mkdir(exist_ok=True)
    
    for i in range(5):
        image, _ = generate_red_square_image(img_size)
        test_path = test_dir / f'test_image_{i+1}.jpg'
        cv2.imwrite(str(test_path), image)
    
    print(f"✅ 测试图片已保存到 test_images/ 目录！")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("📊 数据集生成完成！")
    print(f"  📁 数据集位置: {base_dir.absolute()}")
    print(f"  📷 训练集: {num_train} 张图片")
    print(f"  📷 验证集: {num_val} 张图片")
    print(f"  🖼️  图片尺寸: {img_size}x{img_size}")
    print(f"  🎯 类别: red_square (ID=0)")
    print("\n💡 下一步：")
    print("  1. 运行训练: python3 train_custom_yolo4.py")
    print("  2. 或先查看生成的图片确认效果")
    print("=" * 60)

def preview_samples(num_samples=5):
    """
    预览生成的样本
    """
    print("\n👀 生成预览样本...")
    
    # 创建 samples 目录
    samples_dir = Path('samples')
    samples_dir.mkdir(exist_ok=True)
    
    for i in range(num_samples):
        image, labels = generate_red_square_image(640)
        
        # 在图片上绘制标注框（用于预览）
        for label in labels:
            _, cx, cy, w, h = label
            # 转换为像素坐标
            x1 = int((cx - w/2) * 640)
            y1 = int((cy - h/2) * 640)
            x2 = int((cx + w/2) * 640)
            y2 = int((cy + h/2) * 640)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, "red_square", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        preview_path = samples_dir / f'preview_sample_{i+1}.jpg'
        cv2.imwrite(str(preview_path), image)
    
    print(f"✅ 已生成 {num_samples} 张预览图片（samples/preview_sample_*.jpg）")
    print("   绿色框表示标注位置")

if __name__ == "__main__":
    # 先生成预览样本
    print("🎨 红色方块数据集生成器")
    print("=" * 60)
    
    # 生成预览
    preview_samples(5)
    
    # 询问是否继续生成完整数据集
    print("\n" + "=" * 60)
    print("📋 接下来将生成完整数据集：")
    print("   - 训练集: 200 张")
    print("   - 验证集: 50 张")
    print("   - 测试图片: 5 张")
    print("   - 总计约: 255 张图片")
    print("=" * 60)
    
    # 自动生成（如果要手动确认，可以添加 input()）
    save_dataset(num_train=200, num_val=50, img_size=640)
    
    print("\n🎉 全部完成！现在可以开始训练了！")
