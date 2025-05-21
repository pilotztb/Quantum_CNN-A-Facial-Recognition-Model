import os
from PIL import Image # 使用Pillow库
import numpy as np
import pandas as pd
from tqdm import tqdm # 保留进度条

# --- 配置 ---
input_dataset_base_folder = "Datasets" # 指向包含 lgr, ztb 等子文件夹的父文件夹
output_csv_file = "data.csv" # 您希望输出的CSV文件名 (加了_pil以区分)
img_size_width = 100  # 预处理后的图像宽度
img_size_height = 100 # 预处理后的图像高度

# --- 主逻辑 ---
data_rows = [] # 用于存储所有图像数据和标签
label_map = {} # 将文件夹名（人名）映射到数字标签
current_label_id = 0

print(f"开始从 '{input_dataset_base_folder}' 读取预处理好的图像并转换为CSV (使用Pillow)...")

# 遍历 'Datasets' 文件夹下的每个人物子文件夹 (lgr, ztb, ...)
# sorted确保每次运行时标签分配顺序一致
for person_name in sorted(os.listdir(input_dataset_base_folder)):
    person_folder_path = os.path.join(input_dataset_base_folder, person_name)

    if not os.path.isdir(person_folder_path): # 跳过非文件夹项
        continue

    # 为该人物分配一个数字标签
    if person_name not in label_map:
        label_map[person_name] = current_label_id
        print(f"人物: '{person_name}'  =>  标签ID: {current_label_id}")
        current_label_id += 1
    
    assigned_label = label_map[person_name]

    print(f"正在处理文件夹: '{person_folder_path}' (标签: {assigned_label})...")
    image_files = [f for f in os.listdir(person_folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))] # 支持的图像格式

    for image_filename in tqdm(image_files, desc=f"处理 {person_name}"):
        img_path = os.path.join(person_folder_path, image_filename)
        
        try:
            img = Image.open(img_path)
            
            # 1. 转换为灰度图 (确保单通道)
            # 'L'模式为8位像素灰度图
            img_gray = img.convert('L')
            
            # 2. 确认/调整图像尺寸 (可选，因为您说已预处理好，但作为保险措施)
            if img_gray.size != (img_size_width, img_size_height):
                print(f"警告: 图像 '{img_path}' 的尺寸是 {img_gray.size} 而不是期望的 ({img_size_width},{img_size_height})。正在调整大小...")
                img_gray = img_gray.resize((img_size_width, img_size_height), Image.Resampling.LANCZOS) # 使用高质量的缩放算法
            
            # 3. 获取像素数据并展平
            # img.getdata() 返回一个扁平的像素值序列 (对于'L'模式，是整数)
            flattened_pixels = list(img_gray.getdata())
            
            # 检查像素数量是否正确
            expected_pixel_count = img_size_width * img_size_height
            if len(flattened_pixels) != expected_pixel_count:
                print(f"警告: 图像 '{img_path}' 的像素数量 ({len(flattened_pixels)}) 与期望值 ({expected_pixel_count}) 不符。已跳过。")
                continue

            row = flattened_pixels + [assigned_label] # 附加类别标签
            data_rows.append(row)
            
        except FileNotFoundError:
            print(f"错误: 文件未找到 '{img_path}'，已跳过。")
        except Exception as e:
            print(f"处理图像 '{img_path}' 时发生错误: {e}，已跳过。")


if not data_rows:
    print("错误: 未能从指定文件夹中处理任何图像数据。")
else:
    # 创建DataFrame
    num_pixels = img_size_width * img_size_height
    column_names = [f'pixel_{i}' for i in range(num_pixels)] + ['class']
    df_output = pd.DataFrame(data_rows, columns=column_names)

    # 保存为CSV文件
    df_output.to_csv(output_csv_file, index=False)
    print(f"\n任务完成！CSV文件已保存到: '{output_csv_file}'")
    print(f"共处理了 {len(df_output)} 张图像。")
    print(f"标签映射: {label_map}")