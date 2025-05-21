import os
import shutil
# import cv2 # 不再使用 cv2
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_lfw_people # 用于下载LFW数据集
from tqdm import tqdm
from PIL import Image, ImageFile # 导入 PIL Image 和 ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True # 如果遇到截断图像问题，可以尝试打开此选项

# --- 配置参数 ---
# LFW 数据集下载和选择参数
MIN_FACES_PER_PERSON = 40  # 选择至少有多少张照片的人物
LFW_RESIZE_FACTOR = 0.4    # LFW原始图像较大，此缩放因子使其接近100x100
TARGET_IMG_SIZE = (100, 100) # 最终人脸图像保存的统一尺寸 (宽度, 高度)

# 选择要用于二分类的两个人名 (从LFW数据集中选取)
SELECTED_PERSON_NAMES = [] # 例如: ['George W Bush', 'Colin Powell'] 

# 输出文件夹路径
LFW_RAW_SAVE_DIR = "LFW_raw_images_pil"          # 保存从LFW下载的原始(但已由sklearn处理过的)选择人物的图像 (PIL版本)
LFW_PROCESSED_SAVE_DIR = "LFW_processed_images_pil" # 保存经过裁剪、灰度化、调整大小后的图像 (PIL版本)
OUTPUT_CSV_FILENAME = "lfw_subset_data_pil.csv" # 最终生成的CSV文件名 (PIL版本)

# 人脸检测器不再使用cv2.CascadeClassifier

# --- 步骤1: 下载LFW数据集，选择人物，并保存原始选择的图像 ---
def download_and_save_lfw_subset():
    print(f"--- 步骤1: 下载LFW数据集并保存选定人物的原始图像到 '{LFW_RAW_SAVE_DIR}' (使用Pillow) ---")
    
    if os.path.exists(LFW_RAW_SAVE_DIR):
        print(f"警告: 文件夹 '{LFW_RAW_SAVE_DIR}' 已存在。其中的内容可能会被覆盖或混合。")
    if not os.path.exists(LFW_RAW_SAVE_DIR):
        os.makedirs(LFW_RAW_SAVE_DIR)

    try:
        lfw_people = fetch_lfw_people(
            min_faces_per_person=MIN_FACES_PER_PERSON, 
            resize=LFW_RESIZE_FACTOR,
            color=False, # 确保灰度图
            slice_=(slice(70, 195, None), slice(78, 172, None)),
            funneled=True
        )
    except Exception as e:
        print(f"下载或加载LFW数据集失败: {e}")
        print("请确保网络连接正常，或者scikit-learn版本支持。")
        return None

    n_samples_total, h_lfw, w_lfw = lfw_people.images.shape
    all_target_names = lfw_people.target_names
    n_classes_total = all_target_names.shape[0]

    print(f"LFW初步加载完成: 共 {n_samples_total} 张图像, 每张尺寸约 {h_lfw}x{w_lfw} (经过resize和slice)。")
    print(f"共找到 {n_classes_total} 位至少有 {MIN_FACES_PER_PERSON} 张照片的人物。")

    selected_indices_target_ids = []
    if SELECTED_PERSON_NAMES and len(SELECTED_PERSON_NAMES) >= 2:
        print(f"尝试根据指定名称选择人物: {SELECTED_PERSON_NAMES}")
        temp_indices = []
        for name in SELECTED_PERSON_NAMES:
            try:
                idx = list(all_target_names).index(name)
                temp_indices.append(idx)
            except ValueError:
                print(f"警告: 人物 '{name}' 未在加载的LFW数据中找到 (基于min_faces_per_person={MIN_FACES_PER_PERSON})。")
        if len(temp_indices) >= 2:
            selected_indices_target_ids = temp_indices[:2]
            print(f"已选择的人物ID (target_names索引): {selected_indices_target_ids}")
        else:
            print("指定的人物不足两个或未找到，将自动选择图片最多的两个人物。")
            selected_indices_target_ids = [] 
    
    if not selected_indices_target_ids and n_classes_total >= 2 :
        print("自动选择图片数量最多的两个人...")
        counts = np.bincount(lfw_people.target)
        selected_indices_target_ids = np.argsort(counts)[-2:]
        print(f"自动选择的人物ID (target_names索引): {selected_indices_target_ids}")
    elif n_classes_total < 2:
        print("错误：LFW加载的人物不足2类，无法进行二分类。请调整 fetch_lfw_people 参数。")
        return None
            
    person1_id, person2_id = selected_indices_target_ids[0], selected_indices_target_ids[1]
    person1_name = all_target_names[person1_id]
    person2_name = all_target_names[person2_id]
    
    print(f"将处理的人物1: {person1_name} (原始ID: {person1_id})")
    print(f"将处理的人物2: {person2_name} (原始ID: {person2_id})")

    mask = np.logical_or(lfw_people.target == person1_id, lfw_people.target == person2_id)
    X_subset_images = lfw_people.images[mask]
    y_subset_original_target = lfw_people.target[mask]

    saved_counts = {person1_name: 0, person2_name: 0}
    for i in range(len(X_subset_images)):
        # fetch_lfw_people 返回的是0-1的浮点数灰度图 NumPy 数组
        img_array_float = X_subset_images[i] 
        # 转换为0-255的uint8类型，这是图像文件的常见格式
        img_array_uint8 = (img_array_float * 255).astype(np.uint8)
        
        current_person_original_id = y_subset_original_target[i]
        current_person_name = all_target_names[current_person_original_id]
        
        person_raw_dir = os.path.join(LFW_RAW_SAVE_DIR, current_person_name)
        if not os.path.exists(person_raw_dir):
            os.makedirs(person_raw_dir)
        
        img_filename = f"{current_person_name}_{saved_counts[current_person_name]:04d}.png"
        save_path = os.path.join(person_raw_dir, img_filename)
        
        try:
            # 从NumPy数组创建PIL图像对象 (模式'L'代表灰度图)
            pil_img = Image.fromarray(img_array_uint8, mode='L')
            pil_img.save(save_path)
            saved_counts[current_person_name] += 1
        except Exception as e:
            print(f"保存图像 {save_path} 时出错: {e}")
            
    print(f"为 '{person1_name}' 保存了 {saved_counts[person1_name]} 张原始(sklearn处理后)图像。")
    print(f"为 '{person2_name}' 保存了 {saved_counts[person2_name]} 张原始(sklearn处理后)图像。")
    
    return [person1_name, person2_name]

# --- 步骤2: 对LFW_RAW_SAVE_DIR中的图像进行标准化预处理 ---
def preprocess_saved_lfw_images(person_names_list):
    print(f"\n--- 步骤2: 对 '{LFW_RAW_SAVE_DIR}' 中的图像进行标准化预处理 (调整到{TARGET_IMG_SIZE}) (使用Pillow) ---")
    
    if not os.path.exists(LFW_PROCESSED_SAVE_DIR):
        os.makedirs(LFW_PROCESSED_SAVE_DIR)

    total_processed_count = 0
    for person_name in person_names_list:
        raw_person_folder = os.path.join(LFW_RAW_SAVE_DIR, person_name)
        processed_person_folder = os.path.join(LFW_PROCESSED_SAVE_DIR, person_name)

        if not os.path.exists(raw_person_folder):
            print(f"警告: 文件夹 '{raw_person_folder}' 不存在，跳过对 {person_name} 的预处理。")
            continue
        if not os.path.exists(processed_person_folder):
            os.makedirs(processed_person_folder)

        person_processed_count = 0
        print(f"正在预处理 '{person_name}' 的图像...")
        for image_filename in tqdm(os.listdir(raw_person_folder), desc=f"预处理 {person_name}"):
            img_path = os.path.join(raw_person_folder, image_filename)
            if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                continue

            try:
                img_pil = Image.open(img_path)
                # 确保是灰度图
                img_pil_gray = img_pil.convert('L')
                
                # fetch_lfw_people 已经做了初步的人脸对齐和裁剪
                # 我们直接resize到目标尺寸
                resized_face_pil = img_pil_gray.resize(TARGET_IMG_SIZE, Image.Resampling.LANCZOS)
                
                save_path = os.path.join(processed_person_folder, image_filename)
                resized_face_pil.save(save_path)
                person_processed_count += 1
            except FileNotFoundError:
                 print(f"警告: 图像文件未找到 {img_path}，跳过。")
            except Image.UnidentifiedImageError:
                 print(f"警告: Pillow无法识别图像文件 {img_path}，可能已损坏或格式不受支持，跳过。")
            except Exception as e:
                print(f"警告: 处理图像 {img_path} 时出错: {e}，跳过。")
            
        print(f"为 '{person_name}' 处理并保存了 {person_processed_count} 张图像到 '{processed_person_folder}'")
        total_processed_count += person_processed_count
        
    print(f"LFW子集图像预处理完成。总共处理了 {total_processed_count} 张图像。")

# --- 步骤3: 将预处理后的LFW图像转换为CSV文件 ---
def convert_processed_lfw_to_csv(person_names_list):
    print(f"\n--- 步骤3: 将 '{LFW_PROCESSED_SAVE_DIR}' 中的图像转换为CSV文件 '{OUTPUT_CSV_FILENAME}' (使用Pillow) ---")
    
    data_rows = []
    label_map = {} 
    current_label_id = 0

    for person_name in person_names_list:
        person_folder_path = os.path.join(LFW_PROCESSED_SAVE_DIR, person_name)

        if not os.path.isdir(person_folder_path):
            print(f"警告: 找不到文件夹 '{person_folder_path}'，跳过。")
            continue

        if person_name not in label_map:
            label_map[person_name] = current_label_id
            print(f"CSV标签映射: 人物 '{person_name}'  =>  标签ID: {current_label_id}")
            current_label_id += 1
        assigned_label = label_map[person_name]

        print(f"正在从 '{person_folder_path}' 读取图像并转换为CSV行 (标签: {assigned_label})...")
        image_files = [f for f in os.listdir(person_folder_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]

        for image_filename in tqdm(image_files, desc=f"转换 {person_name} 到CSV"):
            img_path = os.path.join(person_folder_path, image_filename)
            
            try:
                img_pil = Image.open(img_path)
                # 确保是灰度图并符合尺寸 (尽管上一步已处理，这里再确认一次)
                img_pil_gray = img_pil.convert('L')
                if img_pil_gray.size != TARGET_IMG_SIZE:
                    img_pil_gray = img_pil_gray.resize(TARGET_IMG_SIZE, Image.Resampling.LANCZOS)
                
                flattened_pixels = list(img_pil_gray.getdata())
                
                expected_pixel_count = TARGET_IMG_SIZE[0] * TARGET_IMG_SIZE[1]
                if len(flattened_pixels) != expected_pixel_count:
                    print(f"警告: 图像 '{img_path}' 的像素数量 ({len(flattened_pixels)}) 与期望值 ({expected_pixel_count}) 不符。已跳过。")
                    continue

                row = flattened_pixels + [assigned_label]
                data_rows.append(row)
            except FileNotFoundError:
                 print(f"警告: 图像文件未找到 {img_path} (CSV转换中)，跳过。")
            except Image.UnidentifiedImageError:
                 print(f"警告: Pillow无法识别图像文件 {img_path} (CSV转换中)，可能已损坏或格式不受支持，跳过。")
            except Exception as e:
                print(f"处理图像 '{img_path}' 转换为CSV时发生错误: {e}，已跳过。")

    if not data_rows:
        print("错误: 未能从预处理文件夹中转换任何图像数据到CSV。")
        return

    num_pixels = TARGET_IMG_SIZE[0] * TARGET_IMG_SIZE[1]
    column_names = [f'pixel_{i}' for i in range(num_pixels)] + ['class']
    df_output = pd.DataFrame(data_rows, columns=column_names)

    try:
        df_output.to_csv(OUTPUT_CSV_FILENAME, index=False)
        print(f"\nCSV转换完成！文件已保存到: '{OUTPUT_CSV_FILENAME}'")
        print(f"共 {len(df_output)} 张图像的数据已写入CSV。")
        print(f"CSV中使用的标签映射: {label_map}")
    except Exception as e:
        print(f"保存CSV文件 '{OUTPUT_CSV_FILENAME}' 时出错: {e}")


# --- 主执行流程 ---
if __name__ == "__main__":
    selected_persons = download_and_save_lfw_subset()

    if selected_persons:
        preprocess_saved_lfw_images(selected_persons)
        convert_processed_lfw_to_csv(selected_persons)
        
        print(f"\n数据预处理流程全部完成。现在您可以使用 '{OUTPUT_CSV_FILENAME}' 文件。")
    else:
        print("未能成功选择LFW人物或下载数据，后续步骤已跳过。")