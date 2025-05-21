import os
import shutil
import cv2 # 用于人脸检测、图像处理
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_lfw_people # 用于下载LFW数据集
from tqdm import tqdm
from PIL import Image # 用于最后转CSV时读取图像
import sys # 用于 sys.exit()

# --- 配置参数 ---
NUM_CLASSES_TO_SELECT = 3  # 您希望识别的总人数（类别数）
MIN_FACES_PER_PERSON = 50  # 选择至少有多少张照片的人物
LFW_RESIZE_FACTOR = 0.4    # LFW原始图像较大，此缩放因子使其接近100x100
TARGET_IMG_SIZE = (100, 100) # 最终人脸图像保存的统一尺寸 (宽度, 高度)

# 可选：手动指定人物名称列表，如果为空列表，则自动选择图片最多的N个人物
# 例如: SELECTED_PERSON_NAMES = ['George W Bush', 'Colin Powell', 'Donald Rumsfeld']
SELECTED_PERSON_NAMES = [] 

LFW_RAW_SAVE_DIR = "LFW_raw_images_multiclass"
LFW_PROCESSED_SAVE_DIR = "LFW_processed_images_multiclass"
OUTPUT_CSV_FILENAME = f"lfw_{NUM_CLASSES_TO_SELECT}class_data.csv"

# 人脸检测器
try:
    face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    if not os.path.exists(face_cascade_path):
        face_cascade_path = 'haarcascade_frontalface_default.xml' 
        if not os.path.exists(face_cascade_path):
            print(f"错误: 无法找到 'haarcascade_frontalface_default.xml'. 请确保文件路径正确。")
            sys.exit()
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
except Exception as e:
    print(f"加载 CascadeClassifier 出错: {e}")
    sys.exit()

def download_and_save_lfw_subset():
    print(f"--- 步骤1: 下载LFW数据集并保存选定 {NUM_CLASSES_TO_SELECT} 个人物的原始图像到 '{LFW_RAW_SAVE_DIR}' ---")
    
    if os.path.exists(LFW_RAW_SAVE_DIR):
        print(f"警告: 文件夹 '{LFW_RAW_SAVE_DIR}' 已存在。建议手动删除或备份后重新运行以确保数据清洁。")
        # shutil.rmtree(LFW_RAW_SAVE_DIR) # 如果需要每次都重新创建
    if not os.path.exists(LFW_RAW_SAVE_DIR):
        os.makedirs(LFW_RAW_SAVE_DIR)

    try:
        lfw_people = fetch_lfw_people(
            min_faces_per_person=MIN_FACES_PER_PERSON, 
            resize=LFW_RESIZE_FACTOR,
            color=False, 
            slice_=(slice(70, 195, None), slice(78, 172, None)),
            funneled=True
        )
    except Exception as e:
        print(f"下载或加载LFW数据集失败: {e}")
        return None

    n_samples_total, h_lfw, w_lfw = lfw_people.images.shape
    all_target_names = lfw_people.target_names
    n_classes_total_available = all_target_names.shape[0]

    print(f"LFW初步加载完成: 共 {n_samples_total} 张图像, 每张尺寸约 {h_lfw}x{w_lfw}。")
    print(f"共找到 {n_classes_total_available} 位至少有 {MIN_FACES_PER_PERSON} 张照片的人物。")

    selected_person_indices_in_lfw = []
    actual_selected_person_names = []

    if SELECTED_PERSON_NAMES and len(SELECTED_PERSON_NAMES) >= NUM_CLASSES_TO_SELECT:
        print(f"尝试根据指定名称选择人物: {SELECTED_PERSON_NAMES[:NUM_CLASSES_TO_SELECT]}")
        temp_indices = []
        for name in SELECTED_PERSON_NAMES[:NUM_CLASSES_TO_SELECT]:
            try:
                idx = list(all_target_names).index(name)
                temp_indices.append(idx)
            except ValueError:
                print(f"警告: 人物 '{name}' 未在加载的LFW数据中找到。")
        if len(temp_indices) == NUM_CLASSES_TO_SELECT:
            selected_person_indices_in_lfw = temp_indices
        else:
            print(f"指定的人物不足 {NUM_CLASSES_TO_SELECT} 个或未全部找到，将自动选择。")
            selected_person_indices_in_lfw = [] 

    if not selected_person_indices_in_lfw:
        if n_classes_total_available >= NUM_CLASSES_TO_SELECT:
            print(f"自动选择图片数量最多的 {NUM_CLASSES_TO_SELECT} 个人...")
            counts = np.bincount(lfw_people.target)
            selected_person_indices_in_lfw = np.argsort(counts)[-NUM_CLASSES_TO_SELECT:]
        else:
            print(f"错误：LFW加载的人物不足 {NUM_CLASSES_TO_SELECT} 类。请调整 MIN_FACES_PER_PERSON 参数。")
            return None
            
    actual_selected_person_names = [all_target_names[i] for i in selected_person_indices_in_lfw]
    print(f"最终选择的人物: {actual_selected_person_names}")
    print(f"对应的人物原始LFW ID: {selected_person_indices_in_lfw}")

    mask = np.isin(lfw_people.target, selected_person_indices_in_lfw)
    X_subset_images = lfw_people.images[mask]
    y_subset_original_target = lfw_people.target[mask]

    saved_counts = {name: 0 for name in actual_selected_person_names}
    for i in range(len(X_subset_images)):
        img_data = (X_subset_images[i] * 255).astype(np.uint8)
        current_person_original_id = y_subset_original_target[i]
        current_person_name = all_target_names[current_person_original_id]
        
        person_raw_dir = os.path.join(LFW_RAW_SAVE_DIR, current_person_name.replace(" ", "_")) # 替换空格
        if not os.path.exists(person_raw_dir):
            os.makedirs(person_raw_dir)
        
        img_filename = f"{current_person_name.replace(' ', '_')}_{saved_counts[current_person_name]:04d}.png"
        cv2.imwrite(os.path.join(person_raw_dir, img_filename), img_data)
        saved_counts[current_person_name] += 1
    
    for name, count in saved_counts.items():
        print(f"为 '{name}' 保存了 {count} 张原始(sklearn处理后)图像。")
    
    return actual_selected_person_names

def preprocess_saved_lfw_images(person_names_list_to_process):
    print(f"\n--- 步骤2: 对 '{LFW_RAW_SAVE_DIR}' 中的图像进行标准化预处理 (调整到{TARGET_IMG_SIZE}) ---")
    
    if not os.path.exists(LFW_PROCESSED_SAVE_DIR):
        os.makedirs(LFW_PROCESSED_SAVE_DIR)

    for person_name_orig in person_names_list_to_process:
        person_name_fs = person_name_orig.replace(" ", "_") # 文件系统安全名
        raw_person_folder = os.path.join(LFW_RAW_SAVE_DIR, person_name_fs)
        processed_person_folder = os.path.join(LFW_PROCESSED_SAVE_DIR, person_name_fs)

        if not os.path.exists(raw_person_folder):
            print(f"警告: 文件夹 '{raw_person_folder}' 不存在，跳过对 {person_name_orig} 的预处理。")
            continue
        if not os.path.exists(processed_person_folder):
            os.makedirs(processed_person_folder)

        person_processed_count = 0
        print(f"正在预处理 '{person_name_orig}' 的图像...")
        for image_filename in tqdm(os.listdir(raw_person_folder), desc=f"预处理 {person_name_orig}"):
            img_path = os.path.join(raw_person_folder, image_filename)
            if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
            if img is None: continue

            resized_face = cv2.resize(img, TARGET_IMG_SIZE, interpolation=cv2.INTER_AREA)
            
            save_path = os.path.join(processed_person_folder, image_filename)
            cv2.imwrite(save_path, resized_face)
            person_processed_count += 1
        
        print(f"为 '{person_name_orig}' 处理并保存了 {person_processed_count} 张图像到 '{processed_person_folder}'")
        
    print(f"LFW子集图像预处理完成。")

def convert_processed_lfw_to_csv(person_names_list_to_process):
    print(f"\n--- 步骤3: 将 '{LFW_PROCESSED_SAVE_DIR}' 中的图像转换为CSV文件 '{OUTPUT_CSV_FILENAME}' ---")
    
    data_rows = []
    label_map = {} 
    current_label_id = 0

    # 使用 person_names_list_to_process 来保证标签分配顺序与文件夹一致
    for person_name_orig in person_names_list_to_process:
        person_name_fs = person_name_orig.replace(" ", "_") # 文件系统安全名
        person_folder_path = os.path.join(LFW_PROCESSED_SAVE_DIR, person_name_fs)

        if not os.path.isdir(person_folder_path):
            print(f"警告: 找不到文件夹 '{person_folder_path}'，跳过。")
            continue

        if person_name_orig not in label_map: # 使用原始名称作为map的key
            label_map[person_name_orig] = current_label_id
            print(f"CSV标签映射: 人物 '{person_name_orig}'  =>  标签ID: {current_label_id}")
            current_label_id += 1
        assigned_label = label_map[person_name_orig]

        image_files = [f for f in os.listdir(person_folder_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_filename in tqdm(image_files, desc=f"转换 {person_name_orig} 到CSV"):
            img_path = os.path.join(person_folder_path, image_filename)
            try:
                img_pil = Image.open(img_path).convert('L')
                if img_pil.size != TARGET_IMG_SIZE:
                    img_pil = img_pil.resize(TARGET_IMG_SIZE, Image.Resampling.LANCZOS)
                
                flattened_pixels = list(img_pil.getdata())
                expected_pixel_count = TARGET_IMG_SIZE[0] * TARGET_IMG_SIZE[1]
                if len(flattened_pixels) != expected_pixel_count: continue

                row = flattened_pixels + [assigned_label]
                data_rows.append(row)
            except Exception as e:
                print(f"处理图像 '{img_path}' 转换为CSV时发生错误: {e}，已跳过。")

    if not data_rows:
        print("错误: 未能从预处理文件夹中转换任何图像数据到CSV。")
        return None # 返回None表示失败

    num_pixels = TARGET_IMG_SIZE[0] * TARGET_IMG_SIZE[1]
    column_names = [f'pixel_{i}' for i in range(num_pixels)] + ['class']
    df_output = pd.DataFrame(data_rows, columns=column_names)
    df_output.to_csv(OUTPUT_CSV_FILENAME, index=False)
    print(f"\nCSV转换完成！文件已保存到: '{OUTPUT_CSV_FILENAME}'")
    print(f"共 {len(df_output)} 张图像的数据已写入CSV。")
    print(f"CSV中使用的标签映射: {label_map}")
    return label_map # 返回标签映射，FR_Qcnn.py可能会用到

if __name__ == "__main__":
    actual_persons_used = download_and_save_lfw_subset()
    if actual_persons_used:
        preprocess_saved_lfw_images(actual_persons_used)
        label_mapping_for_csv = convert_processed_lfw_to_csv(actual_persons_used)
        if label_mapping_for_csv:
            # (可选) 将标签映射保存到json文件，供FR_Qcnn.py读取
            with open('lfw_label_map.json', 'w') as f_map:
                json.dump(label_mapping_for_csv, f_map)
            print("标签映射已保存到 'lfw_label_map.json'")
            print(f"\n数据预处理流程全部完成。现在您可以使用 '{OUTPUT_CSV_FILENAME}' 和 'lfw_label_map.json' 文件作为 FR_Qcnn.py 的输入。")
    else:
        print("未能成功选择LFW人物或下载数据，后续步骤已跳过。")