# 使用ADAM优化器
import matplotlib
matplotlib.use('Agg') # 必须在导入pyplot之前
import matplotlib.pyplot as plt
import os
import sys
import logging
from datetime import datetime
import json # 用于加载标签映射
import numpy as np # 确保导入numpy

# --- 日志记录设置 ---
log_folder = "log_multiclass" # 新的日志文件夹
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
log_filename_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filepath = os.path.join(log_folder, f"fr_qcnn_multiclass_run_{log_filename_timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
# --- 日志记录设置结束 ---

import tensorflow as tf
from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers import ADAM # <<< 导入 ADAM 优化器
# from qiskit_algorithms.optimizers import COBYLA # 如果需要可以保留作为对比
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
# from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient # <<< 在当前环境下无法使用，注释掉

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd

# 设置 Matplotlib 支持中文显示
try:
    font_list = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
    plt.rcParams['font.sans-serif'] = font_list
    plt.rcParams['axes.unicode_minus'] = False
    logging.info(f"Matplotlib 中文字体尝试列表设置为: {font_list}")
except Exception as e_font:
    logging.warning(f"设置 Matplotlib 中文字体失败: {e_font}。中文可能无法正常显示。")

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

algorithm_globals.random_seed = 1

# --- 全局配置 ---
FORCE_RETRAIN_CNN = False
num_qnn_input_features = 8 # 与量子比特数一致
logging.info(f"QNN将使用 {num_qnn_input_features} 个输入特征 (量子比特)。")
output_image_folder = "所有输出图像_multiclass"
if not os.path.exists(output_image_folder):
    os.makedirs(output_image_folder)
    logging.info(f"文件夹 '{output_image_folder}' 已创建。")

objective_func_vals = []
def simple_text_callback(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)
    logging.info(f"迭代次数: {len(objective_func_vals)}, 当前目标函数值: {obj_func_eval:.4f}")

def stable_softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# === 阶段一：加载多分类人脸数据集 ===
logging.info("--- 阶段一：加载多分类人脸数据集 ---")
num_expected_classes_from_config = 5
csv_filename = f"lfw_{num_expected_classes_from_config}class_data.csv"

try:
    df = pd.read_csv(csv_filename)
    logging.info(f"成功从 '{csv_filename}' 加载数据，共 {len(df)} 条记录。")
except FileNotFoundError:
    logging.error(f"错误: '{csv_filename}' 文件未找到。请先运行数据准备脚本。")
    sys.exit()

label_map_filepath = 'lfw_label_map.json'
actual_person_names_map = None
id_to_name_map = {}
num_unique_classes = None

if os.path.exists(label_map_filepath):
    try:
        with open(label_map_filepath, 'r', encoding='utf-8') as f_map:
            actual_person_names_map = json.load(f_map)
            if actual_person_names_map:
                id_to_name_map = {v: k for k, v in actual_person_names_map.items()} # 反转映射以供显示
                num_classes_from_map = len(actual_person_names_map)
                if num_classes_from_map != num_expected_classes_from_config:
                    logging.warning(f"配置文件中期望类别数 ({num_expected_classes_from_config}) 与标签映射文件中的类别数 ({num_classes_from_map}) 不符。")
                num_unique_classes = num_classes_from_map
                logging.info(f"成功加载标签映射: {actual_person_names_map}。识别到 {num_unique_classes} 个类别。")
            else:
                logging.warning(f"标签映射文件 '{label_map_filepath}' 为空或格式不正确。")
    except Exception as e_load_map:
        logging.error(f"加载标签映射文件 '{label_map_filepath}' 失败: {e_load_map}")

X_from_csv = df.iloc[:, :-1].values.reshape(-1, 100, 100, 1)
y_labels_from_csv = df['class'].values

if num_unique_classes is None: # 如果从json加载失败或文件不存在
    num_unique_classes = len(np.unique(y_labels_from_csv))
    logging.info(f"从CSV数据中推断出的类别数量: {num_unique_classes}")
    # 创建一个临时的 id_to_name_map
    if not id_to_name_map:
         for i in range(num_unique_classes):
             id_to_name_map[i] = f"类别_{i}"
         logging.info(f"已创建临时标签名称映射: {id_to_name_map}")


if num_unique_classes < 2:
    logging.error(f"错误：最终确定的数据集中类别少于2 ({num_unique_classes})。")
    sys.exit()
y_one_hot_from_csv = to_categorical(y_labels_from_csv, num_classes=num_unique_classes)
logging.info(f"加载的数据形状: X={X_from_csv.shape}, y_one_hot={y_one_hot_from_csv.shape}")

# --- (绘制CSV随机图像样本的代码，与之前相同，此处省略以节省篇幅) ---
plt.figure(figsize=(6,6))
q_csv_idx = np.random.randint(len(X_from_csv))
plt.imshow(X_from_csv[q_csv_idx,:,:,0], cmap='gray')
display_label_id = y_labels_from_csv[q_csv_idx]
display_label_name = id_to_name_map.get(display_label_id, f"ID {display_label_id}") # 使用get避免KeyError
plt.title(f'来自CSV的随机图像 - 标签: {display_label_name}')
plt.axis('off')
try:
    plt.savefig(os.path.join(output_image_folder, "01_CSV多分类随机图像样本.png"))
except Exception as e_save_fig:
    logging.error(f"保存CSV随机图像样本失败: {e_save_fig}")
plt.close()

X_train_csv, X_test_csv, y_train_csv, y_test_csv = train_test_split(
    X_from_csv, y_one_hot_from_csv, test_size=0.2,
    random_state=algorithm_globals.random_seed, stratify=y_one_hot_from_csv
)
logging.info(f'用于经典CNN的训练集尺寸 - {X_train_csv.shape}, {y_train_csv.shape}')
logging.info(f'用于经典CNN的测试集尺寸 - {X_test_csv.shape}, {y_test_csv.shape}')

# === 阶段二：训练经典CNN (直接输出低维特征) ===
# --- (此部分与之前基本相同，确保模型结构和保存/加载逻辑正确，此处省略详细代码) ---
logging.info("\n--- 阶段二：训练经典CNN (新结构) 并提取特征 ---")
models_dir = "models_multiclass"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
fixed_best_cnn_model_filename = f"cnn_best_{num_unique_classes}class_model.keras"
checkpoint_path_cnn_to_load_or_save = os.path.join(models_dir, fixed_best_cnn_model_filename)
model = None
history_cnn_dict = None
# ... (加载或训练CNN模型的逻辑，同前) ...
if not FORCE_RETRAIN_CNN and os.path.exists(checkpoint_path_cnn_to_load_or_save):
    logging.info(f"检测到已训练的CNN模型: '{checkpoint_path_cnn_to_load_or_save}'，尝试加载...")
    try:
        model = tf.keras.models.load_model(checkpoint_path_cnn_to_load_or_save)
        logging.info("已成功加载预训练的CNN模型。将跳过CNN训练步骤。")
        history_path = checkpoint_path_cnn_to_load_or_save.replace(".keras", "_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as f_hist:
                    history_cnn_dict = json.load(f_hist)
                logging.info(f"已从 '{history_path}' 加载CNN训练历史。")
            except Exception as e_load_hist:
                logging.warning(f"加载CNN训练历史 '{history_path}' 失败: {e_load_hist}")
    except Exception as e_load:
        logging.warning(f"加载预训练的CNN模型 '{checkpoint_path_cnn_to_load_or_save}' 失败: {e_load}。将重新训练CNN。")
        model = None
else:
    if FORCE_RETRAIN_CNN:
        logging.info("FORCE_RETRAIN_CNN 为 True，将重新训练CNN。")
    else:
        logging.info(f"未找到预训练的CNN模型 '{checkpoint_path_cnn_to_load_or_save}' 或 FORCE_RETRAIN_CNN 为 True。将构建并训练新的CNN模型。")

if model is None:
    train_datagen = ImageDataGenerator(rescale=1./255., rotation_range=10, width_shift_range=0.1,
                                       height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
    valid_datagen = ImageDataGenerator(rescale=1./255.)
    model_train_name = f'FaceRec_CNN_MultiClass_{num_unique_classes}class_DirectLowDim_'+datetime.now().strftime("%Y%m%d_%H%M%S")
    model = Sequential(name = model_train_name)
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(100, 100, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(num_qnn_input_features, activation='tanh', name='dense_for_qnn_features'))
    model.add(BatchNormalization()) # 在tanh之后添加BN可能有助于稳定QNN的输入
    model.add(Dense(num_unique_classes, activation='softmax', name='cnn_output_layer'))
    # ... (保存模型摘要，编译，回调，训练，保存历史的逻辑，同前) ...
    original_stdout_cnn_summary = sys.stdout
    summary_cnn_log_path = os.path.join(log_folder, f"model_summary_{model.name}.txt")
    try:
        with open(summary_cnn_log_path, 'w', encoding='utf-8') as f_summary:
            sys.stdout = f_summary; model.summary(); sys.stdout = original_stdout_cnn_summary
        logging.info(f"Keras主模型摘要已保存到: {summary_cnn_log_path}")
    except Exception as e_save_summary: logging.error(f"保存Keras主模型摘要到文件失败: {e_save_summary}")
    finally: sys.stdout = original_stdout_cnn_summary
    learning_rate_cnn = 0.001
    optimizer_cnn = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_cnn)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_cnn, metrics=['accuracy'])
    ch_cnn = ModelCheckpoint(checkpoint_path_cnn_to_load_or_save, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    es_cnn = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20) # 增加patience
    learning_rate_reduction_cnn = ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1, factor=0.2) # 增加patience
    callbacks_list_cnn = [ch_cnn, es_cnn, learning_rate_reduction_cnn]
    logging.info("开始训练经典CNN模型 (多分类，用于直接提取低维特征)...")
    epochs_cnn = 100 # 可以调整
    batch_size_cnn = 32
    if len(X_train_csv) < batch_size_cnn : batch_size_cnn = max(1, len(X_train_csv) // 4); logging.warning(f"警告：训练样本数 ({len(X_train_csv)}) 小于batch_size，已将batch_size调整为 {batch_size_cnn}")
    history_cnn_obj = model.fit(train_datagen.flow(X_train_csv, y_train_csv, batch_size=batch_size_cnn),epochs=epochs_cnn,validation_data=valid_datagen.flow(X_test_csv, y_test_csv),callbacks=callbacks_list_cnn,verbose=1)
    history_cnn_dict = history_cnn_obj.history
    logging.info("经典CNN模型训练完成。")
    history_path_to_save = checkpoint_path_cnn_to_load_or_save.replace(".keras", "_history.json")
    try: # 保存history
        serializable_history = {k: [float(val) for val in v_list] for k, v_list in history_cnn_dict.items()}
        with open(history_path_to_save, 'w', encoding='utf-8') as f_hist_save: json.dump(serializable_history, f_hist_save)
        logging.info(f"CNN训练历史已保存到 {history_path_to_save}。")
    except Exception as e_save_hist: logging.error(f"保存CNN训练历史失败: {e_save_hist}")
    logging.info(f"从 {checkpoint_path_cnn_to_load_or_save} 加载刚刚训练的最佳经典CNN模型...")
    try: model = tf.keras.models.load_model(checkpoint_path_cnn_to_load_or_save); logging.info("刚刚训练的最佳模型加载成功。")
    except Exception as e: logging.error(f"加载刚刚训练的最佳模型失败: {e}. 将使用训练结束时的模型。")

# --- (绘制CNN训练历史图的代码，与之前相同，此处省略) ---
if history_cnn_dict:
    try:
        plt.figure(figsize=(12, 4)); plt.subplot(1, 2, 1)
        plt.plot(history_cnn_dict['accuracy'], label='训练准确率'); plt.plot(history_cnn_dict['val_accuracy'], label='验证准确率')
        plt.title('经典CNN(多分类) 准确率'); plt.xlabel('Epoch'); plt.ylabel('准确率'); plt.legend(); plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(history_cnn_dict['loss'], label='训练损失'); plt.plot(history_cnn_dict['val_loss'], label='验证损失')
        plt.title('经典CNN(多分类) 损失'); plt.xlabel('Epoch'); plt.ylabel('损失'); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_image_folder, "02_经典CNN_多分类_训练历史.png")); plt.close()
    except Exception as e_plot_hist: logging.error(f"绘制CNN训练历史图失败: {e_plot_hist}")
else: logging.info("未进行CNN训练或未能加载训练历史，跳过绘制CNN训练曲线。")

if model is None:
    logging.error("错误：CNN模型 (model) 未能加载或训练。无法继续。"); sys.exit("CNN模型处理失败。")
# --- (特征提取器模型的创建，与之前相同，此处省略) ---
feature_extractor_layer_name = 'dense_for_qnn_features'
feature_extractor_model = None
logging.info(f"尝试从模型 '{model.name}' 创建特征提取器，目标层: '{feature_extractor_layer_name}'")
try:
    fe_input = tf.keras.Input(shape=(100, 100, 1), name="feature_extractor_new_input")
    temp_model_for_fe = tf.keras.models.clone_model(model) # 使用新变量名以避免混淆
    temp_model_for_fe.build((None, 100, 100, 1))
    temp_model_for_fe.set_weights(model.get_weights())
    current_output = fe_input; found_fe_target_layer = False; target_fe_output = None
    for layer_in_model in temp_model_for_fe.layers:
        current_output = layer_in_model(current_output)
        if layer_in_model.name == feature_extractor_layer_name:
            target_fe_output = current_output; found_fe_target_layer = True
            logging.info(f"已为特征提取器连接到目标层 '{layer_in_model.name}'。"); break
    if not found_fe_target_layer or target_fe_output is None:
        logging.error(f"错误: 未能在模型 '{model.name}' 中找到或连接到名为 '{feature_extractor_layer_name}' 的层。"); sys.exit("无法创建特征提取器。")
    feature_extractor_model = tf.keras.Model(inputs=fe_input, outputs=target_fe_output, name="FeatureExtractor_Reconstructed_Robust")
    logging.info(f"特征提取模型 '{feature_extractor_model.name}' 创建成功。")
    # ... (保存特征提取器摘要) ...
    original_stdout_fe_summary = sys.stdout
    base_model_name_for_summary = model.name.split('_')[-1] if model.name.startswith('FaceRec_CNN_MultiClass_DirectLowDim_') else model.name
    if len(base_model_name_for_summary) > 20 : base_model_name_for_summary = base_model_name_for_summary[:20]
    summary_fe_log_path = os.path.join(log_folder, f"feature_extractor_summary_{base_model_name_for_summary}.txt")
    try:
        with open(summary_fe_log_path, 'w', encoding='utf-8') as f_fe_summary:
            sys.stdout = f_fe_summary; feature_extractor_model.summary(); sys.stdout = original_stdout_fe_summary
        logging.info(f"特征提取器模型摘要已保存到: {summary_fe_log_path}")
    except Exception as e_save_fe_summary: logging.error(f"保存特征提取器模型摘要到文件失败: {e_save_fe_summary}")
    finally: sys.stdout = original_stdout_fe_summary
except Exception as e_build_fe: logging.error(f"构建或获取特征提取模型失败: {e_build_fe}"); logging.exception("详细错误信息:"); sys.exit("无法继续构建特征提取器。")

logging.info("开始从人脸图像中提取低维深度特征...")
all_face_low_dim_features = feature_extractor_model.predict(X_from_csv / 255.0) # 确保数据已归一化
logging.info(f"提取到的低维深度特征形状: {all_face_low_dim_features.shape}")


# === 阶段三：对CNN输出的低维特征进行缩放 ===
logging.info("\n--- 阶段三：对CNN输出的低维特征进行缩放 ---")
scaler = MinMaxScaler(feature_range=(0, np.pi))
scaled_features_for_qnn = scaler.fit_transform(all_face_low_dim_features)
logging.info(f"CNN直接输出并缩放后的特征形状: {scaled_features_for_qnn.shape}")

# === 阶段四：准备QNN的标签并分割数据 ===
logging.info("\n--- 阶段四：准备QNN的标签并分割数据 ---")
y_qnn_target_labels = y_labels_from_csv
logging.info(f"QNN目标标签示例 (整数 0 to N-1, 前20个): {y_qnn_target_labels[:20]}")
X_qnn_train, X_qnn_test, y_qnn_train, y_qnn_test = train_test_split(
    scaled_features_for_qnn, y_qnn_target_labels, test_size=0.3, # 可以调整测试集比例
    random_state=algorithm_globals.random_seed, stratify=y_qnn_target_labels
)
logging.info(f"QNN训练数据形状: X={X_qnn_train.shape}, y={y_qnn_train.shape}")
logging.info(f"QNN测试数据形状: X={X_qnn_test.shape}, y={y_qnn_test.shape}")

# === 阶段五：定义和构建QNN的量子电路 (多分类调整) ===
logging.info("\n--- 阶段五：定义和构建QNN的量子电路 (多分类) ---")
num_qnn_qubits = num_qnn_input_features

# --- 参数用于更清晰的图像 ---
high_dpi = 300
image_scale = 1.0 # Qiskit的draw scale, default is 0.7.
# 动态调整figsize，确保电路图不会太拥挤
custom_figsize_fm = (max(10, num_qnn_qubits * 0.8), max(5, num_qnn_qubits * 0.4))
custom_figsize_ansatz = (max(10, num_qnn_qubits * 1.0), max(6, num_qnn_qubits * 0.6))


# 定义 FeatureMap 和 Ansatz
# 注意：这里的 reps 值应该与您期望的电路复杂度一致
fm_reps = 2 # 特征映射重复次数
ans_reps = 4 # Ansatz重复次数

feature_map = ZZFeatureMap(feature_dimension=num_qnn_qubits, reps=fm_reps, entanglement='linear')
feature_map.name = f"ZZFeatureMap_MultiClass_reps{fm_reps}" # 动态名称
if num_qnn_qubits <= 10: # 限制绘图的量子比特数，避免过大电路绘图时间过长或内存问题
    try:
        # PNG
        fig_fm_png, ax_fm_png = plt.subplots(figsize=custom_figsize_fm)
        feature_map.decompose().draw("mpl", ax=ax_fm_png, fold=-1, scale=image_scale)
        png_path_fm = os.path.join(output_image_folder, f"03_QNN_{feature_map.name}({num_qnn_qubits}qubits)_dpi{high_dpi}.png")
        fig_fm_png.savefig(png_path_fm, dpi=high_dpi, bbox_inches='tight')
        plt.close(fig_fm_png)
        logging.info(f"QNN 特征映射 ({feature_map.name}) PNG图像已保存到: {png_path_fm}")
        # SVG
        fig_fm_svg, ax_fm_svg = plt.subplots(figsize=custom_figsize_fm)
        feature_map.decompose().draw("mpl", ax=ax_fm_svg, fold=-1, scale=image_scale)
        svg_path_fm = os.path.join(output_image_folder, f"03_QNN_{feature_map.name}({num_qnn_qubits}qubits).svg")
        fig_fm_svg.savefig(svg_path_fm, format='svg', bbox_inches='tight')
        plt.close(fig_fm_svg)
        logging.info(f"QNN 特征映射 ({feature_map.name}) SVG图像已保存到: {svg_path_fm}")
    except Exception as e_draw_fm:
        logging.error(f"绘制 FeatureMap ({feature_map.name}) 出错: {e_draw_fm}. 跳过绘图。")
else:
    logging.info(f"QNN 特征映射 ({feature_map.name}) 已创建 (量子比特数 {num_qnn_qubits} 较大，跳过分解绘图)。")

ansatz = RealAmplitudes(num_qnn_qubits, reps=ans_reps, entanglement='full') # entanglement='linear' 或 'full'
ansatz.name = f"RealAmplitudes_MultiClass_reps{ans_reps}" # 动态名称
if num_qnn_qubits <= 10:
    try:
        # PNG
        fig_ans_png, ax_ans_png = plt.subplots(figsize=custom_figsize_ansatz)
        ansatz.decompose().draw("mpl", ax=ax_ans_png, fold=-1, scale=image_scale)
        png_path_ans = os.path.join(output_image_folder, f"04_QNN_{ansatz.name}({num_qnn_qubits}qubits)_dpi{high_dpi}.png")
        fig_ans_png.savefig(png_path_ans, dpi=high_dpi, bbox_inches='tight')
        plt.close(fig_ans_png)
        logging.info(f"QNN Ansatz ({ansatz.name}) PNG图像已保存到: {png_path_ans}")
        # SVG
        fig_ans_svg, ax_ans_svg = plt.subplots(figsize=custom_figsize_ansatz)
        ansatz.decompose().draw("mpl", ax=ax_ans_svg, fold=-1, scale=image_scale)
        svg_path_ans = os.path.join(output_image_folder, f"04_QNN_{ansatz.name}({num_qnn_qubits}qubits).svg")
        fig_ans_svg.savefig(svg_path_ans, format='svg', bbox_inches='tight')
        plt.close(fig_ans_svg)
        logging.info(f"QNN Ansatz ({ansatz.name}) SVG图像已保存到: {svg_path_ans}")
    except Exception as e_draw_ansatz:
        logging.error(f"绘制 Ansatz ({ansatz.name}) 出错: {e_draw_ansatz}. 跳过绘图。")
else:
    logging.info(f"QNN Ansatz ({ansatz.name}) 已创建 (量子比特数 {num_qnn_qubits} 较大，跳过分解绘图)。")

qnn_hybrid_circuit = QuantumCircuit(num_qnn_qubits)
qnn_hybrid_circuit.compose(feature_map, inplace=True)
qnn_hybrid_circuit.compose(ansatz, inplace=True)
qnn_hybrid_circuit.name = f"HybridQNN_Circuit_MultiClass_FM{fm_reps}_Ans{ans_reps}"
logging.info(f"混合量子电路 '{qnn_hybrid_circuit.name}' 已构建。")


observables_list = []
for i in range(num_unique_classes):
    pauli_string_list = ['I'] * num_qnn_qubits
    # 将可观测量分布在不同的量子比特上，如果类别数超过量子比特数则循环使用
    qubit_index_for_obs = i % num_qnn_qubits
    pauli_string_list[qubit_index_for_obs] = 'Z' # 通常使用 Z 算符
    observables_list.append(SparsePauliOp("".join(pauli_string_list)))
    if i >= num_qnn_qubits and num_qnn_qubits > 0 : # 仅在类别数多于量子比特时记录警告
        logging.warning(f"类别 {i} 的可观测量复用了量子比特 {qubit_index_for_obs} 的Z算符，因为量子比特数 ({num_qnn_qubits}) 少于类别数 ({num_unique_classes})。")
logging.info(f"为 {num_unique_classes} 个类别创建了 {len(observables_list)} 个可观测量。")


# --- EstimatorQNN 定义 ---
# 由于无法使用 ParamShiftEstimatorGradient，我们将 gradient 参数留空 (或显式设为 None)
# EstimatorQNN 默认的 gradient 参数就是 None
qnn_estimator = EstimatorQNN(
    circuit=qnn_hybrid_circuit.decompose(), # 确保使用分解后的电路以获得最佳性能
    observables=observables_list,
    input_params=feature_map.parameters, # 特征映射的参数作为QNN的输入参数
    weight_params=ansatz.parameters,     # Ansatz的参数作为QNN的可训练权重
    # gradient=gradient_calculator # <<< 移除或注释掉这一行
    gradient=None # 显式设置 gradient=None
)
logging.info(f"EstimatorQNN 已定义，使用 {len(observables_list)} 个可观测量，未显式配置梯度计算方法 (gradient=None)。")

initial_point_qnn = None # 对于ADAM，通常不需要手动设置初始点，它会随机初始化或有自己的策略
logging.info("QNN将使用随机初始点 (或由优化器决定)。")

# --- 优化器和分类器定义 ---
# 使用 ADAM 优化器
adam_maxiter = 100 # 可以调整ADAM的迭代次数
adam_lr = 0.01   # 可以调整ADAM的学习率
qnn_optimizer_adam = ADAM(maxiter=adam_maxiter, lr=adam_lr)
logging.info(f"QNN 将使用 ADAM 优化器: maxiter={adam_maxiter}, lr={adam_lr}")

qnn_classifier = NeuralNetworkClassifier(
    neural_network=qnn_estimator, # qnn_estimator 现在 gradient=None
    optimizer=qnn_optimizer_adam,
    callback=simple_text_callback,
    loss='cross_entropy', # 内置交叉熵损失，适用于多分类
    one_hot=True,         # 标签是one-hot编码的
    initial_point=initial_point_qnn,
)
logging.info(f"NeuralNetworkClassifier (多分类，使用内置 'cross_entropy' 损失，one_hot=True, ADAM优化器，QNN gradient=None) 已定义。")

# === 阶段六：使用真实人脸特征训练和评估QNN (多分类) ===
logging.info("\n--- 阶段六：使用真实人脸特征训练和评估QNN (多分类) ---")
logging.info(f"QNN训练数据样本: X={X_qnn_train.shape}, y={y_qnn_train.shape}")
logging.info(f"QNN训练标签示例 (整数 0 to N-1, 前20个): {y_qnn_train[:20]}") # y_qnn_train 是整数标签
logging.info(f'开始QNN模型拟合 (在真实人脸特征上，使用 ADAM 优化器, QNN gradient=None)...')
objective_func_vals.clear() # 清除之前的迭代记录

try:
    # 注意：NeuralNetworkClassifier 的 fit 方法期望 y 是类别索引 (0, 1, ..., num_classes-1)
    # 如果 y_qnn_train 是 one-hot 编码的，需要转换或确保NNC正确处理
    # 从您的代码看 y_qnn_target_labels = y_labels_from_csv 是整数标签，所以 y_qnn_train 也是整数标签
    qnn_classifier.fit(X_qnn_train, y_qnn_train)
    logging.info('QNN模型拟合完成。')
except Exception as e_qnn_fit:
    logging.error(f"QNN模型拟合过程中发生错误: {e_qnn_fit}")
    logging.exception("详细错误信息:")


if objective_func_vals:
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(objective_func_vals) + 1), objective_func_vals, marker='o', linestyle='-')
        
        optimizer_params_str = f"ADAM_iter{adam_maxiter}_lr{adam_lr}"
        circuit_params_str = f"FM{fm_reps}_Ans{ans_reps}"
        gradient_status_str = "QNNGradNone" # 表示 EstimatorQNN 的 gradient=None
        
        plot_title = f'QNN (多分类, 内置CE, {circuit_params_str}, {optimizer_params_str}, {gradient_status_str}) 训练目标函数值'
        plt.title(plot_title)
        plt.xlabel('迭代次数'); plt.ylabel('目标函数值 (交叉熵)'); plt.grid(True)
        
        save_filename = f"06_QNN_多分类_{circuit_params_str}_{optimizer_params_str}_{gradient_status_str}_目标函数值.png"
        plt.savefig(os.path.join(output_image_folder, save_filename))
        plt.close()
        logging.info(f"QNN目标函数曲线图已保存为: {save_filename}")
    except Exception as e_plot_qnn_hist:
        logging.error(f"绘制QNN目标函数曲线失败: {e_plot_qnn_hist}")
else:
    logging.info("未记录QNN目标函数值 (objective_func_vals为空)，无法绘制曲线。")

try:
    accuracy_qnn_train_real = np.round(100 * qnn_classifier.score(X_qnn_train, y_qnn_train), 2)
    logging.info(f"QNN在多分类训练集上的准确率 (ADAM, QNNGradNone): {accuracy_qnn_train_real}%")

    # y_qnn_predict_real = qnn_classifier.predict(X_qnn_test) # .predict() 返回类别索引
    accuracy_qnn_test_real = np.round(100 * qnn_classifier.score(X_qnn_test, y_qnn_test), 2)
    logging.info(f"QNN在多分类测试集上的准确率 (ADAM, QNNGradNone): {accuracy_qnn_test_real}%")

    logging.info(f"\nQNN在多分类测试集上的部分预测结果 (ADAM, QNNGradNone):")
    num_samples_to_show_qnn_pred = min(10, len(X_qnn_test))
    
    # 确保 id_to_name_map 已经根据 num_unique_classes 正确填充
    # sorted_person_names_by_id = [id_to_name_map.get(i, f"ID_{i}_无名") for i in range(num_unique_classes)]
    # 修正：确保 id_to_name_map 的键是整数
    temp_id_to_name_map = {int(k): v for k,v in id_to_name_map.items()} if id_to_name_map else {}
    sorted_person_names_by_id = [temp_id_to_name_map.get(i, f"类别_{i}") for i in range(num_unique_classes)]


    logging.info(f"真实标签 vs 预测标签 (类别索引 0-{num_unique_classes-1})")
    y_qnn_predict_indices = qnn_classifier.predict(X_qnn_test)

    for i_pred_final in range(num_samples_to_show_qnn_pred):
        true_label_index = y_qnn_test[i_pred_final] # y_qnn_test 已经是整数标签
        pred_label_index = y_qnn_predict_indices[i_pred_final]

        true_l_display = sorted_person_names_by_id[true_label_index] if 0 <= true_label_index < len(sorted_person_names_by_id) else f"越界真实ID {true_label_index}"
        pred_l_display = sorted_person_names_by_id[pred_label_index] if 0 <= pred_label_index < len(sorted_person_names_by_id) else f"越界预测ID {pred_label_index}"
        
        logging.info(f"样本 {i_pred_final}: 真实类别 = {true_l_display} (ID:{true_label_index}), QNN预测 = {pred_l_display} (ID:{pred_label_index})")

except Exception as e_qnn_eval:
    logging.error(f"QNN模型评估或预测过程中发生错误: {e_qnn_eval}")
    logging.exception("详细错误信息:")

logging.info(f"\n所有分析和图像保存已完成。请检查 '{output_image_folder}' 和 '{log_folder}' 文件夹。")