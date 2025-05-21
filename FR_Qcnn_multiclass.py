# FR_Qcnn_multiclass.py (方案C: CNN直接输出低维特征 + 日志记录 + 多分类)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import logging
from datetime import datetime
import json # 用于加载标签映射

# --- 日志记录设置 ---
log_folder = "log_multiclass" # 使用新的日志文件夹
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

import tensorflow as tf
from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, EfficientSU2 # 添加EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
# from qiskit_machine_learning.neural_networks import SamplerQNN # 如果要尝试SamplerQNN
# from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss # 如果需要自定义损失

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop 

import pandas as pd
import numpy as np

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

algorithm_globals.random_seed = 1

# --- 全局配置 ---
num_qnn_input_features = 8 
logging.info(f"QNN将使用 {num_qnn_input_features} 个输入特征 (量子比特)。")
output_image_folder = "所有输出图像_multiclass" # 使用新的图像输出文件夹
if not os.path.exists(output_image_folder):
    os.makedirs(output_image_folder)
    logging.info(f"文件夹 '{output_image_folder}' 已创建。")

objective_func_vals = [] 
def simple_text_callback(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)
    logging.info(f"迭代次数: {len(objective_func_vals)}, 当前目标函数值: {obj_func_eval:.4f}")

# === 阶段一：加载多分类人脸数据集 ===
logging.info("--- 阶段一：加载多分类人脸数据集 ---")
csv_filename = f"lfw_{json.load(open('lfw_label_map.json')) if os.path.exists('lfw_label_map.json') else 3}class_data.csv" # 动态或固定文件名
# 为了简单，我们直接使用 prepare_lfw_multiclass_data.py 生成的固定文件名
# 您也可以从 lfw_label_map.json 中读取类别数来动态构建文件名
num_expected_classes = 3 # 假设我们用了3分类，这个值应与prepare脚本中的NUM_CLASSES_TO_SELECT一致
csv_filename = f"lfw_{num_expected_classes}class_data.csv"

try:
    df = pd.read_csv(csv_filename) 
    logging.info(f"成功从 '{csv_filename}' 加载数据，共 {len(df)} 条记录。")
except FileNotFoundError:
    logging.error(f"错误: '{csv_filename}' 文件未找到。请先运行 'prepare_lfw_multiclass_data.py'。")
    sys.exit()

# 加载标签映射 (如果存在)
label_map_filepath = 'lfw_label_map.json'
actual_person_names_map = None
if os.path.exists(label_map_filepath):
    with open(label_map_filepath, 'r') as f_map:
        actual_person_names_map = json.load(f_map) # {'PersonA': 0, 'PersonB': 1, ...}
        logging.info(f"成功加载标签映射: {actual_person_names_map}")
else:
    logging.warning(f"警告: 未找到标签映射文件 '{label_map_filepath}'。预测结果将只显示类别ID。")


X_from_csv = df.iloc[:, :-1].values.reshape(-1, 100, 100, 1) 
y_labels_from_csv = df['class'].values # 这些是 0, 1, 2, ... N-1 整数标签
num_unique_classes = len(np.unique(y_labels_from_csv))
logging.info(f"从CSV加载的类别数量 (num_unique_classes): {num_unique_classes}")
if num_unique_classes < 2:
    logging.error(f"错误：数据集中类别少于2 ({num_unique_classes})。")
    sys.exit()
y_one_hot_from_csv = to_categorical(y_labels_from_csv, num_classes=num_unique_classes)
logging.info(f"加载的数据形状: X={X_from_csv.shape}, y_one_hot={y_one_hot_from_csv.shape}")

# ... (绘制随机图像样本的代码可以保留) ...
plt.figure(figsize=(6,6))
q_csv_idx = np.random.randint(len(X_from_csv))
plt.imshow(X_from_csv[q_csv_idx,:,:,0], cmap='gray')
#尝试显示人名
display_label_name = "未知"
if actual_person_names_map:
    for name, idx in actual_person_names_map.items():
        if idx == y_labels_from_csv[q_csv_idx]:
            display_label_name = name
            break
plt.title(f'来自CSV的随机图像 - 标签: {y_labels_from_csv[q_csv_idx]} ({display_label_name})')
plt.axis('off')
plt.savefig(os.path.join(output_image_folder, "01_CSV多分类随机图像样本.png"))
plt.close()


X_train_csv, X_test_csv, y_train_csv, y_test_csv = train_test_split(
    X_from_csv, y_one_hot_from_csv, test_size=0.2, 
    random_state=algorithm_globals.random_seed, stratify=y_one_hot_from_csv
)
logging.info(f'用于经典CNN的训练集尺寸 - {X_train_csv.shape}, {y_train_csv.shape}')
logging.info(f'用于经典CNN的测试集尺寸 - {X_test_csv.shape}, {y_test_csv.shape}')

# === 阶段二：训练经典CNN (直接输出低维特征) ===
logging.info("\n--- 阶段二：训练经典CNN (新结构) 并提取特征 ---")
train_datagen = ImageDataGenerator(rescale=1./255., rotation_range=10, width_shift_range=0.1, 
                                   height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255.)

model_name = 'FaceRec_CNN_MultiClass_DirectLowDim_'+datetime.now().strftime("%Y%m%d_%H%M%S")
model = Sequential(name = model_name)
# ... (CNN结构与之前方案C类似，确保Dense(num_qnn_input_features,...) 和 Dense(num_unique_classes,...) 正确) ...
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
model.add(Dense(num_qnn_input_features, activation='relu', name='dense_for_qnn_features'))
model.add(BatchNormalization())
model.add(Dense(num_unique_classes, activation='softmax', name='cnn_output_layer')) # 输出层单元数为num_unique_classes
model.summary(print_fn=logging.info) # 直接用logging.info打印

learning_rate_cnn = 0.001
optimizer_cnn = RMSprop(learning_rate=learning_rate_cnn)
model.compile(loss='categorical_crossentropy', optimizer=optimizer_cnn, metrics=['accuracy']) # 损失函数不变

models_dir = "models_multiclass" # 新的模型保存路径
if not os.path.exists(models_dir): os.makedirs(models_dir)
checkpoint_path_cnn = os.path.join(models_dir, model_name + "_cnn_best.keras")
# ... (回调函数 ch_cnn, es_cnn, learning_rate_reduction_cnn 定义不变) ...
ch_cnn = ModelCheckpoint(checkpoint_path_cnn, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
es_cnn = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20) # 增加一点patience
learning_rate_reduction_cnn = ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1, factor=0.2)
callbacks_list_cnn = [ch_cnn, es_cnn, learning_rate_reduction_cnn]

logging.info("开始训练经典CNN模型 (多分类，用于直接提取低维特征)...")
epochs_cnn = 60 # 对于多分类和可能更多的数据，可能需要更多epochs
batch_size_cnn = 32 
# ... (batch_size_cnn 动态调整逻辑不变) ...
if len(X_train_csv) < batch_size_cnn :
    batch_size_cnn = max(1, len(X_train_csv) // 4) 
    logging.warning(f"警告：训练样本数 ({len(X_train_csv)}) 小于batch_size，已将batch_size调整为 {batch_size_cnn}")


history_cnn = model.fit( # ... (model.fit 调用不变) ...
    train_datagen.flow(X_train_csv, y_train_csv, batch_size=batch_size_cnn),
    epochs=epochs_cnn,
    validation_data=valid_datagen.flow(X_test_csv, y_test_csv),
    callbacks=callbacks_list_cnn,
    verbose=1
)
logging.info("经典CNN模型训练完成。")
# ... (绘制并保存经典CNN训练历史图的代码不变，文件名可以加上multiclass标识) ...
# 例如: "02a_经典CNN(多分类)训练历史_准确率.png"

logging.info(f"从 {checkpoint_path_cnn} 加载效果最好的经典CNN模型...")
try:
    model = tf.keras.models.load_model(checkpoint_path_cnn) 
    logging.info("模型加载成功。")
except Exception as e:
    logging.error(f"加载完整模型失败: {e}.")
    sys.exit("无法继续。")

feature_extractor_layer_name = 'dense_for_qnn_features' 
# ... (创建 feature_extractor_model 的代码不变) ...
new_input = tf.keras.Input(shape=(100, 100, 1), name="new_feature_extractor_input")
x = new_input; found_target_layer = False; target_layer_output = None
for layer in model.layers:
    try:
        x = layer(x) 
        if layer.name == feature_extractor_layer_name:
            target_layer_output = x; found_target_layer = True; break 
    except Exception: pass # 简化错误处理
if not found_target_layer : sys.exit("特征提取层未找到")
feature_extractor_model = tf.keras.Model(inputs=new_input, outputs=target_layer_output, name="FeatureExtractor_Reconstructed_LowDim_Multi")
feature_extractor_model.summary(print_fn=logging.info)


logging.info("开始从人脸图像中提取低维深度特征...")
all_face_low_dim_features = feature_extractor_model.predict(X_from_csv / 255.0) 
logging.info(f"提取到的低维深度特征形状: {all_face_low_dim_features.shape}")

# === 阶段三：对CNN输出的低维特征进行缩放 ===
logging.info("\n--- 阶段三：对CNN输出的低维特征进行缩放 ---")
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_features_for_qnn = scaler.fit_transform(all_face_low_dim_features) 
logging.info(f"CNN直接输出并缩放后的特征形状: {scaled_features_for_qnn.shape}")

# === 阶段四：准备QNN的标签并分割数据 ===
logging.info("\n--- 阶段四：准备QNN的标签并分割数据 ---")
# 对于多分类QNN，我们直接使用整数标签 0, 1, ..., N-1
y_qnn_target_labels = y_labels_from_csv # <<< 直接使用整数标签
logging.info(f"QNN目标标签示例 (整数 0 to N-1, 前20个): {y_qnn_target_labels[:20]}")
X_qnn_train, X_qnn_test, y_qnn_train, y_qnn_test = train_test_split(
    scaled_features_for_qnn, y_qnn_target_labels, test_size=0.3, 
    random_state=algorithm_globals.random_seed, stratify=y_qnn_target_labels 
)
logging.info(f"QNN训练数据形状: X={X_qnn_train.shape}, y={y_qnn_train.shape}")
logging.info(f"QNN测试数据形状: X={X_qnn_test.shape}, y={y_qnn_test.shape}")

# === 阶段五：定义和构建QNN的量子电路 (多分类调整) ===
logging.info("\n--- 阶段五：定义和构建QNN的量子电路 (多分类) ---")
num_qnn_qubits = num_qnn_input_features 

feature_map = ZZFeatureMap(feature_dimension=num_qnn_qubits, reps=2, entanglement='linear') 
feature_map.name = "ZZFeatureMap_MultiClass"
# ... (特征映射绘图代码不变，文件名可加MultiClass) ...

ansatz = RealAmplitudes(num_qnn_qubits, reps=4, entanglement='full') 
ansatz.name = "RealAmplitudes_MultiClass"
# ... (Ansatz绘图代码不变，文件名可加MultiClass) ...

qnn_hybrid_circuit = QuantumCircuit(num_qnn_qubits)
qnn_hybrid_circuit.compose(feature_map, inplace=True)
qnn_hybrid_circuit.compose(ansatz, inplace=True)
qnn_hybrid_circuit.name = "HybridQNN_Circuit_MultiClass"

# --- QNN可观测量和分类器修改为多分类 ---
# 方法1: 每个类别一个可观测量 (One-vs-Rest style output)
# QNN的输出将是一个N维向量，每个维度对应一个类别的“分数”
# 量子比特数至少应等于类别数，或者使用更复杂的编码
if num_qnn_qubits < num_unique_classes:
    logging.warning(f"警告: QNN量子比特数({num_qnn_qubits}) 小于类别数({num_unique_classes})。 "
                    f"为多分类设计的One-vs-Rest可观测量可能不理想或无法完全实现。")
    # 可以考虑只取前 num_qnn_qubits 个类别，或者让一些类别共享可观测量（更复杂）
    # 作为简化，我们这里可能只为能覆盖的类别创建独立可观测量
    # 或者，如果用 EstimatorQNN 的默认行为，它可能将单可观测量输出通过后续网络层处理

# 简单策略：为每个类别定义一个在不同量子比特上的Z可观测量 (如果qubits >= classes)
# 或者，更通用的做法是让EstimatorQNN输出一个期望值向量，然后NeuralNetworkClassifier处理
# 对于 EstimatorQNN 进行多分类，一种常见方法是输出每个类别的期望值。
# 我们需要 num_unique_classes 个可观测量。
observables_list = []
for i in range(num_unique_classes):
    # 构建一个对角算子，其中一个元素是Z，其他是I，或者更复杂的设计
    # 简单示例: 如果量子比特足够，Z在第i个qubit上 (需要num_qnn_qubits >= num_unique_classes)
    if i < num_qnn_qubits:
        pauli_string = ['I'] * num_qnn_qubits
        pauli_string[i] = 'Z'
        observables_list.append(SparsePauliOp("".join(pauli_string)))
    else: 
        # 如果类别数 > 量子比特数，需要更复杂的策略，或重复使用。
        # 这里简单重复第一个，但这通常不是最优的，表明需要更多qubits或不同策略。
        logging.warning(f"类别 {i} 没有唯一的量子比特用于可观测量，将复用第一个量子比特的Z算符。")
        observables_list.append(SparsePauliOp("Z" + "I" * (num_qnn_qubits - 1)))

qnn_estimator = EstimatorQNN(
    circuit=qnn_hybrid_circuit.decompose(), 
    observables=observables_list, # <<< 使用可观测量列表
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    # input_gradients=False # 如果优化器不需要梯度
)
logging.info(f"EstimatorQNN 已定义，使用 {len(observables_list)} 个可观测量。")

initial_point_qnn = None 
logging.info("QNN将使用随机初始点。")

# NeuralNetworkClassifier for multi-class
# Qiskit ML >= 0.6.0 可以直接使用 loss='cross_entropy'
# 标签 y_qnn_train 应该是 0, 1, ..., N-1 的整数
qnn_classifier = NeuralNetworkClassifier(
    neural_network=qnn_estimator, # neural_network 参数接收QNN
    optimizer=COBYLA(maxiter=300), # 根据需要调整迭代次数
    callback=simple_text_callback, 
    initial_point=initial_point_qnn,
    loss='cross_entropy', # 适用于多分类的交叉熵损失
    output_shape=num_unique_classes # 明确指定输出形状为类别数
)
logging.info("NeuralNetworkClassifier (多分类) 已定义。")


# === 阶段六：使用真实人脸特征训练和评估QNN (多分类) ===
logging.info("\n--- 阶段六：使用真实人脸特征训练和评估QNN (多分类) ---")
logging.info(f"QNN训练数据样本: X={X_qnn_train.shape}, y={y_qnn_train.shape}")
logging.info(f"QNN训练标签示例 (整数 0 to N-1, 前20个): {y_qnn_train[:20]}")
logging.info('开始QNN模型拟合 (在真实人脸特征上)...')
objective_func_vals.clear() 

qnn_classifier.fit(X_qnn_train, y_qnn_train) # y_qnn_train 是 0,1,2... 标签
logging.info('QNN模型拟合完成。')

# ... (绘制QNN目标函数曲线的代码不变, 文件名可加MultiClass) ...
if objective_func_vals:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(objective_func_vals) + 1), objective_func_vals, marker='o', linestyle='-')
    plt.title('QNN (多分类) 训练过程中的目标函数值')
    plt.xlabel('迭代次数'); plt.ylabel('目标函数值'); plt.grid(True)
    plt.savefig(os.path.join(output_image_folder, "06_QNN_多分类_训练目标函数值曲线.png"))
    plt.close()

accuracy_qnn_train_real = np.round(100 * qnn_classifier.score(X_qnn_train, y_qnn_train), 2)
logging.info(f"QNN在多分类训练集上的准确率: {accuracy_qnn_train_real}%")

y_qnn_predict_real = qnn_classifier.predict(X_qnn_test) # 输出将是 0, 1, ... N-1
accuracy_qnn_test_real = np.round(100 * qnn_classifier.score(X_qnn_test, y_qnn_test), 2)
logging.info(f"QNN在多分类测试集上的准确率: {accuracy_qnn_test_real}%")

# 打印部分预测结果 (多分类)
logging.info("\nQNN在多分类测试集上的部分预测结果:")
num_samples_to_show_qnn_pred = min(10, len(X_qnn_test))

# 构建从标签ID到人物名称的列表 (顺序必须与CSV生成时一致)
# 这个 actual_person_names_list 需要在脚本开头根据 label_map_filepath 构建
# 为简单起见，我们假设 prepare_lfw_multiclass_data.py 中的 label_map
# 是 {'PersonA':0, 'PersonB':1, 'PersonC':2}
# 那么 actual_person_names_list 就应该是 ['PersonA', 'PersonB', 'PersonC']

# 尝试从 lfw_label_map.json 加载人物名称列表
sorted_person_names_by_id = ["未知类别"] * num_unique_classes # 初始化
if actual_person_names_map:
    # actual_person_names_map 是 {'Name': ID}，我们需要反向或排序
    # 按ID排序
    try:
        # sorted_map = sorted(actual_person_names_map.items(), key=lambda item: item[1])
        # sorted_person_names_by_id = [name for name, id_val in sorted_map]
        
        # 更直接的方式：创建一个长度为num_unique_classes的列表，用名字填充
        temp_names_list = [""] * num_unique_classes
        for name, id_val in actual_person_names_map.items():
            if 0 <= id_val < num_unique_classes:
                temp_names_list[id_val] = name
            else:
                logging.warning(f"标签映射中的ID {id_val} 超出范围 {num_unique_classes}")
        if all(temp_names_list): # 确保所有位置都被填充
             sorted_person_names_by_id = temp_names_list
        else:
            logging.warning("未能从label_map.json完全构建人物名称列表，将使用类别ID。")

    except Exception as e_map_sort:
        logging.warning(f"从label_map.json构建人物名称列表时出错: {e_map_sort}，将使用类别ID。")


logging.info(f"真实标签 vs 预测标签 (类别索引 0-{num_unique_classes-1})")
for i_pred_final in range(num_samples_to_show_qnn_pred):
    true_label_index = y_qnn_test[i_pred_final] # 这是整数标签 0, 1, ...
    pred_label_index = y_qnn_predict_real[i_pred_final] # 这也是整数标签 0, 1, ...
    
    true_l_display = sorted_person_names_by_id[true_label_index] if 0 <= true_label_index < len(sorted_person_names_by_id) else f"ID {true_label_index}"
    pred_l_display = sorted_person_names_by_id[pred_label_index] if 0 <= pred_label_index < len(sorted_person_names_by_id) else f"ID {pred_label_index}"
    
    logging.info(f"样本 {i_pred_final}: 真实类别 = {true_l_display}, QNN预测 = {pred_l_display}")

logging.info(f"\n所有分析和图像保存已完成。请检查 '{output_image_folder}' 文件夹。")