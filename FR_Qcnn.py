# FR_Qcnn.py (整合版 - 方案C: CNN直接输出低维特征 + 日志记录)

# 1. 导入os 和 设置Matplotlib后端 (放在脚本最顶部)
import matplotlib
matplotlib.use('Agg') # 必须在导入pyplot之前设置
import matplotlib.pyplot as plt
import os
import sys # 用于sys.exit
import logging # <<< 新增导入
from datetime import datetime # <<< datetime 已在下面导入，确保在日志设置前导入

# --- 日志记录设置 ---
log_folder = "log"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
    # print(f"文件夹 '{log_folder}' 已创建。") # 用logging替代

log_filename_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filepath = os.path.join(log_folder, f"fr_qcnn_run_{log_filename_timestamp}.log")

# 配置logging
logging.basicConfig(
    level=logging.INFO, # 设置日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath, encoding='utf-8'), # 输出到文件
        logging.StreamHandler(sys.stdout) # 同时输出到控制台
    ]
)
# --- 日志记录设置结束 ---


import tensorflow as tf
import json
from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
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
# from datetime import datetime # 已移到日志设置之前

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

algorithm_globals.random_seed = 1

num_qnn_input_features = 8
logging.info(f"QNN将使用 {num_qnn_input_features} 个输入特征 (量子比特)。") # <<< print 改为 logging.info

output_image_folder = "所有输出图像"
if not os.path.exists(output_image_folder):
    os.makedirs(output_image_folder)
    logging.info(f"文件夹 '{output_image_folder}' 已创建。") # <<< print 改为 logging.info

objective_func_vals = [] 
def simple_text_callback(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)
    logging.info(f"迭代次数: {len(objective_func_vals)}, 当前目标函数值: {obj_func_eval:.4f}") # <<< print 改为 logging.info

logging.info("--- 阶段一：加载真实人脸数据集 ---")
csv_filename = 'lfw_subset_data.csv'
try:
    df = pd.read_csv(csv_filename) 
    logging.info(f"成功从 '{csv_filename}' 加载数据，共 {len(df)} 条记录。") # <<<
except FileNotFoundError:
    logging.error(f"错误: '{csv_filename}' 文件未找到。请确保您已正确生成此文件。") # <<< print 改为 logging.error
    sys.exit()

X_from_csv = df.iloc[:, :-1].values.reshape(-1, 100, 100, 1) 
y_labels_from_csv = df['class'].values 
num_unique_classes = len(np.unique(y_labels_from_csv))
logging.info(f"从CSV加载的类别数量: {num_unique_classes}") # <<<
if num_unique_classes < 2:
    logging.error(f"错误：数据集中类别少于2 ({num_unique_classes})，无法进行分类任务。请检查 '{csv_filename}'。") # <<<
    sys.exit()
y_one_hot_from_csv = to_categorical(y_labels_from_csv, num_classes=num_unique_classes)
logging.info(f"加载的数据形状: X={X_from_csv.shape}, y_one_hot={y_one_hot_from_csv.shape}") # <<<

plt.figure(figsize=(6,6))
q_csv_idx = np.random.randint(len(X_from_csv))
plt.imshow(X_from_csv[q_csv_idx,:,:,0], cmap='gray')
plt.title(f'来自CSV的随机图像 - 标签: {y_labels_from_csv[q_csv_idx]} (One-hot: {y_one_hot_from_csv[q_csv_idx]})')
plt.axis('off')
plt.savefig(os.path.join(output_image_folder, "01_CSV单张随机图像样本.png"))
plt.close()

X_train_csv, X_test_csv, y_train_csv, y_test_csv = train_test_split(
    X_from_csv, y_one_hot_from_csv, 
    test_size=0.2, 
    random_state=algorithm_globals.random_seed, 
    stratify=y_one_hot_from_csv
)
logging.info(f'用于经典CNN的训练集尺寸 - {X_train_csv.shape}, {y_train_csv.shape}') # <<<
logging.info(f'用于经典CNN的测试集尺寸 - {X_test_csv.shape}, {y_test_csv.shape}') # <<<

logging.info("\n--- 阶段二：训练经典CNN (新结构) 并提取特征 ---") # <<<
train_datagen = ImageDataGenerator(rescale=1./255., rotation_range=10, width_shift_range=0.1, 
                                   height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255.)

model_name = 'FaceRec_CNN_DirectLowDim_'+datetime.now().strftime("%Y%m%d_%H%M%S") # datetime已在日志设置前导入
model = Sequential(name = model_name)
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
model.add(Dense(num_unique_classes, activation='softmax', name='cnn_output_layer')) 

# 重定向 model.summary() 的输出到日志
original_stdout = sys.stdout # 保存原始stdout
summary_log_path = os.path.join(log_folder, f"model_summary_{model_name}.txt")
with open(summary_log_path, 'w', encoding='utf-8') as f_summary:
    sys.stdout = f_summary # 重定向stdout到文件
    model.summary()
sys.stdout = original_stdout # 恢复原始stdout
logging.info(f"Keras模型摘要已保存到: {summary_log_path}")
# 同时也在日志中打印（对于较短的摘要）
# model.summary(print_fn=lambda x: logging.info(x)) # 这种方式更优雅

learning_rate_cnn = 0.001
optimizer_cnn = RMSprop(learning_rate=learning_rate_cnn)
model.compile(loss='categorical_crossentropy', optimizer=optimizer_cnn, metrics=['accuracy'])

models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
checkpoint_path_cnn = os.path.join(models_dir, model_name + "_cnn_best.keras")

ch_cnn = ModelCheckpoint(checkpoint_path_cnn, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
es_cnn = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15) 
learning_rate_reduction_cnn = ReduceLROnPlateau(monitor='val_loss', patience=7, verbose=1, factor=0.2)
callbacks_list_cnn = [ch_cnn, es_cnn, learning_rate_reduction_cnn]

logging.info("开始训练经典CNN模型 (新结构，用于直接提取低维特征)...") # <<<
epochs_cnn = 50 
batch_size_cnn = 32 
if len(X_train_csv) < batch_size_cnn :
    batch_size_cnn = max(1, len(X_train_csv) // 4) 
    logging.warning(f"警告：训练样本数 ({len(X_train_csv)}) 小于batch_size，已将batch_size调整为 {batch_size_cnn}") # <<<

history_cnn = model.fit(
    train_datagen.flow(X_train_csv, y_train_csv, batch_size=batch_size_cnn),
    epochs=epochs_cnn,
    validation_data=valid_datagen.flow(X_test_csv, y_test_csv),
    callbacks=callbacks_list_cnn,
    verbose=1 # Keras的verbose会打印到stdout，我们的logging已配置包含stdout
)
logging.info("经典CNN模型训练完成。") # <<<

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'], label='训练准确率')
if 'val_accuracy' in history_cnn.history:
    plt.plot(history_cnn.history['val_accuracy'], label='验证准确率')
plt.title('经典CNN(新结构) 准确率曲线')
plt.xlabel('Epoch'); plt.ylabel('准确率'); plt.legend(); plt.grid(True)
plt.savefig(os.path.join(output_image_folder, "02a_经典CNN(新结构)训练历史_准确率.png"))
plt.close()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 2) 
plt.plot(history_cnn.history['loss'], label='训练损失')
if 'val_loss' in history_cnn.history:
    plt.plot(history_cnn.history['val_loss'], label='验证损失')
plt.title('经典CNN(新结构) 损失曲线')
plt.xlabel('Epoch'); plt.ylabel('损失'); plt.legend(); plt.grid(True)
plt.savefig(os.path.join(output_image_folder, "02b_经典CNN(新结构)训练历史_损失.png"))
plt.close()

logging.info(f"从 {checkpoint_path_cnn} 加载效果最好的经典CNN模型 (包括结构和权重)...") # <<<
try:
    model = tf.keras.models.load_model(checkpoint_path_cnn) 
    logging.info("模型加载成功。") # <<<
except Exception as e:
    logging.error(f"加载完整模型失败: {e}.") # <<<
    sys.exit("无法继续。")

feature_extractor_layer_name = 'dense_for_qnn_features' 
logging.info(f"尝试从已加载模型 '{model.name}' 创建特征提取器，目标层: '{feature_extractor_layer_name}'") # <<<
if model is None: 
    logging.error("错误: 已加载的CNN模型 (model) 为空。") # <<<
    sys.exit("错误: 已加载的CNN模型 (model) 为空。")
new_input = tf.keras.Input(shape=(100, 100, 1), name="new_feature_extractor_input")
x = new_input
found_target_layer = False
target_layer_output = None
for layer in model.layers:
    try:
        x = layer(x) 
        if layer.name == feature_extractor_layer_name:
            target_layer_output = x 
            found_target_layer = True
            logging.info(f"已连接到目标特征提取层 '{layer.name}'") # <<<
            break 
    except Exception as e_layer_conn:
        logging.error(f"连接层 '{layer.name}' 时出错: {e_layer_conn}") # <<<
        sys.exit("特征提取器构建失败。")
if not found_target_layer or target_layer_output is None:
    logging.error(f"错误: 未能在加载的模型中找到或连接到名为 '{feature_extractor_layer_name}' 的层。") # <<<
    sys.exit("无法继续，特征提取层配置错误。")
try:
    feature_extractor_model = tf.keras.Model(inputs=new_input, outputs=target_layer_output, name="FeatureExtractor_Reconstructed_LowDim")
    
    # 将特征提取器模型的摘要也保存到日志文件
    summary_fe_log_path = os.path.join(log_folder, f"feature_extractor_summary_{model_name}.txt")
    with open(summary_fe_log_path, 'w', encoding='utf-8') as f_fe_summary:
        sys.stdout = f_fe_summary
        feature_extractor_model.summary()
    sys.stdout = original_stdout # 恢复
    logging.info(f"特征提取器模型摘要已保存到: {summary_fe_log_path}")
    # logging.info(f"特征提取模型已通过手动重构路径方式创建，输出层为: '{feature_extractor_layer_name}' (输出维度: {target_layer_output.shape[-1]})")
    # 使用下面的方式打印维度，更安全
    if hasattr(target_layer_output, 'shape'):
        logging.info(f"特征提取模型已通过手动重构路径方式创建，输出层为: '{feature_extractor_layer_name}' (输出维度: {target_layer_output.shape[-1]})")
    else:
        logging.info(f"特征提取模型已通过手动重构路径方式创建，输出层为: '{feature_extractor_layer_name}' (输出维度未知)")


except Exception as e_model_create:
    logging.error(f"使用手动重构路径创建特征提取模型失败: {e_model_create}") # <<<
    sys.exit(f"使用手动重构路径创建特征提取模型失败: {e_model_create}")

logging.info("开始从人脸图像中提取低维深度特征...") # <<<
all_face_low_dim_features = feature_extractor_model.predict(X_from_csv / 255.0) 
logging.info(f"提取到的低维深度特征形状: {all_face_low_dim_features.shape}") # <<<

logging.info("\n--- 阶段三：对CNN输出的低维特征进行缩放 ---") # <<<
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_features_for_qnn = scaler.fit_transform(all_face_low_dim_features) 
logging.info(f"CNN直接输出并缩放后的特征形状: {scaled_features_for_qnn.shape}") # <<<

logging.info("\n--- 阶段四：准备QNN的标签并分割数据 ---") # <<<
y_qnn_target_labels = np.array([-1 if label == 0 else 1 for label in y_labels_from_csv])
logging.info(f"映射后的QNN目标标签示例 (前10个): {y_qnn_target_labels[:10]}") # <<<
X_qnn_train, X_qnn_test, y_qnn_train, y_qnn_test = train_test_split(
    scaled_features_for_qnn, 
    y_qnn_target_labels, 
    test_size=0.3, 
    random_state=algorithm_globals.random_seed,
    stratify=y_qnn_target_labels 
)
logging.info(f"QNN训练数据形状: X={X_qnn_train.shape}, y={y_qnn_train.shape}") # <<<
logging.info(f"QNN测试数据形状: X={X_qnn_test.shape}, y={y_qnn_test.shape}") # <<<

logging.info("\n--- 阶段五：定义和构建QNN的量子电路 ---") # <<<
num_qnn_qubits = num_qnn_input_features 

feature_map = ZZFeatureMap(feature_dimension=num_qnn_qubits, reps=2, entanglement='linear') 
feature_map.name = "ZZFeatureMap"
if num_qnn_qubits <= 10:
    try:
        # 对于 circuit.draw，如果 filename 参数无效，可以先绘制到 plt 对象再保存
        fig_fm, ax_fm = plt.subplots()
        feature_map.decompose().draw("mpl", ax=ax_fm, fold=-1)
        fig_fm.savefig(os.path.join(output_image_folder, f"03_QNN_{feature_map.name}({num_qnn_qubits}qubits).png"))
        plt.close(fig_fm)
        logging.info(f"QNN 特征映射 ({feature_map.name}) 已创建并保存图像。") # <<<
    except Exception as e_draw_fm:
        logging.error(f"绘制 FeatureMap 出错: {e_draw_fm}. 跳过绘图。") # <<<
else:
    logging.info(f"QNN 特征映射 ({feature_map.name}) 已创建 (量子比特数 {num_qnn_qubits} 较大，跳过分解绘图)。") # <<<

ansatz = RealAmplitudes(num_qnn_qubits, reps=4, entanglement='full') 
ansatz.name = "RealAmplitudes"
if num_qnn_qubits <= 10:
    try:
        fig_ans, ax_ans = plt.subplots()
        ansatz.decompose().draw("mpl", ax=ax_ans, fold=-1)
        fig_ans.savefig(os.path.join(output_image_folder, f"04_QNN_{ansatz.name}({num_qnn_qubits}qubits).png"))
        plt.close(fig_ans)
        logging.info(f"QNN Ansatz ({ansatz.name}) 已创建并保存图像。") # <<<
    except Exception as e_draw_ansatz:
        logging.error(f"绘制 Ansatz 出错: {e_draw_ansatz}. 跳过绘图。") # <<<
else:
    logging.info(f"QNN Ansatz ({ansatz.name}) 已创建 (量子比特数 {num_qnn_qubits} 较大，跳过分解绘图)。") # <<<

qnn_hybrid_circuit = QuantumCircuit(num_qnn_qubits)
qnn_hybrid_circuit.compose(feature_map, inplace=True)
qnn_hybrid_circuit.compose(ansatz, inplace=True)
qnn_hybrid_circuit.name = "HybridQNN_Circuit_DirectCNNFeatures"

observable = SparsePauliOp.from_list([("Z" + "I" * (num_qnn_qubits - 1), 1)])
qnn_estimator = EstimatorQNN(
    circuit=qnn_hybrid_circuit.decompose(), 
    observables=observable,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
)
logging.info("EstimatorQNN 已定义。") # <<<

initial_point_qnn = None 
logging.info("QNN将使用随机初始点 (不加载 '11_qcnn_initial_point.json')。") # <<<

qnn_classifier = NeuralNetworkClassifier(
    qnn_estimator,
    optimizer=COBYLA(maxiter=300), 
    callback=simple_text_callback, 
    initial_point=initial_point_qnn,
)
logging.info("NeuralNetworkClassifier 已定义。") # <<<

logging.info("\n--- 阶段六：使用真实人脸特征训练和评估QNN ---") # <<<
logging.info(f"QNN训练数据样本 (CNN直接输出低维特征并缩放后): X={X_qnn_train.shape}, y={y_qnn_train.shape}") # <<<
logging.info(f"QNN训练标签示例 (前10个): {y_qnn_train[:10]}") # <<<
logging.info('开始QNN模型拟合 (在真实人脸特征上)...') # <<<
objective_func_vals.clear() 

qnn_classifier.fit(X_qnn_train, y_qnn_train)
logging.info('QNN模型拟合完成。') # <<<

if objective_func_vals:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(objective_func_vals) + 1), objective_func_vals, marker='o', linestyle='-')
    plt.title('QNN (CNN低维特征) 训练过程中的目标函数值')
    plt.xlabel('迭代次数'); plt.ylabel('目标函数值'); plt.grid(True)
    plt.savefig(os.path.join(output_image_folder, "06_QNN_CNN低维特征_训练目标函数值曲线.png"))
    plt.close()
else:
    logging.info("未记录QNN目标函数值，无法绘制曲线。") # <<<

accuracy_qnn_train_real = np.round(100 * qnn_classifier.score(X_qnn_train, y_qnn_train), 2)
logging.info(f"QNN在CNN低维特征训练集上的准确率: {accuracy_qnn_train_real}%") # <<<

y_qnn_predict_real = qnn_classifier.predict(X_qnn_test)
accuracy_qnn_test_real = np.round(100 * qnn_classifier.score(X_qnn_test, y_qnn_test), 2)
logging.info(f"QNN在CNN低维特征测试集上的准确率: {accuracy_qnn_test_real}%") # <<<

logging.info("\nQNN在CNN低维特征测试集上的部分预测结果:") # <<<
num_samples_to_show_qnn_pred = min(10, len(X_qnn_test))
person_for_qnn_label_minus_1 = 'Colin Powell' 
person_for_qnn_label_plus_1 = 'George W Bush' 
logging.info(f"真实标签 (QNN使用-1/1) vs 预测标签 (QNN输出-1/1) -- [{person_for_qnn_label_minus_1}=-1, {person_for_qnn_label_plus_1}=1]") # <<<
for i_pred_final in range(num_samples_to_show_qnn_pred):
    true_l_qnn = y_qnn_test[i_pred_final]
    pred_l_qnn = y_qnn_predict_real[i_pred_final]
    true_l_display = person_for_qnn_label_plus_1 if true_l_qnn == 1 else person_for_qnn_label_minus_1
    pred_l_display = person_for_qnn_label_plus_1 if pred_l_qnn == 1 else person_for_qnn_label_minus_1
    logging.info(f"样本 {i_pred_final}: 真实类别 = {true_l_display}, QNN预测 = {pred_l_display}") # <<<

logging.info(f"\n所有分析和图像保存已完成。请检查 '{output_image_folder}' 文件夹。") # <<<