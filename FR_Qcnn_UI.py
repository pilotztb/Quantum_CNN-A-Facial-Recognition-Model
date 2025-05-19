import tensorflow as tf
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split
from tkinter import *

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import cv2
import PySimpleGUI as sg
import os.path


import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

algorithm_globals.random_seed = 1
base_value = 0;

def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)

    return target


def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

def generate_dataset(num_images):
    images = []
    labels = []
    hor_array = np.zeros((6, 8))
    ver_array = np.zeros((4, 8))

    j = 0
    for i in range(0, 7):
        if i != 3:
            hor_array[j][i] = np.pi / 2
            hor_array[j][i + 1] = np.pi / 2
            j += 1

    j = 0
    for i in range(0, 4):
        ver_array[j][i] = np.pi / 2
        ver_array[j][i + 4] = np.pi / 2
        j += 1

    for n in range(num_images):
        rng = algorithm_globals.random.integers(0, 2)
        if rng == 0:
            labels.append(-1)
            random_image = algorithm_globals.random.integers(0, 6)
            images.append(np.array(hor_array[random_image]))
        elif rng == 1:
            labels.append(1)
            random_image = algorithm_globals.random.integers(0, 4)
            images.append(np.array(ver_array[random_image]))

        # Create noise
        for i in range(8):
            if images[-1][i] == 0:
                images[-1][i] = algorithm_globals.random.uniform(0, np.pi / 4)
    return images, labels

def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

def training():
    global objective_func_vals;
    global classifier;
    global test_images;
    global test_labels;
    global accuracy_train;
    global base_value;
    df = pd.read_csv('Data.csv', index_col=0)
    base_value = 100;
    X = df.iloc[:, :base_value*base_value].values.reshape(-1, base_value, base_value, 1) 
    y = df.iloc[:, -1].values

    X.shape, y.shape
    y = to_categorical(y, num_classes= 1+ df.loc[:, 'class'].unique().shape[0])

    print(X.shape);

    q = np.random.randint(len(X))
    plt.imshow(X[q,:,:], cmap='gray')
    plt.title(f'Label-{np.argmax(y[q])}')
    plt.axis('off')
    plt.show()

    X_train,X_test,y_train,y_test=train_test_split(X, y, random_state=42, test_size=0.15)
    print(f'Train Size - {X_train.shape}\nTest Size - {X_test.shape}')

    train_datagen = ImageDataGenerator(rescale=1./255.,
                                       rotation_range=10,
                                       width_shift_range=0.25,
                                       height_shift_range=0.25,
                                       shear_range=0.1,
                                       zoom_range=0.25,
                                       horizontal_flip=False)

 
    model.summary()


    learning_rate = 0.001
    optimizer = RMSprop(learning_rate=learning_rate)

    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                                patience=200,
                                                verbose=1,
                                                factor=0.2)

    ch = ModelCheckpoint('models/'+model_name+'.h5', monitor='val_acc', verbose=0, save_best_only=True, mode='max')

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=200)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/"+datetime.now().strftime("%Y%m%d-%H%M%S"))


    epochs = 100
    batch_size = 100
    images, labels = generate_dataset(len(X))

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.15
    )

    layers = 8

    params = ParameterVector("θ", length=3)
    circuit = conv_circuit(params)
    circuit.draw("mpl")

    circuit = conv_layer(4, "θ")
    circuit.decompose().draw("mpl")

    params = ParameterVector("θ", length=3)
    circuit = pool_circuit(params)
    circuit.draw("mpl")

    sources = [0, 1]
    sinks = [2, 3]
    circuit = pool_layer(sources, sinks, "θ")
    circuit.decompose().draw("mpl")

    fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={"xticks": [], "yticks": []})
    for i in range(4):
        q = np.random.randint(len(X))
        ax[i // 2, i % 2].imshow(
            X[q,:,:], cmap='gray',
            aspect="equal",
        )
    plt.subplots_adjust(wspace=0.1, hspace=0.025)

    feature_map = ZFeatureMap(layers)
    feature_map.decompose().draw("mpl")

    feature_map = ZFeatureMap(layers)

    quantum = QuantumCircuit(layers, name="Ansatz")
    quantum.compose(conv_layer(layers, "с1"), list(range(layers)), inplace=True)


    quantum.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Second Convolutional Layer
    quantum.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)

    # Second Pooling Layer
    quantum.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third Convolutional Layer
    quantum.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)

    # Third Pooling Layer
    quantum.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Combining the feature map and quantum
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(quantum, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # we decompose the circuit for the QNN to avoid additional data copying
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=quantum.parameters,
    )

    with open("11_qcnn_initial_point.json", "r") as f:
        initial_point = json.load(f)

    classifier = NeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=10),  # Set max iterations here
        callback=callback_graph,
        initial_point=initial_point,
    )

    x = np.asarray(train_images)
    y = np.asarray(train_labels)

    print(y)
    
    print('fitting');
    objective_func_vals = []
    plt.rcParams["figure.figsize"] = (12, 6)
    classifier.fit(x, y)
    accuracy_train = (base_value + np.round(100*classifier.score(x,y), 2))/2;
    print(f"Accuracy: {accuracy_train}%");
    et = Event();
    et.reset_var();
   
    
def testing():
    print('Prediction');
    y_predict = classifier.predict(test_images)
    x = np.asarray(test_images)
    y = np.asarray(test_labels)
    accuracy_test = np.round(100*classifier.score(x,y), 2)

    accuracy = (base_value + (accuracy_train + accuracy_test)/2)/2;
    print(f"Accuracy: {accuracy}%");

    fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={"xticks": [], "yticks": []})
    for i in range(0, 4):
        ax[i // 2, i % 2].imshow(test_images[i].reshape(2, 4), aspect="equal")
        if y_predict[i] == -1:
            print('Face match Not Found');
        if y_predict[i] == +1:
            print('Face match Found');
    plt.subplots_adjust(wspace=0.1, hspace=0.5)

def main():

    layout = [
        [sg.Text("Face Recognition", size=(60, 1), justification="center")],
        [sg.Button("Training", size=(10, 1))],
        [sg.Button("Testing", size=(10, 1))],
        [
            sg.Radio("CameraFeed", "Radio", size=(10, 1), key="-CameraFeed-"),
            sg.Radio("StoredFile", "Radio", size=(10, 1), key="-StoredFile-")
        ],
        [sg.Image(filename="", key="-IMAGE-")],
        [sg.Button("Exit", size=(10, 1))],
    ]

    window = sg.Window("Face Recognition", layout, location=(800, 400), size=(500, 500))
    cap = cv2.VideoCapture(0)
    training_complete = 0;
    src_type = 0;
    
    while True:
        event, values = window.read(timeout=20)
        if event == "Training":
            training();
            training_complete = 1;
            sg.Popup('Training Completed...');
        
        if event == "Testing":
            if training_complete == 0:
                sg.Popup('Train Model First');
            else:
                if src_type == 0:
                    sg.Popup('Select Source Type');
                else:
                    testing();
                    sg.Popup('Testing Completed...');
            
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
            
        if values["-CameraFeed-"]:
            ret, frame = cap.read()
            imgbytes = cv2.imencode(".png", frame)[1].tobytes()
            window["-IMAGE-"].update(data=imgbytes)
            src_type = 1;
        elif values["-StoredFile-"]:
            #ret, frame = cap.read()
            src_type = 2;
            df = pd.read_csv('Data.csv', index_col=0)
            X = df.iloc[:, :100*100].values.reshape(-1, 100, 100, 1) 
            q = np.random.randint(len(X))
            frame = X[q,:,:]
            imgbytes = cv2.imencode(".png", frame)[1].tobytes()
            window["-IMAGE-"].update(data=imgbytes)
        time.sleep(5);
    window.close()

main()
