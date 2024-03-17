import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#NumPy 提供了广泛的数学函数和运算，并针对性能进行了优化。 这些运算包括算术运算、线性代数函数、傅立叶变换、随机数生成
#Matplotlib 提供了广泛的函数和类，用于创建各种类型的绘图和可视化。 其中包括线图、散点图、条形图、直方图、等高线图、3D 图
#Pandas 是一个流行的开源 Python 库，用于数据操作和分析。 它提供了易于使用的数据结构和函数来处理结构化数据，例如表格数据、时间序列等。 
#Pandas 广泛应用于数据科学、机器学习、金融等领域，用于数据清理、准备、探索和分析任务。

# load train data
df_train = pd.read_csv('./trainALL 0.2 1173 - shuffleYYT.csv')

X = df_train.drop('Label', axis=1)
y = df_train['Label']

print(f"X is :\n{X}\n")
print(f"y is :\n{y}\n")


############################################################
# prepare data for ML, by splitting them in a train and validation dataset

from sklearn.model_selection import train_test_split
#sklearn.model_selection 是 scikit-learn（Python 中流行的机器学习库）中的一个模块，
#提供用于分割数据集、交叉验证、参数调整和模型评估的实用函数。

X_train, X_validation, y_train, y_validation = train_test_split(X, y, 
                                                                test_size=0.1, 
                                                                random_state=42,
                                                                stratify=y)

print("-----[define function] apply_preprocessing(X, y)")
def apply_preprocessing(X, y):

    # scale the features
    print(np.max(X))
    X_scaled = np.array(X/255)
    print(f"X_scaled is :\n{X_scaled}\n")
    X_tensor = X_scaled.reshape(len(X_scaled), 50, 50, 1)
    #print(f"X_tensor is :\n{X_tensor}\n")
    
    # apply one-hot encoding to labels
    y_onehot = pd.get_dummies(y)
    print(f"y_onehot is :\n{y_onehot}\n")
    
    return X_tensor, y_onehot


X_train_tensor, y_train_onehot = apply_preprocessing(X_train, y_train)
X_validation_tensor, y_validation_onehot = apply_preprocessing(X_validation, y_validation)


#################################################
# show few digits

#for i in range(9):
#    plt.subplot(3, 3, i+1)
#    plt.imshow(X_train_tensor[i, :])
#plt.show()


for i in range(21):
    plt.subplot(3, 7, i+1)
    plt.imshow(X_train_tensor[i, :])
plt.show()


####################################################
# show the distribution of labels

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
for i in range(10):
    plt.hist(y_train[y_train==i], width=0.5)
plt.xlim(-0.5, 10)
plt.title('Training dataset')
plt.ylabel('Frequency')
plt.xlabel('Label')

plt.subplot(1, 2, 2)
for i in range(10):
    plt.hist(y_validation[y_validation==i], width=0.5)
plt.xlim(-0.5, 10)
plt.title('Validation dataset')
plt.ylabel('Frequency')
plt.xlabel('Label')

plt.tight_layout()
plt.show()


#2. Create ML model
#We create in this section a Convolutional Neural Network, inspired by LeNet5

#We create a CNN inspired by the old architecture called LeNet5, that consists of 2 convolutional layers and 3 fully connected layers.

#We provide a scaling_factor to increase the dimension of the Neural Network and we also use Dropout and EarlyStopping to avoid overfitting.
"""
Dense：此类代表神经网络中的全连接层。
Conv2D：此类表示神经网络中的 2D 卷积层。
Flatten：此类表示将输入数据从多维数组展平为一维数组的层。
AveragePooling2D：此类表示 2D 平均池化层，它使用平均值沿其空间维度对输入进行下采样。
Dropout：此类代表一个 dropout 层，它在训练期间随机将一部分输入单元设置为零，以防止过度拟合。

EarlyStopping：此类实现提前停止，当监控指标停止改善时停止训练。
ModelCheckpoint：如果监控数量有所改善，该类会在每个 epoch 后保存模型，使您可以在训练期间保存最佳模型。
顺序：此类表示神经网络模型中的线性层堆栈。 它是 Keras 中用于构建简单架构的最常见模型类型。
load_model：此函数加载使用 model.save() 保存的 Keras 模型。
"""

from keras.layers import Dense, Conv2D,Flatten, AveragePooling2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model

def get_model(scaling_factor):
    
    assert type(scaling_factor) == int, "The scaling factor must be an integer"
    
    # creation a sequential model
    model = Sequential()
    print("model1")
    print(model)

    # 1st convolutional layer
    # the dimension of the scanner is 5x5. With padding='same' the dimension of the image does not change
    model.add(Conv2D(filters=6*scaling_factor, 
                     kernel_size=(5, 5), 
                     padding='same', 
                     activation='relu', 
                     input_shape=(50, 50, 1)))
    # with average pooling 2D the dimension of the image changes from 28x28 to 14x14
    model.add(AveragePooling2D())
    print("model2")
    print(model)
    print("--------------------scaling_factor")
    print(scaling_factor)
    # 2nd convolutional layers 
    # the dimension of the scanner is 5x5. With padding='valid' and filter = 5x5,
    # the dimension of the image goes from 14x14 to 10x10
    model.add(Conv2D(filters=16*scaling_factor, 
                     kernel_size=(5,5), 
                     padding='valid', 
                     activation='relu'))
    # with average pooling 2D the dimension of the image changes from 10x10 to 5x5
    model.add(AveragePooling2D())
    
    # from 2D to 1D
    model.add(Flatten())

    # 3rd layer
    model.add(Dense(units=120*scaling_factor, 
                    activation='relu'))
    model.add(Dropout(0.55))

    # 4th layer
    model.add(Dense(units=84*scaling_factor, 
                    activation='relu'))
    model.add(Dropout(0.5))

    # 5th layer (output)
    model.add(Dense(units=3, 
                    activation = 'softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

"""
此代码片段定义了一个函数 fit_model，用于配置 Keras 中模型训练的回调。 让我们分解每个部分：
model：要训练的神经网络模型。
train_data：用于训练模型的训练数据。
validation_data：用于在训练期间监控模型性能的验证数据。
耐心：没有改善的时期数，如果使用早期停止，则训练将停止。
epochs：训练模型的纪元数。
batch_size：训练期间使用的小批量的大小。
validation_split：如果未提供validation_data，则用作验证数据的训练数据的分数。


Early_stopping_monitor = EarlyStopping(patience=patience, monitor='val_accuracy')：此行创建一个 EarlyStopping 回调。 
早期停止是一种用于监控模型在验证数据集上的性能并在性能停止改善时停止训练以避免过度拟合的技术。


回调：
ModelCheckpoint 在训练期间保存模型的权重，以便您稍后恢复最佳模型权重。
filepath='best_model.hdf5'：保存最佳模型权重的文件路径。
save_weights_only=False：是否仅保存权重或整个模型。 在这里，它保存了整个模型。
Monitor='val_accuracy'：要监控改进的指标。 这是验证准确性。
mode='auto'：决定最佳模型的模式。 它根据监控的指标（在本例中为准确性）自动在“最大”和“最小”之间进行选择。
save_best_only=True：是否仅保存基于监控指标的最佳模型。
verbose=1：详细模式。 它打印有关保存模型的信息。
这些回调通常在模型训练期间使用，以监视模型的性能并保存最佳模型权重以供将来使用或部署。
"""

def fit_model(model, train_data, validation_data, patience, epochs, batch_size, validation_split):
    print(model)
    print(patience)
    print(epochs)
    print(batch_size)
    print(validation_split)
    # stop the training if the validation accuracy does not improve after a certain number of epochs
    early_stopping_monitor = EarlyStopping(patience=patience,
                                           monitor='val_accuracy')

    # save the best model
    model_checkpoint_callback = ModelCheckpoint(filepath='./PIC_best_model.hdf5',
                                                save_weights_only=False,
                                                monitor='val_accuracy',
                                                mode='auto',
                                                save_best_only=True,
                                                verbose=1)

    print(model.summary())

    X_train, y_train = train_data
    X_validation, y_validation = validation_data
    
    # save the history of the training
    history = model.fit(X_train, y_train,
                        validation_data=(X_validation, y_validation),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[early_stopping_monitor,model_checkpoint_callback])
    
    return model, history
###############################################
#3. Train the model

model_CNN = get_model(scaling_factor=4)

trained_model, history = fit_model(model = model_CNN,
                                   train_data = (X_train_tensor, y_train_onehot), 
                                   validation_data = (X_validation_tensor, y_validation_onehot),
                                   patience=10,#20,#3, #patience=10, 
                                   epochs=200,#100,#3,   #10,   #100, 
                                   batch_size=256, 
                                   validation_split=0.1)



#################################################
#4. Inspect the results and make predictions
#History of training

epochs_grid = np.arange(1, len(history.history['accuracy'])+1)

plt.plot(epochs_grid, history.history['accuracy'], label='training')
plt.plot(epochs_grid, history.history['val_accuracy'], label='validation')
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""
trained_model：用于进行预测的经过训练的神经网络模型。
X_test：包含需要进行预测的输入图像的验证数据集。
"""
#Compute predictions using the validation dataset
def make_predictions(trained_model, X_test):

    X_test = np.array(X_test)/255
    X_test_scaled = X_test.reshape(len(X_test), 50, 50)
    X_test_tensor = X_test.reshape(len(X_test_scaled), 50, 50, 1)
    
    y_pred = np.argmax(trained_model.predict(X_test_tensor), axis=1)
    
    #此行创建一个 DataFrame 来存储预测。 它使用从 1 到验证数据集长度的一系列图像 ID 来初始化 DataFrame。
    df_submission = pd.DataFrame(np.arange(1, len(X_test)+1))
    df_submission['predictions'] = y_pred
    df_submission.columns=['ImageId', 'Label']
    
    return df_submission

# load the best model
best_model = load_model('./PIC_best_model.hdf5')

#此行调用前面定义的 make_predictions 函数，以使用加载的模型 (best_model) 计算验证数据集的预测。 预测标签存储在 y_validation_pred 中。
y_validation_pred = make_predictions(best_model, X_validation)
#此行从 DataFrame y_validation_pred 中提取预测标签并将其转换为 NumPy 数组。
y_validation_pred = np.array(y_validation_pred['Label'])
#此行通过将预测标签 (y_validation_pred) 与验证数据集的真实标签 (y_validation) 进行比较来计算验证分数。
validation_score = np.mean(y_validation_pred == y_validation)

#此行打印最佳模型的验证分数。
print('此行打印最佳模型的验证分数')
print('The validation score of the best model is', validation_score)

#fusion_matrix 函数用于计算混淆矩阵，这是分类模型的性能评估指标。 
#该表通过显示真阳性、真阴性、假阳性和假阴性预测的计数来总结分类算法的性能。 
#该矩阵有助于评估分类模型的准确性并识别模型可能对实例进行错误分类的区域。
from sklearn.metrics import confusion_matrix
"""
Seaborn 是一个基于 Matplotlib 的 Python 数据可视化库，它提供了一个高级接口来绘制有吸引力且信息丰富的统计图形。 
与单独使用 Matplotlib 相比，它通常用于增强绘图的视觉外观并创建更复杂且更具视觉吸引力的可视化效果。 
Seaborn 提供了附加功能，用于设计绘图、创建复杂的多面板可视化以及在绘图中合并统计估计。
"""
import seaborn as sns

#创建混淆矩阵数据框：
df_confusion_matrix = pd.DataFrame(confusion_matrix(y_validation, y_validation_pred))

#此行使用 Seaborn 的热图函数创建混淆矩阵的热图可视化。
"""
df_confusion_matrix：包含混淆矩阵的 DataFrame。
annot=True：表示应在热图上注释（显示）单元格值。
cmap="OrRd"：指定用于热图的颜色图（colormap）。 这里，“OrRd”代表橙红色颜色图。
fmt='g'：指定单元格注释的格式。 'g' 表示值显示为整数。
cbar=False：从热图中删除颜色条。
plt.title('混淆矩阵')：将绘图的标题设置为“混淆矩阵”。
plt.yticks(rotation=0)：将 y 轴刻度标签旋转为水平（0 度旋转）。
plt.show()：显示热图。
"""

print('confusion_matrix show >>>')
sns.heatmap(df_confusion_matrix, annot=True, cmap="OrRd", fmt='g', cbar=False)
plt.title('Confusion matrix')
plt.yticks(rotation=0)
plt.show()


# show few digits of the validation dataset

#for i in range(9):
#    plt.subplot(3, 3, i+1)
#    plt.imshow(X_validation_tensor[i, :])
#    plt.title('Predicted '+str(y_validation_pred[i]))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(X_validation_tensor[i, :])
    plt.title('Predicted '+str(y_validation_pred[i]))

plt.tight_layout()
plt.show()

#Submit predictions
print('Submit predictions 预测 >>>')
df_submission_example = pd.read_csv('sample_submission.csv')
df_submission_example.head()

# test data (there are no labels in the test dataset)
print('read test data >>>')
X_test = pd.read_csv('./testBeta3 for machine.csv')

#此行调用之前定义的 make_predictions 函数，传递最佳训练模型（best_model）和测试数据集（X_test）。 
#它使用经过训练的模型生成测试数据集的预测。
print('使用经过训练的模型生成测试数据集的预测')
df_submission = make_predictions(best_model, X_test)
#此行将 DataFrame df_submission 中生成的预测保存到名为“submission.csv”的 CSV 文件中。 
#index=False 参数表示 DataFrame 索引不应包含在 CSV 文件中。

print('生成的预测')
df_submission.to_csv('PIC-3-submission.csv', index=False)
#此行返回 DataFrame df_submission，其中包含对测试数据集所做的预测。 如果需要，该数据帧可用于进一步分析或检查。
#总体而言，此代码段使用经过训练的模型准备对测试数据集的预测，并将其保存为机器学习竞赛所需的提交格式。 它确保根据竞赛指南提交的预测格式正确。
print('返回 >生成的预测')
df_submission

