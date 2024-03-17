import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import time
import psutil


#print("TensorFlow v" + tf.__version__)
print("pandas v" + pd.__version__)

train=pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
test=pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")

print("train.head():")
print(train.head())
"""
  PassengerId HomePlanet CryoSleep  Cabin  Destination  ...  ShoppingMall     Spa  VRDeck               Name  Transported
0     0001_01     Europa     False  B/0/P  TRAPPIST-1e  ...           0.0     0.0     0.0    Maham Ofracculy        False
1     0002_01      Earth     False  F/0/S  TRAPPIST-1e  ...          25.0   549.0    44.0       Juanna Vines         True
2     0003_01     Europa     False  A/0/S  TRAPPIST-1e  ...           0.0  6715.0    49.0      Altark Susent        False
3     0003_02     Europa     False  A/0/S  TRAPPIST-1e  ...         371.0  3329.0   193.0       Solam Susent        False
4     0004_01      Earth     False  F/1/S  TRAPPIST-1e  ...         151.0   565.0     2.0  Willy Santantines         True
"""
print("train.info():")
print(train.info())
print("train.describe():")
print(train.describe())
"""
train.describe():
               Age   RoomService     FoodCourt  ShoppingMall           Spa        VRDeck
count  8514.000000   8512.000000   8510.000000   8485.000000   8510.000000   8505.000000
mean     28.827930    224.687617    458.077203    173.729169    311.138778    304.854791
std      14.489021    666.717663   1611.489240    604.696458   1136.705535   1145.717189
min       0.000000      0.000000      0.000000      0.000000      0.000000      0.000000
25%      19.000000      0.000000      0.000000      0.000000      0.000000      0.000000
50%      27.000000      0.000000      0.000000      0.000000      0.000000      0.000000
75%      38.000000     47.000000     76.000000     27.000000     59.000000     46.000000
max      79.000000  14327.000000  29813.000000  23492.000000  22408.000000  24133.000000
"""

print("train.isnull().sum()")
print(train.isnull().sum())

#-------------------
columns=['HomePlanet','CryoSleep','VIP','Destination']
for col in columns:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)
    print("train[col].mode()[0]")
    print(train[col].mode()[0])

train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)
print("train['Age'].median()")
print(train['Age'].median())

a=['RoomService','Spa','FoodCourt','ShoppingMall','VRDeck']
for i in a:
    train[i].fillna(train[i].mean(), inplace=True)
    test[i].fillna(test[i].mean(), inplace=True)
    print("train[i].mean()")
    print(train[i].mean())


print("train.isnull().sum() after fillNA")
print(train.isnull().sum())

columnTarget='Age'
# 绘制年龄的柱状图
plt.figure(figsize=(10, 6))  # 设置图形大小
hist = plt.hist(train[columnTarget], bins=20, color='skyblue', edgecolor='black')  # 绘制柱状图

# 在每个柱子上方显示数量
for i in range(len(hist[0])):
    plt.text(hist[1][i], hist[0][i], str(int(hist[0][i])), ha='center', va='bottom')

plt.title('Age Distribution')  # 设置标题
plt.xlabel(columnTarget)  # 设置 x 轴标签
plt.ylabel('Frequency')  # 设置 y 轴标签
plt.grid(True)  # 显示网格线
plt.show()  # 显示图形


pt=PowerTransformer(method='yeo-johnson', standardize=True)
#c=['RoomService','Spa','FoodCourt','ShoppingMall','VRDeck','Age','VIP','HomePlanet_Earth','HomePlanet_Europa', 'HomePlanet_Mars', 'Destination_55 Cancri e',
#    'Destination_PSO J318.5-22', 'Destination_TRAPPIST-1e']
c=['RoomService','Spa','FoodCourt','ShoppingMall','VRDeck','Age','VIP']
for i in c:
    train[i]=pt.fit_transform(train[i].values.reshape(-1,1))
    test[i]=pt.fit_transform(test[i].values.reshape(-1,1))

# change string to float
print("---------------change string to float---------------")
# 创建一个 LabelEncoder 对象
label_encoder = LabelEncoder()
# 对字符串列进行标签编码
# 将标签编码后的列转换为浮点数
#HomePlanet	CryoSleep	Cabin	Destination Name

listChangeStringToFloat=['HomePlanet','CryoSleep','Cabin','Destination','Name']
for i in listChangeStringToFloat:
    train[i] = label_encoder.fit_transform(train[i])
    train[i] = train[i].astype(float)     

    test[i] = label_encoder.fit_transform(test[i])
    test[i] = test[i].astype(float)  




print("train.head(): -------------------")
print(train.head())
"""
  PassengerId  HomePlanet  CryoSleep   Cabin  Destination  ...  ShoppingMall       Spa    VRDeck    Name  Transported
0     0001_01         1.0        0.0   149.0          2.0  ...     -0.715838 -0.756744 -0.730730  5252.0        False
1     0002_01         0.0        0.0  2184.0          2.0  ...      1.151527  1.499083  1.185414  4502.0         True
2     0003_01         1.0        0.0     1.0          2.0  ...     -0.715838  1.653040  1.208024   457.0        False
3     0003_02         1.0        0.0     1.0          2.0  ...      1.559493  1.622783  1.429402  7149.0        False
4     0004_01         0.0        0.0  2186.0          2.0  ...      1.470620  1.501747  0.132530  8319.0         True
"""
print("train.info(): -------------------")
print(train.info())
print("train.describe(): -------------------")
print(train.describe())
"""
train.describe(): -------------------
        HomePlanet    CryoSleep        Cabin  Destination  ...  ShoppingMall           Spa        VRDeck         Name
count  8693.000000  8693.000000  8693.000000  8693.000000  ...  8.693000e+03  8.693000e+03  8.693000e+03  8693.000000
mean      0.649833     0.349362  3227.857702     1.494306  ... -1.003326e-16  5.476402e-17 -6.702462e-17  4331.254113
std       0.795183     0.476796  2018.301775     0.814966  ...  1.000058e+00  1.000058e+00  1.000058e+00  2499.925329
min       0.000000     0.000000     0.000000     0.000000  ... -7.158381e-01 -7.567441e-01 -7.307295e-01     0.000000
25%       0.000000     0.000000  1341.000000     1.000000  ... -7.158381e-01 -7.567441e-01 -7.307295e-01  2166.000000
50%       0.000000     0.000000  3218.000000     2.000000  ... -7.158381e-01 -7.567441e-01 -7.307295e-01  4332.000000
75%       1.000000     1.000000  5018.000000     2.000000  ...  1.282429e+00  1.260823e+00  1.279477e+00  6502.000000
max       2.000000     1.000000  6560.000000     2.000000  ...  1.715039e+00  1.690102e+00  1.698773e+00  8473.000000
"""


sample=train['PassengerId']
sample1test=test['PassengerId']
train.drop('PassengerId', axis=1, inplace=True)
test.drop('PassengerId', axis=1, inplace=True)
x=train.drop('Transported', axis=1)
y=train['Transported']

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=42)

print("x_test.head():  <-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->")
print(x_test.head())
print("y_test.descrheadibe():  <-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->")
print(y_test.head())
print("y_test.head():  <-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->")
"""
      HomePlanet  CryoSleep   Cabin  Destination       Age  ...  FoodCourt  ShoppingMall       Spa    VRDeck    Name
304          2.0        0.0  4027.0          2.0 -0.629978  ...   1.437523      1.598205  0.231683  1.583182  8246.0
2697         0.0        0.0  5749.0          2.0 -0.705607  ...   1.526016     -0.715838 -0.756744 -0.143981  3382.0
8424         0.0        1.0  5209.0          2.0  0.860302  ...  -0.737423     -0.715838 -0.756744 -0.730730  3299.0
1672         0.0        0.0  5500.0          2.0  0.479291  ...   1.433997      1.572116  1.440420 -0.730730   650.0
8458         1.0        1.0  1299.0          2.0  0.984176  ...  -0.737423     -0.715838 -0.756744 -0.730730  2866.0
"""
print(y_test.head())
start_time = time.time()

rf_classifier = RandomForestClassifier()
param_grid = {'n_estimators': [50, 100, 200],'max_depth': [None, 10, 20],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4]}
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, scoring='neg_log_loss', n_jobs=-1)
grid_search.fit(x_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
#the output is :Best Hyperparameters: {'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 200}
best_rf_model = grid_search.best_estimator_





end_time = time.time()
elapsed_time = end_time - start_time
print("***********************       training elapsed_time is ", elapsed_time, "second")

# 获取当前进程的 CPU 和内存使用情况
cpu_percent = psutil.cpu_percent()
memory_percent = psutil.virtual_memory().percent
print("CPU use rate：", cpu_percent, "%")
print("MEM use rate：", memory_percent, "%")


y_pred_xg=best_rf_model.predict(x_test)
print('Accuracy Score:', accuracy_score(y_test,y_pred_xg))
"""
Accuracy Score: 0.79700977573318
"""
print('Confusion Matrix:', confusion_matrix(y_test,y_pred_xg))
"""
Confusion Matrix: 
[[657 204]
 [149 729]]
 """
print('Classification Report:', classification_report(y_test,y_pred_xg))
"""
Classification Report:               precision    recall  f1-score   support

                           False       0.82      0.76      0.79       861
                            True       0.78      0.83      0.81       878

                        accuracy                           0.80      1739
                       macro avg       0.80      0.80      0.80      1739
                    weighted avg       0.80      0.80      0.80      1739
"""


y_pred_test_xg=best_rf_model.predict(test)
print("y_pred_test_xg step1")
print(y_pred_test_xg)
"""
rf_xg=best_rf_model.predict_proba(test)
print("rf_xg step1")
print(rf_xg)

y_pred_test = pd.Series(y_pred_test_xg)
y_pred_test = y_pred_test.map({1:'True', 0:'False'}).tolist()
print("y_pred_test step2")
print(y_pred_test)
#y_pred_test = y_pred_test.to_list()
"""


submission=pd.DataFrame({'PassengerId':sample1test, 'Transported':y_pred_test_xg})
submission.to_csv('/kaggle/output/spaceship-titanic_submission202401302216.csv', index=False)

"""
Your Best Entry!
Your most recent submission scored 0.79027, which is an improvement of your previous score of 0.78793. Great job!
"""