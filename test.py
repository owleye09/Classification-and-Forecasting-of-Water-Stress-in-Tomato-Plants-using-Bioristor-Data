import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Bidirectional,GRU
import os
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Convolution2D
import pickle
'''
d1 = pd.read_csv("Dataset/prospe_IR1.csv")
d1.fillna(0, inplace = True)
Y1 = d1['y']

d2 = pd.read_csv("Dataset/prospe_IR2.csv")
d2.fillna(0, inplace = True)
Y2 = d2['y']

dd = []
for i in range(len(Y1)):
    if Y1[i] == 0:
        dd.append("Healthy")
    elif Y1[i] == 1:
        dd.append("Uncertain")

for i in range(len(Y2)):
    if Y2[i] == 0:
        dd.append("Stress")
    elif Y2[i] == 1:
        dd.append("Recovery")
    
dd = np.asarray(dd)

data = pd.concat([d1, d2])

print(dd.shape)
print(data.shape)

data['status'] = dd
print(data.shape)
data.to_csv("Dataset/Tomato_data.csv", index = False)
'''

le = LabelEncoder()

dataset = pd.read_csv("Dataset/Tomato_data.csv")
dataset.fillna(0, inplace = True)
dataset['status'] = pd.Series(le.fit_transform(dataset['status'].astype(str)))
status = dataset['status'].ravel()
drought = dataset['y'].ravel()

dataset.drop(['y', 'status'], axis = 1,inplace=True) #removing irrelevant columns

X = dataset.values

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
status = status[indices]
drought = drought[indices]

normalized = MinMaxScaler()
X = normalized.fit_transform(X)

pca = PCA(n_components=70)
#X = pca.fit_transform(X)
print(X)
print(np.unique(drought, return_counts=True))
'''
X_train, X_test, y_train, y_test = train_test_split(X, status, test_size=0.2) #split dataset into train and test
X_train = X
y_train = status
dt = DecisionTreeClassifier(criterion='gini')
dt.fit(X_train, y_train)
predict = dt.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)

dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)
predict = dt.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)

param_grid = {
    'n_estimators': [5, 15, 18, 100],
    'max_depth': [2, 5, 7, 9]
}

#rf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)
'''

drought = to_categorical(drought)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X, drought, test_size=0.2) #split dataset into train and test
X_train, X_test1, y_train, y_test1 = train_test_split(X, drought, test_size=0.1) #split dataset into train and test

lstm_model = Sequential()#defining deep learning sequential object
#adding LSTM layer with 100 filters to filter given input X train data to select relevant features
lstm_model.add(LSTM(100,input_shape=(X_train.shape[1], X_train.shape[2])))
#adding dropout layer to remove irrelevant features
lstm_model.add(Dropout(0.5))
#adding another layer
lstm_model.add(Dense(100, activation='relu'))
#defining output layer for prediction
lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
if os.path.exists('model/lstm_weights.hdf5') == False:
    model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
    lstm_model.fit(X_train, y_train, epochs = 60, batch_size = 64, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
else:
    lstm_model = load_model('model/lstm_weights.hdf5')

predict = lstm_model.predict(X_test)
predict = np.argmax(predict, axis=1)
print(predict)
test = np.argmax(y_test, axis=1)
acc = accuracy_score(test, predict)
print(acc)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

cnn_model = Sequential()
cnn_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
cnn_model.add(Flatten())
cnn_model.add(Dense(units = 256, activation = 'relu'))
cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train, y_train, batch_size = 64, epochs = 60, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model = load_model("model/cnn_weights.hdf5")
   
predict = cnn_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(predict)
print(acc)

