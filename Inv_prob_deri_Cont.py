#%%
from numpy import unique
from numpy import argmax
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.losses import Huber
import numpy as np
import pandas as pd
from numpy import arange
from matplotlib import pyplot
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from numpy import percentile
from matplotlib import rc
import h5py
import numpy as np
import array
import matplotlib as mpl
import sys
import matplotlib.gridspec as gridspec 
import math
import matplotlib.font_manager as fm
import importlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn import metrics
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import tensorflow

import matplotlib.font_manager as fm

#import plottools
import importlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from matplotlib import rc

# Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]

# Edit the font, font size, and axes width
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0.5
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# Load Dataset
df = pd.read_csv("E:\\Naveen\Research\\Thesis Codes\\Regular Porous Structures data.csv")

# Derivable Geometrical Parameters as Inputs
X1 = df.iloc[:,12:13] # Strut Thickness
X2 = df.iloc[:,13:14] # Porosity
X3 = df.iloc[:,14:15] # Specific Surface Area

# Controllable geometrical parameters as targets
Y1 = df.iloc[:,0:1] # Pore shape
Y2 = df.iloc[:,9:10] # Number of pores
Y3 = df.iloc[:,387:388] # Pore Distribution Ratio

#%%
# Label Encoding for Categori
label_encoder = preprocessing.LabelEncoder()
Y1_class = label_encoder.fit_transform(Y1)
n_class = len(unique(Y1_class))
Y1_classdf = pd.DataFrame(Y1_class, columns =['PoreShape'])

#%%
# "dataframe" is data frame after concatenating the required Xi's and yi's(with encoded pore shape)
dataframe = pd.concat([X1, X2, X3, Y1_classdf, Y2, Y3], axis=1) 

# Delete Rows with missing values
dataframe.dropna(inplace = True)

#%%

## Function to get the outputs of the model (Droping the outputs from data)
def get_outputs(data):
    
    y1 = data.pop('PoreShape')    
    y1 = np.array(y1)
    
    y2 = data.pop('No.of.pores')    
    y2 = np.array(y2)
    
    y3 = data.pop('PDR')    
    y3 = np.array(y3)
    
    return y1, y2, y3

#%%

# Split the data into train and test with 80/20
train, test = train_test_split(dataframe, test_size=0.2, random_state = 1)

## Getting the outputs for train and test data 
y_train = get_outputs(train)
y_test = get_outputs(test)

###Scaling the inputs
min_max=MinMaxScaler()
# X_train = min_max.fit_transform(train)
# X_test = min_max.transform(test)

Std_scale = StandardScaler()
X_train = Std_scale.fit_transform(train)
X_test = Std_scale.transform(test)

#%%

##Import the libraries for neural networks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense

##define input layer
input_layer = Input(shape=(3,),name='input_layer')

##Defining hidden layers
Layer_1 = Dense(200, activation="relu",name='Layer_1')(input_layer)
Layer_2 = Dense(300, activation="relu",name='Layer_2')(Layer_1)
Layer_3 = Dense(200, activation="relu",name='Layer_3')(Layer_2)

##Defining  output layer y1
y1_output= Dense(n_class, activation="softmax",name='y1_output')(Layer_3)

##Defining Branched layer
Branched_layer1=Dense(200, activation="relu",name='Branched_layer1')(Layer_3)
Branched_layer2=Dense(300, activation="relu",name='Branched_layer2')(Branched_layer1)
Branched_layer3=Dense(200, activation="relu",name='Branched_layer3')(Branched_layer2)

##Defining 2nd output layer y2
y2_output= Dense(1, activation="linear",name='y2_output')(Branched_layer3)

##Defining 2nd output layer y2
y3_output= Dense(1, activation="linear",name='y3_output')(Branched_layer3)

##Defining the model by specifying the input and output layers
model = Model(inputs=input_layer,outputs=[y1_output,y2_output,y3_output])

## defining the optimiser and loss function
model.compile(optimizer='sgd', loss={'y1_output':'sparse_categorical_crossentropy','y2_output':tf.keras.losses.Huber(delta=5), 'y3_output':tf.keras.losses.Huber()}, metrics={'y1_output':'accuracy'})

##training the model
history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_test,y_test))

##model predictions
predictions = model.predict(X_test)

predicted_shape = predictions[0]
predicted_shape_ = argmax(predicted_shape, axis=-1).astype('int')
predicted_NOP = predictions[1]
predicted_PDR = predictions[2]

#%%
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['y2_output_loss'], label='train')
pyplot.plot(history.history['val_y2_output_loss'], label='test')
pyplot.legend()
#%%

#%%

y_shape_true = label_encoder.inverse_transform(y_test[0])

y_shape_pred = label_encoder.inverse_transform(predicted_shape_)

# Print the confusion matrix
print(metrics.confusion_matrix(y_shape_true, y_shape_pred))

confusion_matrix = metrics.confusion_matrix(y_shape_true, y_shape_pred)

# Print the precision and recall, among other metrics
#print(metrics.classification_report(y_shape_true, y_shape_pred, digits=3))

##Plotting actual vs predicted Number of Pores
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(y_test[1],predicted_NOP, edgecolors=(0, 0, 0.5))
ax1.plot([y_test[1].min(), y_test[1].max()], [y_test[1].min(), y_test[1].max()], 'r', lw=2)
ax1.set(xlabel='Actual Number of Pores (X-Dir)', ylabel='Predicted Number of Pores (X-Dir)')
ax1.set_title("Derivable --to-- Controllable", color="red")
#plt.text(2.5, 20 , r'$R^2 \ Score = 0.491$')
plt.savefig('Der_Cont_nopX.png', dpi = 150)


##Plotting actual vs predicted Pore Distribution Ratio
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(y_test[2],predicted_PDR, edgecolors=(0, 0, 0.5))
ax2.plot([y_test[2].min(), y_test[2].max()], [y_test[2].min(), y_test[2].max()], 'r', lw=2)
ax2.set(xlabel='Actual Number of Pores (Y-Dir)', ylabel='Predicted Number of Pores (Y-Dir)')
ax2.set_title("Derivable --to-- Controllable", color="red")
#plt.text(2, 20 , r'$R^2 \ Score = 0.756$')
plt.savefig('Der_Cont_nopY.png', dpi = 150)

# Heat Map/Confusion Matrix for Pore Shape classification
fig3 = plt.figure()
y_unique = np.unique(y_shape_true)
f = sns.heatmap(confusion_matrix, annot=True, fmt='d', xticklabels=y_unique, yticklabels=y_unique)
plt.savefig('Confusion Matrix_DerToCont.png', dpi = 150, bbox_inches='tight')

# Print the precision and recall, among other metrics
#print(metrics.classification_report(y_shape_true, y_shape_pred, digits=3))

# Print the accuracy
print('Accuray for Pore Shape:', accuracy_score(y_shape_true, y_shape_pred))

print('r2 score for NOP_X :',r2_score(y_test[1],predicted_NOP))

print('r2 score for NOP_Y :',r2_score(y_test[2],predicted_PDR))



residuals_NOPX  = y_test[1] - predicted_NOP.ravel()
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.scatter(predicted_NOP,residuals_NOPX, edgecolors=(0, 0, 0.5))
ax4.set(xlabel='Predicted', ylabel='Residual')

plt.show()























