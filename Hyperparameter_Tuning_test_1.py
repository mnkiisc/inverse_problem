#Importing Libraries
import os
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam

from kerashypetune import KerasGridSearch

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.font_manager as fm
import seaborn as sns
sns.set_style('white')
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
from keras.utils.vis_utils import plot_model
import tikzplotlib

#import plottools
import importlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import rc

# Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def set_seed(seed):
    
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

# Load Dataset
df = pd.read_csv("G:\\Data\\Regular Porous Structures data_UnitCellArea.csv")

# Physical properties as Inputs
X1 = df.iloc[:, 21:22] # ETCX
X2 = df.iloc[:, 23:24] # ETCY
X3 = df.iloc[:, 19:20] # ETCZ
X4 = df.iloc[:, 26:27] # Permeability

# Derivable Geometrical Parameters as Inputs
Y1 = df.iloc[:,15:16] # Unit Cell Area
Y2 = df.iloc[:,16:17] # Porosity
Y3 = df.iloc[:,17:18] # Specific Surface Area

# "dataframe" is data frame after concatenating the required Xi's and yi's
dataframe = pd.concat([X1, X2, X3, X4, Y1, Y2, Y3], axis=1) 
# Delete Rows with missing values
dataframe.dropna(inplace = True)

##define a function to get the outputs of the model (Droping the outputs from data)
def get_outputs(data):
    
    y1 = data.pop('PorosityCal')    
    y1 = np.array(y1)
    
    y2 = data.pop('Specific.surface.area')    
    y2 = np.array(y2)
    
    y3 = data.pop('MaterialArea')    
    y3 = np.array(y3)
    
    return y1, y2, y3

# Split the data into train and test with 80 train / 20 test
train, test = train_test_split(dataframe, test_size=0.2, random_state = 1)

##getting the outputs(PE) and (V) of the train and test data 
y_train = get_outputs(train)
y_test = get_outputs(test)

def scale_datasets(x_train, x_test):
  """
  Standard Scale test and train data
  Z - Score normalization
  """
  standard_scaler = StandardScaler()
  x_train_scaled = pd.DataFrame(
      standard_scaler.fit_transform(x_train),
      columns=x_train.columns
  )
  
  x_test_scaled = pd.DataFrame(
      standard_scaler.transform(x_test),
      columns = x_test.columns
  )
  return x_train_scaled, x_test_scaled

# scale the dataset
x_train, x_test = scale_datasets(train, test)



def get_model(param):
    
    set_seed(33)    
  
    ##define input layer
    input_layer = Input(shape=(4,),name='input_layer')

    ##Defining 2 hidden layers
    Layer_1 = Dense(param['unit_hid_1'], activation="relu",name='Layer_1')(input_layer)
    Layer_2 = Dense(param['unit_hid_2'], activation="relu",name='Layer_2')(Layer_1)

    ##Defining  output layer y1
    y1_output= Dense(1, activation="linear",name='y1_output')(Layer_2)

    ##Defining Branched layer
    Branched_layer1 = Dense(param['unit_hid_3'], activation="relu",name='Branched_layer1')(Layer_2)
    Branched_layer2 = Dense(param['unit_hid_4'], activation="relu",name='Branched_layer2')(Branched_layer1)
    Branched_layer3 = Dense(param['unit_hid_5'], activation="relu",name='Branched_layer3')(Branched_layer2)
    Branched_layer4 = Dense(param['unit_hid_6'], activation="relu",name='Branched_layer4')(Branched_layer3)
    Branched_layer5 = Dense(param['unit_hid_7'], activation="relu",name='Branched_layer5')(Branched_layer4)

    ##Defining 2nd output layer y2
    y2_output= Dense(1, activation="linear",name='y2_output')(Branched_layer5)

    ##Defining 2nd output layer y2
    y3_output= Dense(1, activation="linear",name='y3_output')(Branched_layer5)

    ##Defining the model by specifying the input and output layers
    model = Model(inputs=input_layer, outputs=[y1_output,y2_output,y3_output])

    ## defining the optimiser and loss function
    model.compile(optimizer=Adam(learning_rate=param['lr']), loss={'y1_output':'mse','y2_output':'mse', 'y3_output':'mse'})
    
    return model
    
param_grid = {
    'unit_hid_1' : [16, 32, 64, 100, 150, 200],
    'unit_hid_2' : [16, 32, 64, 100, 150, 200],
    'unit_hid_3' : [16, 32, 64, 100, 150, 200],
    'unit_hid_4' : [16, 32, 64, 100, 150, 200],
    'unit_hid_5' : [16, 32, 64, 100, 150, 200],
    'unit_hid_6' : [16, 32, 64, 100, 150, 200],
    'unit_hid_7' : [16, 32, 64, 100, 150, 200],
    'lr': [1e-1, 1e-2,1e-3, 1e-4], 
    'epochs': 1000, 
    'batch_size': 32
}


es = EarlyStopping(patience=5, verbose=1, min_delta=0.001, monitor='val_loss', mode='auto', restore_best_weights=True)

hypermodel = get_model

kgs = KerasGridSearch(hypermodel, param_grid, monitor='val_loss', greater_is_better=False, tuner_verbose=1)
kgs.search(x_train, [x_train,y_train], validation_data=(x_test, [x_test,y_test]), callbacks=[es])