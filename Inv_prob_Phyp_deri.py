#import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib.font_manager as fm
import seaborn as sns
sns.set_style('white')
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
from keras.utils.vis_utils import plot_model
import tikzplotlib


# xyz = pd.Series(list(mpl.rcParams.keys()))

# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['ps.fonttype'] = 42
# mpl.rcParams['font.family'] = 'Computer Modern'


fig = plt.figure(figsize=(10,6))
ax = plt.axes((0.1,0.1,0.5,0.8))

#import plottools
import importlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import rc

# Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]

# Edit the font, font size, and axes width
# plt.rcParams['font.size'] = 12
# plt.rcParams['axes.linewidth'] = 0.5

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
# plt.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

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
#print(dataset.shape)

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

#Scaling the inputs
# min_max=MinMaxScaler()
# X_train=min_max.fit_transform(train)
# X_test=min_max.transform(test)

Std_scale = StandardScaler()
X_train = Std_scale.fit_transform(train)
X_test = Std_scale.transform(test)

##Import the libraries for neural networks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense

##define input layer
input_layer = Input(shape=(4,),name='input_layer')

##Defining 2 hidden layers
Layer_1 = Dense(100, activation="relu",name='Layer_1')(input_layer)
Layer_2 = Dense(100, activation="relu",name='Layer_2')(Layer_1)

##Defining  output layer y1
y1_output= Dense(1, activation="linear",name='y1_output')(Layer_2)

##Defining Branched layer
Branched_layer1 = Dense(150, activation="relu",name='Branched_layer1')(Layer_2)
Branched_layer2 = Dense(150, activation="relu",name='Branched_layer2')(Branched_layer1)
Branched_layer3 = Dense(150, activation="relu",name='Branched_layer3')(Branched_layer2)
Branched_layer4 = Dense(150, activation="relu",name='Branched_layer4')(Branched_layer3)
Branched_layer5 = Dense(150, activation="relu",name='Branched_layer5')(Branched_layer4)



##Defining 2nd output layer y2
y2_output= Dense(1, activation="linear",name='y2_output')(Branched_layer5)

##Defining 2nd output layer y2
y3_output= Dense(1, activation="linear",name='y3_output')(Branched_layer5)

##Defining the model by specifying the input and output layers
model = Model(inputs=input_layer,outputs=[y1_output,y2_output,y3_output])

## defining the optimiser and loss function
#opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer= 'adam', loss={'y1_output':'mse','y2_output':'mae', 'y3_output':'mse'})

##training the model
model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test,y_test))
##model predictions
predictions = model.predict(X_test)

predicted_porosity = predictions[0]
predicted_SSA = predictions[1]
predicted_ST = predictions[2]


##Plotting actual vs predicted Porosity
fig1 = plt.figure(figsize=(10,6))
ax1 = plt.axes((0.1,0.1,0.5,0.8))
ax1.scatter(y_test[0],predicted_porosity, edgecolors=(0, 0, 0.4), s=5, label='data')
ax1.plot([y_test[0].min(), y_test[0].max()], [y_test[0].min(), y_test[0].max()], 'r', lw=1, label='parity')
ax1.set(xlabel='Actual Porosity ($\phi$)', ylabel='Predicted Porosity ($\phi$)')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles=handles, labels=labels)
plt.savefig('predicted__porosity.png', bbox_inches='tight', dpi = 300)
#tikzplotlib.save("predicted__porosity.tex")



##Plotting actual vs predicted SSA
fig2 = plt.figure(figsize=(10,6))
ax2 = plt.axes((0.1,0.1,0.5,0.8))
ax2.scatter(y_test[1],predicted_SSA, edgecolors=(0, 0, 0.4), s=5, label='data')
ax2.plot([y_test[1].min(), y_test[1].max()], [y_test[1].min(), y_test[1].max()], 'r', lw=1.5, label='parity')
ax2.set(xlabel='Actual SSA', ylabel='Predicted SSA')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles, labels=labels)
plt.savefig('predicted__SSA.png', bbox_inches='tight', dpi = 300)

##Plotting actual vs predicted Strut Thickness
fig3 = plt.figure(figsize=(10,6))
ax3 = plt.axes((0.1,0.1,0.5,0.8))
ax3.scatter(y_test[2],predicted_ST, edgecolors=(0, 0, 0.4), s=5, label='data')
ax3.plot([y_test[2].min(), y_test[2].max()], [y_test[2].min(), y_test[2].max()], 'r', lw=1.5, label='parity')
ax3.set(xlabel='Actual ST', ylabel='Predicted ST')
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles=handles, labels=labels)
plt.savefig('predicted__ST.png', bbox_inches='tight', dpi = 300)




print('r2 score for Porosity predicted :',r2_score(y_test[0],predicted_porosity))
print('r2 score for Specific Surface Area predicted :',r2_score(y_test[1],predicted_SSA))
print('r2 score for Strut Thickness predicted :',r2_score(y_test[2],predicted_ST))




# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)
# ax1.scatter(Y2,Y3, edgecolors=(0, 0, 0.4), s=7)
# #ax1.plot(Y2,Y3, 'r', lw =2)
# ax1.set(xlabel='Porosity', ylabel='SSA')