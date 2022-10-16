#%%
#import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib.font_manager as fm
import seaborn as sns
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
from keras.utils.vis_utils import plot_model
import tikzplotlib
sns.set_style('white')

fig = plt.figure(figsize=(10,6))
ax = plt.axes((0.1,0.1,0.5,0.8))

#import plottools
import importlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import rc

# Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
#%%


# Load Dataset
df = pd.read_csv("G:\\Data\\Regular Porous Structures data_UnitCellArea.csv")

# Physical + Derivable Geometrical Parameters as Inputs
X1 = df.iloc[:, 21:22] # ETCX
X2 = df.iloc[:, 23:24] # ETCY
X3 = df.iloc[:, 19:20] # ETCZ
X4 = df.iloc[:, 26:27] # Permeability

X5 = df.iloc[:,15:16] # Interstitial Solid Material Area
X6 = df.iloc[:,16:17] # Porosity
X7 = df.iloc[:,17:18] # Specific Surface Area


# Controllable geometrical parameters as targets
Y1 = df.iloc[:,0:1] # Pore shape
Y2 = df.iloc[:,9:10] # Number of pores
Y3 = df.iloc[:,13:14] # Strut Thickness in X-Direction
Y4 = df.iloc[:,14:15] # Strut Thickness in Y-Direction

#%%
# Label Encoding for Categorical feature
label_encoder = LabelEncoder()
Y1_class = label_encoder.fit_transform(Y1)
n_class = len(np.unique(Y1_class))
Y1_classdf = pd.DataFrame(Y1_class, columns =['PoreShape'])

#%%
# "dataframe" is data frame after concatenating the required Xi's and yi's(with encoded pore shape)
dataframe = pd.concat([X1, X2, X3, X4, X5, X6, X7, Y1_classdf, Y2, Y3, Y4], axis=1) 

# Delete Rows with missing values
dataframe.dropna(inplace = True)

#%%

## Function to get the outputs of the model (Droping the outputs from data)
def get_outputs(data):
    
    y1 = data.pop('PoreShape')    
    y1 = np.array(y1)
    
    y2 = data.pop('No.of.pores')    
    y2 = np.array(y2)
    
    y3 = data.pop('StrutThicknessX')    
    y3 = np.array(y3)
    
    y4 = data.pop('StrutThicknessY')    
    y4 = np.array(y3)
    
    return y1, y2, y3, y4

#%%

# Split the data into train and test with 80/20
train, test = train_test_split(dataframe, test_size=0.2, random_state = 1)

## Getting the outputs for train and test data 
y_train = get_outputs(train)
y_test = get_outputs(test)

##Scaling the inputs
# min_max=MinMaxScaler()
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
input_layer = Input(shape=(7,),name='input_layer')

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

##Defining 2nd output layer y2
y4_output= Dense(1, activation="linear",name='y4_output')(Branched_layer3)

##Defining the model by specifying the input and output layers
model = Model(inputs=input_layer,outputs=[y1_output,y2_output,y3_output, y4_output])

## defining the optimiser and loss function
model.compile(optimizer='adam', loss={'y1_output':'sparse_categorical_crossentropy','y2_output':'mse', 'y3_output': 'mse', 'y4_output': 'mse'}, metrics={'y1_output':'accuracy'})

##training the model
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test,y_test))

##model predictions
predictions = model.predict(X_test)

predicted_shape = predictions[0]
predicted_shape_ = np.argmax(predicted_shape, axis=-1).astype('int')
predicted_NOP = predictions[1]
predicted_STX = predictions[2]
predicted_STY = predictions[3]

#%%
# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['y2_output_loss'], label='train')
plt.plot(history.history['val_y2_output_loss'], label='test')
plt.legend()
#%%

#%%

y_shape_true = label_encoder.inverse_transform(y_test[0])
y_shape_pred = label_encoder.inverse_transform(predicted_shape_)

# Print the confusion matrix
print(metrics.confusion_matrix(y_shape_true, y_shape_pred))
confusion_matrix = metrics.confusion_matrix(y_shape_true, y_shape_pred)



##Plotting actual vs predicted Number of pores
fig1 = plt.figure(figsize=(10,6))
ax1 = plt.axes((0.1,0.1,0.5,0.8))
ax1.scatter(y_test[1],predicted_NOP, edgecolors=(0, 0, 0.5), s=5, label='data')
ax1.plot([y_test[1].min(), y_test[1].max()], [y_test[1].min(), y_test[1].max()], 'r', lw=1.5, label='parity')
ax1.set(xlabel='Actual Number of pores ($n_p$)', ylabel='Predicted Number of pores ($n_p$)')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles=handles, labels=labels)
plt.savefig('predicted_NOP.png', bbox_inches='tight', dpi = 300)
#tikzplotlib.save("predicted__porosity.tex")




##Plotting actual vs predicted Strut Thickness in X-Direction
fig2 = plt.figure(figsize=(10,6))
ax2 = plt.axes((0.1,0.1,0.5,0.8))
ax2.scatter(y_test[2], predicted_STX, edgecolors=(0, 0, 0.5), s=5, label='data')
ax2.plot([y_test[2].min(), y_test[2].max()], [y_test[2].min(), y_test[2].max()], 'r', lw=1.5, label='parity')
ax2.set(xlabel='Actual Strut Thickness X-Dir ($S_x$)', ylabel='Predicted Strut Thickness X-Dir ($S_x$)')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles, labels=labels)
plt.savefig('predicted_STX.png', bbox_inches='tight', dpi = 300)
#tikzplotlib.save("predicted__porosity.tex")


##Plotting actual vs predicted Strut Thickness in Y-Direction
fig3 = plt.figure(figsize=(10,6))
ax3 = plt.axes((0.1,0.1,0.5,0.8))
ax3.scatter(y_test[3], predicted_STY, edgecolors=(0, 0, 0.5), s=5, label='data')
ax3.plot([y_test[3].min(), y_test[3].max()], [y_test[3].min(), y_test[3].max()], 'r', lw=1.5, label='parity')
ax3.set(xlabel='Actual Strut Thickness Y-Dir ($S_y$)', ylabel='Predicted Strut Thickness Y-Dir ($S_y$)')
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles=handles, labels=labels)
plt.savefig('predicted_STY.png', bbox_inches='tight', dpi = 300)
#tikzplotlib.save("predicted__porosity.tex")

# Heat Map/Confusion Matrix for Pore Shape classification
fig3 = plt.figure()
y_unique = np.unique(y_shape_true)
f = sns.heatmap(confusion_matrix, annot=True, fmt='d', xticklabels=y_unique, yticklabels=y_unique)
plt.savefig('Confusion Matrix_DerToCont.png', dpi = 300, bbox_inches='tight')

# Print the precision and recall, among other metrics
# print(metrics.classification_report(y_shape_true, y_shape_pred, digits=3))

# Print the accuracy
print('Accuray for Pore Shape:', accuracy_score(y_shape_true, y_shape_pred))
print('r2 score for NOP :',r2_score(y_test[1],predicted_NOP))
print('r2 score for STX :',r2_score(y_test[2],predicted_STX))
print('r2 score for STY :',r2_score(y_test[2],predicted_STY))



# residuals_NOPX  = y_test[1] - predicted_NOP.ravel()
# fig4 = plt.figure()
# ax4 = fig4.add_subplot(111)
# ax4.scatter(predicted_NOP,residuals_NOPX, edgecolors=(0, 0, 0.5))
# ax4.set(xlabel='Predicted', ylabel='Residual')
# plt.show()























