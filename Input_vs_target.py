# Plotting the inputs vs targets

import numpy as np
#from pylab import *
import seaborn as sns

# Create some test data
##import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tikzplotlib

# Load Dataset
df = pd.read_csv("E:\\Naveen\Research\\Thesis Codes\\Regular Porous Structures data.csv")

# Physical properties as Inputs
X1 = df.iloc[:, 18:19] # ETCX
X2 = df.iloc[:, 20:21] # ETCY
X3 = df.iloc[:, 16:17] # ETCZ
X4 = df.iloc[:, 23:24] # Permeability

# Derivable Geometrical Parameters as Inputs
Y1 = df.iloc[:,12:13] # Strut Thickness
Y2 = df.iloc[:,13:14] # Porosity
Y3 = df.iloc[:,14:15] # Specific Surface Area

# "dataframe" is data frame after concatenating the required Xi's and yi's
dataframe = pd.concat([X1, X2, X3, X4, Y1, Y2, Y3], axis=1) 
#print(dataset.shape)

# Delete Rows with missing values
dataframe.dropna(inplace = True)

sns.displot(x='ETCX', data=dataframe, kde=True);
#sns.displot(x='ETCX', data=dataframe, kind='cdf');

plt.savefig('predicted__porosity.png', dpi = 300)
tikzplotlib.save("ETCX_vs_porosity.tex")


