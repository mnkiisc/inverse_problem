import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load Dataset
df = pd.read_csv("E:\\Naveen\Research\\Thesis Codes\\Regular Porous Structures data.csv")

# Derivable Geometrical Parameters as Inputs
X1 = df.iloc[:,12:13] # Strut Thickness
X2 = df.iloc[:,13:14] # Porosity
X3 = df.iloc[:,14:15] # Specific Surface Area

# Controllable geometrical parameters as targets
Y1 = df.iloc[:,0:1] # Pore shape
Y2 = df.iloc[:,9:10] # Number of pores 
Y3 = df.iloc[:,387:388] # Number of pores in Y-direction



# Encoding the categorical variable - Pore Shape

def prepare_inputs(X_train, X_test):
	ohe = OneHotEncoder()
	ohe.fit(X_train)
	X_train_enc = ohe.transform(X_train)
	X_test_enc = ohe.transform(X_test)
	return X_train_enc, X_test_enc


from testClass import MathsOperations

xyz = MathsOperations(2, 3)
print(xyz.testAddition())