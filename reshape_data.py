import pandas as pd
import data_loading as dl

# Reshape numeric for X_test data
num_data = dl.numeric_data.head()
num_data2 = dl.numeric_data.iloc[0:26048, :]
print(" Old Shape", dl.numeric_data.shape)
print(" New Shape", num_data2.shape)

# Reshape numeric for X_test data
print("Reshape numeric for X_test data")
num_data = dl.numeric_data.head()
num_data3 = dl.numeric_data.iloc[0:6513, :]
print(" Old Shape", dl.numeric_data.shape)
print(" New Shape", num_data3.shape)
