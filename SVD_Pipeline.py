'''Dimension Reduction - SVD 

# Install the required packages if not available

# !pip install feature_engine
# !pip install dtale

# **Importing required packages**
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from kneed import KneeLocator
from sqlalchemy import create_engine

user = 'root' # user name
pw = 'WorkBench1' # password
db = 'univ_db' # database
# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# **Import the data**
University = pd.read_excel(r"C:\Users\Karuna Singh\Downloads\Mar.23\Datasets\clustering\KMeans_material\University_Clustering.xlsx")
University
# dumping data into database 
# name should be in lower case
University.to_sql('university_clustering', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# loading data from database
sql = 'select * from university_clustering'

df = pd.read_sql_query(sql, con = engine)

print(df)

# EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS

# Descriptive Statistics and Data Distribution Function
df.describe()

# Data Preprocessing
# Drop the unwanted features
df1 = df.drop(["UnivID"], axis = 1)

df1.info()

# Checking Null Values
df1.isnull().sum()

# SVD can be implemented on Numeric features
numeric_features = df1.select_dtypes(exclude = ['object']).columns
numeric_features

# Make Pipeline
# Define the Pipeline steps

# Define SVD model
svd = TruncatedSVD(n_components = 5)

# By using Mean imputation null values can be impute

# Data has to be standardized to address the scale difference
num_pipeline = make_pipeline(SimpleImputer(strategy = 'mean'), StandardScaler(), svd)

# Pass the raw data through pipeline
processed = num_pipeline.fit(df1[numeric_features]) 
processed

# ## Save the End to End SVD pipeline with Imputation and Standardization
import joblib
joblib.dump(processed, 'svd_DimRed')

import os 
os.getcwd()

# ## Import the pipeline
model = joblib.load("svd_DimRed")
model

# Apply the saved model on to the Dataset to extract SVD values
svd_res = pd.DataFrame(model.transform(df1[numeric_features]))
svd_res

# SVD weights
svd.components_

# Variance percentage
print(svd.explained_variance_ratio_)

# Cumulative Variance percentage
var1 = np.cumsum(svd.explained_variance_ratio_)
print(var1)

# Variance plot for SVD components obtained 
plt.plot(var1, color = "red")

# KneeLocator
# Refer the link to understand the parameters used: https://kneed.readthedocs.io/en/stable/parameters.html    
# from kneed import KneeLocator
kl = KneeLocator(range(len(var1)), var1, curve = 'concave', direction = "increasing") 
# The line is pretty linear hence Kneelocator is not able to detect the knee/elbow appropriately
kl.elbow
plt.style.use("seaborn")
plt.plot(range(len(var1)), var1)
plt.xticks(range(len(var1)))
plt.ylabel("variance")
plt.axvline(x = kl.elbow, color = 'r', label = 'axvline - full height', ls = '--')
plt.show()

# SVD for Feature Extraction
# Final dataset with manageable number of columns (Feature Extraction)

final = pd.concat([df.Univ, svd_res.iloc[:, 0:3]], axis = 1)
final.columns = ['Univ', 'svd0', 'svd1', 'svd2']
final

# Scatter diagram
ax = final.plot(x = 'svd0', y = 'svd1', kind = 'scatter', figsize = (12, 8))
final[['svd0', 'svd1', 'Univ']].apply(lambda x: ax.text(*x), axis = 1)

# Prediction on new data
newdf = pd.read_excel(r"C:\Users\Karuna Singh\Downloads\Mar.23\Datasets\clustering\SVD\new_Univ_4_pred.xlsx")
newdf

# Drop the unwanted features
newdf1 = newdf.drop(["UnivID"], axis = 1)

num_feat = newdf1.select_dtypes(exclude = ['object']).columns
num_feat

new_res = pd.DataFrame(model.transform(newdf1[num_feat]))
new_res
