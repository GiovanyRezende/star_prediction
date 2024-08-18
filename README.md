# Star Classification Project
*This project has the objective of studying methods to create a ML model for star classification and analyze astronomical data*. It was used Multiple Linear Regression and Neural Network for scientific purposes and comparison, while the entire project was made in Google Colab. 

# The data
*[This is the data of the project](https://www.kaggle.com/datasets/deepu1109/star-dataset)*. The features are:

|Features|
|-|
|Temperature (K)|
|Luminosity (relative of Sun luminosity L0)|
|Radius (relative of Sun radius R0)|
|Absolute Magnitude|
|Color|
|Spectral Class (O, B, A, F, G, K, M)|
|Type (Red Dwarf, Brown Dwarf, White Dwarf, Main Sequence, Supergiants, Hypergiants)|

L0 is the average Sun's luminosity and R0 the average radius:

|Sun values|
|-|
|L0 = 3.828 * 10^26 W|
|R0 = 6.9551 * 10^8 m|

In the Stars.csv file, the columns are:

|Columns|
|-|
|Temperature|
|L|
|R|
|A_M|
|Color|
|Spectral_Class|
|Type|

## The training and target data
*For the Machine Learning, the training columns are Temperature, L, R and A_M, while the target is Type.* The remaining columns were used purely for data analysis. It's interesting to notice that, while ```'Type'``` is an integer column, statistically it describes a categorical variable, the numbers are used only for representation and facilitate classification (because ML models don't work well with string data). 

# Importing libraries and creating Pandas dataframe
```
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score, classification_report
```

Pandas was used for the dataframe and queries, Matplotlib for data visualization, Keras for the Neural Network and scikit-learn for Linear Regression (and main Machine Learning base). For the dataframe ```df``` *(remembering that it was made in Google Colab, so pay attention importing the .csv file for the code)*:
```
df = pd.read_csv('/content/Stars.csv',delimiter=',')
```

When we call ```df.head()```, the output is:

![image](https://github.com/user-attachments/assets/cbe9a9cc-c6a9-4642-ba9c-1156ce02d2bb)

# The Data Analysis
Firstly, we can can call the Pandas method ```.describe()``` in ```df.drop(axis=1,columns='Type').describe()``` (considering it's more important to get the statistics of pure numeric variables). We get the following statistics:

![image](https://github.com/user-attachments/assets/27f5cb07-6fca-459f-8fa6-fd910a2f02ee)

It's notable that the luminosity amplitude and radius amplitude are ***astronomical***. The actual reason for this was evidenced in the scope of the project: luminosity and radius are relative of Sun's values, so it's explained for the fact that exist stars thousands of times bigger than our star and with high temperatures (this is a question answered posteriorly in the analysis). By the way, the columns have high standard-deviations, expressing how diverse is the stellar catalog with such statistical variability.
In the Data Analysis, we can also observe the correlation between the numerical columns (including target). So with ```correl``` as  ```df.drop(columns=['Spectral_Class','Color'],axis=1).corr()```, the output for ```correl``` is:

![image](https://github.com/user-attachments/assets/592df0b8-0356-4000-823f-01911b23c23b)

The absolute magnitude of the star fits almost perfectly with the star's type! A -95,52% correlation coefficient says that the star's shine increases with its class in the dataset (because a low absolute magnitude means a high shine, that's why its inversely proportional). It's a great correlation for, like an example, simple linear regression. It generates a valuable conclusion: the target is an *ordinal* categorical variable, meaning the star classes have a type of hierarchy!


