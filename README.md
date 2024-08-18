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
|Type (Red Dwarf, Brown Dwarf, White Dwarf, Main Sequence, Supergiant, Hypergiant)|

L0 is the average Sun's luminosity and R0 the average radius:

|Sun values|
|-|
|L0 = $$3.828 \times 10^{26} \text{ W}$$|
|R0 = $$6.9551 \times 10^{8} \text{ m}$$|

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
In the Data Analysis, we can also observe the correlation between the numerical columns (including target). So with ```correl``` as ```df.drop(columns ['Spectral_Class','Color'],axis=1).corr()```, the output for ```correl``` is:

![image](https://github.com/user-attachments/assets/592df0b8-0356-4000-823f-01911b23c23b)

The absolute magnitude of the star fits almost perfectly with the star's type! A -95,52% correlation coefficient says that the star's shine increases with its class in the dataset (because a low absolute magnitude means a high shine, that's why its inversely proportional). It's a great correlation for, like an example, simple linear regression. It generates a valuable conclusion: the target is an *ordinal* categorical variable, meaning the star classes have a type of hierarchy! This creates the linear regression thought.

Although the following question is not answering by the dataset Data Analysis, it's worth answering: why luminosity has bad correlation with temperature and radius? This question comes from the mathematical relation between luminosity and temperature/radius. The truth is the calculated correlation is linear, i.e., how close this relation is from a first-degree equation. The star luminosity can be calculated by:

$$
L = 4 \pi R^2 \sigma T^4
$$

Where:
- L is the luminosity of the star (in watts);
- R is the radius of the star (in meters);
- $$\sigma$$ is the Stefan-Boltzmann constant $$\(\sigma = 5.670 \times 10^{-8} \ \text{ W m}^{-2} \text{K}^{-4}\)$$;
- T is the effective surface temperature of the star (in Kelvin).

That's why the correlation is worse between luminosity and temperature than luminosity and radius. The relation with radius is quadratic, while the relation with temperature is in forth power, so far from a first-degree equation (although a quadratic equation is also different from a fisrt-degree). This was an Astrophysics problem answered with Statistics knowledge!

It's also important to verify the correlation between the training columns with the target columns. Except for the temperature, the other columns have a correlation coefficient greater than 50% or less than -50% *(Attention: a negative correlation coefficient does not mean a bad correlation, but means that the two variables are inversely proportional. The worst correlation coefficient is the closest to 0%)*. This seems to be the moment to consider the star's temperature only to Data Analysis, but let's do it temporarily.

## Data visualization
Let's consider the numeric labels as the following dictionary ```{Red Dwarf: 0, Brown Dwarf: 1, White Dwarf: 2, Main Sequence: 3, Supergiant: 4, Hypergiant: 5}```. This has not been changed to string data because of lack of need and to not interfer later in ML. The first chart is a bar chart studying the average temperature by star class with standard-deviation:

```
plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
color = ['r','w','y','g','b','c']

for typ in df['Type'].unique():
  df_typ = df[df['Type'] == typ]
  ax.bar(typ,df_typ['Temperature'].mean(),yerr=df_typ['Temperature'].std(),color=color[typ])
ax.set_ylabel("Temperature (K)")
ax.set_xlabel("Stellar type")
plt.show()
```

![image](https://github.com/user-attachments/assets/2237e087-02e7-4492-853c-f8599cc494a8)

