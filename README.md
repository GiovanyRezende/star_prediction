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

Although the following question is not answering by the dataset Data Analysis, it's worth answering: why luminosity has bad correlation with temperature and radius? This question comes from the mathematical relation between luminosity and temperature/radius. The truth is the calculated correlation is linear, i.e., how close this relation is from a first-degree equation. The star luminosity can be calculated by Stefan-Boltzmann Law:

$$
L = 4 \pi R^2 \sigma T^4
$$

Where:
- L is the luminosity of the star (in watts);
- R is the radius of the star (in meters);
- $$\sigma$$ is the Stefan-Boltzmann constant $$\(\sigma = 5.670 \times 10^{-8} \ \text{ W m}^{-2} \text{K}^{-4}\)$$;
- T is the effective surface temperature of the star (in Kelvin).

That's why the correlation is worse between luminosity and temperature than luminosity and radius. The relation with radius is quadratic, while the relation with temperature is in forth power, so far from a first-degree equation (although a quadratic equation is also different from a fisrt-degree). This was an Astrophysics problem answered with Statistics knowledge!

It's also important to verify the correlation among the training columns with the target columns. Except for the temperature, the other columns have a correlation coefficient greater than 50% or less than -50% *(Attention: a negative correlation coefficient does not mean a bad correlation, but means that the two variables are inversely proportional. The worst correlation coefficient is the closest to 0%)*. This seems to be the moment to consider the star's temperature only to Data Analysis, but let's do it temporarily.

## Data visualization
Let's consider the numeric labels as the following dictionary ```{Red Dwarf: 0, Brown Dwarf: 1, White Dwarf: 2, Main Sequence: 3, Supergiant: 4, Hypergiant: 5}```. This has not been changed to string data because of lack of need and to not interfer later in ML. The first chart is a bar chart studying the average temperature by star class with standard-deviation:

```
plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
color = ['r','y','w','g','b','c']

for typ in df['Type'].unique():
  df_typ = df[df['Type'] == typ]
  ax.bar(typ,df_typ['Temperature'].mean(),yerr=df_typ['Temperature'].std(),color=color[typ])
ax.set_ylabel("Temperature (K)")
ax.set_xlabel("Stellar type")
plt.show()
```

![image](https://github.com/user-attachments/assets/2237e087-02e7-4492-853c-f8599cc494a8)

Red dwarfs and brown dwarfs have relatively low average temperature and low standard-deviation. From white dwarfs to classes beyond, there are higher average temperatures and higher standard-deviations. Despite high dispersion, temperature is a variable that differentiates red and brown dwarfs from the other classes. But the most curious class is the one from the hypergiants. The dispersion is so extreme that it seems like there is star with 0K temperature. It's not true, neither in the dataset, nor in the Physics Laws. The ```.describe()``` in ```df[df['Type'] == 5].Temperature.describe()``` outputs the following table:

![image](https://github.com/user-attachments/assets/c38701c1-db8d-46ba-8d9a-105c9db3c4c2)

The "yerr" bar only evidences the extreme high variability in hypergiants. The minimum temperature is 3399K, while the maximum temperature is 38940K (the "yerr" bar doesn't achive this value in the chart), having a high amplitude. Still studying the stars temperature, using a boxplot chart, we have:

![image](https://github.com/user-attachments/assets/b0c83f4c-a27a-42d4-bf08-0074ebe9abf9)

The only outlier is below the red dwarfs' lower limit. An overall sight reveals no abnormal temperature, in a statistical way, despite the outlier in red dwarfs. The curious boxplot exists in the hypergiants' analysis: the median is basically the first quartile, really close to the lower limit, while the difference between maximum value and third quartile is high.

The second analysis studies the average radius by star class with standard-deviation:

```
plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
color = ['r','y','w','g','b','c']

for typ in df['Type'].unique():
  df_typ = df[df['Type'] == typ]
  ax.bar(typ,df_typ['R'].mean(),yerr=df_typ['R'].std(),color=color[typ])
ax.set_ylabel("Radius (R0)")
ax.set_xlabel("Stellar type")
plt.show()
```

![image](https://github.com/user-attachments/assets/1046c05a-f030-4812-ab14-faea8bb531b0)

```
plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
radius = [df[df['Type'] == typ]['R'] for typ in df['Type'].unique()]
ax.boxplot(radius,labels=df['Type'].unique())
ax.set_ylabel("Radius (R0)")
ax.set_xlabel("Stellar type")
plt.show()
```

![image](https://github.com/user-attachments/assets/9d4d3c65-9dbe-4dea-be1a-a44f6cabf8fc)

Hypergiants are so colossal to the point of supergiants seeming small and the the dwarfs seeming like as we "see" molecules. To study smaller types, we use the following code:

```
df_1 = df[df['Type'] < 4]

plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
color = ['r','y','w','g','b','c']

for typ in df_1['Type'].unique():
  df_typ = df[df['Type'] == typ]
  ax.bar(typ,df_typ['R'].mean(),yerr=df_typ['R'].std(),color=color[typ])
ax.set_ylabel("Radius (R0)")
ax.set_xlabel("Stellar type")
plt.show()
```

![image](https://github.com/user-attachments/assets/62664c2f-1529-450d-8c52-e349b02107bd)

White dwarfs are the smaller stars in this dataset. Another significant insight for Astronomy is the ```Main Sequence``` bar, revealing that our Sun is one of the smallest stars on the Main Sequence.

As demonstrated previously, the luminosity study will provide a similar problem, because of Stefan-Boltzmann Law:

```
plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
color = ['r','y','w','g','b','c']

for typ in df['Type'].unique():
  df_typ = df[df['Type'] == typ]
  ax.bar(typ,df_typ['L'].mean(),yerr=df_typ['L'].std(),color=color[typ])
ax.set_ylabel("Luminosity (L0)")
ax.set_xlabel("Stellar type")
plt.show()
```

![image](https://github.com/user-attachments/assets/e17c74b1-988b-485b-9d8a-171ade02503a)

```
plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
lums = [df[df['Type'] == typ]['L'] for typ in df['Type'].unique()]
ax.boxplot(lums,labels=df['Type'].unique())
ax.set_ylabel("Luminosity (L0)")
ax.set_xlabel("Stellar type")
plt.show()
```

![image](https://github.com/user-attachments/assets/30ca4f65-961d-4c1c-8482-49f780c9b677)

Luminosity study evidences more outliers. This doesn't mean the search is wrong, but means there are values beyond (or below) the calculated statistical range. Outliers are good for decision making depending on the occasion. In this case, the outliers make sense, because of forth-power property of Stefan-Botzmann Law. To study less bright types, we use the following code:

```
df_1 = df[df['Type'] < 3]

plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
color = ['r','y','w','g','b','c']

for typ in df_1['Type'].unique():
  df_typ = df[df['Type'] == typ]
  ax.bar(typ,df_typ['L'].mean(),yerr=df_typ['L'].std(),color=color[typ])
ax.set_ylabel("Luminosity (L0)")
ax.set_xlabel("Stellar type")
plt.show()
```

![image](https://github.com/user-attachments/assets/59ea5e2b-0863-42c3-b78e-8a4dd8144cab)

The relation between absolute magnitude and type is the closest to a first degree equation, as demonstrated in correlation study:

```
plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
color = ['r','y','w','g','b','c']

for typ in df['Type'].unique():
  df_typ = df[df['Type'] == typ]
  ax.bar(typ,df_typ['A_M'].mean(),yerr=df_typ['A_M'].std(),color=color[typ])
ax.set_ylabel("Absolute Magnitude")
ax.set_xlabel("Stellar type")
plt.show()
```

![image](https://github.com/user-attachments/assets/95f126ae-d328-48a5-8c5a-bb64576ef987)


```
plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
ams = [df[df['Type'] == typ]['A_M'] for typ in df['Type'].unique()]
ax.boxplot(ams,labels=df['Type'].unique())
ax.set_ylabel("Absolute Magnitude")
ax.set_xlabel("Stellar type")
plt.show()
```

![image](https://github.com/user-attachments/assets/42d7cbea-be18-4d67-bd08-769ab5536584)

To finish the charts part, it's interesting to study a clustering. Plotting the relation between absolute magnitude and temperature, we create a chart similar to the HR-diagram:

```
plt.style.use('dark_background')
fig, ax = plt.subplots()
color = ['r','y','w','g','b','c']

for typ in df['Type'].unique():
  x = df[df['Type'] == typ].Temperature
  y = df[df['Type'] == typ].A_M
  ax.scatter(x,y,c=color[typ])
ax.set_ylabel("Absolute Magnitude")
ax.set_xlabel("Temperature (K)")
plt.show()
```

![image](https://github.com/user-attachments/assets/9523ef1b-d50f-4add-b5f5-4eb229ff3109)

Considering bright and temperature, red dwarfs and brown dwarfs are close, while supergiants and hypergiants have this property too. White dwarfs and the Main Sequence are very highlighted as a defined group with this variables.

## Groupby queries and data cleansing
