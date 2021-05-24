import  pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv(r'D:\Desktop\Technologies\MachineLearning\Machine-Learning-in-90-days\Day2\train.csv')

print(df.head())
# eda
df.isnull()
# Checks The Missing Data
#  A heatmap for null values using seaborn 
sb.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#  Visualize the data better b bar Charts

# Survival Using Counter Plot , Checks who did not survive in Titanic
sb.set_style('whitegrid')
sb.countplot(x='Survived',data=df)
sb.set_style('whitegrid')
sb.countplot(x='Survived',hue='Sex',data=df)

#  This means further classifying the data based on Sex 
#  Also Checking which class of the Passedngers survived More
sb.set_style('whitegrid')
sb.countplot(x='Survived',hue='Pclass',data=df)
#  Distribution of age on titanice
sb.displot(df['Age'].dropna(),kde=False,bins=30)

# Number of people who has Sibling/Spouse
sb.countplot(x='SibSp',data=df)

#  Histogram of the Train fare
df['Fare'].hist(color='pink',bins=40,figsize=(8,4))

plt.figure(figsize=(12, 7))
sb.boxplot(x='Pclass',y='Age',data=df,palette='winter')
#  This will give a candle chart which will help us determine avergae values
# Then we will replace null with those average values, replacing with 0s and other few thumb rules have also been used empirically
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
#  Now we will see a much smoother heatmap as there are no null Values
