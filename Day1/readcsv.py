import pandas as pd
from io import BytesIO, StringIO
import numpy as np
df=pd.DataFrame(np.arange(0,20).reshape(5,4),index=['Row1','Row2','Row3','Row4','ROW5'],columns=['COL1','COL2','COL3','COL4'])
print(df)
# df.to_csv('Index.csv')
#  To access a series with row 
a=df.loc['Row1']
print (a)
c=df.iloc[0,1]
print(c)
# To get some rows and some colums for
d=df.iloc[0:2,0:2]
print (d)
# To Make dataFrames into Arrrays
array=df.iloc[:,:].values
print(array)
# To chjeck condition of null values
print(df.isnull().sum())
#  To check value counts and unique counts
print(df['COL3'].unique())
#  TO print count values
# print(df['COL1']).value_counts()\

df1=pd.read_csv('mercedesbenz.csv')
# print(df1.head())
#  Sperator can be specified by putting second argument as the seprator 
print(df1['X0'].value_counts())
#  TO check conditonal values and thing
print(df1['y']>100)
#  To Only disp;llay those values use this as a argument/ condition
print(df1[df1['y']>180])
Data=('col1,col2,col3\n'
'1,a,x\n'
'2,b,y\n'
'3,c,z\n')
type(Data)
pd.read_csv(StringIO(Data))
df=pd.read_csv(StringIO(Data),usecols=lambda x: x.upper() in ['COL1', 'COL3'])
df.to_csv('test.csv')

df2=pd.read_csv('test.csv')
print(df2.head())


