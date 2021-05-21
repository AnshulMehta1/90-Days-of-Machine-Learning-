import pandas as pd 
import numpy as np
# Jo karvu hoy to dict ke json same j thaay
# Data can be converted to JSON
Data=('col1,col2,col3\n'
'1,a,x\n'
'2,b,y\n'
'3,c,z\n')
df=pd.DataFrame(np.arange(0,20).reshape(5,4),index=['Row1','Row2','Row3','Row4','ROW5'],columns=['COL1','COL2','COL3','COL4'])
print(df.to_json(orient='records'))
url='https://www.fdic.gov/bank/individual/failed/banklist.html'
dfs=pd.read_html(url)
print(dfs[0])