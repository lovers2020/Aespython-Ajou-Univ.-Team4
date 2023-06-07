import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 전처리
married = pd.read_csv('연령별 혼인률 데이터.csv',encoding='cp949',index_col=0)
income = pd.read_csv('연령별 소득 데이터.csv',encoding='cp949',index_col=0)

# seaborn 한글 깨짐 방지를 위한 글꼴 설정
plt.rcParams['font.family'] = 'Malgun gothic'

df = pd.concat([married,income],axis=1)
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
