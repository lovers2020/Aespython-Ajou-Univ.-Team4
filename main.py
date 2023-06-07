import Keyword_Analisys as ka
import pandas as pd
import numpy as np

import folium
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# 키워드 분석
# ka.get_Keyword_Analisys()
#
# 주택 매매, 전세 가격
# 가족 구성원의 변경 (핵가족화)
# 출산율

# 각 파일 불러오기
nuptiality = pd.read_csv('혼인율.csv',skiprows=[0,1])
birthrate = pd.read_csv('출산율.csv', skiprows=[0,1])
apart_trade = pd.read_csv('주택매매가격.csv',skiprows=[0,1])  # skiprows -> 맨 위의 필요없는 행 삭제
apart_lease = pd.read_csv('주택전세가격.csv',skiprows=[0,1])
houseMember_rate = pd.read_csv('가구구성 변화율.csv',skiprows=[0,1])
plt.rc('font',family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 전처리
# 혼인율 파일
# 필요없는 행 삭제
for i in range(1,4):
    nuptiality = nuptiality.drop(index=i,axis=0)


#열 이름 변경 및 index를 항목으로 변경 / index 이름 변경
nuptiality = nuptiality.rename(columns={'Unnamed: 0' : '항목'}).set_index('항목')
nuptiality = nuptiality.set_index(pd.Index(['조혼인율']))

# 출산율 파일
# # 필요없는 행 삭제

for i in range(1,4):
    birthrate = birthrate.drop(index=i,axis=0)

# 열 이름을 항목으로 변경 / index를 항목으로 변경
birthrate = birthrate.rename(columns={'Unnamed: 0': '항목'}).set_index('항목')
birthrate = birthrate.set_index(pd.Index(['합계출산율']))

# # 주택매매가격 파일
for i in range(1,7):
    apart_trade = apart_trade.drop(index=i,axis=0)

# #열 이름을  지역으로 변경 / index를 지역으로 변경
apart_trade = apart_trade.rename(columns={'Unnamed: 0' : '지역'}).set_index('지역')
apart_trade = apart_trade.set_index(pd.Index(['전국매매가격']))


# 주택전세가격 파일
for i in range(1,7):
    apart_lease = apart_lease.drop(index=i,axis=0)

# 열 이름을  지역으로 변경 / index를 지역으로 변경
apart_lease = apart_lease.rename(columns={'Unnamed: 0' : '지역'}).set_index('지역')
apart_lease = apart_lease.set_index(pd.Index(['전국전세가격']))

# # 가구구성원 파일
# 필요없는 행 삭제
for i in range(7,10):
    houseMember_rate = houseMember_rate.drop(index=i,axis=0)

# 열 이름 변경 / index를 년도로 변경p
houseMember_rate = houseMember_rate.drop('Unnamed: 0', axis=1)
houseMember_rate = houseMember_rate.rename(columns={'Unnamed: 1' : '년도'})
houseMember_rate['년도']= houseMember_rate['년도'].fillna('평균가구원수')

houseMember_rate = houseMember_rate.set_index('년도')

df = pd.concat([nuptiality,birthrate,apart_trade,apart_lease,houseMember_rate], join='outer').reset_index()
df = df.rename(columns={'index' : ' '}).set_index(' ')

df = df.astype(float)
df = df.transpose()
df['평균가구원수'] = df['평균가구원수'].fillna(method='ffill')
df['1인가구'] = df['1인가구'].fillna(method='ffill')
df['2인가구'] = df['2인가구'].fillna(method='ffill')
df['3인가구'] = df['3인가구'].fillna(method='ffill')
df['4인가구'] = df['4인가구'].fillna(method='ffill')
df['5인가구'] = df['5인가구'].fillna(method='ffill')
df['6인이상가구'] = df['6인이상가구'].fillna(method='ffill')

corr = df.corr(method='pearson')
df = df.transpose()

#
# all_columns = list(set(df['항목']))
# year = list(map(str,([i for i in range(2000,2023)])))
# df2 = df[['항목']+year]
#
#
# for i in all_columns:
#     out = df2[df2.항목 == i]
#     out = out[year]
#
#     corr2 = out.corr()
#
#
#     pass
##################


# 히트맵
plt.figure(figsize=(12,12))
plt.title('혼인율 상관관계',fontsize= 20)
heatmap = sns.heatmap(corr,annot=True,linewidths=0.01,linecolor='skyblue',cbar=False) #cmap='RdYlBu'
sns.set_palette('pastel')
plt.yticks(rotation= 0)
plt.xticks(rotation= -10)

# 그래프
year = [i for i in range(2000,2023)]

plt.figure(figsize=(14,8))

plt.plot(year,df.iloc[0],'ro-',label="혼인율")
plt.plot(year,df.iloc[1],'ko-',label= "출산율")
plt.plot(year,df.iloc[2]/2,'y-',label= "전국매매가격")
plt.plot(year,df.iloc[3]/2,label= "전국전세가격")
plt.plot(year,df.iloc[5]/10,'bo-',label= "1인가구 수")

plt.legend()
plt.xlabel('년도')
plt.show()




























