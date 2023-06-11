import Keyword_Analisys as ka
import pandas as pd
import numpy as np
import folium
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

'''
# 키워드 분석
ka.get_Keyword_Analisys()
'''

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 각 파일 불러오기 # skiprows -> 맨 위의 필요없는 행 삭제
nuptiality = pd.read_csv('혼인율.csv', skiprows=[0, 1])
birthrate = pd.read_csv('출산율.csv', skiprows=[0, 1])
apart_trade = pd.read_csv('주택매매가격.csv',skiprows=[0, 1])
apart_lease = pd.read_csv('주택전세가격.csv',skiprows=[0, 1])
houseMember_rate = pd.read_csv('가구구성 변화율.csv', skiprows=[0, 1])
house_supply = pd.read_csv('주택보급률.csv', engine='python', skiprows=[0, 1])
income = pd.read_csv('1인당국민총소득 2000~2022.csv', encoding='cp949')
job_rate = pd.read_csv('취업,실업률.csv', engine='python', skiprows=[0, 1])
economic_pop = pd.read_csv('경제활동인구수sv.csv', engine='python')
price_rate = pd.read_csv('소비자물가지수.csv', engine='python', skiprows=[0 ,1])
base_rate = pd.read_csv('한국은행 기준금리.csv', engine='python')

############################## 데이터 전처리 ##################################
# 혼인율 파일 # 필요없는 행 삭제

nuptiality = nuptiality.drop(labels=range(1, 4))

#열 이름 변경 및 index를 항목으로 변경 / index 이름 변경
nuptiality = nuptiality.rename(columns={'Unnamed: 0' : '항목'}).set_index('항목')
nuptiality = nuptiality.set_index(pd.Index(['조혼인율']))

# 출산율 파일 # 필요없는 행 삭제
birthrate = birthrate.drop(labels=range(1, 4))

# 열 이름을 항목으로 변경 / index를 항목으로 변경
birthrate = birthrate.rename(columns={'Unnamed: 0': '항목'}).set_index('항목')
birthrate = birthrate.set_index(pd.Index(['합계출산율']))

# 주택매매가격 파일# 필요없는 행 삭제
apart_trade = apart_trade.drop(labels=range(1, 7))

# #열 이름을  지역으로 변경 / index를 지역으로 변경
apart_trade = apart_trade.rename(columns={'Unnamed: 0' : '지역'}).set_index('지역')
apart_trade = apart_trade.set_index(pd.Index(['전국매매가격']))

# 주택전세가격 파일 # 필요없는 행 삭제
apart_lease = apart_lease.drop(labels=range(1, 7))

# 열 이름을  지역으로 변경 / index를 지역으로 변경
apart_lease = apart_lease.rename(columns={'Unnamed: 0' : '지역'}).set_index('지역')
apart_lease = apart_lease.set_index(pd.Index(['전국전세가격']))

# 가구구성원 파일# 필요없는 행 삭제
houseMember_rate = houseMember_rate.drop(labels=range(7, 10))

# 열 이름 변경 / index를 년도로 변경p
houseMember_rate = houseMember_rate.drop('Unnamed: 0', axis=1)
houseMember_rate = houseMember_rate.rename(columns={'Unnamed: 1': '년도'})
houseMember_rate['년도'] = houseMember_rate['년도'].fillna('평균가구원수')
houseMember_rate = houseMember_rate.set_index('년도')

# 소득 파일
income = income.drop(1, axis=0)
income = income.set_index('계정항목별')
income = income.rename(index={'1인당 국민총소득(명목, 원화표시) (만원)': '1인당 국민총소득', '국내총생산(실질성장률) (%)': '국내 총생산', '국내총소득 (%)': '국내 총소득', '국민총소득 (%)': '국민 총소득', '최종소비지출(실질증감률) (%)': '최종소비지출', '전국혼인율 (%)': '혼인율'})

# 주택 보급률
house_supply = house_supply.drop(columns=['Unnamed: 1','20102)월']).drop(labels=range(1, 21), axis=0)
house_supply = house_supply.rename(columns={'Unnamed: 0': '지역'}).set_index('지역')
house_supply = house_supply.rename({'전국': '주택보급률'})

for i in range(2004, 1999,-1):
    house_supply.insert(0, str(i), house_supply['2005'])
house_supply['2022'] = house_supply['2021']

# 취업/ 실업자
job_rate = job_rate.drop(labels=range(1, 5), axis=0).drop(labels=range(8, 13), axis=0)
job_rate = job_rate.drop(6,axis=0).rename(columns={'Unnamed: 0': '항목'}).set_index('항목')
job_rate = job_rate.astype(float)

# 경제활동 인구 수 (취업률 구하기 위함)
economic_pop = economic_pop.drop(0, axis=0).rename(columns={'성별': '계'}).set_index('계').rename({'계': '경제활동인구'}, axis=0)
economic_pop = economic_pop.astype(int)


df2 = (job_rate.iloc[0]*10000) / (economic_pop.iloc[0] * 1000) * 100
df2 = pd.DataFrame(df2).transpose()
df2.iloc[0] = round(df2.iloc[0], 2)
df2.insert(0, '취업률', 0)
df2 = df2.set_index('취업률').rename({0: '취업률'})

job_rate = job_rate.drop(index=job_rate[job_rate['2000'] == 88.2].index)
job_rate = pd.concat([job_rate, df2], join='outer')
job_rate = job_rate.rename(index={'실업률(%)': '실업률', '청년실업률(%)': '청년실업률'})
# 취업/ 실업률 합치기 완료

#소비자 물가지수 파일

price_rate = price_rate.drop(labels=range(2, 5)).rename(columns={'Unnamed: 0': '항목'}).set_index('항목')
price_rate = price_rate.rename(index= {'소비자물가상승률 2)': '물가 상승률'})
price_rate = price_rate.drop(index=price_rate[price_rate['2000'] == '63.151'].index)

# 기준금리 파일
base_rate = base_rate.set_index('단위')
base_rate = base_rate.rename(index={'연%': '기준금리'})

########################### 데이터 합치고 상관계수 구하기 ########################
#기준금리 파일은 출처 한국은행 나머지는 통계청
#혼인율, 출산율, 주택매매가격, 주택전세가격, 가구구성원, 국민총소득, 주택보급률, 취업/실업률, 물가지수, 기준금리
# 청년실업자 및 청년실업률 연령 기준 : 15~29세
df = pd.concat([nuptiality, birthrate, apart_trade, apart_lease, houseMember_rate, income,house_supply, job_rate, price_rate, base_rate], join='outer').reset_index()
df = df.rename(columns={'index': ' '}).set_index(' ')

df = df.astype(float)
df = df.T
df['평균가구원수'] = df['평균가구원수'].fillna(method='ffill')
df['1인가구'] = df['1인가구'].fillna(method='ffill')
df['2인가구'] = df['2인가구'].fillna(method='ffill')
df['3인가구'] = df['3인가구'].fillna(method='ffill')
df['4인가구'] = df['4인가구'].fillna(method='ffill')
df['5인가구'] = df['5인가구'].fillna(method='ffill')
df['6인이상가구'] = df['6인이상가구'].fillna(method= 'ffill')

corr = df.corr(method='pearson')
'''
###################################### 석태형 코드 ###############################
'''

# 정권별 분석
data = pd.read_csv('혼인율 1988~2023.csv', encoding='cp949')
data = data.replace('-', np.nan)
'''
1988-02-25~1993-02-24	노태우	진보
#1993-02-25 ~ 1998-02-24	김영삼	보수
#1993-02-25 ~ 2003-02-24	김대중	진보
#2003-02-25 ~ 2004-03-12	노무현	진보
#2004-05-14 ~ 2008-02-24		
#2008-02-25 ~ 2013-02-24	이명박	보수
#2013-02-25~ 2016-12-09	박근혜	보수
#2016-12-09~2017-05-10	황교안	보수
#2017-05-10 ~ 2022-05-09	문재인	진보
#2022-05-10 ~ Ing	윤석열	보수
'''

#1988-02-25~1993-02-24	노태우	진보
TW = pd.concat([data["행정구역별(1)"], data.iloc[1:, 1:6]], axis=1)
TW = TW.drop(0).reset_index(drop=True)

#1993-02-25 ~ 1998-02-24	김영삼	보수
YS = pd.concat([data["행정구역별(1)"], data.iloc[1:, 6:11]], axis=1)
YS = YS.drop(0).reset_index(drop=True)

# 1993-02-25 ~ 2003-02-24	김대중	진보
DJ = pd.concat([data["행정구역별(1)"], data.iloc[1:, 11:16]], axis=1)
DJ = DJ.drop(0).reset_index(drop=True)

#2003-02-25 ~ 2004-03-12, 2004-05-14 ~ 2008-02-24	노무현	진보

MH = pd.concat([data["행정구역별(1)"], data.iloc[1:, 16:21]], axis=1)
MH = MH.drop(0).reset_index(drop=True)

#2008-02-25 ~ 2013-02-24	이명박	보수
MB = pd.concat([data["행정구역별(1)"], data.iloc[1:, 21:26]], axis=1)
MB = MB.drop(0).reset_index(drop=True)

#2013-02-25~ 2016-12-09	박근혜	보수
GH = pd.concat([data["행정구역별(1)"], data.iloc[1:, 26:30]], axis=1)
GH = GH.drop(0).reset_index(drop=True)

#2016-12-09~2017-05-10	황교안	보수
KA = pd.concat([data["행정구역별(1)"], data.iloc[1:, 30:31]], axis=1)
KA = KA.drop(0).reset_index(drop=True)

#2017-05-10 ~ 2022-05-09	문재인	진보
MUN = pd.concat([data["행정구역별(1)"], data.iloc[1:, 31:35]], axis=1)
MUN = MUN.drop(0).reset_index(drop=True)

#2022-05-10 ~ Ing	윤석열	보수
SY = pd.concat([data["행정구역별(1)"], data.iloc[1:, 35]], axis=1)
SY = SY.drop(0).reset_index(drop=True)

# 정권별 혼인율 그래프

# TW.iloc[:, 1:6] = TW.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce')
# YS.iloc[:, 1:6] = YS.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce')
# DJ.iloc[:, 1:6] = DJ.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce')
# MH.iloc[:, 1:6] = MH.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce')
# MB.iloc[:, 1:6] = MB.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce')
# GH.iloc[:, 1:6] = GH.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce')
# KA.iloc[:, 1:6] = KA.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce')
# MUN.iloc[:, 1:6] = MUN.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce')
# SY.iloc[:, 1:6] = SY.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce')
#
# # 그래프
# plt.figure(figsize=(14,8))
#
# plt.plot(TW['행정구역별(1)'], TW.iloc[:, 1:6].sum(axis=1), label='TW')
# plt.plot(YS['행정구역별(1)'], YS.iloc[:, 1:6].sum(axis=1), label='YS')
# plt.plot(DJ['행정구역별(1)'], DJ.iloc[:, 1:6].sum(axis=1), label='DJ')
# plt.plot(MH['행정구역별(1)'], MH.iloc[:, 1:6].sum(axis=1), label='MH')
# plt.plot(MB['행정구역별(1)'], MB.iloc[:, 1:6].sum(axis=1), label='MB')
# plt.plot(GH['행정구역별(1)'], GH.iloc[:, 1:6].sum(axis=1), label='GH')
# plt.plot(KA['행정구역별(1)'], KA.iloc[:, 1:6].sum(axis=1), label='KA')
# plt.plot(MUN['행정구역별(1)'], MUN.iloc[:, 1:6].sum(axis=1), label='MUN')
# plt.plot(SY['행정구역별(1)'], SY.iloc[:, 1:6].sum(axis=1), label='SY')
#
# plt.xlabel('임기&지역')
# plt.ylabel('혼인건수',rotation= 0)
# plt.yticks(rotation= 0)
# plt.xticks(rotation= -20)
# plt.title('대통령 별 혼인건수 추이',fontsize= 20)
# plt.legend()
#
# # 상관관계 히트맵
# plt.figure(figsize=(21,12))
# plt.title('Correlation Matrix',fontsize= 20)
# heatmap = sns.heatmap(corr,annot=True,linewidths=0.01,linecolor='skyblue',cbar=False,cmap='RdYlBu') #cmap='RdYlBu'
# plt.yticks(rotation= 0)
# plt.xticks(rotation= -20)
#
# list = list(df.columns)
#
# list_2 = ['조혼인율', '합계출산율', '전국매매가격', '전국전세가격', '평균가구원수', '1인가구', '2인가구']
# list_3 = ['조혼인율','1인당 국민총소득', '국내 총생산', '국내 총소득', '국민 총소득', '최종소비지출', '주택보급률']
# list_4 = ['조혼인율','실업률', '청년실업률', '취업률', '물가 상승률', '기준금리']
#
# # 산점도
# sns.pairplot(data=df[list_2], kind="scatter", diag_kind="kde")
# sns.pairplot(data=df[list_3], kind="scatter", diag_kind="kde")
# sns.pairplot(data=df[list_4], kind="scatter", diag_kind="kde")

# plt.show()

######################### 회귀 분석 ###################

x = df.drop(['조혼인율'], axis= 1, inplace= False).values

y = df.조혼인율.values

# print(x)

result = sm.OLS(y,x).fit().summary()
# print(result)

lr = LinearRegression()

# print(x,y)

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
list = list(df.columns)


# model = sm.OLS.from_formula("조혼인율 ~ 합계출산율 + 전국매매가격 + 전국전세가격 + 평균가구원수",data=df)
# model2 = sm.OLS.from_formula("조혼인율 ~  1인가구 + 2인가구 + 3인가구 + 4인가구 + 5인가구 + 6인이상가구 + 1인당 국민총소득 + 국내 총생산 + 국내 총소득 + 국민 총소득 + 최종소비지출 + 주택보급률 + 실업률 + 청년실업률 + 취업률 + 물가 상승률 + 기준금리",data=df)
# result2 = model2.fit().summaty()
# result = model.fit().summary()
# print(result)
# print(result2)

#
# y_predict = lr.predict(x_test)
# mse = mean_squared_error(y_test,y_predict)
# rmse = np.sqrt(mse)
#
# print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
# print('R^2(Variance score) : {0:.3f}'.format(r2_score(y_test, y_predict)))
#
# print('회귀 계수 값: ',np.round(lr.coef_, 2))
#
# coef = pd.Series(data= np.round(lr.coef_,2), index=x.columns)
# coef.sort_values(ascending=False)
# # print(coef)
#
# x_train2 = sm.add_constant(x_train)
# model = sm.OLS(y_train,x_train2)
# result = model.fit().summary()
# print(result)



