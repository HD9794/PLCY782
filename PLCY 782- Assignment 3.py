# -*- coding = utf-8 -*-
# @Time : 2022/4/10 16:40
# @Author : 华定
# @File : PLCY 782- Assignment 3.py
# @Software: PyCharm

import pandas as pd
import zipfile
import numpy as np
import math
import matplotlib.pyplot as plt

#%% load & read data
filepath=r'C:\Users\wwwhd\OneDrive\桌面\PLCY782-International development economics\Empirical Exercise 3\hh02dta_b3a.zip'
datafile=zipfile.ZipFile(filepath)
print(datafile.namelist())
tb=datafile.open('hh02dta_b3a/iiia_tb.dta')
df_tb=pd.read_stata(tb)
tb.close()

#%% create df
# create dummy of df_tb
df_tb['dummy_lfp']=np.where(df_tb['tb02_1'].isna(),np.nan, np.where(df_tb['tb02_1']==1, 1,0)) #warning: DataFrame is highly fragmented...
# method2: df_tb['dummy_lfp']=df_tb['tb02_1'].apply(lambda x: 1 if x==1 else (np.nan if math.isnan(x) else 0))
df_tb['dummy_lfp'].isnull().value_counts()
df_tb['dummy_lfp'].unique()



#import age and gender data
age=datafile.open('hh02dta_b3a/iiia_portad.dta')
gender=datafile.open('hh02dta_b3a/iiia_hm.dta')

df_age=pd.read_stata(age)
df_gender=pd.read_stata(gender)

age.close()
gender.close()


#rename age and gender columns
df_age.rename(columns={"edad":"age"},inplace=True)
df_gender.rename(columns={"hm16":"gender"},inplace=True)

#merge tb with age and gender data
df=df_tb.merge(df_age,on=["folio","ls"]).merge(df_gender,on=["folio","ls"])


#create male dummy
df['gender'].isnull().value_counts()
df["dummy_male"]=np.where(df["gender"].isna(),np.nan,np.where(df["gender"]==1, 1,0))


#%% Calculate labor market participation rate of men and women aged 16 to 65
df_16to65_gender=df[(df["age"]>=16) & (df["age"]<65)& (df["dummy_lfp"].notnull())]

df_gender_marketparticipation=df_16to65_gender.groupby("gender")["dummy_lfp"].mean().reset_index()
df_gender_marketparticipation.columns=["gender","labor_market_participation"]
print(df_gender_marketparticipation)


#create bar chart for comparison
gender=['men','women']
gender_participation=df_gender_marketparticipation["labor_market_participation"]
plt.bar(['men','women'],gender_participation,color = ['blue','green'])
plt.xlabel("gender")
plt.ylabel("labor_market_participation")
plt.title("labor market participation rate of men and women aged 16 to 65")
for a, b in zip(gender, gender_participation):
    plt.text(a, b, format(b,".2%"), ha='center', va='bottom', fontsize=8)
plt.show()


#%% Calculate labor market participation of indigenous versus non indigenous individuals aged 16 to 65

#import indigenous data
indigenous=datafile.open('hh02dta_b3a/iiia_ed.dta')
df_indigenous=pd.read_stata(indigenous)

indigenous.close()
datafile.close()

#create indigenous dummy
df_indigenous["ed03"].unique()
df_indigenous["dummy_indigenous"]=np.where(df_indigenous["ed03"].isnull(),np.nan,np.where(df_indigenous['ed03']==1, 1,0))
df_indigenous["dummy_indigenous"].isnull().value_counts()

#merge with indigenous data
df=df.merge(df_indigenous,on=["folio","ls"])
df_16to65_indigenous=df[(df["age"]>=16) & (df["age"]<65) & (df["dummy_indigenous"].notnull())]

#calculate labor particiapation rate in indigenous and non indigenous group
df_indigenous_marketparticipation=df_16to65_indigenous.groupby("dummy_indigenous")["dummy_lfp"].mean().reset_index()
df_indigenous_marketparticipation.columns=["group","labor_market_participation"]
print(df_indigenous_marketparticipation)


#create bar chart for comparison
group=['non-indigenous','indigenous']
group_pariticipation=df_indigenous_marketparticipation["labor_market_participation"]

plt.bar(group,group_pariticipation,color = ['blue','green'])
plt.xlabel("living group")
plt.ylabel("labor_market_participation")
plt.title("labor market participation of indigenous versus non indigenous individuals")
for a, b in zip(group, group_pariticipation):
    plt.text(a, b, format(b,".2%"), ha='center', va='bottom', fontsize=8)

plt.show()


#%% concstruct variable of TOTAL EARNINGS
df_wage = df_tb[["folio", "ls", "tb35a_2"]]
df_wage.rename(columns={"tb35a_2":"total_consum_final"},inplace=True) #has a warning:A value is trying to be set on a copy of a slice from a DataFrame.

df_hours = df_tb[["folio", "ls", "tb27p"]]
df_hours.rename(columns={"tb27p":"hrs_week"},inplace=True) #has a warning
df_hours["hrs_month"]=df_hours["hrs_week"]*4.3             #has a warning

# Merge with wage data
df_wage=df_wage.merge(df_hours,on=["folio", "ls"])
df=df.merge(df_wage,on=["folio", "ls"])


#%% concstruct variable of levels of schooling
df["ed06"].value_counts()
df["edu_primary_less"]=np.where((df["ed06"].isnull())|(df["ed06"]==98),np.nan, np.where((df["ed06"]==2)|(df["ed06"]==3),1,0))
df["edu_secondary"]=np.where((df["ed06"].isnull())|(df["ed06"]==98),np.nan, np.where((df["ed06"]>3) & (df["ed06"]<11), 1,0))


#%% Question 3
# 3.1 What are the average and median labor market earnings for women and men in the previous month from their main job?
df1=df.dropna(subset='total_consum_final')
df1['hourly_wage']=df1['total_consum_final']/df1['hrs_month']   #warning: A value is trying to be set on a copy of a slice from a DataFrame.
df1.replace([np.inf, -np.inf], np.nan,inplace=True)
df1.dropna(subset='hourly_wage',inplace=True)

##monthly earning
df_ave_monthly_earning_gender=df1.groupby('gender')['total_consum_final'].mean().reset_index()
df_ave_monthly_earning_gender['gender'].replace({1:"men",3:"women"},inplace=True)
df_ave_monthly_earning_gender.rename(columns={'total_consum_final':'average_monthly_earning'},inplace=True)
print(df_ave_monthly_earning_gender)

df_median_monthly_earning_gender=df1.groupby('gender')['total_consum_final'].median().reset_index()
df_median_monthly_earning_gender['gender'].replace({1:"men",3:"women"},inplace=True)
df_median_monthly_earning_gender.rename(columns={'total_consum_final':'median_monthly_earning'},inplace=True)
print(df_median_monthly_earning_gender)

##weekly earning
df_ave_weekly_earning_gender=df1.groupby('gender')['hourly_wage'].mean().reset_index()
df_ave_weekly_earning_gender['gender'].replace({1:"men",3:"women"},inplace=True)
df_ave_weekly_earning_gender.rename(columns={'hourly_wage':'average_weekly_earning'},inplace=True)
print(df_ave_weekly_earning_gender)

df_median_weekly_earning_gender=df1.groupby('gender')['hourly_wage'].median().reset_index()
df_median_weekly_earning_gender['gender'].replace({1:"men",3:"women"},inplace=True)
df_median_weekly_earning_gender.rename(columns={'hourly_wage':'median_weekly_earning'},inplace=True)
print(df_median_weekly_earning_gender)


#%%
# 3.2 What are the average and median labor market earnings for indigenous and non-indigenous individuals in the previous month from their main job?

##monthly earning
df_ave_monthly_earning_indig=df1.groupby('dummy_indigenous')['total_consum_final'].mean().reset_index()
df_ave_monthly_earning_indig['dummy_indigenous'].replace({0:"non-indigenous",1:"indigenous"},inplace=True)
df_ave_monthly_earning_indig.rename(columns={'total_consum_final':'average_monthly_earning'},inplace=True)
print(df_ave_monthly_earning_indig)

df_median_monthly_earning_indig=df1.groupby('dummy_indigenous')['total_consum_final'].median().reset_index()
df_median_monthly_earning_indig['dummy_indigenous'].replace({0:"non-indigenous",1:"indigenous"},inplace=True)
df_median_monthly_earning_indig.rename(columns={'total_consum_final':'median_monthly_earning'},inplace=True)
print(df_median_monthly_earning_indig)

##weekly earning
df_ave_weekly_earning_indig=df1.groupby('dummy_indigenous')['hourly_wage'].mean().reset_index()
df_ave_weekly_earning_indig['dummy_indigenous'].replace({0:"non-indigenous",1:"indigenous"},inplace=True)
df_ave_weekly_earning_indig.rename(columns={'hourly_wage':'average_weekly_earning'},inplace=True)
print(df_ave_weekly_earning_indig)

df_median_weekly_earning_indig=df1.groupby('dummy_indigenous')['hourly_wage'].median().reset_index()
df_median_weekly_earning_indig['dummy_indigenous'].replace({0:"non-indigenous",1:"indigenous"},inplace=True)
df_median_weekly_earning_indig.rename(columns={'hourly_wage':'median_weekly_earning'},inplace=True)
print(df_median_weekly_earning_indig)


#%%
# 3.3 Regression
#Carry out a regression of earnings per hour as a function: of age, education, gender and indigenous status
### Create separate data frame

df2=df1.dropna(subset=['age','dummy_male','dummy_indigenous','edu_secondary'])
reg=df2[['hourly_wage','age','dummy_male','dummy_indigenous','edu_secondary']]


import seaborn as sns
sns.pairplot(reg, x_vars=['age','dummy_male','dummy_indigenous','edu_secondary'], y_vars='hourly_wage', height=7, aspect=0.8,kind='reg')


#%% sklearn training

feature_cols=['age','dummy_male','dummy_indigenous','edu_secondary']

X = reg[feature_cols]
y = reg['hourly_wage']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()   #create a model

linreg.fit(X_train, y_train)  #training

print (linreg.intercept_)     #intercept

for a,b in zip(feature_cols, linreg.coef_):
    print(a, b)              #coefficient


## calculate mean squared error(MSE) to verify the model
y_pred = linreg.predict(X_test)

from sklearn import metrics
print (np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


## Conduct OLS Regression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as sm

linear_model = sm.ols(formula='hourly_wage ~ age+dummy_male + dummy_indigenous + edu_secondary', data=reg)
results = linear_model.fit()
print(results.summary())
