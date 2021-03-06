# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 00:00:28 2022

@author: Ding Hua
"""
import pandas as pd
import zipfile



#%%  load i_cs

# path_cs is where the downloaded zip file is located in your computer
path_cs=r'C:\Users\wwwhd\OneDrive\桌面\PLCY782-International development economics\Empirical Exercise 2\data_hh02dta_b1.zip'
zfile = zipfile.ZipFile(path_cs)
print(zfile.namelist())
cs = zfile.open('hh02dta_b1/i_cs.dta')
cs1 = zfile.open('hh02dta_b1/i_cs1.dta')
df_cs=pd.read_stata(cs)
df_cs1=pd.read_stata(cs1)
print(df_cs.head(5))
print(df_cs1.head(5))
cs.close()
cs1.close()
zfile.close()


#%% load ls(family member) data

path_ls=r'C:\Users\wwwhd\OneDrive\桌面\PLCY782-International development economics\Empirical Exercise 1\hh02dta_bc.zip'
zfile2 = zipfile.ZipFile(path_ls)
print(zfile2.namelist())
ls = zfile2.open('hh02dta_bc/c_ls.dta')
df_ls = pd.read_stata(ls)
print(df_ls.head(5))
ls.close()
zfile2.close()


#%% prepare the consumption data
df_month=df_cs.filter(regex='cs16._2|folio')
print(df_month)
    # method 2:
    # df_cs.loc[:,df_cs.columns.str.match('folio|cs16._2')]


# load weekly data and transfrom it into montly data
df_week=df_cs.filter(regex='cs02.*2$|folio')
df_week.loc[:,'cs02a_12':'cs02e_32']*=4.3       #note: has a warning
    # method 2: df_week.loc[:,'cs02a_12':'cs02e_32']=df_week.loc[:,'cs02a_12':'cs02e_32'].apply(lambda x: x*4.3)

print(df_week)

# load 3-month data and transfrom it into montly data
df_3_month=df_cs1.filter(regex='cs22._2|folio')
df_3_month.loc[:,'cs22a_2':'cs22h_2']/=3        #note: has a warning
    # method 2: df_3_month.loc[:,'cs22a_2':'cs22h_2']=df_3_month.loc[:,'cs22a_2':'cs22h_2'].apply(lambda x: x/3)

print(df_3_month)


#%% merge all consumption data

df_cons=df_month.merge(df_week,on='folio',how='outer').merge(df_3_month,on='folio',how='outer')


# #replace NA with 0
df_cons=df_cons.fillna(0)

#%% Calculate total consumption, create a new column
df_cons['total_cons']=df_cons.loc[:,'cs16a_2':'cs22h_2'].sum(axis=1).round(2)
print(df_cons[['folio','total_cons']])


#%% Calculate per capita consumption

# merge consumption data with family member 
df_ls_uniquefolio=df_ls.groupby('folio').count()
df_ls_uniquefolio=df_ls_uniquefolio.loc[:,'ls']
df_cons_with_member = df_cons.merge(df_ls_uniquefolio, on='folio', how='left')


#rename 'ls' to 'family_members'
df_cons_with_member.rename(columns={'ls':'family_members'},inplace=True)


#Calculate per capita consumption
df_cons_with_member['cons_percapita']=df_cons_with_member['total_cons']/df_cons_with_member['family_members']
print(df_cons_with_member.loc[:,['folio','family_members','total_cons','cons_percapita']])


#%% poverty rate calculation

poverty_line=500

#calculate household poverty rate
df_cons_with_member['pov_dummy']=df_cons_with_member.cons_percapita.apply(lambda x: 1 if x<poverty_line else 0)

household_pov_rate = df_cons_with_member['pov_dummy'].mean()
# household_pov_rate1 = format(household_pov_rate,'.2%')
household_pov_rate = format(df_cons_with_member['pov_dummy'].mean(),'.2%')
print(f'The household poverty rate is {household_pov_rate}')


# calculate individual poverty rate
all_people = df_cons_with_member['family_members'].sum()
df_househould_in_pov = df_cons_with_member[df_cons_with_member['pov_dummy']==1]
individual_pov_rate = format(df_househould_in_pov['family_members'].sum()/all_people,'.2%')
print(f'The individual poverty rate is  {individual_pov_rate}')

# method 2:
# df_cons_with_member.groupby('pov_dummy').agg({'family_members':'sum'})


#%% calculate average poverty gap and average poverty gap squared
Gi=poverty_line-df_househould_in_pov['cons_percapita']

average_pov_gap=sum((Gi/poverty_line)*df_househould_in_pov['family_members'])/all_people
average_pov_gap_squared=sum(((Gi/poverty_line)**2*df_househould_in_pov['family_members']))/all_people

average_pov_gap=format(average_pov_gap,'.2%')
average_pov_gap_squared=format(average_pov_gap_squared,'.2%')

print(f'The average poverty gap is {average_pov_gap}')
print(f'The average poverty gap squared is {average_pov_gap_squared}')


#%% load data of resident area

# load portad (living area) data
path_area=r'C:\Users\wwwhd\OneDrive\桌面\PLCY782-International development economics\Empirical Exercise 1\hh02dta_bc.zip'
zfile2 = zipfile.ZipFile(path_area)
portad = zfile2.open('hh02dta_bc/c_portad.dta')
df_portad = pd.read_stata(portad)

portad.close()
zfile2.close()
df_portad = df_portad.loc[:,['folio','estrato']]


#%% calculate hosuehold poverty rate in different areas
 
#merge consumption data with portad data 
df_cons_all=df_cons_with_member.merge(df_portad,on='folio',how='left')

#create area dummy(1 if city; 0 if rural )
df_cons_all['area_dummy']=df_cons_all.estrato.apply(lambda x: 1 if x==1 or x==2 or x==3 else 0)


#calculate household poverty rate in city
df_cons_city=df_cons_all.loc[df_cons_all['area_dummy']==1]
city_household_pov_rate = format(df_cons_city['pov_dummy'].mean(),'.2%')
# city_household_pov_rate = df_cons_city['pov_dummy'].mean()

print(f'The city household poverty rate is  {city_household_pov_rate}')


# calculate individual poverty rate in city
city_all_people = df_cons_city['family_members'].sum()
df_city_househould_in_pov = df_cons_city[df_cons_city['pov_dummy']==1]
city_individual_pov_rate = format(df_city_househould_in_pov['family_members'].sum()/city_all_people,'.2%')
print(f'The city individual poverty rate is  {city_individual_pov_rate}')


#calculate household poverty rate in rural area                                                                                    
df_cons_rural=df_cons_all.loc[df_cons_all['area_dummy']==0]
rural_household_pov_rate = format(df_cons_rural['pov_dummy'].mean(),'.2%')
# rural_household_pov_rate = df_cons_rural['pov_dummy'].mean()
print(f'The rural household poverty rate is  {rural_household_pov_rate}')


# calculate individual poverty rate in rural area 
rural_all_people = df_cons_rural['family_members'].sum()
df_rural_househould_in_pov = df_cons_rural[df_cons_rural['pov_dummy']==1]
rural_individual_pov_rate = format(df_rural_househould_in_pov['family_members'].sum()/rural_all_people,'.2%')
print(f'The rural individual poverty rate is  {rural_individual_pov_rate}')


#%% calculate average poverty gap and average poverty gap squared in different areas


#calculate average poverty gap and average poverty gap squared in city
Gi_city=poverty_line-df_city_househould_in_pov['cons_percapita']

city_average_pov_gap=sum((Gi_city/poverty_line)*df_city_househould_in_pov['family_members'])/city_all_people
city_average_pov_gap_squared=sum(((Gi_city/poverty_line)**2*df_city_househould_in_pov['family_members']))/city_all_people

city_average_pov_gap=format(city_average_pov_gap,'.2%')
city_average_pov_gap_squared=format(city_average_pov_gap_squared,'.2%')

print(f'The average poverty gap in city is {city_average_pov_gap}')
print(f'The average poverty gap squared in city is {city_average_pov_gap_squared}')


# calculate average poverty gap and average poverty gap squared in rural area
Gi_rural=poverty_line-df_rural_househould_in_pov['cons_percapita']

rural_average_pov_gap=sum((Gi_rural/poverty_line)*df_rural_househould_in_pov['family_members'])/rural_all_people
rural_average_pov_gap_squared=sum(((Gi_rural/poverty_line)**2*df_rural_househould_in_pov['family_members']))/rural_all_people

rural_average_pov_gap=format(rural_average_pov_gap,'.2%')
rural_average_pov_gap_squared=format(rural_average_pov_gap_squared,'.2%')

print(f'The average poverty gap in rural area is {rural_average_pov_gap}')
print(f'The average poverty gap squared in rural area is {rural_average_pov_gap_squared}')


#%%  bar chart of poverty rate in different regions


import matplotlib.pyplot as plt
import matplotlib

label_list = ['all regions', 'urban area', 'rural area']    # label of x axis
num_list1 = [float(household_pov_rate.strip("%")), float(city_household_pov_rate.strip("%")), float(rural_household_pov_rate.strip("%"))]    # 纵坐标值1 household
num_list2 = [float(individual_pov_rate.strip("%")), float(city_individual_pov_rate.strip("%")), float(rural_individual_pov_rate.strip("%"))]     # 纵坐标值2  individual
x = range(len(num_list1))

rects1 = plt.bar(x, height=num_list1, width=0.4, alpha=0.8, color='red', label="household level")
rects2 = plt.bar(x=[i + 0.4 for i in x], height=num_list2, width=0.4, color='green', label="individual level")                                                                

plt.ylim(0, 100)       # value range of y axis
plt.ylabel("poverty rate(%)")        


plt.xticks([index + 0.2 for index in x], label_list)
plt.xlabel("Region")
plt.title("Poverty rate in different regions")
plt.legend()     

# edit the text on bars
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
plt.show()

#%% bar chart of Average Poverty Gap(Squared) in Different Regions

label_list1 = ['all region', 'urban area', 'rural area']    # label of x axis
num_lista = [float(average_pov_gap.strip("%")), float(city_average_pov_gap.strip("%")), float(rural_average_pov_gap.strip("%"))]      # 纵坐标值1  gap
num_listb = [float(average_pov_gap_squared.strip("%")), float(city_average_pov_gap_squared.strip("%")), float(rural_average_pov_gap_squared.strip("%"))]   # 纵坐标值2   gap squared
x = range(len(num_list1))

rectsa = plt.bar(x=x, height=num_lista, width=0.4, alpha=0.8, color='blue', label="gap")
rectsb = plt.bar(x=[i + 0.4 for i in x], height=num_listb, width=0.4, color='yellow', label="gap squared")                                                                

plt.ylim(0, 100)     # value range of y axis
plt.ylabel("avarege poverty gap(squared)(%)")        


plt.xticks([index + 0.2 for index in x], label_list1)
plt.xlabel("Region")
plt.title("Average Poverty Gap(Squared) in Different Regions")
plt.legend()     

# edit the text on bars
for rect in rectsa:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
for rect in rectsb:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
plt.show()                                                                                