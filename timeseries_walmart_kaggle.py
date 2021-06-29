# code based on Caio's noteobok 👏 https://www.kaggle.com/avelinocaio/walmart-store-sales-forecasting

import pandas as pd 
path= "./data/"
csv_features= "features.csv.zip"
csv_stores= "stores.csv" 
csv_train= "train.csv.zip"
csv_test= "test.csv.zip"

## EDA and cleaning

### Merge two datasets: features and stores
#### - the common key/column: Stores

features= pd.read_csv(path+csv_features)
stores= pd.read_csv(path+csv_stores)
train= pd.read_csv(path+csv_train)
test= pd.read_csv(path+csv_test)

feat_stor= features.merge(stores,how="inner",on="Store")
feat_stor.head(1)

#### Check dtypes
feat_stor.dtypes

#### Convert `Date` dtype to datetime
feat_stor.Date= pd.to_datetime(feat_stor.Date)
train.Date= pd.to_datetime(train.Date)
test.Date= pd.to_datetime(test.Date)

### ==================================================
### Add new columns: `Year`,`Week`
feat_stor["Year"]= feat_stor.Date.dt.isocalendar().year
feat_stor["Week"]= feat_stor.Date.dt.isocalendar().week

### ==================================================
### Merge store details with train/test
train_detail= train.merge(feat_stor,how="inner",on=["Store","Date","IsHoliday"]).sort_values(by=["Store","Dept","Date"]).reset_index(drop=True)

test_detail= test.merge(feat_stor,how="inner",on=["Store","Date","IsHoliday"]).sort_values(by=["Store","Dept","Date"]).reset_index(drop=True)

# del features,stores,train,test

### ==================================================
### Check null values
cols_null= (train_detail.isnull().sum()/train_detail.shape[0]).sort_values(ascending=False).index # returns the column names MarkDown2 (index) 0.736110 (sorted_col)

null_data_df= pd.concat([
    train_detail.isna().sum(),
    (train_detail.isna().sum()/train_detail.shape[0]).sort_values(ascending=False),
    train_detail.loc[:,train_detail.columns.isin(list(cols_null))].dtypes],axis=1)
null_data_df= null_data_df.rename(columns={
    0:"# null",
    1:"% null",
    2:"type"}).sort_values(ascending=False,by="% null")
null_data_df= null_data_df[null_data_df["# null"]!=0]
null_data_df

### ==================================================
### Find the week number and day that the four major holidays fall on
from pandasql import sqldf
pysql= lambda q: sqldf(q,globals())

pysql(
    """SELECT T.*,
    case 
    when ROW_NUMBER() OVER(partition by Year order by Week) = 1 then 'Super Bowl'
    when ROW_NUMBER() OVER(partition by Year order by Week) = 2 then 'Labor Day'
    when ROW_NUMBER() OVER(partition by Year order by Week) = 3 then 'Thanksgiving'
    when ROW_NUMBER() OVER(partition by Year order by Week) = 4 then 'Christmas'
    end as Holiday,
    case
    when ROW_NUMBER() OVER(partition by Year order by Week) = 1 then 'Sunday'
    when ROW_NUMBER() OVER(partition by Year order by Week) = 2 then 'Monday'
    when ROW_NUMBER() OVER(partition by Year order by Week) = 3 then 'Thursday'
    when ROW_NUMBER() OVER(partition by Year order by Week) = 4 and Year = 2010 then 'Saturday'
    when ROW_NUMBER() OVER(partition by Year order by Week) = 4 and Year = 2011 then 'Sunday'
    when ROW_NUMBER() OVER(partition by Year order by Week) = 4 and Year = 2012 then 'Thuesday'
    end as Day
    from 
        (SELECT DISTINCT Year,Week,
            case when Date <= '2012-11-01' then 'Train Data' else 'Test Data' end as Data_type
        FROM feat_stor
        WHERE IsHoliday = True) as T""")

### ==================================================
### Check average weekly sales per year
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
plt.style.use("ggplot")
sns.set_style("darkgrid") #whitegrid
weekly_sales_2010= train_detail[train_detail.Year==2010].Weekly_Sales.groupby(train_detail.Week).mean()
weekly_sales_2011= train_detail[train_detail.Year==2011].Weekly_Sales.groupby(train_detail.Week).mean()
weekly_sales_2012= train_detail[train_detail.Year==2012].Weekly_Sales.groupby(train_detail.Week).mean()
plt.figure(figsize=(10,5))
sns.lineplot(x=weekly_sales_2010.index,y=weekly_sales_2010.values)
sns.lineplot(x=weekly_sales_2011.index,y=weekly_sales_2011.values)
sns.lineplot(x=weekly_sales_2012.index,y=weekly_sales_2012.values)
plt.xticks(np.arange(1,53))
plt.legend(["2010","2011","2012"],loc="best",fontsize=11)
plt.title("Average Weekly Sales, 2010-2012",fontsize=12)
plt.ylabel("sales")
plt.xlabel("week")
plt.tight_layout()
plt.grid()
plt.show()

### ==================================================
### Change IsHoliday flag to true for Easter week
train_detail.loc[(train_detail.Year==2010)&(train_detail.Week==13),"IsHoliday"]= True
train_detail.loc[(train_detail.Year==2011)&(train_detail.Week==16),"IsHoliday"]= True
train_detail.loc[(train_detail.Year==2012)&(train_detail.Week==14),"IsHoliday"]= True
test_detail.loc[(test_detail.Year==2013)&(test_detail.Week==13),"IsHoliday"]= True

### ==================================================
### Sales median and mean values
### - When data is significantly *skewed*—i.e., the distribution deviates from or distorts the symmetrical bell curve—then the median becomes far more representative of what’s ‘typical’ than the mean because it splits the data exactly in two, with 50% being above the median and 50% below.
weekly_sales_mean= train_detail.Weekly_Sales.groupby(train_detail.Date).mean()
weekly_sales_median= train_detail.Weekly_Sales.groupby(train_detail.Date).median()
plt.figure(figsize=(10,5))
sns.lineplot(x=weekly_sales_mean.index,y=weekly_sales_mean.values)
sns.lineplot(x=weekly_sales_median.index,y=weekly_sales_median.values)
plt.grid()
plt.legend(["Mean","Median"])
plt.title("Weekly Sales: Mean and Median")
plt.ylabel("Sales")
plt.xlabel("Date")
plt.tight_layout()
plt.show()

### ==================================================
### Average sales per store
weekly_sales= train_detail.Weekly_Sales.groupby(train_detail.Store).mean()
plt.figure(figsize=(10,5))
sns.barplot(x=weekly_sales.index,y=weekly_sales.values,palette="winter") # spring, summer, autumn, winter,gnuplot2, copper,Blues_d,Spectral
plt.title("Average Sales per Store")
plt.ylabel("Sales")
plt.xlabel("Store")
plt.show()

### ==================================================
### Average sales per department
weekly_sales= train_detail.Weekly_Sales.groupby(train_detail.Dept).mean()
plt.figure(figsize=(18,5))
sns.barplot(x=weekly_sales.index,y=weekly_sales.values,palette="copper_r")
plt.title("Average Sales per Dept")
plt.ylabel("Sales")
plt.xlabel("Dept")
plt.show()

### ==================================================
### Feature correlation
### - Pearson correlation
###     * 0: none
###     * [0,.3]: weak
###     * [.3,.7]: moderate
###     * [.7,1]: strong
sns.set_style("whitegrid")
corr= round(train_detail.corr(),2)
mask= np.triu(np.ones_like(corr,dtype=np.bool))
fig,ax= plt.subplots(figsize=(12,8))
cmap= sns.diverging_palette(240, 10,as_cmap=True)
plt.title("Correlation Matrix")
sns.heatmap(corr,mask=mask,cmap=cmap,vmax=.3,center=0,square=True,linewidths=.5,cbar_kws={"shrink":.5},annot=True)
plt.show()

### ==================================================
### Drop insignificant columns
### - MarkDown1,2,3,4,5 have very little correlation with sales
### - Fuel_Price is also linearly dependent on Year
train_detail.drop(columns=["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5","Fuel_Price"],inplace=True)
test_detail.drop(columns=["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5","Fuel_Price"],inplace=True)

## EDA
### - For discrete variables: boxplot, stripplot
### - For continuous variables: boxcox, scatterplot
from matplotlib.gridspec import GridSpec
def plot_discrete(series):
    fig= plt.figure(figsize=(12,5))
    gs= GridSpec(1, 2) #row 1,col 2
    # boxplot
    sns.boxplot(y=train_detail.Weekly_Sales,x=train_detail[series],ax=fig.add_subplot(gs[0,0]))
    plt.ylabel("Sales")
    plt.xlabel(series)
    # stripplot
    sns.stripplot(y=train_detail.Weekly_Sales,x=train_detail[series],ax=fig.add_subplot(gs[0,1]))
    plt.ylabel("Sales")
    plt.xlabel(series)
    # fig.show()
    plt.show()

from scipy import stats
from scipy.special import boxcox1p
def plot_continuous(series):
    fig= plt.figure(figsize=(12,8))
    gs= GridSpec(2, 2) #row 2, col 2

    ax= sns.scatterplot(y=train_detail.Weekly_Sales.values,x=boxcox1p(train_detail[series],.15),ax=fig.add_subplot(gs[0,1]),palette="blue")
    plt.title("BoxCos .15\n"+"Corr: "+str(round(train_detail.Weekly_Sales.corr(boxcos1p(train_detail[series],.15)),2))+", Skew: "+str(round(stats.skew(boxcox1p(train_detail[series],.15),nan_policy="omit"),2)))

    ax= sns.scatterplot(y=train_detail.Weekly_Sales,x=boxcox1p(train_detail[series],.25),ax=fig.add_subplot(gs[1,0]),palette="blue")
    plt.title("BoxCos .25\n"+"Corr: "+str(round(train_detail.Weekly_Sales.corr(boxcos1p(train_detail[series],.25)),2))+", Skew: "+str(round(stats.skew(boxcox1p(train_detail[series],.25),nan_policy="omit"),2)))

    ax= sns.displot(train_detail[series],ax=fig.add_subplot(gs[1,1]),color="green")
    plt.title("Distribution\n")

    ax= sns.scatterplot(y=train_detail.Weekly_Sales,x=train_detail[series],ax=fig.add_subplot(gs[0,0]),color="red")
    plt.title("Linear\n"+"Corr: "+str(round(train_detail.Weekly_Sales.corr(train_detail[series]),2))+", Skew: "+str(round(stats.skew(train_detail[series],.25,nan_policy="omit"),2)))

    # fig.show()
    plt.show()
    # End of function

### Feature correlation
### - weekly_sales X isHoliday
plot_discrete("IsHoliday")