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
plt.grid()
plt.xticks(np.arange(1,53))
plt.legend(["2010","2011","2012"],loc="best",fontsize=11)
plt.title("Average Weekly Sales, 2010-2012",fontsize=12)
plt.ylabel("sales")
plt.xlabel("week")
plt.tight_layout()
plt.show()
