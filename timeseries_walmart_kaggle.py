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
### Add a new column: `Week`
feat_stor["Week"]= feat_stor.Date.dt.isocalendar().week

### ==================================================
### Merge store details with train/test
train_detail= train.merge(feat_stor,how="inner",on=["Store","Date","IsHoliday"]).sort_values(by=["Store","Dept","Date"]).reset_index(drop=True)

test_detail= test.merge(feat_stor,how="inner",on=["Store","Date","IsHoliday"]).sort_values(by=["Store","Dept","Date"]).reset_index(drop=True)

del features,stores,train,test

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
from pandasql import sqldf
pysql= lambda q: sqldf(q,globals())

