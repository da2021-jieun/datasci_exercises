import pandas as pd 
import matplotlib.pyplot as plt 

def show_graph(df_1,df_2,title):
    data= pd.concat([df_1,df_2])
    data.reset_index(inplace=True,drop=True)
    for col in data.columns:
        if col.lower().startswith("pred"):
            data[col].plot(label=col,linestyle="dashed")
        else:
            data[col].plot(label=col)
    plt.title(title)
    plt.legend()
    plt.show()

# Timeseries Models
# - based on Masa's notebook 👏 https://www.kaggle.com/sajikim/time-series-forecasting-methods-example-python
# 1. Autoregression, AR
# 2. Moving Average, MA
# 3. Autoregressive Moving Average, ARMA
# 4. Autoregressive Integrated Moving Average, ARIMA

## =======================
## Autoregression, AR
## =======================
## - models the next step in the sequence as a linear function of the **observations** at prior time steps.
## - AR(p) where p is the order of the model; AR(1) is a first-order AR model
## - suitable for univariate timeseries without trend and seasonality    

from statsmodels.tsa.ar_model import AutoReg
from random import random
def AR_model(train,test):
    model= AutoReg(train["Truth"], lags=1,old_names=False)
    model_fit= model.fit()
    print(model_fit.summary())
    # predict
    yhat= model_fit.predict(len(train),len(train)+len(test)-1)
    res= pd.DataFrame({"Pred":yhat,"Truth":test["Truth"].values})
    return res

train= pd.DataFrame([x+random()*10 for x in range(0,100)],columns=["Truth"])
test= pd.DataFrame([x+random()*10 for x in range(101,201)],columns=["Truth"])
df_res= AR_model(train, test)
show_graph(train, df_res, "Autoregression (AR)")

## =======================
## Moving Average, MA
## =======================
## - models the next step in the sequence as a linear function of the **residual errors** from a mean process at prior time steps.
## - MA(q) where q is the order of the model; MA(1) is a first-order MA model
## - suitable for univariate timeseries without trend and seasonality
import matplotlib.pyplot as plt 

def show_graph(df_1,df_2,title):
    data= pd.concat([df_1,df_2])
    data.reset_index(inplace=True,drop=True)
    for col in data.columns:
        if col.lower().startswith("pred"):
            data[col].plot(label=col,linestyle="dashed")
        else:
            data[col].plot(label=col)
    plt.title(title)
    plt.legend()
    plt.show()

from statsmodels.tsa.arima.model import ARIMA
from random import random
def MA_model(train,test):    
    model= ARIMA(train["Truth"])
    model_fit= model.fit()
    print(model_fit.summary())
    # predict
    yhat= model_fit.predict(len(train),len(train)+len(test)-1)
    res= pd.DataFrame({"Pred":yhat,"Truth":test["Truth"].values})
    return res
import pandas as pd 
train= pd.DataFrame([x+random()*10 for x in range(0,100)],columns=["Truth"])
test= pd.DataFrame([x+random()*10 for x in range(101,201)],columns=["Truth"])
df_res= MA_model(train, test)
show_graph(train, df_res, "Moving Average (MA)")

## ==============================================
## Autoregressive Moving Average, ARMA
## ==============================================
## - models the next step in the sequence as a linear function of (i) the observations and (ii) resiudal errors at prior time steps.
## - combines AR and MA models
## - ARMA(p,q)
## - suitable for univariate time series without trend and seasonality
import matplotlib.pyplot as plt 

def show_graph(df_1,df_2,title):
    data= pd.concat([df_1,df_2])
    data.reset_index(inplace=True,drop=True)
    for col in data.columns:
        if col.lower().startswith("pred"):
            data[col].plot(label=col,linestyle="dashed")
        else:
            data[col].plot(label=col)
    plt.title(title)
    plt.legend()
    plt.show()

from statsmodels.tsa.arima.model import ARIMA
from random import random
def ARMA_model(train,test):
    # moving average models: MA(q)
    # mixed autoregressive moving average: ARMA(p, q)
    # integration models: ARIMA(p, d, q)
    model= ARIMA(train["Truth"],order=(1,0,2)) #p,d,q
    model_fit= model.fit()
    # Covariance Type: opg, outer product of gradients 
    print(model_fit.summary())
    # predict
    yhat= model_fit.predict(len(train),len(train)+len(test)-1)
    res= pd.DataFrame({"Pred":yhat,"Truth":test["Truth"].values})
    return res
import pandas as pd 
train= pd.DataFrame([x+random()*10 for x in range(0,100)],columns=["Truth"])
test= pd.DataFrame([x+random()*10 for x in range(101,201)],columns=["Truth"])
df_res= ARMA_model(train, test)
show_graph(train, df_res, "Autoregressive Moving Average (ARMA)")

## ===============================================
## Autoregressive Integrated Moving Average, ARIMA
## ===============================================
## - models the next step in the sequence as a linear function of (i) the **differenced** observations and (ii) residual errors at prior time steps.
## - combines AR and MA models and adds a preprocessing step of differencing, aka integration, to make the sequence stationary
## - AR(p), I(d), MA(q)
## - suitable for univariate time series with trend and without seasonality
import matplotlib.pyplot as plt 

def show_graph(df_1,df_2,title):
    data= pd.concat([df_1,df_2])
    data.reset_index(inplace=True,drop=True)
    for col in data.columns:
        if col.lower().startswith("pred"):
            data[col].plot(label=col,linestyle="dashed")
        else:
            data[col].plot(label=col)
    plt.title(title)
    plt.legend()
    plt.show()

# from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm 
from random import random
def ARIMA_model(train,test):
    # from statsmodels.tsa.arima.model import ARIMA
    # model= sm.tsa.arima.ARIMA(train["Truth"],order=(1,1,1))
    from statsmodels.tsa.arima_model import ARIMA
    model= ARIMA(train["Truth"],order=(1,1,1))
    model_fit= model.fit(disp=False)
    print(model_fit.summary())
    # predict
    yhat= model_fit.predict(len(train),len(train)+len(test)-1,typ="levels") # ; Predict the levels of the original endogenous variables
    res= pd.DataFrame({"Pred":yhat,"Truth":test["Truth"].values})
    return res
import pandas as pd 
train= pd.DataFrame([x+random()*10 for x in range(0,100)],columns=["Truth"])
test= pd.DataFrame([x+random()*10 for x in range(101,201)],columns=["Truth"])
df_res= ARIMA_model(train, test)
show_graph(train, df_res, "Autoregressive Integrated Moving Average (ARIMA)")