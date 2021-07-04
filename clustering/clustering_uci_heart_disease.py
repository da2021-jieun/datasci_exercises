# Clustering with UCI Heart Disease Dataset 

file= "heart.csv"
import pandas as pd
data= pd.read_csv(file)
heart= data.copy()
heart.info()
heart.columns

heart['sex'] = heart['sex'].map({0:'female',1:'male'})

heart['chest_pain_type'] = heart['cp'].map({3:'asymptomatic', 1:'atypical_angina', 2:'non_anginal_pain', 0:'typical_angina'})

heart['fbs'] = heart['fbs'].map({0:'less_than_120mg/ml',1:'greater_than_120mg/ml'})

heart['restecg'] = heart['restecg'].map({0:'normal',1:'ST-T_wave_abnormality',2:'left_ventricular_hypertrophy'})

heart['exang'] = heart['exang'].map({0:'no',1:'yes'})

heart['slope'] = heart['slope'].map({0:'upsloping',1:'flat',2:'downsloping'})

heart['thal'] = heart['thal'].map({1:'fixed_defect',0:'normal',2:'reversable_defect'})

heart['target'] = heart['target'].map({0:'no_disease', 1:'has_disease'})

heart.isna().sum()

## =====================================================
## Separate categorical and non-categorical variables
## - categorical if nunique() <=10
## - non-categorical if nunique() >10
categorical = [i for i in heart.loc[:,heart.nunique()<=10]]
continuous = [i for i in heart.loc[:,heart.nunique()>=10]]

# plot
def plot_distribution(df,cols,hue=None,r=4,c=3):
    import matplotlib.pyplot as plt 
    import seaborn as sns 
    plt.style.use("ggplot")
    fig,ax= plt.subplots(r,c,figsize=(15,22))
    ax= ax.flatten() #?
    for col,axis in zip(df[cols].columns,ax):
        sns.countplot(x=col,data=df,hue=hue,ax=axis,orient=df[col].value_counts().index)
        axis.set_title(f"{col.capitalize()} distribution")
        total= float(len(df[col])) # n_rows
        for patch in axis.patches:
            h= patch.get_height()
            axis.text(patch.get_x()+patch.get_width()/2,h/2,"{:1.2f}%".format((h/total)*100),ha="center")
        # plt.tight_layout()
    plt.show()

plot_distribution(heart, categorical)

## =====================================================
# plot non-categorical variables
# continuous = [i for i in heart.loc[:,heart.nunique()>=10]]
import matplotlib.gridspec as gridspec 
fig= plt.figure(constrained_layout=True,figsize=(14,10))
grid= gridspec.GridSpec(nrows=3, ncols=6,figure=fig)

plot_distribution(heart, categorical[:-1],"target",4,2) #hue=target,r=4,c=2


## =====================================================
## multivariate analysis
## sns.pairplot
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.style.use("ggplot")
plt.figure(figsize=(14,9))
sns.pairplot(data=heart[continuous+["target"]],hue="target",markers=["o","D"])
plt.savefig("pairplot_continuous_var_target.png")
plt.show()
## =====================================================
## 3d-scatterplot
import plotly.express as px
fig= px.scatter_3d(heart,x="chol",y="thalach",z="age",size="oldpeak",color="target",opacity=.8)
fig.update_layout(margin=dict(l=0,r=0,b=0,t=0))
fig.show()

# point-plot
# - A point plot represents an estimate of central tendency for a numeric variable by the position of scatter plot points and provides some indication of the uncertainty around that estimate using error bars.
# - The lines that join each point from the same hue level allow interactions to be judged by differences in slope, which is easier for the eyes than comparing the heights of several groups of points or bars.
# - It is important to keep in mind that a point plot shows only the mean (or other estimator) value, but in many cases it may be more informative to show the distribution of values at each level of the categorical variables. In that case, other approaches such as a _box_ or _violin plot_ may be more appropriate.
# https://seaborn.pydata.org/generated/seaborn.pointplot.html
def plot_frequency(df,cols,x_i,hue=None,row=4,col=1):
    fig,ax= plt.subplots(row,col,figsize=(16,12),sharex=True)
    ax= ax.flatten()
    for col,axis in zip(df[cols].columns,ax):
        sns.pointplot(x=x_i,y=col,data=df,hue=hue,ax=axis)
        plt.tight_layout()
    plt.show()

## =====================================================
plot_frequency(heart, continuous[1:], "age",hue="target",row=4,col=1) #continuous without age

## =====================================================
## corr. matrix
import numpy as np 
corr_mat= heart.corr()
mask= np.triu(corr_mat.corr())
plt.figure(figsize=(16,10))
sns.heatmap(corr_mat,annot=True,fmt=".2f",cmap="Spectral",linewidths=1,cbar=True)
plt.show()

## =====================================================
##

# === models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score,cross_validate,RandomizedSearchCV,train_test_split, KFold

from sklearn.metrics import plot_confusion_matrix

clf_cb= GradientBoostingClassifier(random_state=0)
clf_kn= KNeighborsClassifier()
clf_dt= DecisionTreeClassifier(random_state=0)
clf_sv= SVC()
clf_rf= RandomForestClassifier(random_state=0)
clf_ada= AdaBoostClassifier(random_state=0)
clf_gnb= GaussianNB()
clf_xgb= XGBClassifier()
clf_lgbm= LGBMClassifier()
clf_catb= CatBoostClassifier()
cv= KFold(5,shuffle=True,random_state=0)
classifiers= [clf_cb,clf_kn,clf_dt,clf_sv,clf_rf,clf_ada,clf_gnb,clf_xgb,clf_lgbm,clf_catb]

def check_model(X,y,classifiers,cv):
    """
    Tests multiple models
    Returns: several metrics
    """
    model_tbl= pd.DataFrame()
    row_idx=0
    for clf in classifiers:
        name= clf.__class__.__name__
        model_tbl.loc[row_idx,"Model Name"]= name
        cv_res= cross_validate(clf,X,y,cv=cv,scoring=("accuracy","f1","roc_auc"),return_train_score=True,n_jobs=-1)

        model_tbl.loc[row_idx,"train_rocauc_mean"]= cv_res["train_roc_auc"].mean()
        model_tbl.loc[row_idx,"test_rocauc_mean"]= cv_res["test_roc_auc"].mean()
        model_tbl.loc[row_idx,"test_rocauc_std"]= cv_res["test_roc_auc"].std()

        model_tbl.loc[row_idx,"train_accuracy_mean"]= cv_res["train_accuracy"].mean()
        model_tbl.loc[row_idx,"test_accuracy_mean"]= cv_res["test_accuracy"].mean()
        model_tbl.loc[row_idx,"test_accuracy_std"]= cv_res["test_accuracy"].std()

        model_tbl.loc[row_idx,"train_f1_mean"]= cv_res["train_f1"].mean()
        model_tbl.loc[row_idx,"test_f1_mean"]= cv_res["test_f1"].mean()
        model_tbl.loc[row_idx,"test_f1_std"]= cv_res["test_f1"].std()
        
        model_tbl.loc[row_idx,"fit_time"]= cv_res["fit_time"].mean()

        row_idx+=1
    model_tbl.sort_values(by=["test_f1_mean"],ascending=False,inplace=True)
    return model_tbl

## =====================================================
## Baseline performance
## cv= KFold(5,shuffle=True,random_state=0)
X= data.drop("target",axis=1)
y= data.target
df_models_baseline= check_model(X, y, classifiers, cv)
display(df_models_baseline)

## =====================================================
## plot feature importance
def plot_feature_importance(classifiers,X,y,bins):
    import matplotlib.pyplot as plt 
    import seaborn as sns 
    fix,ax= plt.subplots(1,2,figsize=(15,4))
    ax= ax.flatten()
    for axis,clf in zip(ax,classifiers):
        try:
            clf.fit(X,y)
            importance= pd.DataFrame(sorted(zip(clf.feature_importances_,X.columns)),columns=["Value","Feature"])
            sns.barplot(x="Value",y="Feature",data=importance.sort_values(by="Value",ascending=False),ax=axis,palette="plasma")
            plt.title("Feature Importance")
            plt.tight_layout()
            axis.set(title=f"{clf.__class__.__name__} Feature Importance")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=bins))
        except:
            continue
    plt.savefig(f"feature_importance_comparison.png")
    plt.show()

plot_feature_importance([clf_rf,clf_dt], X, y, 6)

## =====================================================
## anomaly detection
## - find outliers using Mahalanobis distance
## - clustering using KMeans
## - DBSCAN
## - isolation forest
from sklearn.ensemble import IsolationForest
isolf= IsolationForest(contamination=.1,random_state=0)
yhat= isolf.fit_predict(X)
mask= (yhat!=-1)
X_isol= X.loc[mask,:]
y_isol= y[mask]
isolf_models= check_model(X_isol, y_isol, classifiers, cv)
display(isolf_models)
## === Elliptic Envelope
from sklearn.covariance import EllipticEnvelope # for detecting outliers in a Gaussian distributed dataset.
ee= EllipticEnvelope(contamination=.1,assume_centered=True,random_state=0)
yhat= ee.fit_predict(X)
mask= (yhat!=-1)
X_ee= X.loc[mask,:]
y_ee= y[mask]
ee_models= check_model(X_ee, y_ee, classifiers, cv)
display(ee_models)

## ====================================================
## KBinsDiscretizer
## - Bin continuous data into intervals.
from sklearn.preprocessing import KBinsDiscretizer
def discretize_kbins(col,X,nbins=5):
    binit= KBinsDiscretizer(n_bins=nbins,encode="onehot",strategy="kmeans")
    binned= binit.fit_transform(X[col].values.reshape(-1,1))
    binned= pd.DataFrame(binned.toarray())
    bin_n= [f"bin_{str(i)}" for i in range(nbins)]
    binned.columns= [i.replace("bin",f"{str(col)}") for i in bin_n]
    binned= binned.astype(int)
    return binned

continuous = [i for i in heart.loc[:,heart.nunique()>=10]]
X_bin= X_ee.copy()
y_bin= y_ee.copy()
for col in continuous:
    X_bin= X_bin.join(discretize_kbins(col, X,5))
    X_bin.drop(col,axis=1,inplace=True)
kbin_models= check_model(X_bin,y_bin,classifiers,cv)
display(kbin_models)

## ====================================================
## Learning Curves
## - Determines cross-validated training and test scores for different training set sizes.
## - A cross-validation generator splits the whole dataset k times in training and test data. Subsets of the training set with varying sizes will be used to train the estimator, and a score for each training subset size and the test set will be computed. Afterwards, the scores will be averaged over all k runs for each training subset size.
## https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html
import math
import numpy as np
from sklearn.model_selection import learning_curve
from matplotlib.ticker import MaxNLocator
def plot_learnig_curve(classifiers,X,y,ylim=None,cv=None,n_jobs=None,train_sizes=np.linspace(.1,1.,5)):
    fig,ax= plt.subplots(math.ceil(len(classifiers)/2),2,figsize=(20,40))
    ax=ax.flatten()
    ## To be continued . . . 