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

# plot non-categorical var
