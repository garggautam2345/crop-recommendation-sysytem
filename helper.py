from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import numpy as np
le=LabelEncoder()
preprocess=ColumnTransformer(
    transformers=[
        ('Standard Scaler',StandardScaler(),[0,1,2,3,4,5,6])
    ],
    remainder='passthrough'
)

def x_and_y(df):
    x=df.drop(columns=['label'])
    y=df['label']
    y=le.fit_transform(y)

    return train_test_split(x,y,test_size=0.3,random_state=42)

def scalling(x_train,x_test):
    return preprocess.fit_transform(x_train),preprocess.transform(x_test)

def model(x_train_scalled,y_train):
    gnb=GaussianNB()
    return gnb.fit(x_train_scalled,y_train)

def prediction(N,P,K,temperature,humidity,ph,rainfall,model):
    data=np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    ans=model.predict(data)
    return(le.inverse_transform(ans))


