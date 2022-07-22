# -*- coding: utf-8 -*-
"""
"""

import os 
import re
import pickle
import datetime 
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Dense,Dropout
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping

def plot_con_graph(con_cols,df):
    '''
    This function is meant to plot continous data using seaborn distplot
    function

    Parameters
    ----------
    con : list
        con contains the names of the continous columns
    df : DataFrame
        df contains all data

    Returns
    -------
    None.

    '''
    for i in con_cols:
        plt.figure()
        sns.distplot(df[i])
        plt.show()

def plot_cat_graph(cat_cols,data):
    '''
    This function is meant to plot categorical data using seaborn countplot
    function

    Parameters
    ----------
    cat : list
        con contains the names of the categorical columns
    df : DataFrame
        df contains all data

    Returns
    -------
    None.

    '''
    for i in cat_cols:
        plt.figure()
        sns.countplot(data[i])
        plt.show()

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

# 1) Data Loading

CSV_PATH = os.path.join(os.getcwd(),'Train.csv')

df = pd.read_csv(CSV_PATH)

# 2) Data Inspection 

df.describe()
df.info()

# Th id column is dropped as it will not served any meaning foe the training
df=df.drop(labels='id',axis=1)

df.dtypes

# List out the categorical columns 
# The term_deposit_subscribed is the targeted column 
cat_cols = list(df.columns[df.dtypes == 'object'])
cat_cols.append('term_deposit_subscribed')

# List out the continous columns
con_cols = list(df.columns[df.dtypes != 'object'])
con_cols.remove('term_deposit_subscribed')

# Data visualization 

plot_con_graph(con_cols,df)

plot_cat_graph(cat_cols,df)

fig, axes = plt.subplots(7,figsize=(8,20))
for i,c in enumerate(df[con_cols]):
    df[[c]].boxplot(ax=axes[i], vert=False)

# 3) Data Cleaning

df.isna().sum()

#Calculate the percentage of null values
miss_df = df.isna().sum()
percentage_miss_df = df.isna().mean().round(4)*100
df_per = {'Missing Values' : miss_df,
      'percentage missing values' : percentage_miss_df}
df_per = pd.DataFrame(df_per)
df_per = df_per.sort_values(by='Missing Values',ascending=False)
df_per

# The days_since_prev_campaign_contact has 81%, 25831 of NaN, thus drop the column

df = df.drop(labels='days_since_prev_campaign_contact',axis=1)

for i in cat_cols:
    le = LabelEncoder()
    temp = df[i]
    temp[temp.notnull()] = le.fit_transform(temp[df[i].notnull()])
    df[i] = pd.to_numeric(df[i],errors='coerce')
        
    PICKLE_SAVE_PATH = os.path.join(os.getcwd(),'sample_data','Models', i+'_LEncoder.pkl')
    with open(PICKLE_SAVE_PATH, 'wb') as file:
        pickle.dump(le,file)

# Handling NaNS
col_names= df.columns

knn = KNNImputer()
df = knn.fit_transform(df)
df = pd.DataFrame(df)
df.columns = col_names

df['personal_loan'] = np.floor(df['personal_loan'])

df.describe().T

con_cols.remove('days_since_prev_campaign_contact')

plot_con_graph(con_cols,df)

plot_cat_graph(cat_cols,df)

df.isna().sum()
df.boxplot()

#%% 4) Features Selection 
# con vs cat

for i in con_cols:
    print(i)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[i], axis=-1),df['term_deposit_subscribed'])
    print(lr.score(np.expand_dims(df[i],axis=-1),df['term_deposit_subscribed']))

# cat vs cat 
# cramer's v

for i in cat_cols:
    print(i)
    confusion_matrix = pd.crosstab(df[i],df['term_deposit_subscribed']).to_numpy()
    print(cramers_corrected_stat(confusion_matrix))

#Deep Learning

X = df.loc[:,con_cols]
y = df['term_deposit_subscribed']

ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y,axis=-1))

#Model saving
OHE_SAVE_PATH = os.path.join(os.getcwd(),'sample_data','Models','ohe.pkl')
with open(OHE_SAVE_PATH,'wb') as file:
  pickle.dump(ohe,file)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,
                                               random_state=1)

model = Sequential()
model.add(Input(shape=np.shape(X_train)[1:]))
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=['acc'])

log_dir = os.path.join("logs_seg", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_callback = EarlyStopping(monitor='val_loss',patience=10)

hist = model.fit(X_train,y_train,
                epochs=128,verbose=1,
                validation_data=(X_test,y_test),
                 callbacks=[tensorboard_callback,early_callback])

# Model evaluation
print(hist.history.keys())

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel('epoch')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.xlabel('epoch')
plt.legend(['Training Acc', 'Validation Acc'])
plt.show()

print(model.evaluate(X_test,y_test))

#Model Analysis
pred_y = model.predict(X_test)
pred_y = np.argmax(pred_y,axis=1)
true_y = np.argmax(y_test,axis=1)

# Print classification report
cr = classification_report(true_y,pred_y)
print(cr)

# Commented out IPython magic to ensure Python compatibility.
# Load the TensorBoard notebook extension
# %load_ext tensorboard
# %tensorboard --logdir logs_seg

# Model
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'sample_data','Models',
                               'model.h5')
model.save(MODEL_SAVE_PATH)