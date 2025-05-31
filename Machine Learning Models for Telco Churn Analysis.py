#!/usr/bin/env python
# coding: utf-8

# # 
# !pip install -U scikit-learn imbalanced-learn
# !pip install --upgrade imbalanced-learn
# 
# 

# In[2]:


pip uninstall -y scikit-learn imbalanced-learn


# In[3]:


pip install scikit-learn==1.3.2 imbalanced-learn==0.11.0


# In[43]:


import sklearn
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN


# In[45]:


print("pandas version:", pd.__version__)
print("scikit-learn version:", sklearn.__version__)


# In[5]:


df=pd.read_csv("tel_churn.csv")


# In[6]:


df.head(5)


# In[7]:


x=df.drop('Churn', axis=1)
print(x)


# In[8]:


##Creating X and Y variable
y=df['Churn']
y


# In[9]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size =.2)


# # Decision Tree Classifier

# In[10]:


model_dt=DecisionTreeClassifier(criterion='gini',random_state=100, max_depth=6, min_samples_leaf=8)


# In[11]:


model_dt.fit(x_train,y_train)


# In[12]:


y_pred=model_dt.predict(x_test)


# In[13]:


y_pred


# In[14]:


y_test


# In[15]:


model_dt.score(x_test,y_test)


# In[ ]:





# In[16]:


print(classification_report(y_test,y_pred, labels=[0,1]))


# In[17]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[18]:


sm=SMOTEENN()
X_resampled,y_resampled=sm.fit_resample(x,y)


# In[19]:


xr_train,xr_test,yr_train,yr_test=train_test_split(X_resampled,y_resampled,test_size =.2)


# In[20]:


model_dt_smote=DecisionTreeClassifier(criterion='gini',random_state=100, max_depth=6, min_samples_leaf=8)


# In[21]:


model_dt_smote.fit(xr_train,yr_train)


# In[22]:


y_pred_smote=model_dt_smote.predict(xr_test)


# In[23]:


print(classification_report(yr_test,y_pred_smote, labels=[0,1]))


# In[24]:


print(confusion_matrix(yr_test,y_pred_smote))


# # Random Forest Classifier

# In[25]:


from sklearn.ensemble import  RandomForestClassifier


# In[26]:


model_rf=RandomForestClassifier(n_estimators=100,criterion='gini',random_state=100, max_depth=6, min_samples_leaf=8)
model_rf.fit(x_train,y_train)
y_pred_rf=model_rf.predict(x_test)


# In[27]:


print(classification_report(y_test,y_pred_rf,labels=[0,1]))


# In[28]:


sm=SMOTEENN()
x_resampled,y_resampled=sm.fit_resample(x,y)


# In[29]:


xr_train,xr_test,yr_train,yr_test=train_test_split(x_resampled,y_resampled,test_size =.2)


# In[30]:


model_smote_rf=RandomForestClassifier(criterion='gini',random_state=100, max_depth=6, min_samples_leaf=8)


# In[31]:


model_smote_rf.fit(xr_train,yr_train)


# In[32]:


y_pred_smote_rf=model_smote_rf.predict(xr_test)


# In[33]:


print(classification_report(yr_test,y_pred_smote_rf, labels=[0,1]))


# In[34]:


print(confusion_matrix(yr_test,y_pred_smote_rf))


# In[35]:


import pickle


# In[36]:


filename='model.sav'


# In[37]:


pickle.dump(model_smote_rf,open(filename,'wb'))


# In[38]:


load_model=pickle.load(open(filename,'rb'))


# In[39]:


load_model.score(xr_test,yr_test)


# In[ ]:




