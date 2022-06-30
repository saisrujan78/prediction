
import numpy as np
import pandas as pd


# In[3]:


data = pd.read_csv(r"C:\Users\srujan\Downloads\gre\gre\admission.csv")


# In[4]:


data.head()


data.shape


# In[6]:


data.isnull().sum()


# In[7]:


data.describe()


# In[8]:


data.dtypes


# In[9]:


data.drop(['Serial No.'],axis=1,inplace=True)


# In[10]:


data.dtypes


# In[11]:


data.corr()


# In[12]:


data.info()


# In[13]:


Y = data.iloc[:,-1]


# In[14]:


Y.head()


# In[15]:


X = data.iloc[:,1:-1]


# In[16]:


X.head()


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)


# In[18]:


from sklearn.linear_model import LinearRegression as lr
lr = lr()


# In[19]:


lr.fit(X_train, Y_train)


# In[20]:


lr.score(X_test,Y_test)


# In[21]:


gre = 335
pred_chance = lr.predict(np.array([[gre,115,4,5,4,9.80,0]]))


# In[22]:


print("Chance of admission:", pred_chance[0]*100)


# In[23]:


pred_chance = lr.predict(np.array([[335,115,4,5,4,9.80,1]]))


# In[24]:


print("Chance of admission:", pred_chance[0]*100)


# In[25]:


#random forest model


# In[26]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=7)


# In[27]:


from sklearn.ensemble import RandomForestRegressor


# In[28]:


rs = RandomForestRegressor(max_depth=2,random_state=0,n_estimators=5)


# In[29]:


rs.fit(X_train,Y_train)


# In[30]:


rs.score(X_test,Y_test)


# In[31]:


rs_pred_chance = rs.predict(np.array([[335,115,4,5,4,9.80,1]]))
print("Chance of admission:", rs_pred_chance[0]*100)


# In[32]:


import pickle


# In[33]:


file = open('predict_lr.pkl','wb')


# In[34]:


pickle.dump(lr,file)


# In[35]:


file.close()


# In[36]:


model = open('predict_lr.pkl','rb')


# In[37]:


chance = pickle.load(model)


# In[38]:


from sklearn import metrics


# In[39]:


y_prediction = chance.predict(X_test)


# In[40]:


metrics.r2_score(Y_test, y_prediction)

