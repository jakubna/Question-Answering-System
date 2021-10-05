#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ast
import pandas as pd


# In[2]:


data = pd.read_csv("data/processed.csv").reset_index(drop=True)


# ## Create features

# In[4]:


def count_sentences(dist):
    return len(eval(dist))


# In[5]:


#remove records with too many sentences
data["no_of_sentences"] = data["eucl_dis"].apply(count_sentences)
data = data[data["no_of_sentences"] < 13]
del data["no_of_sentences"]
data = data.reset_index(drop=True)


# In[8]:


features = pd.DataFrame()
     
for i in range(len(data["eucl_dis"])):
    eucl_dis = eval(data["eucl_dis"][i])
    for j in range(12):
        if j < len(eucl_dis):
            features.loc[i, "eucl_dis_" + str(j)] = eucl_dis[j]
        else: 
            features.loc[i, "eucl_dis_" + str(j)] = 10


# In[9]:


for i in range(len(data["cosine_dis"])):
    cosine_dis = eval(data["cosine_dis"][i].replace("nan","1"))
    for j in range(12):
        if j < len(cosine_dis):
            features.loc[i, "cosine_dis_" + str(j)] = cosine_dis[j]
        else:
            features.loc[i, "cosine_dis_" + str(j)] = 1


# In[10]:


features["ans_sentence"] = data["ans_sentence"]


# In[11]:


features


# ## Split data and run models

# In[43]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features.iloc[:,:-1])


# In[44]:


train_feat, test_feat, train_ans, test_ans = train_test_split(features_scaled, features.iloc[:,-1])


# In[45]:


from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# In[46]:


regression = linear_model.LogisticRegression(solver='saga', multi_class='multinomial')
regression.fit(train_feat, train_ans)

regression_train_predictions = regression.predict(train_feat)
regression_test_predictions = regression.predict(test_feat)

regression_train_acc = metrics.accuracy_score(train_ans, regression_train_predictions)
regression_test_acc = metrics.accuracy_score(test_ans, regression_test_predictions)

print("Logistic regression results:\ntrain set: " + str(regression_train_acc) + "\ntest set: " + str(regression_test_acc))


# In[47]:


forest = RandomForestClassifier(criterion='entropy')
forest.fit(train_feat, train_ans)

forest_train_predictions = forest.predict(train_feat)
forest_test_predictions = forest.predict(test_feat)

forest_train_acc = metrics.accuracy_score(train_ans, forest_train_predictions)
forest_test_acc = metrics.accuracy_score(test_ans, forest_test_predictions)

print("Random forest results:\ntrain set: " + str(forest_train_acc) + "\ntest set: " + str(forest_test_acc))


# In[51]:


xg = xgb.XGBClassifier(max_depth = 5)
xg.fit(train_feat, train_ans)
xg_train_predictions = xg.predict(train_feat)
xg_test_predictions = xg.predict(test_feat)

xg_train_acc = metrics.accuracy_score(train_ans, xg_train_predictions)
xg_test_acc = metrics.accuracy_score(test_ans, xg_test_predictions)

print("XGBoost results:\ntrain set: " + str(xg_train_acc) + "\ntest set: " + str(xg_test_acc))


# In[ ]:




