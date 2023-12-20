#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import pandas as pd


# In[50]:


df1 = pd.read_csv("C:\\Users\\welcome\\Downloads\\DataFrame\\IMDB Dataset.csv")


# In[51]:


df1.shape


# In[52]:


df1['review'][0]


# # Text Claening
# 
# 

# 1.Sample 1000 rows
# 2.Remove html tags
# 3.Remove special characters
# 4.converting every thing to lower case
# 5.removing stop words
# 6.Stemming

# In[53]:


df = df1.sample(1000)


# In[54]:


df.shape


# In[55]:


df.info()


# In[56]:


df['sentiment'].replace({'positive':1,'negative':0},inplace=True)


# In[57]:


df.head()


# In[58]:


import re


# In[59]:


# Function to clean html tags

def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean,'',text)


# In[60]:


df['review']=df['review'].apply(clean_html)


# In[61]:


#Converting everything to lower
def convert_lower(text):
    return text.lower()


# In[62]:


df['review'] = df['review'].apply(convert_lower)


# In[63]:


#function to remove special characters

def remove_special(text):
    x = ''
    
    for i in text:
        if i.isalnum():
            x = x+i
        else:
            x = x + ' '
    return x        


# In[64]:


remove_special('One of the other reviewers has mentioned th#@at after watching just 1 Oz episode you')


# In[65]:


df['review'] = df['review'].apply(remove_special)


# In[66]:


#Remove the stop words
import nltk


# In[67]:


from nltk.corpus import stopwords


# In[68]:


stopwords.words('english')


# In[69]:


df


# In[70]:


def remove_stopwords(text):
    x = []
    for i in text.split():
        if i not in stopwords.words('english'):
            x.append(i)
    y = x[:]      
    x.clear()
    return y


# In[71]:


df['review'] = df['review'].apply(remove_stopwords)


# In[72]:


df


# In[73]:


# performing stemming

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[74]:


y=[]
def stem_words(text):
    for i in text:
        y.append(ps.stem(i))
        
    z = y[:]    
    y.clear()
    return z


# In[75]:


stem_words({'I','loved','loving'})


# In[76]:


df['review'] = df['review'].apply(stem_words)


# In[77]:


#join back

def join_back(list_input):
    return " ".join(list_input)


# In[78]:


df['review'] = df['review'].apply(join_back)


# In[79]:


df


# In[80]:


# join back

#def join_back(list_input):
#    return " ".join(list_input)


# In[81]:


#df['review']= df['review'].apply(join_back)


# In[82]:


df['review']


# In[84]:


x = df.iloc[:,0].values


# In[96]:


print(x[0].mean())
print(x.shape)


# In[86]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=500)


# In[87]:


x = cv.fit_transform(df['review']).toarray()


# In[88]:


x.shape


# In[89]:


y = df.iloc[:,-1].values


# In[97]:


y   


# In[98]:


y.shape


# In[99]:


#x,y
# Training set
# Test Set(Already Know the result)


# In[100]:


from sklearn.model_selection import train_test_split

x_train ,x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2, random_state=34)


# In[101]:


x_train.shape


# In[102]:


x_test.shape


# In[104]:


y_train.shape,y_test.shape


# In[105]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB


# In[106]:


clf1 = GaussianNB()
clf2 = MultinomialNB()
clf3 = BernoulliNB()


# In[110]:


print(clf1.fit(x_train,y_train))
print(clf2.fit(x_train,y_train))
print(clf3.fit(x_train,y_train))


# In[111]:


y_pred1 = clf1.predict(x_test)
y_pred2 = clf2.predict(x_test)
y_pred3 = clf3.predict(x_test)


# In[112]:


from sklearn.metrics import accuracy_score


# In[113]:


print('Gaussian',accuracy_score(y_test,y_pred1))
print('Multinomial',accuracy_score(y_test,y_pred2))
print('Bernoulli',accuracy_score(y_test,y_pred3))


# In[ ]:




