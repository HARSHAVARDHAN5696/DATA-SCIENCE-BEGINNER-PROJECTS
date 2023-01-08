#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

data = pd.read_csv(r"C:\Users\HARSHA VARDHAN\Desktop\Credit-Score-Data\Credit Score Data\train.csv")
print(data.head())


# In[2]:


print(data.info())


# In[3]:


#lets look if the data set have any null values or not
print(data.isnull().sum())


# In[4]:


data['Credit_Score'].value_counts()


# In[6]:


#This data set has a lot of features so we can make a machine learning model. Lets freakin go...
#Lets start by exploring the occupation feature, so that it will whether affect the credit score or not
fig = px.box(data, x='Occupation', color='Credit_Score', title = 'credit score based on occupation', color_discrete_map = {'Poor':'red', 'Standard':'Yellow', 'Good':'Green'})
fig.show()


# In[10]:


# Since it is not showing much difference, lets explore the annual income whether it actually affects the credit score or not.
fig = px.box(data, x='Credit_Score', y ='Annual_Income', color = 'Credit_Score', title = 'Credit scores based on annual income', color_discrete_map = {'Poor':'red', 'Standard':'yellow', 'Good':'green'})
fig.update_traces(quartilemethod = 'exclusive')
fig.show()


# In[11]:


# Acooeding to the above plot we can say that the more you earn the good your score will be.
#Now let's compare whether monthly_in_hand salary impacts your score or not
fig = px.box(data, 
             x="Credit_Score", 
             y="Monthly_Inhand_Salary", 
             color="Credit_Score",
             title="Credit Scores Based on Monthly Inhand Salary", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[12]:


# Is goes goes same with the annual income. so lets see whether the number of bank accounts will impact score or not
fig = px.box(data, 
             x="Credit_Score", 
             y="Num_Bank_Accounts", 
             color="Credit_Score",
             title="Credit Scores Based on Number of Bank Accounts", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[13]:


# Having more bank accounts is also a problem, so you must have 4 or 5. Now lets check whether the number of credit cards affect the credit score or not
fig = px.box(data, 
             x="Credit_Score", 
             y="Num_Credit_Card", 
             color="Credit_Score",
             title="Credit Scores Based on Number of Credit cards", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[14]:


# The result is same as bank accounts, lesser the credit cards more the credit score. Now lets check whether the average intrest you pay pn the loan and emi will affect your score or not
fig = px.box(data, 
             x="Credit_Score", 
             y="Interest_Rate", 
             color="Credit_Score",
             title="Credit Scores Based on the Average Interest rates", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[15]:


#Having more intrest  rate is bad for you. Now lets check how many loans you can take at a time  for a good credit score
fig = px.box(data, 
             x="Credit_Score", 
             y="Num_of_Loan", 
             color="Credit_Score", 
             title="Credit Scores Based on Number of Loans Taken by the Person",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[16]:


# To have a good credit score you should not take more than 1-3 loans at a time. # Now lest check whether delaying payements on due date will affect your score or not.
fig = px.box(data, 
             x="Credit_Score", 
             y="Delay_from_due_date", 
             color="Credit_Score",
             title="Credit Scores Based on Average Number of Days Delayed for Credit card Payments", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[17]:


#you can delay your payement upto 15 days maxx, if you delay after that it will impact. Now lets see if frequently delaying payements will affect the credit score or not
fig = px.box(data, 
             x="Credit_Score", 
             y="Num_of_Delayed_Payment", 
             color="Credit_Score", 
             title="Credit Scores Based on Number of Delayed Payments",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[18]:


#So delaying 4 – 12 payments from the due date will not affect your credit scores. But delaying more than 12 payments from the due date will affect your credit scores negatively. Now let’s see if having more debt will affect credit scores or not:
fig = px.box(data, 
             x="Credit_Score", 
             y="Outstanding_Debt", 
             color="Credit_Score", 
             title="Credit Scores Based on Outstanding Debt",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[19]:


#Having an outstanding debt of more than $1100 wil affect your score. Now lets check if having a high credit utilization ratio will affect credit score or not
fig = px.box(data, 
             x="Credit_Score", 
             y="Credit_Utilization_Ratio", 
             color="Credit_Score",
             title="Credit Scores Based on Credit Utilization Ratio", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[20]:


#from the above figure it says that it cant affect your score. Now lets check if the credit history age of a person affects credit scores:
fig = px.box(data, 
             x="Credit_Score", 
             y="Credit_History_Age", 
             color="Credit_Score", 
             title="Credit Scores Based on Credit History Age",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[21]:


#Having a less vredit history will impact your credit score. Now lets see  how many EMI'S you can have in a month for a good credit score
fig = px.box(data, 
             x="Credit_Score", 
             y="Total_EMI_per_month", 
             color="Credit_Score", 
             title="Credit Scores Based on Total Number of EMIs per Month",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[22]:


#The amount you are paying on EMI per month doesnt affect much to your credit score. Now lets see if your monthly investments affect your credit score
fig = px.box(data, 
             x="Credit_Score", 
             y="Amount_invested_monthly", 
             color="Credit_Score", 
             title="Credit Scores Based on Amount Invested Monthly",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[23]:


#The amount of money you invest monthly doesn’t affect your credit scores a lot. Now let’s see if having a low amount at the end of the month affects credit scores or not:
fig = px.box(data, 
             x="Credit_Score", 
             y="Monthly_Balance", 
             color="Credit_Score", 
             title="Credit Scores Based on Monthly Balance Left",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[71]:


# So havin montly high balance left will get you a good credit score. Since credit mix is a categorical feature i will convert it into numerical so it will bes easier in creating a ML model.
data['Credit_Mix']= data['Credit_Mix'].map({'Poor':0, 'Standard':1, 'Good':2})



# In[74]:


from sklearn.model_selection import train_test_split
x= np.array(data[["Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", 
                   "Delay_from_due_date", "Num_of_Delayed_Payment"
                   ]])
y = np.array(data[["Credit_Score"]])


# In[75]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.33, random_state = 42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtrain, ytrain)


# In[48]:


print("Credit Score Prediction : ")
a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit cards: "))
e = float(input("Interest rate: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed by the person: "))
h = float(input("Number of delayed payments: "))

features = np.array([[a, b, c, d, e, f, g, h]])
print("Predicted Credit Score = ", model.predict(features))


# In[ ]:




