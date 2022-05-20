from dataclasses import dataclass
from select import select
from turtle import title
import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import linregress
import statsmodels

st.title('Incident Dashboard')
st.sidebar.title('Incident Analysis and Visualisation')

st.markdown('A dashboard to visualise incident data 📈')
st.sidebar.markdown('This application is a dashboard to visualise and anlyse Incident data. 📈')

Data_url = 'Incident Dump/Hackathon-Data-200 records v2.0.xlsx'
@st.cache(persist = True)
def load_data():

    data = pd.read_excel(Data_url, sheet_name='Data')
    data['Priority'] = data['Priority'].apply(lambda s: str(s))
    data['Priority'] = data['Priority'].apply(lambda s: re.sub('P','',s))
    data['Resolved'] = data['Resolved'].fillna('Open')
    data['Resolved'] = data['Resolved'].apply(lambda s: str(s))
    data['City'] = data['City'].fillna('Delhi')
    data['Priority'] = data['Priority'].apply(lambda s: str(s))
    data['Country'] = data['Countries'].fillna('India')
    return data

data = load_data()

st.sidebar.subheader("Show Open Incidents")
status = st.sidebar.radio('Status', ('Resolved', 'Closed', 'Hold', 'In Progress'))
st.dataframe(data.query('Status == @status')[['Number', 'Status', 'Service/Application Criticality', 'Short Description']])

st.sidebar.markdown('### Frequency counts for the Assignment Type')
select = st.sidebar.selectbox('Visualisation Type', ['Bar-Chart', 'Pie-Chart'])

st.markdown('### Total Impacted Application counts')
application_count = data['Impacted application'].value_counts()
application_count = pd.DataFrame({'Effected Application count':application_count.index, 'Application': application_count.values})

if st.sidebar.checkbox("Hide", True):
    st.markdown("No of affected applications over the last six months")
    if select == 'Bar-Chart':
        fig = px.bar(application_count, x = 'Effected Application count', y = 'Application', height = 500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(application_count, values = 'Application', names = 'Effected Application count')        
        st.plotly_chart(fig)

st.sidebar.subheader("Priority of the issues.")
#priority = st.sidebar.slider('Priority of issues raised:', 1, 5)
priority = st.sidebar.number_input('Priority of Issues Raised:', min_value = 1, max_value  = 5) ##
priority = data['Priority'].value_counts()
priority_count = pd.DataFrame({'Priority levels':priority.index, 'Prioritiy count': priority.values})

if st.sidebar.checkbox("Show Priority", True):
    st.markdown("High Priority incidents over the last six months")
    fig = px.pie(priority_count, values = 'Prioritiy count', names = 'Priority levels')        
    st.plotly_chart(fig)

new = data[['Impacted application', 'Opened']]
new['month'] = new['Opened'].dt.month
new['month'] = new['month'].apply(lambda s: int(s))

new.set_index('month')
pred_model = new.groupby(new['month']).count()

pred_model.reset_index(inplace = True)




x = np.array(pred_model['month'])
y = np.array(pred_model['Impacted application'])

st.markdown('## Predictive Analysis of the no of Incidents occurring in the months')
slope, intercept, r_value, p_value, std_err = linregress(x, y)
st.markdown("slope: %f, intercept: %f" % (slope, intercept))
st.markdown("R-squared: %f" % r_value**2)

st.markdown('#### Trend of number of Incidents in months')

fig11 = px.bar(pred_model, x = 'month', y = 'Impacted application')
st.plotly_chart(fig11)

data['month'] = data['Opened'].dt.month
new_data = data.groupby(['month', 'Impacted application']).count()



st.markdown('#### Trendline of incidents over months')

fig3 = px.scatter(x, y, trendline = 'lowess')
st.plotly_chart(fig3)

  
#px.figure(figsize=(15, 5))
#fig1 = px.line(x, y)
#st.plotly_chart(fig1)

st.markdown('#### Prediction of Incident Occurence Trend')
a = intercept + slope*x
fig2 = px.line(x, a)
st.plotly_chart(fig2)






    


