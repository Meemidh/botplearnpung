import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
from datetime import datetime as dt

pd.options.mode.chained_assignment = None  

cred = credentials.Certificate('plenpung-firebase-adminsdk-3kbnb-3d11f00d77.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
docs = firestore.client().collection('UserSentimentRecord').stream()
box = pd.DataFrame(columns=['datetime','result','text'])

for doc in docs: 
    userid = doc
    data = doc.to_dict()  
    box = box.append({'datetime':data['datetime'],'result': data[u'result'],'test': data[u'text']}, ignore_index=True)


box['datetime'] = pd.to_datetime(box['datetime']).dt.floor('H')

date_time_range = pd.date_range(start="2021-08-17", end="2021-10-25" , freq="60min" )

df = pd.DataFrame(data=date_time_range,columns=['datetime'])
df['negative'] = 0
df['neutral'] = 0
df['positive'] = 0


for i in range(len(df)):
    df['negative'][i] = len(box[(box['result']==-1) & (box['datetime'] == df['datetime'][i])])
    df['neutral'][i]  = len(box[(box['result']==0) & (box['datetime']  == df['datetime'][i])])
    df['positive'][i] = len(box[(box['result']==1) & (box['datetime']  == df['datetime'][i])])

# col_options = [dict(label=x, value=x) for x in df.columns]
# dimensions = ["x", "y", "sentiment", "facet_col", "facet_row"]

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Demo: Plotly Express in Dash with Tips Dataset"),
    
    dcc.DatePickerRange(
        id='my-date-picker-range',
        min_date_allowed = dt(2021, 8, 1),
        max_date_allowed = dt(2022, 12, 31),
        start_date = dt(2021, 8, 1),
        end_date = dt(2021, 10, 1),
        start_date_placeholder_text ='DD/MM/YYYY'
    ),
    html.H2("  "),
    
    dcc.Graph(id="graph", style={"width": "95%" ,"display": "inline-block"})
    
])


@app.callback(Output("graph", "figure"),   
              [Input("my-date-picker-range","start_date"),
                Input("my-date-picker-range","end_date")]
              )

def make_figure(start_date,end_date):
        
    date_time_range = pd.date_range(start= start_date , end= end_date , freq="60min" )

    df = pd.DataFrame(data=date_time_range,columns=['datetime'])
    df['negative'] = 0
    df['neutral'] = 0
    df['positive'] = 0


    for i in range(len(df)):
        df['negative'][i] = len(box[(box['result']==-1) & (box['datetime'] == df['datetime'][i])])
        df['neutral'][i]  = len(box[(box['result']==0) & (box['datetime']  == df['datetime'][i])])
        df['positive'][i] = len(box[(box['result']==1) & (box['datetime']  == df['datetime'][i])])

    return px.line( 
        df,
        x='datetime',
        y=['negative','neutral','positive'],
        labels={
                     "datetime": "Day ,time(hr) ",
                     "value": "Count",
                     "variable" : "Sentiment"                                     
                 },
        
        color_discrete_map={
                 "negative": "red",
                 "neutral": "green",
                 "positive": "blue"
             },
            
        height=700,
    )
    
app.run_server(debug=True)
