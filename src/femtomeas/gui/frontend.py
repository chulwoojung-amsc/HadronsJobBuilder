import time
import dash
from dash import callback, html, Input, Output, State, dcc
from dash_chat import ChatComponent
import dash_extensions as de
from dash.exceptions import PreventUpdate
import dash_ag_grid as dag
import dash_bootstrap_components as dbc

app_dash = dash.Dash(__name__)#   , external_stylesheets=[dbc.themes.SLATE]) #mount on / instead

app_dash.layout = html.Div([
    de.WebSocket(url="ws://localhost:8050/ws", id="ws"),    
    dcc.Tabs(id="tabs", value="chatbot", children=[
        dcc.Tab(label="Workflow Creation Agent", value="chatbot"),
        dcc.Tab(label="Job Monitor", value="job_status", children=[
            dag.AgGrid(id="transfer_monitor", columnDefs=[ {"field" : "ID"}, {"field" : "origin"}, {"field" : "destination"}, {"field": "status"} ])
        ])
    ]),
    html.Div(id="chat-panel", children=[        
    ChatComponent(
        id="chat-component",
        messages=[],
        class_name = 'my-chat'
    )
    ])
])

@callback( Output("chat-component", "messages", allow_duplicate=True),
           Input("ws", "message"),
           State("chat-component", "messages"),
           prevent_initial_call=True
          )
def receiveAgentResponse(agent_message, messages):
    if agent_message:
        print("RECEIVED AGENT RESPONSE", agent_message)
        return messages + [ { "role" : "assistant", "content" : agent_message['data'] } ]
    raise PreventUpdate


@callback(
    [ Output("chat-component", "messages"), Output("ws","send",allow_duplicate=True) ],
    Input("chat-component", "new_message"),
    State("chat-component", "messages"),
    prevent_initial_call=True,
)
def handleUserInput(new_message, messages):
    print("USER MESSAGE",new_message)    
    return messages + [ new_message ], new_message['content']


@callback(Output("chat-panel", "hidden"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "chatbot":
        return False
    else:
        return True
