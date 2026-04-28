import time
import dash
from dash import callback, html, Input, Output, State, dcc
from dash_chat import ChatComponent
import dash_extensions as de
from dash.exceptions import PreventUpdate
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import json
from femtomeas.workflow_manager.manager_config import parseManagerConfigStr
import base64
from .utils import make_scroll_callback

app_dash = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

startup_controls = dbc.Container(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Label("State file", html_for="state-file"),
                    dcc.Store(id="wfman-config-store", storage_type="local"), #persist in browser cache
                    dbc.Input(
                        id="state-file",
                        type="text",
                        placeholder="ckpoint_state.json",
                        value="",
                        debounce=True,
                    ),
                    dbc.Checklist(
                        id="options-switches",
                        options=[{"label": "Reload the state file if it exists", "value": "reload_state"},
                                 {"label": "Run the workflow agent", "value": "use_agent"},
                                 {"label": "Run the workflow manager", "value": "use_workflow_manager"}
                                 ],
                        value=["use_agent", "use_workflow_manager"],
                        switch=True,
                        className="mt-3",
                    ),
                    dcc.Upload(
                        id="wfman-config-upload",
                        children=dbc.Button(
                            id="wfman-config-upload-button",
                            #children="Choose the workflow manager configuration file (JSON)",
                            color="secondary",
                            className="mt-3",
                        ),
                        multiple=False,
                        style={"display": "inline-block"}
                    ),
                    dbc.Button(
                        "Clear cache",
                        id="clear-cache",
                        color="primary",
                        className="mt-3"
                    ),
                    dbc.Button(
                        "Start",
                        id="start",
                        color="primary",
                        className="mt-3"#,
                        #disabled=True #start disabled because use_workflow_manager is enabled by default and we need to upload a config file to use it
                    )
                ]
            ),
            className="mb-3",
        ),
    ],
    fluid=True,
    className="py-3",
)


chat_tab = dbc.Container(
    [
      ChatComponent(
        id="chat-component",
        messages=[],
        class_name="my-chat",
      )
    ],
    fluid=True,
    className="py-3",
)

job_tab = dbc.Container(
    [
        # Row 1: Tables
        dbc.Row(
            [                
                dbc.Col([
                    html.H5("Transfer Monitor", className="mb-2"),
                    dag.AgGrid(
                        id="transfer-monitor",
                        columnDefs=[
                            {"field": "job ID", "flex" : 1},
                            {"field": "api ID", "flex" : 1},
                            {"field": "origin", "flex" : 2},
                            {"field": "destination", "flex" : 3},
                            {"field": "status", "flex" : 1},
                        ],                        
                        defaultColDef={
                            "wrapText": True,
                            "autoHeight": True,
                        },
                        getRowId="params.data.ID",
                        style={"height": 400, "width": "100%"},
                    )],
                    width=6,
                ),
                dbc.Col([
                    html.H5("Compute Monitor", className="mb-2"),
                    dag.AgGrid(
                        id="compute-monitor",
                        columnDefs=[
                            {"field": "job ID", "flex" : 1},
                            {"field": "api ID", "flex" : 1},
                            {"field": "machine", "flex" : 1},
                            {"field": "queue", "flex" : 1},
                            {"field": "time", "flex" : 1},
                            {"field": "status", "flex" : 1},
                        ],
                        defaultColDef={
                            "wrapText": True,
                            "autoHeight": True,
                            },
                        getRowId="params.data.ID",
                        #columnSize="autoSize",
                        style={"height": 400, "width": "100%"},
                    )],
                    width=6,
                ),
            ],
            className="mb-3",
        ),

        # Row 2: Logs
        dbc.Row(
            [
                dbc.Col([
                    html.H5("Workflow Manager Logs", className="mb-2"),
                    dcc.Textarea(
                        id='wfman-log',
                        style={'width': '100%', 'height': 300},
                        disabled=True,
                        readOnly=True
                    )],
                    width=6,
                ),
                dbc.Col(
                    [
                    html.H5("API Logs", className="mb-2"),
                    dcc.Textarea(
                        id='wfapi-log',
                        style={'width': '100%', 'height': 300},
                        disabled=True,
                        readOnly=True
                    )],
                    width=6,
                ),
            ]
        ),
    ],
    fluid=True,
    className="py-3",
)

app_dash.layout = html.Div([
    de.WebSocket(url="ws://localhost:8050/ws", id="ws"),
    html.Div(id="startup-controls", children=startup_controls),
    
    html.Div(id="tabs-wrapper", children=[
      dcc.Tabs(id="tabs"),
      html.Div(id="chat-tab-pane", children=chat_tab),
      html.Div(id="job-tab-pane", children=job_tab)    
    ], style={"display": "none"}),

    html.Div(id="error-display-wrapper",
             children = [
                 html.H5("Server errors", className="mb-2"),
                 dcc.Textarea(
                        id='error-display',
                        disabled=True,
                        readOnly=True
                    )
                 ],
             style={"display": "none"}
             )
             
])

@callback(
    Output("wfman-config-store", "clear_data"),
    Input("clear-cache", "n_clicks"), 
    prevent_initial_call=True
    )
def clear_wfman_cfg_cache(n_clicks):
    if n_clicks and n_clicks > 0:
        return True
    raise PreventUpdate

@callback(
    Output("wfman-config-store", "data"),
    Input("wfman-config-upload", "filename"),
    State("wfman-config-upload","contents"),
    prevent_initial_call=True)
def upload_wfman_cfg(filename, contents):
    if filename:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')        
        config = parseManagerConfigStr(decoded)        
        return { "origin" : filename, "config" : config.model_dump_json() }
    raise PreventUpdate

@callback(
    Output("wfman-config-upload-button", "children"),
    Input("wfman-config-store", "data")
    )
def set_upload_button_text_to_filename(store_data):
    if store_data:
        return store_data["origin"]
    else:
        return "Choose the workflow manager configuration file (JSON)"




#If we we have the use-workflow-manager toggle enabled, we cannot start unless the wfman configuration exists
@callback(
Output("start", "disabled"),
Input("options-switches","value"),
Input("wfman-config-store", "data"),
State("options-switches","value"),
State("wfman-config-store", "data"),
)
def disable_start_until_wfman_cfg_upload(option_values_trigger, store_trigger, options_values, store_data):   
    if "use_workflow_manager" in options_values and not store_data:
        return True
    else:
        return False


@callback(
    Output("chat-tab-pane", "style"),
    Output("job-tab-pane", "style"),
    Input("tabs", "value"),
)
def toggle_tabs(active_tab):
    if active_tab == "chatbot":
        return {"display": "block"}, {"display": "none"}
    return {"display": "none"}, {"display": "block"}

def setupTabs(use_agent: bool, use_wflow_man : bool):
    tabs_children = []
    if use_agent and use_wflow_man:
        tabs_children.append( dcc.Tab(label="Workflow Creation Agent", value="chatbot") )        
        tabs_children.append( dcc.Tab(label="Job Monitor", value="job_status") )
    tabs_value = "chatbot" if use_agent else "job_status"
    return tabs_children, tabs_value

@callback(
    Output("tabs-wrapper", "style"), #switch on the tabs
    Output("tabs", "children"), #populate the active tabs
    Output("tabs", "value"), #set default tab
    Output("startup-controls", "style"), #hide control panel
    Output("ws","send",allow_duplicate=True), #sent start signal to backend
    Input("start", "n_clicks"),
    State("state-file", "value"),
    State("options-switches", "value"),
    State("wfman-config-store", "data"),
    prevent_initial_call=True,
)
def start_agent(n_clicks, state_file, option_values, wfman_config):
    if not n_clicks:
        raise PreventUpdate

    option_values = option_values or []
    
    state_file = (state_file or "").strip() or "ckpoint_state.json"
    reload_state = "reload_state" in option_values
    use_agent = "use_agent" in option_values
    use_workflow_manager = "use_workflow_manager" in option_values

    workflow_config = {"state_file": state_file, "reload_state": reload_state, "use_agent" : use_agent, "use_workflow_manager" : use_workflow_manager, "workflow_manager_config" : wfman_config["config"] }

    tabs_children, tabs_value = setupTabs(use_agent, use_workflow_manager)
    
    return {"display": "block"}, tabs_children, tabs_value,  {"display" : "none" },  json.dumps({'task' : 'start', 'content' : json.dumps(workflow_config)})

@callback( Output("chat-component", "messages", allow_duplicate=True),
           Input("ws", "message"),
           State("chat-component", "messages"),
           prevent_initial_call=True
          )
def receiveAgentResponse(server_message, messages):
    if server_message and (j := json.loads(server_message['data']))['task'] == 'agent_output':
        agent_message = j['content']        
        return messages + [ { "role" : "assistant", "content" : agent_message } ]
    raise PreventUpdate


@callback(
    [ Output("chat-component", "messages"), Output("ws","send",allow_duplicate=True) ],
    Input("chat-component", "new_message"),
    State("chat-component", "messages"),
    prevent_initial_call=True,
)
def handleUserInput(new_message, messages):
    if new_message == None:
        raise PreventUpdate
    print("CHAT COMPONENT DETECTED USER RESPONSE", new_message['content'])
    
    return messages + [ new_message ], json.dumps({ "task" : "user_response", "content" : new_message['content'], "nonce": time.time_ns()  })  #use a timestamp to ensure each message is unique. This prevents the stupid socket library from dropping responses with the same content

@callback( Output("wfman-log", "value"),
           Input("ws", "message"),
           State("wfman-log", "value")
          )
def receiveWfmanLogOutput(server_message, log):
    if server_message and (j := json.loads(server_message['data']))['task'] == 'wfman_log':
        return (log if log else "") + j['content']
    raise PreventUpdate

@callback( Output("wfapi-log", "value"),
           Input("ws", "message"),
           State("wfapi-log", "value")
          )
def receiveWfapiLogOutput(server_message, log):
    if server_message and (j := json.loads(server_message['data']))['task'] == 'wfapi_log':
        return (log if log else "") + j['content']
    raise PreventUpdate


@callback( Output("transfer-monitor", "rowTransaction"),
           Input("ws", "message")
          )
def addOrUpdateTransferEntry(server_message):
    if server_message and (j := json.loads(server_message['data']) )['task'] in ('add_transfer', 'update_transfer'):
        print("TRANSFER INSTRUCTION",j)
        content = json.loads(j['content'])
        
        row = { "job ID" : content['job_id'], "api ID" : content['api_key'], "origin" : content['origin'], "destination" : content['destination'], "status" : content['api_status'] }
        cmd = 'add' if j['task'] == 'add_transfer' else 'update'
        print("TRANSFER CMD",cmd,"UPDATED ROW",row)
        
        return { cmd : [ row ] }
    raise PreventUpdate


@callback( Output("compute-monitor", "rowTransaction"),
           Input("ws", "message")
          )
def addOrUpdateComputeEntry(server_message):
    if server_message and (j := json.loads(server_message['data']) )['task'] in ('add_compute', 'update_compute'):
        print("COMPUTE INSTRUCTION",j)
        content = json.loads(j['content'])
        
        row = { "job ID" : content['job_id'], "api ID" : content['api_key'], "machine" : content['machine'], "queue" : content['queue'], "time" : content['time'], "status" : content['api_status'] }
        cmd = 'add' if j['task'] == 'add_compute' else 'update'
        print("COMPUTE CMD",cmd,"UPDATED ROW",row)
        
        return { cmd : [ row ] }
    raise PreventUpdate


@callback( Output("error-display", "value"),
           Output("error-display", "style"),
           Output("error-display-wrapper", "style"),
           Input("ws", "message"),
           State("error-display", "value")
          )
def receiveServerError(server_message, log):
    if server_message and (j := json.loads(server_message['data']))['task'] == 'server_error':
        err = json.loads(j['content'])
        content = (log if log else "") + json.dumps(err,indent=4)
        lines = content.count("\n") + 1 if content else 1
        height = max(300, lines * 20)
        
        return content , {'width': '100%', 'height': height, 'color' : 'red'},   {"display" : "block"}
    raise PreventUpdate


make_scroll_callback('wfman-log', app_dash)
make_scroll_callback('wfapi-log', app_dash)
