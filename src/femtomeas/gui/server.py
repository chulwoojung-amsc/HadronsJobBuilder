from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi import WebSocket, WebSocketDisconnect

import io
import time
import asyncio
import websockets
import json

app = FastAPI()

main_loop = None
@app.on_event("startup")
async def capture_loop():
   global main_loop
   main_loop = asyncio.get_running_loop()
   
####Send tasks on same queue but with task information attached 
send_queue = asyncio.Queue()

def sendToFrontend(task: str, content: str):
   """
   Send data via the websocket to the frontend
   task: description of the data
   content: the data
   """
   asyncio.run_coroutine_threadsafe(send_queue.put({"task" : task, "content" : content }), main_loop).result()
   time.sleep(0.3) #allows successive calls to print to all render correctly (seems to be an inherent limitation in dash that callbacks in quick succession do not work well)
   
def printToString(*args, **kwargs):
   buf = io.StringIO()
   print(*args, *kwargs, file=buf)
   return buf.getvalue()
   
def agentPrint(*args, **kwargs):
   sendToFrontend("agent_output", printToString(*args, *kwargs))

def workflowManagerLogGUI(*args, **kwargs):
   sendToFrontend("wfman_log", printToString(*args, *kwargs))
    
def workflowAPIlogGUI(*args, **kwargs):
   sendToFrontend("wfapi_log", printToString(*args, *kwargs))

    
####Receiver tasks on different queues
agent_recv_user_queue = asyncio.Queue()
    
def agentQuery(query):
    sendToFrontend("agent_output",query)
    user_response = asyncio.run_coroutine_threadsafe(agent_recv_user_queue.get(), main_loop).result().strip()
    print("AGENT QUERY FUNCTION DETECTED USER RESPONSE", user_response)
    return user_response
   
server_workflow = None

#Allow overriding server workflow
def setServerWorkflow(workflow):
    global server_workflow
    server_workflow = workflow

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    assert server_workflow != None
    
    await websocket.accept()
    global agent_recv_user_queue, send_queue
    agent_recv_user_queue = asyncio.Queue()
    send_queue = asyncio.Queue()
    
    async def sender():
       while True:
          json = await send_queue.get()
          print("SENDER",json)
          
          await websocket.send_json(json)

    #Task that receives input from the frontend and redirects as appropriate
    #Arguments to agent start
    workflow_config = None
    start_event = asyncio.Event()
    
    async def receiver():
        while True:
            print("RECEIVER TASK WAITING FOR MESSAGE")
            msg = await websocket.receive_json()
            print("RECEIVER TASK RECEIVED MESSAGE",msg)
            if msg['task'] == 'user_response':
               print("RECEIVER TASK DETECTED USER RESPONSE", msg['content'])
               await agent_recv_user_queue.put(msg['content'])
            elif msg['task'] == 'start':
               nonlocal workflow_config
               workflow_config = json.loads(msg['content'])
               start_event.set()
         

    try:
        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())

        await start_event.wait()
        await main_loop.run_in_executor(None, server_workflow, workflow_config)
        
        sender_task.cancel()
        receiver_task.cancel()
        
        while True:
            await asyncio.sleep(3600)
        

    except WebSocketDisconnect:
        print("Client disconnected")
