from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi import WebSocket, WebSocketDisconnect

import io
import time
import asyncio
import websockets

app = FastAPI()

main_loop = None
@app.on_event("startup")
async def capture_loop():
   global main_loop
   main_loop = asyncio.get_running_loop()

send_queue = asyncio.Queue()
recv_queue = asyncio.Queue()
    
def printToSocket(*args, **kwargs):
    buf = io.StringIO()
    print(*args, *kwargs, file=buf)
    msg = buf.getvalue()
    
    asyncio.run_coroutine_threadsafe(send_queue.put(msg), main_loop).result()
    time.sleep(0.3) #allows successive calls to print to all render correctly
    
def inputFromSocket(query):
    msg = query
    asyncio.run_coroutine_threadsafe(send_queue.put(msg), main_loop).result()
    return asyncio.run_coroutine_threadsafe(recv_queue.get(), main_loop).result().strip()

server_workflow = None

def setServerWorkflow(workflow):
    global server_workflow
    server_workflow = workflow

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    assert server_workflow != None
    
    await websocket.accept()
    global send_queue, recv_queue
    send_queue = asyncio.Queue()
    recv_queue = asyncio.Queue()
    
    async def sender():
        while True:
            msg = await send_queue.get()
            await websocket.send_text(msg)

    async def receiver():
        while True:
            msg = await websocket.receive_text()
            await recv_queue.put(msg)

    try:
        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())
        await main_loop.run_in_executor(None, server_workflow)
        
        sender_task.cancel()
        receiver_task.cancel()
        
        while True:
            await asyncio.sleep(3600)
        

    except WebSocketDisconnect:
        print("Client disconnected")
