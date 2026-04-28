from femtomeas.gui.server import app, setServerWorkflow, agentPrint, agentQuery, workflowManagerLogGUI, workflowAPIlogGUI, sendToFrontend
from femtomeas.gui.frontend import app_dash
from fastapi.middleware.wsgi import WSGIMiddleware
import femtomeas.meas_config_agent.common as common
from femtomeas.workflow_manager.manager_config import readManagerConfigStr
from femtomeas.workflow_manager.manager import JobManager, ActionClass
from femtomeas.workflow_manager.hadrons_workflow import hadronsSubmissionAgent

import traceback
import time
import os
import json
import threading
import femtomeas.workflow_manager.logging as wfman_logging

from langchain_openai import ChatOpenAI
from femtomeas.meas_config_agent import agent

common.print_func = agentPrint
common.input_func = agentQuery
common.output_style = "markdown"

wfman_logging.wfman_log_func = workflowManagerLogGUI
wfman_logging.api_log_func = workflowAPIlogGUI
wfman_logging.update_gui_func = sendToFrontend #allow the manager to send action status updates to the GUI tables

amsc_llm_0t = ChatOpenAI(
    model="gpt-oss-120b",
    base_url="https://api.i2-core.american-science-cloud.org/",
    temperature=0
)

llm = amsc_llm_0t

#def global_exception_handler(exctype, value, traceback):
#    print(f"Caught global error: {exctype}:{value}:{traceback}")
#sys.excepthook = global_exception_handler

def thread_error_handler(args):
    tb_list = traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
    sendToFrontend("server_error", json.dumps(tb_list))

threading.excepthook = thread_error_handler

def workflow(config : dict):
    jman = None

    if config["use_workflow_manager"]:
        db_file = "jobs.db"
        workflowManagerLogGUI(f"Starting job manager with database {db_file} and config:\n{ json.dumps( json.loads(config["workflow_manager_config"]), indent=4) }")
        
        readManagerConfigStr(config["workflow_manager_config"])
        jman = JobManager(db_file)
        jman.start()

        #Inform the frontend about active transfers
        with jman as jd:
            transfer_status = jd.getActiveActions(ActionClass.TRANSFER)            
            compute_status = jd.getActiveActions(ActionClass.COMPUTE)
            for t in transfer_status:
                sendToFrontend('add_transfer', json.dumps(t))
            for c in compute_status:
                sendToFrontend('add_compute', json.dumps(t))


                
    if config["use_agent"]:
        query = "" if config["reload_state"] else agentQuery("Describe the observables you wish to compute")
        state = agent(query, llm, ckpoint_file=config["state_file"], reload_state=config["reload_state"])
       
        if config["use_workflow_manager"]:
            hadronsSubmissionAgent(state, jman, llm)

    if config["use_agent"]:
        agentPrint(f"Agent interaction has finished. Final state information has been stored in {config['state_file']}. Goodbye")            

            
    if jman:
        time.sleep(10)
        jman.stop() #will wait until the job queue is complete
        workflowManagerLogGUI(f"Job manager has finished")



setServerWorkflow(workflow)

app.mount("/", WSGIMiddleware(app_dash.server))

