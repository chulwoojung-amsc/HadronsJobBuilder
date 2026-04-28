from dataclasses import dataclass
import pickle
import sqlite3
import tempfile
import os
from pathlib import Path
from typing import List, Tuple
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
import time
import threading
import json

from .api_general import *
from .hadrons import submitHadronsJob
from . import globals
from .logging import wfmanLog, updateGUI

from enum import Enum
import re

def replaceJobIdSubstring(in_str, job_id):
    """Replace instances of <JOBID> with the job index in path strings"""
    return re.sub(r'<JOBID>', str(job_id), in_str)

class HadronsJobSpec:
    job_rundir : str #Job run directory. The <JOBID> substring will be replaced by the job index if present
    xml_spec : bytes #The HadronsXML spec as a bytestring
    grid : Tuple[int,int,int,int]

    def __init__(self, job_rundir, xml : HadronsXML, grid):
        self.job_rundir = job_rundir
        self.xml_spec = xml.toBytes()
        self.grid = grid

    def __repr__(self):
        xml = HadronsXML()
        xml.fromBytes(self.xml_spec)
        xmlstr = xml.toString()
        
        return f"HadronsJobSpec(job_rundir={self.job_rundir}, xml_spec={xmlstr}, grid={self.grid})"

    def writeXML(self, filename):
        xml = HadronsXML()
        xml.fromBytes(self.xml_spec)
        xml.write(filename)
        wfmanLog("XML written to",filename)
        
@dataclass
class TransferActionBase:
    def getInfo(self)->dict:
        """
        Return the transfer information in a common dictionary format with entries {"origin", "destination"}
        """
        raise NotImplementedError("Derived class must implement getTransferInfo")

@dataclass
class TransferToAction(TransferActionBase):
    #The <JOBID> substring will be replaced by the job index if present in the path strings
    source_endpoint: str 
    source_path: str
    machine: str
    dest_path: str

    def initiateAction(self, job_id)-> str:
        assert self.machine in globals.remote_workdir
        source_path = replaceJobIdSubstring(self.source_path, job_id)
        dest_path = replaceJobIdSubstring(self.dest_path, job_id)
        
        return globusCopyToMachine(self.machine, dest_path, self.source_endpoint, source_path)

    def getInfo(self)->dict:
        """
        Return the transfer information in a common dictionary format with entries {"origin", "destination"}
        """
        return { "origin" : f"{self.source_endpoint}:{self.source_path}",  "destination" : f"{self.machine}:{self.dest_path}" }

    
@dataclass
class TransferFromAction(TransferActionBase):
    #The <JOBID> substring will be replaced by the job index if present in the path strings
    machine: str
    source_path: str
    dest_endpoint: str
    dest_path: str
    
    def initiateAction(self, job_id)-> str:
        assert self.machine in globals.remote_workdir
        source_path = replaceJobIdSubstring(self.source_path, job_id)
        dest_path = replaceJobIdSubstring(self.dest_path, job_id)
        
        return globusCopyFromMachine(self.dest_endpoint, dest_path, self.machine, source_path)

    def getInfo(self)->dict:
        """
        Return the transfer information in a common dictionary format with entries {"origin", "destination"}
        """
        return { "origin" : f"{self.machine}:{self.source_path}", "destination" : f"{self.dest_endpoint}:{self.dest_path}" }


    
    
@dataclass
class ComputeActionBase:
    machine : str
    account : str
    queue : str
    time : str

    def getInfo(self)->dict:
        """
        Return the transfer information in a common dictionary format with entries {"machine", "queue", "time"}
        """
        return {"machine" : self.machine, "queue" : self.queue, "time" : self.time}

    
@dataclass
class HadronsComputeAction(ComputeActionBase):
    spec :  HadronsJobSpec
    mpi : Tuple[int, int, int, int]
   
    def initiateAction(self, job_id)->str:
        _, xml_file = tempfile.mkstemp(prefix="hadrons_xml_", text=True, dir="/tmp", suffix=".xml")
        self.spec.writeXML(xml_file)
        assert self.machine in globals.remote_workdir
        
        rundir = replaceJobIdSubstring(self.spec.job_rundir, job_id)
        wfmanLog(f"Job {job_id} machine {self.machine} rundir {rundir}")
        return submitHadronsJob(self.machine, xml_file, rundir, self.account, self.queue, self.time, self.spec.grid, self.mpi, delete_xml_after_upload = True)

class ActionClass(Enum):
    NONE = 0
    TRANSFER = 1
    COMPUTE = 2
    
def actionClass(action):
    if issubclass(type(action), TransferActionBase):
        return ActionClass.TRANSFER
    elif issubclass(type(action), ComputeActionBase):
        return ActionClass.COMPUTE
    else:
        raise Exception("Unknown action type",type(action))    
    
def _ser(obj):
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
def _unser(ser):
    return pickle.loads(ser)

class ActionStatus(Enum):
    PENDING = 0 #not yet started
    ACTIVE = 1 #a live action (any status not failed or completed, e.g. queued, new, etc)
    COMPLETED = 2 #action completed successfully
    FAILED = 3 #action failed
        
class ActionManager:
    def _queryStatusInternal(self, machine, api_key):
        """Return the API status"""        
        raise NotImplementedError("Derived class must implement _queryStatusInternal")
    
    def __init__(self, connection : sqlite3.Connection, table_name, api_action_status_map):
        self.conn = connection
        self.table_name = table_name
        self.api_action_status_map = api_action_status_map #map between return status from the API to an ActionStatus
        
        with self.conn as conn:        
            conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
            action_id INTEGER PRIMARY KEY,
            machine TEXT,
            details BLOB,
            api_key TEXT,
            api_status TEXT,
            action_status TEXT,
            last_update INTEGER,
            job_id INTEGER
            )
            """)

    def startAction(self, action, job_id):
        api_key = action.initiateAction(job_id)
        api_status = self._queryStatusInternal(action.machine, api_key)
        action_status = self.api_action_status_map[api_status]
        
        with self.conn as conn:        
            cur = conn.execute(f"INSERT INTO {self.table_name}(machine, details, api_key, api_status, action_status, last_update, job_id) VALUES (?,?,?,?,?,?,?)",
                               (action.machine, _ser(action), api_key, api_status, action_status.name, int(time.time()), job_id)
                               )
            action_id = cur.lastrowid
            return action_id


    def updateStatuses(self):
        """Force update of all active statuses"""
        with self.conn as conn:
            actions = conn.execute(f"SELECT action_id, api_key, machine FROM {self.table_name} WHERE action_status = ?", (ActionStatus.ACTIVE.name,) ).fetchall()
            for t in actions:
                api_status = self._queryStatusInternal(t['machine'],t['api_key'])
                action_status = self.api_action_status_map[api_status]

                conn.execute(f"UPDATE {self.table_name} SET api_status = ?, action_status = ?, last_update = ? WHERE action_id = ?", (api_status, action_status.name, int(time.time()), t['action_id']) )

    def __str__(self):
        with self.conn as conn:
            actions = conn.execute(f"SELECT api_key, action_status, api_status, last_update FROM {self.table_name}").fetchall()
            out = ""
            for t in actions:
                out = out + f"({t['api_key']},{t['action_status']}[ {t['api_status']} ],{t['last_update']})\n"
            return out
                    
            
    def queryStatus(self, action_id, update_freq=30, force_update=False):
        """
        Query the status of an action by id. This will use the last known status unless it has been more than update_freq seconds since the last poll or force_update == True
        Return: action_status, api_status
        """
        with self.conn as conn:
            action = conn.execute(f"SELECT action_status, api_status, last_update, machine, api_key FROM {self.table_name} WHERE action_id = ?", (action_id,) ).fetchone()
            action_status = getattr(ActionStatus, action['action_status'],None)
            api_status = action['api_status']
            if force_update or (int(time.time()) > action['last_update'] + update_freq):
                api_status = self._queryStatusInternal(action['machine'],action['api_key'])
                action_status = self.api_action_status_map[api_status]
                
                conn.execute(f"UPDATE {self.table_name} SET api_status = ?, action_status = ?, last_update = ? WHERE action_id = ?", (api_status, action_status.name, int(time.time()), action_id) )
            return action_status, api_status
        
    def waitForAction(self, action_id, check_freq=30):
        """Blocking wait until the action either completes or fails. Status checks are performed every check_freq seconds. Return the final status."""
        while( (action_status := self.queryStatus(action_id,force_update=True)[0] ) == ActionStatus.ACTIVE):
            time.sleep(check_freq)
        return action_status

    def getActiveActions(self):
        """
        Get all active actions and returning a list of dictionaries, each containing "api_key", "api_status" and other custom fields defined on a per-action basis
        """
        out = []
        with self.conn as conn:
            entries = conn.execute(f"SELECT job_id, api_key, api_status, details FROM {self.table_name} WHERE action_status = ?", (ActionStatus.ACTIVE.name,) ).fetchall()
            for entry in entries:
                action = _unser(entry['details'])
                dc = action.getInfo()
                dc['job_id'] = entry['job_id']
                dc['api_key'] = entry['api_key']
                dc['api_status'] = entry['api_status']
                out.append(dc)
        return out

    def getActionInfo(self, action_id)->dict:
        """
        For the given action, return a dictionary containing "api_key", "api_status" and other custom fields defined on a per-action basis
        """        
        with self.conn as conn:
            entry = conn.execute(f"SELECT job_id, details, api_status, api_key FROM {self.table_name} WHERE action_id = ?", (action_id,) ).fetchone()
            action = _unser(entry['details'])
            dc = action.getInfo()
            dc['job_id'] = entry['job_id']
            dc['api_key'] = entry['api_key']
            dc['api_status'] = entry['api_status']
            return dc
            
    
class DataTransfers(ActionManager):
    def __init__(self, connection : sqlite3.Connection):
        #"ACTIVE"  The task is in progress.
        #"INACTIVE" The task has been suspended and will not continue without intervention. Currently, only credential expiration will cause this state.
        #"SUCCEEDED"  The task completed successfully.
        #"FAILED"  The task or one of its subtasks failed, expired, or was canceled.
        smap = { "ACTIVE" : ActionStatus.ACTIVE, "INACTIVE" : ActionStatus.FAILED, "SUCCEEDED" : ActionStatus.COMPLETED, "FAILED" : ActionStatus.FAILED }    
        super().__init__(connection, "transfers", smap)
    def _queryStatusInternal(self, machine, api_key):
        return globusTransferStatus(machine, api_key)

class ComputeActions(ActionManager):
    def __init__(self, connection : sqlite3.Connection):
        #IRI API: 
        #0"new"
        #1"queued"
        #2"active"
        #3"completed"
        #4"failed"
        #5"canceled"
        smap = {"new": ActionStatus.ACTIVE, "queued" : ActionStatus.ACTIVE, "active" : ActionStatus.ACTIVE,
                "completed" : ActionStatus.COMPLETED, "failed" : ActionStatus.FAILED, "canceled" : ActionStatus.FAILED }
        super().__init__(connection, "computes", smap)
    def _queryStatusInternal(self, machine, api_key):
        return getJobState(machine, api_key)

    
class JobData:
    def __init__(self, filename: str | None = None, max_workflows_active=10):
        db_path = ":memory:" if filename is None else str(Path(filename).expanduser())
        self.max_workflows_active = max_workflows_active
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        self.action_man = { ActionClass.TRANSFER : DataTransfers(self.conn),
                            ActionClass.COMPUTE : ComputeActions(self.conn) }

        with self.conn as conn:        
            conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
            job_id INTEGER PRIMARY KEY,
            job_group TEXT,
            workflow BLOB NOT NULL,
            workflow_stage INTEGER NOT NULL,
            head_action_type TEXT,
            head_action_class TEXT,
            head_action_status TEXT,
            head_action_id INTEGER,
            last_status_change INTEGER
            )
            """)
       
    def enqueueJob(self, workflow, job_group = None):
        assert len(workflow) > 0
        with self.conn as conn:        
            cur = conn.execute("INSERT INTO jobs(job_group, workflow, workflow_stage, head_action_type, head_action_class, head_action_status, last_status_change) VALUES (?,?,?,?,?,?,?)",
                               (job_group, _ser(workflow), -1,
                                type(workflow[0]).__name__,
                                actionClass(workflow[0]).name,
                                ActionStatus.PENDING.name,
                                int(time.time())
                                ) )
            job_id = cur.lastrowid
            return job_id
   
    def jobStatus(self, job_id):
        with self.conn as conn:
            row = dict(conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone())
            row['head_action_status'] = ActionStatus[row['head_action_status']]
            row['head_action_class'] = ActionClass[row['head_action_class']]
            return row


    def progressWorkflows(self, condition: Tuple[str, list | None]):
        """
        Find actions matching a certain condition and initiate the next stage of the workflow
        condition:
           ("COMPLETE",None) - Progress all active workflows whose head action status is ActionStatus.COMPLETED and for which there are remaining workflow actions
           ("VALID_IN", [job_id1, job_id2, ...]) - Progress valid workflows (those whose head action status is either ActionStatus.PENDING or ActionStatus.COMPLETED, not in a failure state) based on a list of job indices.
        """

        pending_actions = []        
        completed_actions = [] #list of action ids of completed actions
        
        with self.conn as conn:
            if condition[0] == "COMPLETE" and condition[1] == None:
                progress_actions = conn.execute("SELECT job_id, workflow, workflow_stage, head_action_type, head_action_status, head_action_id, head_action_class FROM jobs WHERE head_action_class != ? AND head_action_status = ?",
                                                (ActionClass.NONE.name,ActionStatus.COMPLETED.name)).fetchall()
            elif condition[0] == "VALID_IN" and isinstance(condition[1],list):
                placeholders = ",".join("?" for _ in condition[1])
                progress_actions = conn.execute(f"SELECT job_id, workflow, workflow_stage, head_action_type, head_action_status, head_action_id, head_action_class FROM jobs WHERE head_action_class != ? AND job_id IN ({placeholders}) AND head_action_status IN (?,?)",
                                                ( ActionClass.NONE.name, *condition[1], ActionStatus.PENDING.name, ActionStatus.COMPLETED.name )
                                                )
            else:
                raise Exception("Unknown condition" + str(condition))
                
            for a in progress_actions:              
                #Get information on the next workflow task
                workflow = _unser(a['workflow'])
                workflow_stage = a['workflow_stage']

                #Record completed actions so we can update any monitors
                if a['head_action_status'] == ActionStatus.COMPLETED.name:
                    completed_actions.append(  (a['head_action_id'], getattr(ActionClass, a['head_action_class'], None) ) )
                
                job_id = a['job_id']
                next_workflow_stage = workflow_stage+1

                next_action = None if next_workflow_stage == len(workflow) else workflow[next_workflow_stage]
                next_action_class = ActionClass.NONE if next_action == None else actionClass(next_action)
                next_action_status = ActionStatus.COMPLETED if next_action == None else ActionStatus.PENDING
                
                wfmanLog(f"Progressing job {job_id} action {a['head_action_type']} status {a['head_action_status']} to action {type(next_action).__name__}")
                
                #Update the next action and put into pending status
                conn.execute("UPDATE jobs SET head_action_type = ?, head_action_class = ?, head_action_status = ?, head_action_id = ?, last_status_change = ?, workflow_stage = ? WHERE job_id = ?",
                             (type(next_action).__name__,  next_action_class.name, next_action_status.name, -1, int(time.time()), next_workflow_stage, job_id )
                              )

                #Gather information to initiate next action
                if next_action_status == ActionStatus.PENDING:
                    pending_actions.append( (next_action_class, next_action, job_id ) )


        #Inform GUI regarding completed actions (requires database activity)
        for action_id, action_class in completed_actions:
            info = self.action_man[action_class].getActionInfo(action_id)
            if action_class == ActionClass.TRANSFER:
                updateGUI('update_transfer', json.dumps(info))
            elif action_class == ActionClass.COMPUTE:
                updateGUI('update_compute', json.dumps(info))

        #Initiate the required actions
        head_action_updates = [] #(job_id, head_action_id, head_action_status, head_action_class)

        for action_class, action, job_id in pending_actions:
            wfmanLog(f"Initiating action of type {action_class.name} for {job_id}")
            aman = self.action_man[action_class]
            action_id = aman.startAction(action, job_id)
            action_status, _ = aman.queryStatus(action_id)
            head_action_updates.append( (job_id, action_id, action_status, action_class) )
            
        ####WARNING: If the manager is killed here we can think the action is pending but it is already underway. The action DBs will know about it but not the main DB. How to fix?
        #Maybe instead of PENDING we have some other marker, e.g. SCHEDULING. Then if we come across an entry with this status we will know to check the action DB to see if it was actually scheduled
        
        #Update job state DB
        with self.conn as conn:
            for job_id, action_id, status, _ in head_action_updates:
                conn.execute("UPDATE jobs SET head_action_status = ?, head_action_id = ?, last_status_change = ? WHERE job_id = ?",
                             (status.name, action_id, int(time.time()), job_id )
                             )

        #Inform GUI regarding new actions (requires database activity)
        for _, action_id, _, action_class in head_action_updates:
            info = self.action_man[action_class].getActionInfo(action_id)
            if action_class == ActionClass.TRANSFER:
                updateGUI('add_transfer', json.dumps(info))
            elif action_class == ActionClass.COMPUTE:
                updateGUI('add_compute', json.dumps(info))

                


    def startWorkflows(self, job_ids : list | None = None):
        """
        Start the workflows specified by the list of job ids. If None, additional workflows will be started until the total number of active workflows reaches the maximum
        """
        if job_ids == None:        
            with self.conn as conn:
                count = int(conn.execute("SELECT COUNT(*) FROM jobs WHERE head_action_status = ?", (ActionStatus.ACTIVE.name,) ).fetchone()[0])
                rem =  self.max_workflows_active - count

                if rem > 0:
                    toschedule = conn.execute("SELECT job_id FROM jobs WHERE head_action_status = ? ORDER BY job_id ASC LIMIT ?", (ActionStatus.PENDING.name, rem)).fetchall()                   
                    job_ids = [ j[0] for j in toschedule ]
                    if len(job_ids) > 0:
                        wfmanLog("Number of active workflows",count,"want to activate",rem,"more.\nActivating",len(job_ids),"workflows with job ids", job_ids)

        if job_ids:
            self.progressWorkflows(("VALID_IN",job_ids))
                        


    def progressActiveWorkflows(self):
        """
        Find COMPLETE actions and initiate the next stage of the workflow
        """
        self.progressWorkflows(("COMPLETE",None))
        
    def progressActiveActions(self, poll_freq=30, force_poll=False):
        """
        Find active actions nd try to progress their status.
        poll_freq: control the minimum time lag between manager polls of the API for status updates. Queries within this period return only the cached status.
        force_poll: force the manager to poll the API for status updates, use wisely!

        Return: dict  job_id -> new status
        """       
        with self.conn as conn:
            active_actions = conn.execute("SELECT head_action_id, job_id, head_action_type, head_action_class FROM jobs WHERE head_action_status = ?", (ActionStatus.ACTIVE.name, )).fetchall()

        updates = {}
        for t in active_actions:
            job_id = t['job_id']
            action_class = getattr(ActionClass, t['head_action_class'], None)
            aman = self.action_man[action_class] 
            action_status, _ = aman.queryStatus(t['head_action_id'], update_freq=poll_freq, force_update = force_poll)
            if action_status != ActionStatus.ACTIVE:
                updates[job_id] = action_status
            if job_id in updates:
                wfmanLog(f"Progressed job {job_id} action {t['head_action_type']} of class {action_class.name} to {updates[job_id].name}")
                
        #Update head action state
        if len(updates) > 0:
            with self.conn as conn:
                for job_id, status in updates.items():
                    conn.execute("UPDATE jobs SET head_action_status = ? WHERE job_id = ?",(status.name, job_id))
        return updates


    def progressActiveState(self, poll_freq=30, force_poll=False):
        """
        Updates knowledge of action state and then progresses the workflow for those actions that have completed
        poll_freq: control the minimum time lag between manager polls of the API for status updates. Queries within this period return only the cached status.
        force_poll: force the manager to poll the API for status updates, use wisely!"""
        self.progressActiveActions(poll_freq=poll_freq, force_poll=force_poll)
        self.progressActiveWorkflows()
    
    def countWorkflowsWithStatus(self, statuses : ActionStatus | list[ActionStatus]):
        with self.conn as conn:
            if isinstance(statuses, ActionStatus):            
                return int(conn.execute("SELECT COUNT(*) FROM jobs WHERE head_action_status = ?", (statuses.name,) ).fetchone()[0])
            elif isinstance(statuses, list):
                placeholders = ",".join("?" for _ in statuses)
                names = [s.name for s in statuses]                
                return int(conn.execute(f"SELECT COUNT(*) FROM jobs WHERE head_action_status IN ({placeholders})", (*names,) ).fetchone()[0])
            else:
                raise Exception("Unexpected type for 'statuses'", type(statuses))

    def getActiveActions(self, action_class : ActionClass)->dict:
        """
        Get all active actions and returning a list of dictionaries, each containing "api_key", "api_status" and other custom fields defined on a per-action basis
        """
        return self.action_man[action_class].getActiveActions()


            
class JobManager:
    def __init__(self, filename: str | None = None, poll_freq=30, max_workflows_active=10):
        """
        poll_freq: how often the action monitors poll the API for status updates
        max_workflows_active: if >0, the manager will attempt to maintain this many active workflows, activating more when others finish; if 0, they must be activated manually
        """
        
        self.job_data = JobData(filename, max_workflows_active=max_workflows_active)
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self.poll_freq = poll_freq

    def isAlive(self):
        return self._thread is not None and self._thread.is_alive()
        
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self, wait_until_done=True):
        """
        Ask the manager thread to stop and wait until it does.
        wait_until_done : block until there are no more active or pending workflows before stopping (default True)
        """
        def __nincomplete():
            with self._lock:
                return self.job_data.countWorkflowsWithStatus([ ActionStatus.PENDING, ActionStatus.ACTIVE ])
        
        if wait_until_done:
            while(__nincomplete() > 0):
                time.sleep(2)
        
        self._stop.set()

        if self._thread is not None:
            self._thread.join()

    def _run(self):
        self._lock.acquire()
        
        while not self._stop.is_set():
            #A safe checkpoint for allowing the user to obtain a lock and modify the state (e.g. manually activating workflows, restarting after failure, etc)
            self._lock.release()
            time.sleep(0.5)
            self._lock.acquire()
            
            if self._stop.is_set():
                break

            self.job_data.startWorkflows() #start new workflows as required
            self.job_data.progressActiveState(poll_freq=self.poll_freq) #attempt to progress active workflows
            time.sleep(2)
        self._lock.release()

    def __call__(self, op_lambda):
        """
        Perform an operation on the JobData database under lock
        """
        with self._lock:
            return op_lambda(self.job_data)
        
    def __enter__(self):
        """
        Allow the user to acquire a lock on the database for manipulation using 'with'
        """
        self._lock.acquire()
        return self.job_data
        
    def __exit__(self,exc_type, exc_val, exc_tb):
        self._lock.release() #unlock before exception!
        if exc_type:
            raise Exception("Caught exception",exc_type,exc_val,exc_tb)
            

    
