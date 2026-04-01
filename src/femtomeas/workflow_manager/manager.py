from dataclasses import dataclass
import pickle
import sqlite3
from pathlib import Path
from typing import List, Tuple
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
import time
import threading

from .api_general import *
from .hadrons import submitHadronsJob
from . import globals

from enum import Enum

class HadronsJobSpec:
    job_subdir : str #Job directory relative to base sandbox. The actual job will be executed within a subdirectory of this named by the id to ensure uniqueness
    xml_spec : bytes #The XML spec as a bytestring
    grid : Tuple[int,int,int,int]

    def __init__(self, job_subdir, xml : HadronsXML, grid):
        self.job_subdir = job_subdir
        self.xml_spec = xml.toBytes()
        self.grid = grid

    def __repr__(self):
        xml = HadronsXML()
        xml.fromBytes(self.xml_spec)
        xmlstr = xml.toString()
        
        return f"HadronsJobSpec(job_subdir={self.job_subdir}, xml_spec={xmlstr}, grid={self.grid})"

    def writeXML(self, filename):
        xml = HadronsXML()
        xml.fromBytes(self.xml_spec)
        xml.write(filename)
        print("XML written to",filename)
        
@dataclass
class TransferActionBase:
    pass
        
@dataclass
class TransferToAction(TransferActionBase):
    source_endpoint: str
    source_path: str
    machine: str
    dest_path: str

    def initiateAction(self, job_id)-> str:
        return globusCopyToMachine(self.machine, self.dest_path, self.source_endpoint, self.source_path)
    
@dataclass
class TransferFromAction(TransferActionBase):
    machine: str
    source_path: str
    dest_endpoint: str
    dest_path: str

    def initiateAction(self, job_id)-> str:
        return globusCopyFromMachine(self.dest_endpoint, self.dest_path, self.machine, self.source_path)
    
@dataclass
class ComputeActionBase:
    machine : str
    account : str
    queue : str
    time : str
    
@dataclass
class HadronsComputeAction(ComputeActionBase):
    spec :  HadronsJobSpec
    mpi : Tuple[int, int, int, int]
   
    def initiateAction(self, job_id)->str:
        xml_file = f"/tmp/hadrons_xml.{job_id}"
        self.spec.writeXML(xml_file)
        assert self.machine in globals.remote_workdir
        
        rundir = globals.remote_workdir[self.machine] + "/" + self.spec.job_subdir + f"/{job_id}"
        print(f"Job {job_id} machine {self.machine} rundir {rundir}")
        return submitHadronsJob(self.machine, xml_file, rundir, self.account, self.queue, self.time, self.spec.grid, self.mpi)

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
        return status

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
    def __init__(self, filename: str | None = None):
        db_path = ":memory:" if filename is None else str(Path(filename).expanduser())
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
        
        with self.conn as conn:
            if condition[0] == "COMPLETE" and condition[1] == None:
                progress_actions = conn.execute("SELECT job_id, workflow, workflow_stage, head_action_type, head_action_status FROM jobs WHERE head_action_class != ? AND head_action_status = ?",
                                                (ActionClass.NONE.name,ActionStatus.COMPLETED.name)).fetchall()
            elif condition[0] == "VALID_IN" and isinstance(condition[1],list):
                placeholders = ",".join("?" for _ in condition[1])
                progress_actions = conn.execute(f"SELECT job_id, workflow, workflow_stage, head_action_type, head_action_status FROM jobs WHERE head_action_class != ? AND job_id IN ({placeholders}) AND head_action_status IN (?,?)",
                                                ( ActionClass.NONE.name, *condition[1], ActionStatus.PENDING.name, ActionStatus.COMPLETED.name )
                                                )
            else:
                raise Exception("Unknown condition")
                
            for a in progress_actions:
                #Get information on the next workflow task
                workflow = _unser(a['workflow'])
                workflow_stage = a['workflow_stage']
               
                job_id = a['job_id']
                next_workflow_stage = workflow_stage+1

                next_action = None if next_workflow_stage == len(workflow) else workflow[next_workflow_stage]
                next_action_class = ActionClass.NONE if next_action == None else actionClass(next_action)
                next_action_status = ActionStatus.COMPLETED if next_action == None else ActionStatus.PENDING
                
                print(f"Progressing job {job_id} action {a['head_action_type']} status {a['head_action_status']} to action {type(next_action).__name__}")
                
                #Update the next action and put into pending status
                conn.execute("UPDATE jobs SET head_action_type = ?, head_action_class = ?, head_action_status = ?, head_action_id = ?, last_status_change = ?, workflow_stage = ? WHERE job_id = ?",
                             (type(next_action).__name__,  next_action_class.name, next_action_status.name, -1, int(time.time()), next_workflow_stage, job_id )
                              )

                #Gather information to initiate next action
                if next_action_status == ActionStatus.PENDING:
                    pending_actions.append( (next_action_class, next_action, job_id ) )

        #Initiate the required actions
        head_action_updates = [] #(job_id, head_action_id, head_action_status)                
        #Initiate pending actions
        for action_class, action, job_id in pending_actions:
            print(f"Setting up new {action_class.name} for {job_id}:", action)
            aman = self.action_man[action_class]
            action_id = aman.startAction(action, job_id)
            action_status, _ = aman.queryStatus(action_id)
            head_action_updates.append( (job_id, action_id, action_status) )
            
        ####WARNING: If the manager is killed here we can think the action is pending but it is already underway. The action DBs will know about it but not the main DB. How to fix?
        #Maybe instead of PENDING we have some other marker, e.g. SCHEDULING. Then if we come across an entry with this status we will know to check the action DB to see if it was actually scheduled
        
        #Update job state DB
        with self.conn as conn:
            for job_id, action_id, status in head_action_updates:
                conn.execute("UPDATE jobs SET head_action_status = ?, head_action_id = ?, last_status_change = ? WHERE job_id = ?",
                             (status.name, action_id, int(time.time()), job_id )
                             )
 


    def startWorkflows(self, job_ids : list):
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
                print(f"Progressed job {job_id} action {t['head_action_type']} of class {action_class.name} to {updates[job_id].name}")
                
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
    
        
class JobManager:
    def __init__(self, filename: str | None = None, poll_freq=30):
        """
        poll_freq: how often the action monitors poll the API for status updates
        """
        
        self.job_data = JobData(filename)
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self.poll_freq = poll_freq
    
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        self._stop.set()

        if self._thread is not None:
            self._thread.join()

    def _run(self):
        self._lock.acquire()
        
        while not self._stop.is_set():
            #A safe checkpoint for allowing the user to modify the state
            self._lock.release()
            time.sleep(0.5)
            self._lock.acquire()
            
            if self._stop.is_set():
                break

            self.job_data.progressActiveState(poll_freq=self.poll_freq)
            time.sleep(2)
        self._lock.release()

    def __call__(self, op_lambda):
        """
        Perform an operation on the JobData database under lock
        """
        with self._lock:
            return op_lambda(self.job_data)
        
