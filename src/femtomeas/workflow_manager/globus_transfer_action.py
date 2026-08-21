from dataclasses import dataclass
from .actions_base import replaceJobIdSubstring, TransferActionBase
from .api_general import globusCopyFromMachine, globusCopyToMachine, globusTransferStatus
from .action_manager import ActionManager, ActionStatus
import sqlite3

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

class DataTransfers(ActionManager):
    """Action manager for Globus transfers"""
    def __init__(self, connection : sqlite3.Connection):
        #"ACTIVE"  The task is in progress.
        #"INACTIVE" The task has been suspended and will not continue without intervention. Currently, only credential expiration will cause this state.
        #"SUCCEEDED"  The task completed successfully.
        #"FAILED"  The task or one of its subtasks failed, expired, or was canceled.
        smap = { "ACTIVE" : ActionStatus.ACTIVE, "INACTIVE" : ActionStatus.FAILED, "SUCCEEDED" : ActionStatus.COMPLETED, "FAILED" : ActionStatus.FAILED }    
        super().__init__(connection, "transfers", smap)
    def _queryStatusInternal(self, machine, api_key):
        return globusTransferStatus(machine, api_key)