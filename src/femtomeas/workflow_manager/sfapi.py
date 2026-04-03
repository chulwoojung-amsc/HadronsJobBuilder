import sfapi_client
from sfapi_client import Client
from sfapi_client.compute import Machine
import sfapi_client.paths
import pathlib
import io
import json
from typing import Literal, Union, List, Optional, Tuple
from .utils import checkSafePath
from . import globals
import time

known_machines = {  "Perlmutter" : { "sfapi_enum" : Machine.perlmutter }  }

sfapi_key_path=None

def setupWorkflowAgent(_sfapi_key_path: str, iriapi_key_path : str, work_dir : dict):
    """
    Setup the workflow agent
    Args:
       _sfapi_key_path: The full path to the key file in .pem format (with username on the first line) as per https://nersc.github.io/sfapi_client/quickstart/#__tabbed_9_2  "Storing keys in files"
       iriapi_key_path: Ignored
       work_dir: The remote work directories, by machine as a dict, e.g. { "Perlmutter" : "/path/to/dir" }.  The agent is only allowed to modify the contents of files within this directory or its children
    """
    global sfapi_key_path
    sfapi_key_path = _sfapi_key_path
    globals.remote_workdir=work_dir

sfapi_clients = {}
    
def sfAPIclientExecute(machine: str, what):
    if machine not in known_machines:
        raise Exception("Invalid machine name")
    m = known_machines[machine]["sfapi_enum"]

    global sfapi_clients
    if machine in sfapi_clients:
        client = sfapi_clients[machine]
    else:    
        if sfapi_key_path == None:
            raise Exception("setupWorkflowAgent has not been called")
        client = Client(key = sfapi_key_path)
        sfapi_clients[machine] = client
    return what(client, m)


def remoteRun(machine: str, args : str | List[str] )-> str:
    """
    Run a command on the remote machine login node
    """
    return sfAPIclientExecute(machine, lambda client, m: client.compute(m).run(args))


def remoteLs(machine: str, path: str)-> List[str]:
    """
    Query the contents of a path on a given machine
    Args:
       machine - The name of the machine. Valid values are 'Perlmutter'
       path - The absolute path. If a directory, ensure the path ends in '/' to get the directory contents.
    Return:
       A list of files in the directory
    """
    def _doit(client, m):
        pths = client.compute(m).ls(path)
        return [ p.name for p in pths ]
        
    return sfAPIclientExecute(machine, _doit)


def remoteMkdir(machine: str, path: str, create_parents = True, allow_unsafe = False)-> int:
    """
    Create a directory on the remote machine. This is an unsafe action as it is not confined to the sandbox directory, and thus should not be exposed as a tool without safeguards
    Args:
           allow_unsafe - Allow uploading to directories other than within the sandbox
    Return: 0 if the operation failed, 1 if the directory was created, 2 if it already existed
    
    """
    if not allow_unsafe and not checkSafePath(machine, path):
        raise Exception("Path is not a subdirectory of the sandbox path")

    if not pathlib.Path(path).is_absolute():
        raise Exception("Path must be absolute")

    cmd = f"[ -d '{path}' ] && echo 'Directory exists' || mkdir {'-pv' if create_parents else '-v'} {path}"
    
    #ret = remoteRun(machine, [ "mkdir", "-pv" if create_parents else "-v", path ]).strip().split('\n')
    ret = remoteRun(machine, cmd).strip().split('\n')

    if ret[-1] == 'Directory exists':
        return 2
    elif ret[-1] == f"mkdir: created directory '{path}'":
        return 1
    else:
        print(ret)
        return 0



def queryMachineStatus(machine: str)-> bool:
    """
    Query the status of a machine
    Args:
       machine - The name of the machine. Valid values are 'Perlmutter'
    Return:
       A bool indicating whether the machine up (True) or down (False)
    """
    def _doit(client, m):
        status = client.compute(m)
        return True if status.status == sfapi_client.StatusValue.active else False
        
    return sfAPIclientExecute(machine, _doit)


def uploadBytes(machine: str, remote_path: str, content: io.BytesIO, allow_unsafe = False) -> bool:
    """
    Upload file contents as bytes to a remote path
    Args:
       machine - The name of the machine. Valid values are 'Perlmutter'
       remote_path - The absolute path on the remote machine
       content - The file contents as binary
       allow_unsafe - Allow uploading to directories other than within the sandbox
    Return:
       True if successful, False otherwise
    """
    if not allow_unsafe and not checkSafePath(machine, remote_path):
        raise Exception("Path is not below the privileged directory")
    
    def _doit(client, m):
        pth = sfapi_client.paths.RemotePath(path=remote_path, compute=client.compute(m))        
        return pth.upload(content).is_file()
    return sfAPIclientExecute(machine, _doit)


#SFAPI returns job handles, IRI uses job IDs
sfapi_jobs = {}

def executeBatchJob(machine: str, script: str) -> str:
    """
    Execute a batch script on the machine
    """    
    def doit_(client, m):
        return client.compute(m).submit_job(script)

    job = sfAPIclientExecute(machine, doit_)
    global sfapi_jobs
    sfapi_jobs[str(job.jobid)] = job
    return str(job.jobid)

sfapi_state_map = { "CONFIGURING" : "new", "PENDING" : "queued", "RUNNING" : "active", "COMPLETED" : "completed", "FAILED" : "failed", "CANCELLED" : "cancelled", "COMPLETING" : "active" }

    
    #IRI API: 
    #0"new"
    #1"queued"
    #2"active"
    #3"completed"
    #4"failed"
    #5"canceled"

    #SFAPI:
    # CONFIGURING = "CONFIGURING"

    # RESV_DEL_HOLD = "RESV_DEL_HOLD"
    # REQUEUE_FED = "REQUEUE_FED"
    # REQUEUE_HOLD = "REQUEUE_HOLD"
    # REQUEUED = "REQUEUED"

    # CANCELLED = "CANCELLED"
    # COMPLETED = "COMPLETED"
    # COMPLETING = "COMPLETING"

    # DEADLINE = "DEADLINE"
    # FAILED = "FAILED"
    # NODE_FAIL = "NODE_FAIL"
    # OUT_OF_MEMORY = "OUT_OF_MEMORY"
    # PENDING = "PENDING"
    # PREEMPTED = "PREEMPTED"
    # RUNNING = "RUNNING"
    # RESIZING = "RESIZING"
    # REVOKED = "REVOKED"
    # SIGNALING = "SIGNALING"
    # SPECIAL_EXIT = "SPECIAL_EXIT"
    # STAGE_OUT = "STAGE_OUT"
    # STOPPED = "STOPPED"
    # SUSPENDED = "SUSPENDED"
    # BOOT_FAIL = "BOOT_FAIL"
    # TIMEOUT = "TIMEOUT"

   
def getJobState(machine: str, jobid: str) -> str:
    if jobid not in sfapi_jobs.keys():
        raise Exception("Job is not in the list of known jobs")
    job = sfapi_jobs[jobid]
    job.update()
    if job.state not in sfapi_state_map.keys():
        raise Exception(f"Unknown job state: {str(job.state)}")
    return sfapi_state_map[job.state]

def cancelJob(machine: str, jobid: str):
    if jobid not in sfapi_jobs.keys():
        raise Exception("Job is not in the list of known jobs")
    job = sfapi_jobs[jobid]
    job.cancel()
    

machine_globus_endpoints = { "Perlmutter" : "perlmutter" }

def globusTransferStatus(machine, transfer_id)-> str:
    """
    Query the status of a Globus transfer initiated from the API on the given machine, with the provided transfer_id
    Returns the status from the following (cf. https://docs.globus.org/api/transfer/task/):
    "ACTIVE"  The task is in progress.
    "INACTIVE" The task has been suspended and will not continue without intervention. Currently, only credential expiration will cause this state.
    "SUCCEEDED"  The task completed successfully.
    "FAILED"  The task or one of its subtasks failed, expired, or was canceled.
    """
    def doit_(client, m):
        return client.get("storage/globus/transfer/" + transfer_id)
    
    ret = sfAPIclientExecute(machine, doit_)

    if ret.status_code == 200:
        j = json.loads(ret.text)
        return j["globus_status"]
    else:
        raise Exception("Globus transfer query, response content: " + ret.text)


def _globusCopy(source_endpoint, dest_endpoint, source_path, dest_path, machine, block_until_complete=False):
    trans_args = { "source_uuid" : source_endpoint, "target_uuid" : dest_endpoint,
                   "source_dir" : source_path, "target_dir" : dest_path }
    def doit_(client, m):
        return client.post("storage/globus/transfer", data=trans_args)
    
    ret = sfAPIclientExecute(machine, doit_)
    if ret.status_code == 200:
        j = json.loads(ret.text)
        tid = j["transfer_id"]

        if block_until_complete:
            while (status := globusTransferStatus(machine, tid)) == "ACTIVE":
                print(".", end="")
                time.sleep(20)
            print(status)

        return tid
    else:
        raise Exception("Globus transfer failed, response content: " + ret.text)
    
    
def globusCopyToMachine(machine: str, dest_path : str,
                        source_endpoint: str, source_path : str,
                        allow_unsafe=False,
                        block_until_complete=False)-> str:
    """
    Perform a Globus transfer to the machine
    Args:
       machine: The destination machine
       dest_path: The destination path on that machine (directory)
       source_endpoint: The name/tag of the Globus source endpoint
       source_path : The path on that endpoint
       allow_unsafe : Allow movement to paths outside of the sandbox
       block_until_complete : Poll the transfer status every 20s until the transfer is complete before returning
    Return:
       The transfer ID as a string

    Notes:
       If source_path is a filename, only that file will be copied. If it is a directory name only the contents of that directory will be copied, not the directory itself (even if there is no trailing /)
    """
    
    if not allow_unsafe and not checkSafePath(machine, dest_path):
        raise Exception("Attempting to copy data to a location outside of the sandbox")
    if machine not in machine_globus_endpoints.keys():
        raise Exception("Unknown machine endpoint")

    return _globusCopy(source_endpoint, machine_globus_endpoints[machine], source_path, dest_path, machine, block_until_complete)

    
def globusCopyFromMachine(dest_endpoint: str, dest_path : str,
                          machine: str, source_path : str,                          
                          allow_unsafe=False,
                          block_until_complete=False)-> str:
    """
    Perform a Globus transfer from the machine
    Args:
       dest_endpoint: The name/tag of the Globus target endpoint
       dest_path : The path on that endpoint (directory)
       machine: The source machine
       source_path: The source path on that machine
       allow_unsafe : Allow movement from paths outside of the sandbox
       block_until_complete : Poll the transfer status every 20s until the transfer is complete before returning
    Return:
       The transfer ID as a string

    Notes:
       If source_path is a filename, only that file will be copied. If it is a directory name only the contents of that directory will be copied, not the directory itself (even if there is no trailing /)
    """
    
    if not allow_unsafe and not checkSafePath(machine, source_path):
        raise Exception("Attempting to copy data from a location outside of the sandbox")
    if machine not in machine_globus_endpoints.keys():
        raise Exception("Unknown machine endpoint")

    return _globusCopy(machine_globus_endpoints[machine], dest_endpoint, source_path, dest_path, machine, block_until_complete)

    
