import random
import time as timemodule
import io
from . import globals

def setupWorkflowAgent(key_path, work_dir : dict):
    globals.remote_workdir=work_dir

tid = 0
transfers = { }

def _fakeGlobusCopy():
    #Assign the transfer a fake active time
    active_time = random.randint(3,8)

    #Generate a unique_key
    global tid
    key = f"transfer_{tid}"
    tid +=1

    transfers[key] = timemodule.time() + active_time
    print("Fake transfer",key,"time",active_time)
    
    return key

def globusCopyFromMachine(dest_endpoint: str, dest_path : str,
                          machine: str, source_path : str):
    return _fakeGlobusCopy()
                    
def globusCopyToMachine(machine: str, dest_path : str,
                        source_endpoint: str, source_path : str):
    return _fakeGlobusCopy()
    
def globusTransferStatus(machine, transfer_id):
    if timemodule.time() >= transfers[transfer_id]:
        return "SUCCEEDED"
    else:
        return "ACTIVE"
    
def remoteMkdir(machine: str, path: str, create_parents = True, allow_unsafe = False)-> int:
    return 1

def uploadBytes(machine: str, remote_path: str, content: io.BytesIO, allow_unsafe = False) -> bool:
    return True

jid=0
compute_jobs = { }

def executeBatchJobCompat(machine: str, script_body: str,
                    nodes : int, ranks_per_node : int, gpus_per_rank : int,
                    time : str, queue : str, account : str,
                    job_run_dir : str, exclusive=True, allow_unsafe=False) -> str:
    #Assign the job a fake active time
    active_time = random.randint(3,8)

    #Generate a unique_key
    global jid
    key = f"compute_{jid}"
    jid +=1

    compute_jobs[key] = timemodule.time() + active_time
    print("Fake compute",key,"time",active_time)
    
    return key

def getJobState(machine: str, jobid: str) -> str:
    if timemodule.time() >= compute_jobs[jobid]:
        return "completed"
    else:
        return "active"
