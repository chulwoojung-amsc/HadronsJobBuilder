import pathlib
import io
import time

default_iri_api=True

if default_iri_api:
    print("Using IRI API")
    from .iri_api import setupWorkflowAgent, remoteLs, remoteMkdir, uploadBytes, executeBatchJobCompat, remoteChmod, getJobState, cancelJob, queryMachineStatus
    from .iri_api import known_machines
else:
    print("Using Superfacility API")
    from .sfapi import setupWorkflowAgent, remoteLs, remoteMkdir, uploadBytes, executeBatchJob, getJobState, cancelJob, queryMachineStatus
    from .sfapi import known_machines
    
from . import globals

def testExecutablePrivileges(machine: str)-> bool:
    try:
        ret = remoteRun(machine, ["echo","'TEST'"])
        if ret.strip() == "TEST":
            return True
        else:
            return False
    except Exception as e:
        return False    

    
def uploadSmallFile(machine: str, remote_path: str, local_path: str, allow_unsafe = False) -> bool:
    """
    Upload a small file to a remote path
    Args:
       machine - The name of the machine. Valid values are 'Perlmutter'
       remote_path - The absolute path on the remote machine
       local_path - The path on the local machine
       allow_unsafe - Allow uploading to directories other than within the sandbox
    Return:
       True if successful, False otherwise
    """
    with open(local_path, "rb") as fh:
        return uploadBytes(machine, remote_path, io.BytesIO( fh.read() ) )

def watchJobStatus(machine, jobid, howlong=300, poll_freq=10):
    t0=int(time.time())
    while( int(time.time()) - t0 < howlong  ):
        state = getJobState(machine, jobid)
        print(state)
        if state not in ("new", "queued", "active"):
            print("Detected job completion")
            break    
        time.sleep(poll_freq)
        
