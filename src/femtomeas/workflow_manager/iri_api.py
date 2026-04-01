from authlib.integrations.requests_client import OAuth2Session
from authlib.oauth2.rfc7523 import PrivateKeyJWT
import httpx
from . import  globals
import json
from typing import Literal, Union, List, Optional, Tuple
import time
import pathlib
import io
from .utils import checkSafePath

known_machines = {  "Perlmutter" : { "iriapi_base" : "https://api.iri.nersc.gov/api/v1", "iriapi_group" : "perlmutter", "sfapi_base" : "https://api.nersc.gov/api/v1.2" } }

sfapi_session = None
sfapi_client = None

def setupWorkflowAgent(key_path, work_dir : dict):
    """
    Setup the workflow agent
    Args:
       key_path: The full path to the key file in .pem format (with username on the first line) as per https://nersc.github.io/sfapi_client/quickstart/#__tabbed_9_2  "Storing keys in files"
       work_dir: The remote work directories, by machine as a dict, e.g. { "Perlmutter" : "/path/to/dir" }.  The agent is only allowed to modify the contents of files within this directory or its children
    """
    globals.remote_workdir=work_dir
    token_url = "https://oidc.nersc.gov/c2id/token"    
    with open(key_path, 'r') as f:
        client_id = f.readline().strip()
    
        is_open=False
        private_key = ""
        is_complete=False
        
        for line in f:
            line = line.strip()
            if line == "-----BEGIN RSA PRIVATE KEY-----":
                is_open = True
                private_key = private_key + line
            elif line == "-----END RSA PRIVATE KEY-----":
                is_open = False
                private_key = private_key + line
                is_complete = True
            elif is_open:
                private_key = private_key + line

        if not is_complete:
            raise Exception("Unable to read key from file", key_path)

        # private_key = key_path
        global sfapi_session
        sfapi_session = OAuth2Session(
            client_id, 
            private_key, 
            PrivateKeyJWT(token_url),
            grant_type="client_credentials",
            token_endpoint=token_url
        )

        #Check token accessible              
        tok = sfapi_session.fetch_token()
        if 'access_token' not in tok.keys():
            raise Exception("Unable to fetch token; response was", tok)

        global sfapi_client
        sfapi_client = httpx.Client()
        

def get(machine, suburl, params = None, base='iriapi_base'):
    assert sfapi_session != None and sfapi_client != None
    assert machine in known_machines
    assert base in known_machines[machine]
    
    base = known_machines[machine][base]
    resp = sfapi_client.get(base + '/' + suburl, headers={ "accept" : "application/json", "Authorization" : f"Bearer {sfapi_session.fetch_token()['access_token']}" }, params=params, timeout=300  )
    if resp.status_code == 200:
        j = json.loads(resp.text)
        return j
    else:
        raise Exception(f"Get operation failed with code {resp.status_code} and text {resp.text}")
        
iri_api_project_map = {}

def getUserProjectIDmap(machine):
    """
    Obtain the mapping between project name and id
    Return:
       dict name -> id
    """
    global iri_api_project_map

    if machine not in iri_api_project_map:
        j = get(machine, "account/projects")
        pmap = dict()
        for acct in j:
            pmap[acct['name']] = acct['id']
        iri_api_project_map[machine] = pmap            
    
    return iri_api_project_map[machine]


iri_api_resource_map = {}

def getResourceID(machine, rtype="compute"):
    """
    Get the resource ID associated with the resource
    rtype: "compute" or "login"
    """
    global iri_api_resource_map
    if rtype not in ["compute","login"]:
        raise Exception("Invalid resource type")
    
    if machine not in iri_api_resource_map:
        j = get(machine, "status/resources", params={"group" : known_machines[machine]['iriapi_group'], "resource_type" : "compute"})
        
        iri_api_resource_map[machine] = dict()
            
        #The API does not distinguish between login and compute nodes; both are "compute" resources, but the login node does not have any capabilities listed (unclear if this will change)
        #For now, use the capabilities to distinguish as the names are likely arbitrary
        compute = None
        login=None
        for r in j:
            print(json.dumps(r))
            cpu = False
            gpu =False
            for cap in r['capability_uris']:
                if 'capabilities/cpu' in cap:
                    cpu = True
                if 'capabilities/gpu' in cap:
                    gpu = True
            if cpu and gpu:
                print(f"Identified resource {json.dumps(r,indent=2)} as compute backend")
                iri_api_resource_map[machine]["compute"] = r["id"]
            elif not cpu and not gpu:
                print(f"Identified resource {json.dumps(r,indent=2)} as login frontend")
                iri_api_resource_map[machine]["login"] = r["id"]
            else:
                print(f"Warning: unidentified resource {json.dumps(r,indent=2)}")
        
    
    return iri_api_resource_map[machine][rtype]


def queryMachineStatus(machine: str, rtype="compute")-> bool:
    """
    Query the status of a machine
    Args:
       machine - The name of the machine
       rtype - The resource type (compute/login)
    Return:
       A bool indicating whether the machine up (True) or down (False)
    """
    rid = getResourceID(machine, rtype)
    j = get(machine, f"status/resources/{rid}")
    print(json.dumps(j,indent=2))
    return j['current_status'] == 'up'
    

def waitTask(machine, task_id, poll_freq=4):
    j = get(machine, f"task/{task_id}")
    
    while(j['status'] == "active"):
        time.sleep(poll_freq)
        j = get(machine, f"task/{task_id}")

    if j['status'] != "completed":
        raise Exception(f"Task not completed, response {json.dumps(j,indent=2)}")

    return j

def remoteLs(machine: str, path: str)-> List[str]:
    """
    Query the contents of a path on a given machine
    Args:
       machine - The name of the machine. Valid values are 'Perlmutter'
       path - The absolute path
    Return:
       A list of files in the directory

    TODO: Explore behavior of trailing slashes, and the fact that the directory name itself seems to be listed among the directory content; should we unify the behavior with SFAPI?
    """
    assert sfapi_session != None and sfapi_client != None
    assert machine in known_machines
    
    rid = getResourceID(machine, rtype="login")

    j = get(machine, f"filesystem/ls/{rid}", params={"path" : path})
    tid = j['task_id']
    j = waitTask(machine, tid)

    files = [ f['name'] for f in j['result']['output'] ]
    return files


def put(machine, suburl, data = None, params=None):
    assert sfapi_session != None and sfapi_client != None
    assert machine in known_machines
    base = known_machines[machine]['iriapi_base']
    headers={ "accept" : "application/json", "Authorization" : f"Bearer {sfapi_session.fetch_token()['access_token']}" }
    
    resp = sfapi_client.put(base + '/' + suburl, headers=headers, json=data, params=params, timeout=300)
    return json.loads(resp.text), resp.status_code



def remoteChmod(machine: str, path : str, mode : str, allow_unsafe = False) -> bool: 
    if not allow_unsafe and not checkSafePath(machine, path):
        raise Exception("Path is not a subdirectory of the sandbox path")

    if not pathlib.Path(path).is_absolute():
        raise Exception("Path must be absolute")

    rid = getResourceID(machine, rtype="login")
    j, status = put(machine, f"filesystem/chmod/{rid}", data={"path" : path, "mode" : mode})
    tid = j['task_id']
    j = waitTask(machine, tid)
    
    if j["status"] == "completed":
        return True
    else:
        print(json.dumps(j,indent=2))
        return False
    

def post(machine, suburl, data = None, params=None, files=None, base='iriapi_base', data_is_json=True):
    assert sfapi_session != None and sfapi_client != None
    assert machine in known_machines
    assert base in known_machines[machine]
    
    base = known_machines[machine][base]
    headers={ "accept" : "application/json", "Authorization" : f"Bearer {sfapi_session.fetch_token()['access_token']}" }
    
    resp = sfapi_client.post(base + '/' + suburl, headers=headers, json=data if data_is_json else None, data=data if not data_is_json else None, params=params, files=files, timeout=300 )
    return json.loads(resp.text), resp.status_code
    
def remoteMkdir(machine: str, path: str, create_parents = True, allow_unsafe = False)-> int:
    """
    Create a directory on the remote machine. This is an unsafe action as it is not confined to the sandbox directory, and thus should not be exposed as a tool without safeguards
    Args:
           allow_unsafe - Allow uploading to directories other than within the sandbox
    Return: 0 if the operation failed, 1 if the directory was created, 2 if it already existed

    TODO: Doesn't seem to be a way to check if the directory already existed; save doing and waiting on an ls, for now we just always return 1 if success
    """
    if not allow_unsafe and not checkSafePath(machine, path):
        raise Exception("Path is not a subdirectory of the sandbox path")

    if not pathlib.Path(path).is_absolute():
        raise Exception("Path must be absolute")

    rid = getResourceID(machine, rtype="login")
        
    j, status = post(machine, f"filesystem/mkdir/{rid}", data={"path" : path, "parent" : create_parents})
    print(status, json.dumps(j,indent=2))
    tid = j['task_id']
    j = waitTask(machine, tid)
    print(json.dumps(j,indent=2))

    if j["status"] == "completed":
        return 1
    #elif status == 304:  #This does not seem to work
    #    return 2
    else:
        print(status, json.dumps(j,indent=2))
        return 0


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
    
    if not pathlib.Path(remote_path).is_absolute():
        raise Exception("Path must be absolute")
    
    rid = getResourceID(machine, rtype="login")

    j, status = post(machine, f"filesystem/upload/{rid}", params={'path' : remote_path}, files={'file': content})
    print(json.dumps(j,indent=2))

    tid = j['task_id']
    j = waitTask(machine, tid)
    print(json.dumps(j,indent=2))

    if j["status"] == "completed":
        return 1
    else:
        print(status, json.dumps(j,indent=2))
        return 0



    
def executeBatchJobCompat(machine: str, script_body: str,
                    nodes : int, ranks_per_node : int, gpus_per_rank : int,
                    time : str, queue : str, account : str,
                    job_run_dir : str, exclusive=True, allow_unsafe=False) -> str:
    """
    Execute batch script on the machine
    script_body: The content of the batch script. If you are executing an existing remote script, use "source /path/to/script"    
    Note that any SLURM/PBS headers will be ignored; ensure that SLURM headers that are usually passed to srun are manually passed instead
    """
    
    if not allow_unsafe and not checkSafePath(machine, job_run_dir):
        raise Exception("Path is not below the privileged directory")

    resources = {
        "node_count" : nodes,
        "processes_per_node" : ranks_per_node,
        "exclusive_node_use": exclusive,
        "gpu_cores_per_process" : gpus_per_rank
        }       
        
    attributes = {
        "duration" : time,
        "account": account,
        "queue_name": queue
        }
    spec = {
        "executable": "date", 
        "directory" : job_run_dir,
        "inherit_environment": True,
        "stdin_path" : None,
        "stdout_path": f"{job_run_dir}/run.log",
        "stderr_path": f"{job_run_dir}/err.log",
        "resources" : resources,
        "attributes" : attributes,
        "launcher" : "srun",
        "pre_launch" : f"cd {job_run_dir}\n" + script_body,  #hijack the pre_launch to run the actual batch script content to avoid the jobspec restrictions
        "post_launch" : None
        }
    #Hopefully they will allow arbitrary batch script submission as part of the API eventually!
    
    print("JSON",json.dumps(spec,indent=2))
    
    rid = getResourceID(machine, rtype="compute")

    j, status = post(machine, f"compute/job/{rid}", data=spec)
    print(json.dumps(j,indent=2))
    if status == 200:
        return j['id']
    else:
        raise Exception("Job submission failed")


def executeBatchJobTest(machine: str, job_run_dir):
    spec = {
        "name": "iri-sample-job",
        "executable": "/bin/hostname",
        "arguments": [],
        "directory": job_run_dir,
        "stdout_path": f"{job_run_dir}/iri-job.out",
        "stderr_path": f"{job_run_dir}/iri-job.err",
        "launcher": "srun",
        "resources": {
            "node_count": 1,
            "process_count": 1,
            "processes_per_node": 1,
            "cpu_cores_per_process": 1,
            "gpu_cores_per_process" : 1
        },
        "attributes": {
            "duration": 300,
            "queue_name": "debug",
            "account": "amsc013_g"
        },
        "environment": {
            "OMP_NUM_THREADS": "1"
        }
    }
    
    print("JSON",json.dumps(spec,indent=2))
    
    rid = getResourceID(machine, rtype="compute")

    print(f"Post to compute/job/{rid}")
    
    j, status = post(machine, f"compute/job/{rid}", data=spec)
    print(json.dumps(j,indent=2))

def getJobState(machine: str, jobid: str) -> str:
    rid = getResourceID(machine, rtype="compute")
    j = get(machine, f"compute/status/{rid}/{jobid}", params = { "historical" : True })
    print(json.dumps(j,indent=2))
    return j['status']['state']
    
    
def delete(machine, suburl, params = None):
    assert sfapi_session != None and sfapi_client != None
    assert machine in known_machines
    base = known_machines[machine]['iriapi_base']
    resp = sfapi_client.delete(base + '/' + suburl, headers={ "accept" : "*/*", "Authorization" : f"Bearer {sfapi_session.fetch_token()['access_token']}" }, params=params, timeout=300  )
    return {} if resp.text == "" else resp.json(), resp.status_code

def cancelJob(machine: str, jobid: str):
    rid = getResourceID(machine, rtype="compute")
    j, status = delete(machine, f"compute/cancel/{rid}/{jobid}")
    if status != 204:
        raise Exception("Job cancellation failed:",json.dumps(j))




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
    j = get(machine, f"storage/globus/transfer/{transfer_id}", base='sfapi_base')
    return j["globus_status"]


def _globusCopy(source_endpoint, dest_endpoint, source_path, dest_path, machine, block_until_complete=False):
    trans_args = { "source_uuid" : source_endpoint, "target_uuid" : dest_endpoint,
                   "source_dir" : source_path, "target_dir" : dest_path }

    j, status=post(machine, "storage/globus/transfer", data=trans_args, base='sfapi_base', data_is_json=False)
    if status == 200:
        tid = j["transfer_id"]

        if block_until_complete:
            while (status := globusTransferStatus(machine, tid)) == "ACTIVE":
                print(".", end="")
                time.sleep(20)
            print(status)

        return tid
    else:
        raise Exception("Globus transfer failed, response content: " + json.dumps(j))

    
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
    
