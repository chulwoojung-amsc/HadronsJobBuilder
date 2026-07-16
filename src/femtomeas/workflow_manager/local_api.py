"""
Local API implementation: executes jobs as real processes on the local machine.

Unlike the SPOOF API, all operations have real effects: directories are created,
files are written, batch scripts are executed with bash, and "Globus" transfers
are performed with local filesystem copies (endpoint UUIDs are ignored).

Job state is persisted to <sandbox>/.local_api/jobs.json so that getJobState
works across restarts of the workflow manager (e.g. --skip-agent resume).
"""
import io
import os
import json
import time as timemodule
import shutil
import getpass
import subprocess
from pathlib import Path
from typing import List, Tuple
from . import globals
from .logging import wfapiLog
from .utils import checkSafePath


def listSpecialGlobusEndpoints():
    #"local" always means "the file is already on this machine, just copy it".
    #When real Globus transfers are enabled the named remote collections known
    #to the IRI backend (dtn, perlmutter, ...) are usable as sources too.
    if _globus["endpoint"]:
        from . import iri_api
        return ["local"] + list(iri_api.special_globus_endpoints.keys())
    return ["local"]

def setupWorkflowAgent(sfapi_key_path: str, iriapi_key_path : str, work_dir : dict):
    """Key paths are accepted for config compatibility but ignored."""
    globals.remote_workdir = work_dir
    for machine, d in work_dir.items():
        os.makedirs(d, exist_ok=True)
        os.makedirs(_stateDir(machine), exist_ok=True)
    wfapiLog("Using LOCAL api with workdir", globals.remote_workdir)


#################### persistent job/transfer registry ####################

def _stateDir(machine: str) -> Path:
    return Path(globals.remote_workdir[machine]) / ".local_api"

def _registryFile(machine: str) -> Path:
    return _stateDir(machine) / "jobs.json"

def _loadRegistry(machine: str) -> dict:
    f = _registryFile(machine)
    if f.exists():
        with open(f) as fh:
            return json.load(fh)
    return {}

def _saveRegistry(machine: str, reg: dict):
    f = _registryFile(machine)
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, 'w') as fh:
        json.dump(reg, fh, indent=2)


#################### filesystem operations ####################

def remoteMkdir(machine: str, path: str, create_parents = True, allow_unsafe = False) -> int:
    if not allow_unsafe and not checkSafePath(machine, path):
        raise Exception(f"Path {path} is not in the sandbox")
    wfapiLog(f"Creating directory {machine}:{path}")
    if create_parents:
        os.makedirs(path, exist_ok=True)
    else:
        os.mkdir(path)
    return 1

def uploadBytes(machine: str, remote_path: str, content: io.BytesIO, allow_unsafe = False) -> bool:
    if not allow_unsafe and not checkSafePath(machine, remote_path):
        raise Exception(f"Path {remote_path} is not in the sandbox")
    wfapiLog(f"Writing binary data to {machine}:{remote_path}")
    os.makedirs(os.path.dirname(remote_path), exist_ok=True)
    with open(remote_path, 'wb') as fh:
        fh.write(content.getbuffer())
    return True

def downloadFile(machine: str, remote_path: str) -> str:
    wfapiLog(f"Reading file {machine}:{remote_path}")
    with open(remote_path) as fh:
        return fh.read()

def remoteRun(machine: str, args : str | List[str] ) -> str:
    if isinstance(args, list):
        cmd = ";".join(args)
    else:
        cmd = args
    wfapiLog(f"Executing command {cmd} on local machine (as machine {machine})")
    ret = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    return ret.stdout


#################### "Globus" transfers (local copies) ####################

def _localCopy(source_path: str, dest_path: str) -> str:
    """Copy a file or directory tree; return SUCCEEDED/FAILED api status."""
    try:
        src = Path(source_path)
        dst = Path(dest_path)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            if dst.exists() and dst.is_dir():
                dst = dst / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return "SUCCEEDED"
    except Exception as e:
        wfapiLog(f"Local copy {source_path} -> {dest_path} failed: {e}")
        return "FAILED"

def _recordTransfer(machine: str, status: str) -> str:
    reg = _loadRegistry(machine)
    seq = reg.get("_next_transfer", 0)
    reg["_next_transfer"] = seq + 1
    key = f"transfer_{seq}"
    reg.setdefault("transfers", {})[key] = status
    _saveRegistry(machine, reg)
    return key

#################### real Globus transfers (optional) ####################
#
# If setupGlobus() has been called with the UUID of a Globus collection on
# this machine (e.g. Globus Connect Personal), transfers to/from a non-null
# remote endpoint are performed as real Globus transfers; otherwise (or for
# endpoint "local"/None) they fall back to local filesystem copies as before.

TRANSFER_RESOURCE_SERVER = "transfer.api.globus.org"
_LOGIN_ATTEMPTS = 3

_globus = {"endpoint": "", "token_path": "", "client": None}

def setupGlobus(local_endpoint: str, token_path: str):
    """Enable real Globus transfers landing on this machine's collection."""
    _globus["endpoint"] = local_endpoint
    _globus["token_path"] = token_path or str(Path.home() / ".femtomeas" / "globus_transfer_tokens.json")
    _globus["client"] = None
    wfapiLog(f"LOCAL api Globus transfers enabled, local endpoint {local_endpoint}, tokens at {_globus['token_path']}")

def _globusLoginScope(iri_api, extra_scopes=()):
    import globus_sdk
    from globus_sdk.scopes import TransferScopes
    data_access = [globus_sdk.scopes.GCSCollectionScopes(mc).data_access
                   for mc in iri_api.special_globus_endpoints.values()]
    scope = TransferScopes.all.with_dependencies(data_access)
    return [scope, *extra_scopes]

def _globusInteractiveLogin(iri_api, client, extra_scopes=()):
    from .logging import wfapiUserQuery
    client.oauth2_start_flow(
        requested_scopes=_globusLoginScope(iri_api, extra_scopes),
        refresh_tokens=True,
    )
    query = f"""Open this URL, login, and consent:
    { client.oauth2_get_authorize_url(query_params={"prompt": "login"}) }

    Enter authorization code"""
    #A mistyped code is worth retrying; no terminal to read from is not.
    for attempt in range(_LOGIN_ATTEMPTS):
        try:
            code = wfapiUserQuery("Globus Transfer login", query).strip()
        except EOFError:
            raise Exception("Globus login requires an interactive terminal to enter the "
                            "authorization code; run 'python3 main/globus_login.py <config>' "
                            "from a terminal once to cache tokens, then re-run this.")
        try:
            token_response = client.oauth2_exchange_code_for_tokens(code)
            break
        except Exception as e:
            wfapiLog(f"Globus login attempt {attempt+1}/{_LOGIN_ATTEMPTS} failed: {e}")
    else:
        raise Exception(f"Globus login failed after {_LOGIN_ATTEMPTS} attempts")
    return token_response.by_resource_server[TRANSFER_RESOURCE_SERVER]

def _transferClient(force_login_scopes=None):
    """Build (or reuse) a TransferClient; force_login_scopes triggers a fresh
    interactive login requesting the given additional scopes (consent flow)."""
    import globus_sdk
    from . import iri_api
    if _globus["client"] is not None and force_login_scopes is None:
        return _globus["client"]

    client = globus_sdk.NativeAppAuthClient(iri_api.GLOBUS_CLIENT_ID)
    token_path = Path(_globus["token_path"])
    auth_data = None
    if force_login_scopes is None:
        stored = iri_api.load_tokens(token_path)
        if stored and stored.get("refresh_token"):
            auth_data = iri_api.refresh_tokens(client, stored["refresh_token"], TRANSFER_RESOURCE_SERVER)
    if auth_data is None:
        auth_data = _globusInteractiveLogin(iri_api, client, force_login_scopes or ())
    iri_api.save_tokens(token_path, auth_data)

    def _on_refresh(token_response):
        iri_api.save_tokens(token_path, token_response.by_resource_server[TRANSFER_RESOURCE_SERVER])

    authorizer = globus_sdk.RefreshTokenAuthorizer(
        auth_data["refresh_token"], client,
        access_token=auth_data.get("access_token"),
        expires_at=auth_data.get("expires_at_seconds"),
        on_refresh=_on_refresh)
    _globus["client"] = globus_sdk.TransferClient(authorizer=authorizer)
    return _globus["client"]

def _resolveEndpoint(endpoint: str) -> str:
    from . import iri_api
    return iri_api.replaceSpecialGlobusEndpoint(endpoint)

def _globusSubmit(source_endpoint: str, source_path: str,
                  dest_endpoint: str, dest_path: str) -> str:
    """Submit a Globus transfer; returns the task ID. Retries once through an
    interactive consent login if the collections need extra data_access scopes."""
    import globus_sdk
    for attempt in range(2):
        tc = _transferClient()
        try:
            try:
                is_dir = tc.operation_stat(source_endpoint, source_path)["type"] == "dir"
            except globus_sdk.TransferAPIError as e:
                if e.info.consent_required:
                    raise
                is_dir = False  # stat can fail on some collections; assume file
            td = globus_sdk.TransferData(source_endpoint=source_endpoint,
                                         destination_endpoint=dest_endpoint,
                                         notify_on_succeeded=False, notify_on_failed=False)
            if is_dir:
                td.add_item(source_path, dest_path, recursive=True)
            else:
                td.add_item(source_path, os.path.join(dest_path, os.path.basename(source_path)))
            task_id = tc.submit_transfer(td)["task_id"]
            wfapiLog(f"Globus transfer {task_id} submitted: {source_endpoint}:{source_path} -> {dest_endpoint}:{dest_path}")
            return task_id
        except globus_sdk.TransferAPIError as e:
            if attempt == 0 and e.info.consent_required:
                wfapiLog(f"Globus consent required for scopes {e.info.consent_required.required_scopes}; re-authenticating")
                _transferClient(force_login_scopes=e.info.consent_required.required_scopes)
                continue
            raise

def _globusEnabledFor(endpoint: str) -> bool:
    return bool(_globus["endpoint"]) and endpoint not in (None, "", "local")

def globusCopyFromMachine(dest_endpoint: str, dest_path : str,
                          machine: str, source_path : str):
    if _globusEnabledFor(dest_endpoint):
        wfapiLog(f"Globus copy from {machine}:{source_path} to {dest_endpoint}:{dest_path}")
        return _globusSubmit(_globus["endpoint"], source_path,
                             _resolveEndpoint(dest_endpoint), dest_path)
    wfapiLog(f"Local copy (endpoint {dest_endpoint} ignored) from {machine}:{source_path} to {dest_path}")
    return _recordTransfer(machine, _localCopy(source_path, dest_path))

def globusCopyToMachine(machine: str, dest_path : str,
                        source_endpoint: str, source_path : str,
                        allow_unsafe = False):
    if _globusEnabledFor(source_endpoint):
        if not allow_unsafe and not checkSafePath(machine, dest_path):
            raise Exception(f"Attempting to copy data to a location outside of the sandbox: {dest_path}")
        wfapiLog(f"Globus copy from {source_endpoint}:{source_path} to {machine}:{dest_path}")
        return _globusSubmit(_resolveEndpoint(source_endpoint), source_path,
                             _globus["endpoint"], dest_path)
    wfapiLog(f"Local copy (endpoint {source_endpoint} ignored) from {source_path} to {machine}:{dest_path}")
    return _recordTransfer(machine, _localCopy(source_path, dest_path))

def globusTransferStatus(machine, transfer_id):
    reg = _loadRegistry(machine)
    if transfer_id in reg.get("transfers", {}):
        return reg["transfers"][transfer_id]
    # not a local-copy record: treat as a real Globus task ID
    if _globus["endpoint"]:
        return _transferClient().get_task(transfer_id)["status"]
    return "FAILED"


#################### machine information ####################

def queryMachineStatus(machine: str, rtype="compute") -> bool:
    return True

def getKnownMachines():
    return list(globals.remote_workdir.keys())

def getUserAccountProjects(machine):
    if machine not in getKnownMachines():
        raise Exception(f"Invalid machine: {machine}")
    return [getpass.getuser()]

def getMachineQueues(machine) -> List[ Tuple[str,str] ]:
    return [ ("local", "jobs run immediately as processes on the local machine") ]


#################### batch job execution ####################

def _parseSeconds(time_str: str) -> int | None:
    """Accept plain seconds or H:MM:SS / MM:SS; return None if unparseable."""
    try:
        return int(time_str)
    except (ValueError, TypeError):
        pass
    try:
        parts = [int(p) for p in str(time_str).split(":")]
        if 2 <= len(parts) <= 3:
            secs = 0
            for p in parts:
                secs = secs * 60 + p
            return secs
    except ValueError:
        pass
    return None

def executeBatchJobCompat(machine: str, script_body: str,
                    nodes : int, ranks_per_node : int, gpus_per_rank : int,
                    time : str, queue : str, account : str,
                    job_run_dir : str, exclusive=True, allow_unsafe=False) -> str:
    """
    Run the script as a local bash process. nodes/ranks_per_node/gpus_per_rank/
    queue/account are accepted for interface compatibility but ignored; 'time'
    is enforced as a wall-clock limit if it parses as seconds or H:MM:SS.
    """
    if not allow_unsafe and not checkSafePath(machine, job_run_dir):
        raise Exception("Path is not below the privileged directory")

    reg = _loadRegistry(machine)
    seq = reg.get("_next_job", 0)
    reg["_next_job"] = seq + 1
    job_id = f"local_{seq}"

    state_dir = _stateDir(machine)
    exit_file = state_dir / f"{job_id}.exit"
    driver_log = state_dir / f"{job_id}.driver.log"

    secs = _parseSeconds(time)
    timeout_prefix = f"timeout {secs}s " if secs else ""
    wrapper = f"{timeout_prefix}bash -c {json.dumps(script_body)}\necho $? > {json.dumps(str(exit_file))}\n"

    wfapiLog(f"Starting local job {job_id} in {job_run_dir} (time limit: {secs}s)")
    with open(driver_log, 'w') as log_fh:
        proc = subprocess.Popen(["bash", "-c", wrapper], cwd=job_run_dir,
                                stdout=log_fh, stderr=subprocess.STDOUT,
                                start_new_session=True)

    reg.setdefault("jobs", {})[job_id] = {
        "pid": proc.pid,
        "exit_file": str(exit_file),
        "job_run_dir": job_run_dir,
        "started": int(timemodule.time()),
    }
    _saveRegistry(machine, reg)
    return job_id

def _pidAlive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False

def getJobState(machine: str, jobid: str) -> str:
    reg = _loadRegistry(machine)
    job = reg.get("jobs", {}).get(jobid)
    if job is None:
        wfapiLog(f"Unknown local job {jobid}")
        return "failed"

    exit_file = Path(job["exit_file"])
    if exit_file.exists():
        try:
            rc = int(exit_file.read_text().strip())
        except ValueError:
            rc = -1
        status = "completed" if rc == 0 else "failed"
    elif _pidAlive(job["pid"]):
        status = "active"
    else:
        #process gone without writing an exit code (killed, or lost across reboot)
        status = "failed"
    wfapiLog(f"Queried job state {machine}:{jobid}, got {status}")
    return status

def cancelJob(machine: str, jobid: str):
    reg = _loadRegistry(machine)
    job = reg.get("jobs", {}).get(jobid)
    if job is not None and _pidAlive(job["pid"]):
        wfapiLog(f"Killing local job {jobid} (pid {job['pid']})")
        try:
            os.killpg(job["pid"], 15)
        except ProcessLookupError:
            pass
