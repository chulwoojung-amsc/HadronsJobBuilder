from . import globals
import pathlib

def checkSafePath(machine: str, path: str):
    if globals.remote_workdir == None:
        raise Exception("setupWorkflowAgent has not been called")
    if machine not in globals.remote_workdir:
        raise Exception("Unknown machine")
    safe_p = pathlib.Path(globals.remote_workdir[machine]).resolve()
    path_p = pathlib.Path(path).resolve()
    tmp_p = pathlib.Path("/tmp")
    return path_p.is_relative_to(safe_p) or path_p.is_relative_to(tmp_p)
