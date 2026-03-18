from .api_general import *
from langchain.tools import tool
from typing import Literal, Union, List, Optional, Tuple

@tool
def queryMachineStatus_t(machine: str)-> bool:
    """
    Query the status of a machine
    Args:
       machine - The name of the machine. Valid values are 'Perlmutter'
    Return:
       A bool indicating whether the machine up (True) or down (False)
    """
    return queryMachineStatus(machine)

@tool
def remoteLs_t(machine: str, path: str)-> List[str]:
    """
    Query the contents of a path on a given machine
    Args:
       machine - The name of the machine. Valid values are 'Perlmutter'
       path - The absolute path. If a directory, ensure the path ends in '/' to get the directory contents.
    Return:
       A list of files in the directory
    """
    return remoteLs(machine, path)

@tool
def remoteMkdir_t(machine: str, path: str, create_parents : bool = True)-> int:
    """
    Create a directory on the remote machine. The path must be a relative of the sandbox directory
    Args:
       machine - The name of the machine. Valid values are 'Perlmutter'
       path - The absolute path on the remote machine
       create_parents - If True, parent directories will be created as needed, if False the new directory must be a child of an existing directory        
    Return: 0 if the operation failed, 1 if the directory was created, 2 if it already existed
    """
    return remoteMkdir(machine, path, create_parents, allow_unsafe = False)

@tool
def uploadSmallFile_t(machine: str, remote_path: str, local_path: str) -> bool:
    """
    Upload a small file to a remote path
    Args:
       machine - The name of the machine. Valid values are 'Perlmutter'
       remote_path - The absolute path on the remote machine
       local_path - The path on the local machine
    Return:
       True if successful, False otherwise
    """
    return uploadSmallFile(machine, remote_path, local_path, allow_unsafe=False)

