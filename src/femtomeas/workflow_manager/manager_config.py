from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from .api_general import setupWorkflowAgent
from .hadrons import setHadronsInfo
from . import globals
from .manager_config_models import WorkflowConfig, ManagerConfig

if globals.api_impl in ("SPOOF", "IRI"):
    def setupManager(config : ManagerConfig):
        setupWorkflowAgent(config.workflow.iriapi_key_path, config.workflow.transferapi_key_path, config.workflow.sandbox_directories)
        setHadronsInfo(config.hadrons)
else:
    raise Exception("Unknown API implementation")

def parseManagerConfigStr(json_str : str)->dict:
    try:
        return ManagerConfig.model_validate_json(json_str)
    except ValidationError as e:
        raise Exception(f"Could not parse manager config {json_str}: {e}")
    
def readManagerConfigFile(filename):
    try:
        config = ManagerConfig.model_validate_json(Path(filename).read_text())
    except ValidationError as e:
        raise Exception(f"Could not parse manager config {filename}: {e}")
    return config
