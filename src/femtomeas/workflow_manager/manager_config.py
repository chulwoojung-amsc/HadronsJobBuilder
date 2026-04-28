from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from .api_general import setupWorkflowAgent
from .hadrons import setHadronsInfo

class WorkflowConfig(BaseModel):
    sfapi_key_path: str = Field(..., description="The path to the Superfacility API key")
    iriapi_key_path: str = Field(..., description="The path to the IRI API key (will be created if doesn't yet exist)")
    sandbox_directories: dict[str, str] = Field(..., description="A map of machine names to base sandbox directories")

class HadronsConfig(BaseModel):
    bin: str = Field(..., description="The path to the 'bin' directory of the Hadrons install")
    env: str = Field("", description="Bash commands required to set up the Hadrons environment")

class ManagerConfig(BaseModel):
    workflow: WorkflowConfig = Field(..., description="General manager arguments")
    hadrons: dict[str, HadronsConfig] = Field(..., description="A map of machine names to HadronsConfig structures")

def setupManager(config : dict):
    setupWorkflowAgent(config.workflow.sfapi_key_path, config.workflow.iriapi_key_path, config.workflow.sandbox_directories)
    setHadronsInfo(config.model_dump()["hadrons"])


def parseManagerConfigStr(json_str : str)->dict:
    try:
        return ManagerConfig.model_validate_json(json_str)
    except ValidationError as e:
        raise Exception(f"Could not parse manager config {json_str}: {e}")

    
def readManagerConfigStr(json_str : str):
    try:
        config = ManagerConfig.model_validate_json(json_str)
    except ValidationError as e:
        raise Exception(f"Could not parse manager config {json_str}: {e}")
    setupManager(config)
    
    
def readManagerConfigFile(filename):
    try:
        config = ManagerConfig.model_validate_json(Path(filename).read_text())
    except ValidationError as e:
        raise Exception(f"Could not parse manager config {filename}: {e}")
    setupManager(config)
