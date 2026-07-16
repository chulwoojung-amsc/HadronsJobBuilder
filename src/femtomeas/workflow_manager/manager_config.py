from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from .api_general import setupWorkflowAgent
from .hadrons import setHadronsInfo
from . import globals

if globals.api_impl in ("SPOOF", "LOCAL", "IRI_SF_HYBRID", "SF"):
    class WorkflowConfig(BaseModel):
        sfapi_key_path: str = Field(..., description="The path to the Superfacility API key")
        iriapi_key_path: str = Field(..., description="The path to the IRI API key (will be created if doesn't yet exist)")
        sandbox_directories: dict[str, str] = Field(..., description="A map of machine names to base sandbox directories")
        local_globus_endpoint: str = Field("", description="(LOCAL api only) UUID of a Globus collection on this machine (e.g. Globus Connect Personal); enables real Globus transfers")
        globus_token_path: str = Field("", description="(LOCAL api only) Path to store Globus transfer tokens; defaults to ~/.femtomeas/globus_transfer_tokens.json")

    def setupManager(config : dict):
        setupWorkflowAgent(config.workflow.sfapi_key_path, config.workflow.iriapi_key_path, config.workflow.sandbox_directories)
        if globals.api_impl == "LOCAL" and config.workflow.local_globus_endpoint:
            from .local_api import setupGlobus
            setupGlobus(config.workflow.local_globus_endpoint, config.workflow.globus_token_path)
        setHadronsInfo(config.model_dump()["hadrons"])
        
elif globals.api_impl == "IRI":
    #For convenience we use separate files for the IRI and data transfer tokens, as the latter will eventually no longer be needed
    class WorkflowConfig(BaseModel):
        iriapi_key_path: str = Field(..., description="The path to the IRI API key (will be created if doesn't yet exist)")
        transferapi_key_path: str = Field(..., description="The path to the Data Transfer API key (will be created if doesn't yet exist)")
        sandbox_directories: dict[str, str] = Field(..., description="A map of machine names to base sandbox directories")

    def setupManager(config : dict):
        setupWorkflowAgent(config.workflow.iriapi_key_path, config.workflow.transferapi_key_path, config.workflow.sandbox_directories)
        setHadronsInfo(config.model_dump()["hadrons"])
else:
    raise Exception("Unknown API implementation")


class HadronsConfig(BaseModel):
    bin: str = Field(..., description="The path to the 'bin' directory of the Hadrons install")
    env: str = Field("", description="Bash commands required to set up the Hadrons environment")

class AgentConfig(BaseModel):
    i2api_key_path: str = Field(..., description="Path to a file containing the AmSC I2 API key")
    
class ManagerConfig(BaseModel):
    workflow: WorkflowConfig = Field(..., description="General manager arguments")
    hadrons: dict[str, HadronsConfig] = Field(..., description="A map of machine names to HadronsConfig structures")
    agent: AgentConfig = Field(..., description="Configuration for the agent")

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
