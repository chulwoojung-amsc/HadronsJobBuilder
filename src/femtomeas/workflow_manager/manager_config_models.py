from . import globals
from pydantic import BaseModel, Field

if globals.api_impl in ("SPOOF", "IRI"):
    #For convenience we use separate files for the IRI and data transfer tokens, as the latter will eventually no longer be needed
    class WorkflowConfig(BaseModel):
        iriapi_key_path: str = Field(..., description="The path to the IRI API key (will be created if doesn't yet exist)")
        transferapi_key_path: str = Field(..., description="The path to the Data Transfer API key (will be created if doesn't yet exist)")
        sandbox_directories: dict[str, str] = Field(..., description="A map of machine names to base sandbox directories")
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