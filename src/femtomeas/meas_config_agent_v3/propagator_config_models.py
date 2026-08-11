from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from .hadrons_xml import HadronsXML

class PropagatorConfig(BaseModel):
    name : str = Field(..., description="The name/tag for the propagator instance")
    source: str = Field(..., description="The name/tag of the propagator's source instance")
    solver: str = Field(..., description="The name/tag of the propagator's solver instance")
    
    def setXML(self,xml):
        opt = xml.addModule(self.name,"MFermion::GaugeProp")
        HadronsXML.setValues(opt, [ ("source",self.source), ("solver",self.solver) ])  
