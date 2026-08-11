from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
import os
from femtomeas.agent_common.common import *
from .hadrons_xml import HadronsXML
from femtomeas.workflow_manager.api_general import listSpecialGlobusEndpoints
from femtomeas.agent_common.agent_base import parameterAgent

class LoadGauge(BaseModel):
    """Load NERSC-format gauge configurations. If the user provides a range, infer from it the start, step and end."""
    type: Literal["gauge-load"] = "gauge-load"
    source_uuid :  str | None = Field(..., description="A Globus endpoint UUID containing the configurations. If the path is local, set the value to None.")
    stub : str = Field(...,description="The path stub of the configurations. A period followed by the configuration index will be appended during the run. If the user provides a complete path including an index (or variable_, remove the period and index")
    start : int = Field(...,description="The index of the first configuration")
    step : int = Field(...,description="The increment between successive configurations")
    end : int = Field(...,description="The index of the last configuration")
    
    def setXML(self,xml):
        opt = xml.addModule("gauge","MIO::LoadNersc")
        HadronsXML.setValue(opt, "file", self.stub)
        xml.setTrajCounter(self.start, self.end + self.step, self.step) #note, the "end" parameter in Hadrons is not actually the last config but rather one step after!

    def setXMLsingle(self,xml,job_index, override_path = None  ):
        """
        Output the XML just for a single configuration.
        job_index : The index of the entry in the range, i.e. 0 -> start, 1 -> start+step,  etc
        override_path : Replace the path in which the file resides, e.g. if it was moved prior to execution
        """
        
        stub = self.stub
        if override_path != None:
            stub = os.path.join(override_path,  os.path.basename(self.stub) )
        ckpoint_idx = self.start + job_index * self.step
        if ckpoint_idx > self.end:
            raise Exception("Configuration index is out of range")           
        
        opt = xml.addModule("gauge","MIO::LoadNersc")
        HadronsXML.setValue(opt, "file", stub)
        xml.setTrajCounter(ckpoint_idx, ckpoint_idx+1, 1)

        
    def getJobConfigurationsAndSource(self):
        """
        Return a list of configuration filenames required for the job and the source endpoint ID. If no actual file is required return a suitable sized list of None for the first argument. If the files are local or no files are required, return None for the second argument.
        """
        return [ self.stub + f".{i}" for i in range(self.start, self.end+self.step, self.step) ], self.source_uuid

        
class UnitGauge(BaseModel):
    """Use a unit gauge configuration"""
    type: Literal["gauge-unit"] = "gauge-unit"

    def setXML(self,xml):
        xml.addModule("gauge","MGauge::Unit")
        xml.setTrajCounter(0,1,1)

    def setXMLsingle(self,xml,job_index, override_path = None  ):
        self.setXML(xml)

    def getJobConfigurationsAndSource(self):
        """
        Return a list of configuration filenames required for the job and the source endpoint ID. If no actual file is required return a suitable sized list of None for the first argument. If the files are local or no files are required, return None for the second argument.
        """
        return [ None ], None


        
class RandomGauge(BaseModel):
    """Use a random gauge configuration"""
    type: Literal["gauge-random"] = "gauge-random"
    def setXML(self,xml):
        xml.addModule("gauge","MGauge::Random")
        xml.setTrajCounter(0,1,1)

    def setXMLsingle(self,xml,job_index, override_path = None  ):
        self.setXML(xml)

    def getJobConfigurationsAndSource(self):
        """
        Return a list of configuration filenames required for the job and the source endpoint ID. If no actual file is required return a suitable sized list of None for the first argument. If the files are local or no files are required, return None for the second argument.
        """
        return [ None ], None
        
        
class GaugeFieldConfig(BaseModel):
    config: Union[LoadGauge,UnitGauge,RandomGauge] = Field(
        ..., description="Information about the gauge configuration(s) that will be computed upon.", discriminator='type'
    )
    Lx : int = Field(..., description="The lattice size in the x-direction in dimensionless (lattice) units.")
    Ly : int = Field(..., description="The lattice size in the y-direction in dimensionless (lattice) units.")
    Lz : int = Field(..., description="The lattice size in the z-direction in dimensionless (lattice) units.")
    Lt : int = Field(..., description="The lattice size in the t-direction in dimensionless (lattice) units.")
    
    def setXML(self,xml):
        self.config.setXML(xml)
        cast = xml.addModule("gaugef", "MUtilities::GaugeSinglePrecisionCast")
        HadronsXML.setValue(cast, "field", "gauge")
        
    def setXMLsingle(self,xml,job_index, override_path = None  ):
        """
        Output the XML just for a single configuration.
        job_index : The index of the entry in the range, i.e. 0 -> start, 1 -> start+step,  etc
        override_path : Replace the path in which the file resides, e.g. if it was moved prior to execution
        """       
        self.config.setXMLsingle(xml, job_index, override_path)

    def getJobConfigurationsAndSource(self):
        """
        Return a list of configuration filenames required for the job and the source endpoint ID. If no actual file is required return a suitable sized list of None for the first argument. If the files are local or no files are required, return None for the second argument.
        """
        return self.config.getJobConfigurationsAndSource()

    def getGrid(self)->Tuple[int,int,int,int]:
        return (self.Lx, self.Ly, self.Lz, self.Lt)
        

def identifyGaugeConfigs(model, user_interactions: list[BaseMessage]) -> GaugeFieldConfig:
    #This version uses the default structured output strategy

    role = "identifying all lattice QCD gauge configurations required for the calculation, based solely on user input."

    parameter_rules = [
        """Lx, Ly, Lz, Lt:
  Do not ask for these parameters separately. Instead, if the user has not already done so in their previous responses, ask the user to specify the lattice size in dimensionless (lattice) units. Use the lattice size query rules specified below.

  Lattice size query rules:    
   - Do not insist on a specific format. Rather, in parentheses list some common formats, e.g. 32^3x64, 32 32 32 64, 32x32x32x32x64.
   - Assume that these strings list the spatial (Lx,Ly,Lz) followed by the time (Lt) dimensions, i.e.. 32^3x64, 32 32 32 64, 32x32x32x32x64 all represent Lx=Ly=Lz=32, Lt=64
   - The user may choose to specify all directions at once or only some; keep asking follow-up questions until all four dimensions have been specified
   - Some actions such as DWF have a fifth dimension; ignore this dimension.""",

       """config:
  - If the user has not already done so in their previous responses, ask the user to specify where to obtain the configurations.
  - The supported options are defined by the models supported by the config field. When describing these options, use full sentences, not abbreviations. e.g. "You can use the unit gauge, a random gauge configuration or load a gauge field."
  - The goal of your question is to identify which of these options the user desires; *do not* list the parameters associated with those choices.""",

       f"""LoadGauge.source_uuid: If loading gauge configurations, determine whether the configuration files are local or remote. If remote, a Globus endpoint must be provided and entered into the source_uuid field; if local, this field should be set to None.

       The API also accepts special UUID tags {listSpecialGlobusEndpoints()} in place of regular ID strings. List these special tags as part of your question.""",

       """LoadGauge.{start, step, end}: Do not ask the user to specify these one at a time. Instead ask them to specify the configuration range and use that to populate these numbers."""
    ]

    return parameterAgent(model, GaugeFieldConfig, role, tools=[], tool_rules=[], parameter_rules=parameter_rules, input_messages=user_interactions)
