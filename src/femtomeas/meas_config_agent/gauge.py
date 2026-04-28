from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

import os
import json
from .common import *
from .hadrons_xml import HadronsXML

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
        xml.setTrajCounter(self.start, self.end, self.step)

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
        xml.setTrajCounter(ckpoint_idx, ckpoint_idx, 1)

        
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

    sys = """
You are an assistant responsible for identifying all lattice QCD gauge configurations required for the calculation, based solely on user input.

The configuration information is parameterized by a GaugeFieldConfig structure. Your workflow is:
1) if the user has not already done so in their previous responses, ask the user to specify the lattice size in dimensionless (lattice) units. Use the lattice size query rules specified below.
1) if the user has not already done so in their previous responses, ask the user to specify where to obtain the configurations. The supported options are defined by the models supported by the GaugeFieldConfig.config field. The goal of your question is to identify which of these options the user desires; *do not* list the parameters associated with those choices.
2) instantiate an GaugeFieldConfig instance with the config field chosen as above, and input the lattice sizes.
3) Populate the parameters of the data structure. If a parameter value is unknown you must ask the user; never guess parameters.
4) If loading gauge configurations, determine whether the configuration files are local or remote. If remote, a Globus endpoint must be provided and entered into the source_uuid field; if local, this field should be set to None. Note that we are using NERSC's SuperFacility API to initiate these transfers, which also accepts special UUIDs "dtn", "hpss" or "perlmutter" in place of regular ID strings.

Lattice size query rules:    
- Do not insist on a specific format. Rather, in parentheses list some common formats, e.g. 32^3x64, 32 32 32 64, 32x32x32x32x64.
- The user may choose to specify all directions at once or only some; keep asking follow-up questions until all four dimensions have been specified
- Some actions such as DWF have a fifth dimension; ignore this dimension
    
User Query rules:
- Use the getUserInput tool
- If the user responds to a query with an invalid response, repeat the query until a valid response is provided. Never accept an invalid response.
- Instead of answering your question, the user might respond to your query with a question. If this occurs, answer the user's question using provideInformationToUser tool and ensure the user is satisfied with a follow-up call to getUserInput. Once satisfied, repeat the original question.

Your output must be in JSON format and adhere to the following schema:    
""" + json.dumps(GaugeFieldConfig.model_json_schema())
   
    agent = create_agent(model=model, tools=[getUserInput,provideInformationToUser], system_prompt=sys, response_format = GaugeFieldConfig)
    
    accepted = False
    obj = None
    while(accepted == False):    
        resp = agent.invoke({ "messages": user_interactions },     {"configurable": {"thread_id": "1"}})
        obj = resp["structured_response"]        
        user_interactions = resp['messages']
        
        output = f"Obtained gauge field parameters\n" + prettyPrintPydantic(obj)
        Print(output)

        accepted = queryYesNo("Is this correct?")
        if(accepted == False):
            reason = Input("Explain what is wrong: ")
            user_interactions.append(HumanMessage(f"Your previous response was not accepted for the following reason: {reason}"))
    return obj










        
#This seems to be another one that confuses ToolStrategy, giving incorrect structured output
@tool
def setUnitGauge(runtime: ToolRuntime) -> None:
    """Use a unit gauge configuration"""
    runtime.store.put(("ns",), "gauge", UnitGauge())
@tool
def setRandomGauge(runtime: ToolRuntime) -> None:
    """Use a random gauge configuration"""
    runtime.store.put(("ns",), "gauge", RandomGauge())

@tool
def setLoadGauge(stub : str, start: int, step: int, end: int, runtime : ToolRuntime) -> None:
    """Load NERSC-format gauge configurations.
    Args:
        stub : The path stub of the configurations. A period followed by the configuration index will be appended during the run. If the user provides a complete path including an index (or variable_, remove the period and index)
        start : The index of the first configuration
        step : The increment between successive configurations
        end : The index of the last configuration
    Instructions:
        If the user provides a range, infer from it the start, step and end.
    """
    runtime.store.put(("ns",), "gauge", LoadGauge(stub=stub, start=start, step=step, end=end))

    

def identifyGaugeConfigsToolBased(model, user_interactions: list[BaseMessage]) -> GaugeFieldConfig:
    #ToolStrategy can be flakey. This version uses explicit tooling
    
    sys = """
You are an assistant responsible for identifying the lattice QCD propagator gauge configuration(s) to use for the calculation, based solely on user input.

Your workflow:
1. Identify the appropriate tool for the gauge configuration specification based on user input. If the user does not specify what configuration(s) to use you must ask the user. Never guess a gauge type.
2. Call the tool with its required parameters. If a parameter value is unknown you must ask the user; never guess parameters.
       
User Query rules:
- Use the getUserInput tool
- If the user responds to a query with an invalid response, repeat the query until a valid response is provided. Never accept an invalid response.
- Instead of answering your question, the user might respond to your query with a question. If this occurs, answer the user's question using provideInformationToUser tool and ensure the user is satisfied with a follow-up call to getUserInput. Once satisfied, repeat the original question.
"""
    
#Your output must be in JSON format and adhere to the following schema:    
#""" + json.dumps(GaugeFieldConfig.model_json_schema())
# Your workflow:
# 1. Identify the appropriate schema for the 'config' field based on user input. If the user does not specify what configuration(s) to use you must ask the user. Never guess a gauge type.
# 2. Fill in all parameters associated with the gauge field instance. If a parameter value is unknown you must ask the user; never guess parameters.

    store = InMemoryStore()

    agent = create_agent(model=model, tools=[getUserInput,provideInformationToUser,setUnitGauge,setRandomGauge,setLoadGauge], system_prompt=sys, store=store) #, response_format=ToolStrategy(GaugeFieldConfig))
    
    accepted = False
    obj = None
    while(accepted == False):    
        resp = agent.invoke({ "messages": user_interactions },     {"configurable": {"thread_id": "1"}})
        config = store.get( ("ns",), "gauge").value
        obj = GaugeFieldConfig(config=config)
        
        Print("Obtained gauge field parameters:", obj.config)

        accepted = queryYesNo("Is this correct? [y/n]: ")
        if(accepted == False):
            reason = Input("Explain what is wrong: ")
            user_interactions.append(HumanMessage(f"Your previous response was not accepted for the following reason: {reason}"))            
    return obj
