from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
from .hadrons_xml import HadronsXML

from femtomeas.agent_common.common import *
from femtomeas.agent_common.python_output_agent import parameterAgent
from .meas_agent_common import Gammas
from .agent_workflow_globals import registerInstanceModel, getInstanceModel

def momentumStr(mom):
    return "0. 0. 0. 0." if mom == None else spaceSeparateSeq(mom)

@registerInstanceModel("sources")
class PointSource(BaseModel):
    """A point or single-location source"""
    type: Literal["point"] = "point"
    location: Tuple[NonNegativeInt,NonNegativeInt,NonNegativeInt,NonNegativeInt] = Field(..., description="The point source 4D location")

    def setXML(self,name,xml):
        opt = xml.addModule(name,"MSource::Point")
        HadronsXML.setValue(opt, "position", spaceSeparateSeq(self.location))

    def check(self, state, src_names):
        return (True, "")
       
@registerInstanceModel("sources")    
class WallSource(BaseModel):
    """A wall or wall-momentum (aka just "momentum") source. A wall source requires just a timeslice, whereas a wall-momentum source needs a momentum also. When describing these sources, list them as two separate types: wall and wall-momentum/momentum"""
    type: Literal["wall"] = "wall"
    timeslice: int = Field(..., description="time slice of the wall")
    momentum: Optional[Tuple[float,float,float,float]] = Field(
        None, description="Optional four-momentum"
    )

    def setXML(self,name,xml):
        opt = xml.addModule(name,"MSource::Wall")
        HadronsXML.setValues(opt, [ ("tW",self.timeslice), ("mom", momentumStr(self.momentum)  ) ])

    def check(self, state, src_names):
        return (True, "")

@registerInstanceModel("sources")
class VolumeMomentumSource(BaseModel):
    """A volume-momentum source aka plane-wave source   src_x = e^{sum_k 2pi i p_k x_k /L_k} """
    type: Literal["volume_momentum"] = "volume_momentum"
    momentum: Tuple[float,float,float,float] = Field(..., description="The four-momentum")

    def setXML(self,name,xml):
        opt = xml.addModule(name,"MSource::Momentum")
        HadronsXML.setValue(opt, "mom", momentumStr(self.momentum))

    def check(self, state, src_names):
        return (True, "")

@registerInstanceModel("sources")
class Z2BandSource(BaseModel):
    """A wall or band source (distinguished by whether it exists on one or more timeslices) with Z2 random numbers"""
    type: Literal["z2"] = "z2"
    tA: NonNegativeInt = Field(..., description="Start timeslice of the band source")
    tB: NonNegativeInt = Field(..., description="End timeslice of the band source")    

    def setXML(self,name,xml):
        opt = xml.addModule(name,"MSource::Z2")
        HadronsXML.setValues(opt, [ ("tA",self.tA), ("tB", self.tB) ])

    def check(self, state, src_names):
        if(self.tA > self.tB):
            return (False, "End timeslice is before start timeslice")
        return (True, "")

@registerInstanceModel("sources")
class GaussianSource(BaseModel):
    """
    A gaussian source centered at some position,      [ 1/(sqrt(2*pi)*width)^3 ] exp(-i sum_{i=0}^{3} (x_i - position_i)^2/(2 width^2)  + 2pi i sum_{i=0}^4 mom_i x_i/L_i )    for  tA <= x_3 <= tB
    If the user wants a point source, use PointSource instead    
    """
    type: Literal["gauss"] = "gauss"
    position: Tuple[NonNegativeInt,NonNegativeInt,NonNegativeInt] = Field(..., description="The spatial position of the center of the Gaussian")
    momentum: Tuple[NonNegativeInt,NonNegativeInt,NonNegativeInt,NonNegativeInt] = Field(..., description="The integer four-momentum")
    tA: NonNegativeInt = Field(..., description="Start timeslice of the source")
    tB: NonNegativeInt = Field(..., description="End timeslice of the source")    
    width: float = Field(..., description="The width of the Gaussian")

    def setXML(self,name,xml):
        opt = xml.addModule(name,"MSource::Gauss")
        HadronsXML.setValues(opt, [ ("position", spaceSeparateSeq(self.position)), ("mom", momentumStr(self.momentum)), ("tA",self.tA), ("tB", self.tB), ("width", self.width) ])

    def check(self, state, src_names):
        if(self.tA > self.tB):
            return (False, "End timeslice is before start timeslice")
        return (True, "")
 





@registerInstanceModel("sources")
class SeqGammaSource(BaseModel):
    """A sequential propagator source where a source is constructed from the slice of a propagator between two timeslices with a specific gamma matrix structure and momentum,
       src_x = q_x * theta(x_3 - tA) * theta(tB - x_3) * gamma * exp(i x.mom)
    """
    type: Literal["seq_gamma"] = "seq_gamma"
    t_a : NonNegativeInt = Field(..., description="Start timeslice of sequential source")
    t_b : NonNegativeInt = Field(..., description="End timeslice of sequential source")
    gamma: Gammas = Field(...,description="Gamma-matrix structure of the sequential source")
    momentum: Optional[Tuple[float,float,float,float]] = Field(
        None, description="Optional four-momentum"
    )
    q: str = Field(..., description="The name of the propagator to construct the sequential source from")
    q_source_name : str = Field(..., description="The name of the source that will be used for this input propagator")
    q_prop_info: str= Field(...,description="Other information associated with the input propagator provided by the user")
    
    def setXML(self,name,xml):
        opt = xml.addModule(name,"MSource::SeqGamma")
        HadronsXML.setValues(opt, 
                             [
                                ("q", self.q),
                                ("tA", self.t_a),
                                ("tB", self.t_b),
                                ("gamma", self.gamma),
                                ("mom", momentumStr(self.momentum) )                                  
                              ])
    
    def check(self, state, src_names):
        if self.q_source_name not in src_names:
            return (False, f"Source {self.q_source_name} for input propagator {self.q} does not exist in the list of sources")
        return (True,"")
        


    



class SourceConfig(BaseModel):
    name : str = Field(..., description="The name/tag for the source")
    source: getInstanceModel("sources") = Field(
        ..., description="Information about the source.", discriminator='type'
    )
    
    def setXML(self,xml):
        self.source.setXML(self.name,xml)

    def check(self, state, all_sources):
        return self.source.check(state, all_sources)
