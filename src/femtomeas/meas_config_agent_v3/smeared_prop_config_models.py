from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
from .hadrons_xml import HadronsXML

from femtomeas.agent_common.common import *
from .meas_agent_common import Gammas
from .source_config import momentumStr

class WallSmear(BaseModel):
    """A wall smearing with optional momentum,    sum_x e^{+i p . x} prop_sol(x)  """
    type: Literal["wall_smear"] = "wall_smear"
    momentum : Tuple[float,float,float,float] | None = Field(..., description="An optional four-momentum")

    def setXML(self,smeared_prop_name,prop_name,xml):
        smr_base_name = smeared_prop_name + "_base"
        base = xml.addModule(smr_base_name, "MSink::Point")
        HadronsXML.setValue(base, "mom", momentumStr(self.momentum))
        
        opt = xml.addModule(smeared_prop_name,"MSink::Smear")
        HadronsXML.setValues(opt, [("sink", smr_base_name), ("q", prop_name)] )

    def check(self, state):
        return (True, "")


class SmearedPropagatorConfig(BaseModel):
    name : str = Field(..., description="The name/tag for the smeared propagator")
    input_prop : str = Field(..., description="The name/tag of the input propagator")
    smearing: Union[WallSmear] = Field(
        ..., description="Information about the smearing type", discriminator='type'
    )
    
    def setXML(self,xml):
        self.smearing.setXML(self.name, self.input_prop,  xml)

    def check(self, state):
        if not state.isValidPropagator(self.input_prop):
            return (False, f"Input propagator {self.input_prop} does not exist")
        return self.smearing.check(state)