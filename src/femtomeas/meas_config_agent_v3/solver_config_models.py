from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
from femtomeas.agent_common.common import *
from .hadrons_xml import HadronsXML
from femtomeas.agent_common.python_output_agent import parameterAgent

class RBPrecCGsolver(BaseModel):
    """red-black preconditioned conjugate gradient (CG) solver"""
    type: Literal["RBPrecCG"] = "RBPrecCG"
    residual: float = Field(...,description="the solver tolerance, residual or stopping condition. Typical values are in the range 1e-6 to 1e-9")
    maxIteration: NonNegativeInt = Field(...,description="maximum number of solver iterations.")
    guesser: str = Field(..., description="guesser instance.")
    
    def setXML(self,name,action,xml):
        opt = xml.addModule(name,"MSolver::RBPrecCG")
        HadronsXML.setValues(opt, [ ("action",action), ("maxIteration", self.maxIteration), ("residual", self.residual), ("guesser", "") ])

    
class SolverConfig(BaseModel):
    name : str = Field(..., description="The name/tag for the solver instance")
    solver_args: Union[RBPrecCGsolver] = Field(..., description="Parameters of the solver. Each item must have a 'type' field. Valid values are: RBPrecCG", discriminator='type')
    action: str = Field(..., description="The name/tag of the action instance to use with the solver.")
    
    def setXML(self,xml):
        self.solver_args.setXML(self.name,self.action,xml)