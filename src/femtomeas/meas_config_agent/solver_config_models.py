from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
from femtomeas.agent_common.common import *
from .hadrons_xml import HadronsXML
from femtomeas.agent_common.python_output_agent import parameterAgent
from .agent_workflow_globals import registerInstanceModel, getInstanceModel

@registerInstanceModel("solvers")
class RBPrecCGsolver(BaseModel):
    """red-black preconditioned conjugate gradient (CG) solver"""
    type: Literal["RBPrecCG"] = "RBPrecCG"
    residual: float = Field(...,description="the solver tolerance, residual or stopping condition. Typical values are in the range 1e-6 to 1e-9")
    maxIteration: NonNegativeInt = Field(...,description="maximum number of solver iterations.")
    guesser: str = Field(..., description="guesser instance.")
    
    def setXML(self,name,action,xml):
        opt = xml.addModule(name,"MSolver::RBPrecCG")
        HadronsXML.setValues(opt, [ ("action",action), ("maxIteration", self.maxIteration), ("residual", self.residual), ("guesser", self.guesser) ])

    def check(self,action,state):
        return (True,"")


@registerInstanceModel("solvers")
class MixedPrecisionRBPrecCGsolver(BaseModel):
    """mixed-precision red-black preconditioned conjugate gradient (CG) solver"""
    type: Literal["MixedPrecisionRBPrecCG"] = "MixedPrecisionRBPrecCG"
    residual: float = Field(...,description="the solver tolerance, residual or stopping condition. Typical values are in the range 1e-6 to 1e-9")
    innerAction: str = Field(..., description="the single-precision inner action")
    maxInnerIteration: NonNegativeInt = Field(...,description="maximum number of inner solver iterations.")
    maxOuterIteration: NonNegativeInt = Field(...,description="maximum number of outer solver iterations.")
    innerGuesser: str = Field(..., description="guesser instance for the inner solver (must be single precision)")
    outerGuesser: str = Field(..., description="guesser instance for the outer solver (must be double precision)")
    
    def setXML(self,name,action,xml):
        opt = xml.addModule(name,"MSolver::RBPrecCG")
        HadronsXML.setValues(opt, [ ("innerAction", self.innerAction), ("outerAction", action), 
                                    ("maxInnerIteration", self.maxInnerIteration), ("maxOuterIteration", self.maxOuterIteration),
                                    ("residual", self.residual), ("innerGuesser", self.innerGuesser), ("outerGuesser", self.outerGuesser) ])


    def check(self,action,state):
        sact_inst = state.getInstance("actions", self.innerAction)
        if sact_inst is None:
            return (False, f"Inner action {self.innerAction} does not exist")
        if sact_inst.precision != "Single":
            return (False, f"Inner action must be single-precision")
        return (True,"")


class SolverConfig(BaseModel):
    name : str = Field(..., description="The name/tag for the solver instance")
    solver_args: getInstanceModel("solvers") = Field(..., description="Parameters of the solver. Each item must have a 'type' field. Valid values are: RBPrecCG", discriminator='type')
    action: str = Field(..., description="The name/tag of the action instance to use with the solver.")
    
    def setXML(self,xml):
        self.solver_args.setXML(self.name,self.action,xml)

    def check(self, state):
        act_inst = state.getInstance("actions", self.action)
        if act_inst is None:
            return (False, f"Action instance {self.action} does not exist")
        if act_inst.precision != "Double":
            return (False, f"SolverConfig.action must refer to a double-precision action instance")

        return self.solver_args.check(self.action, state)
        