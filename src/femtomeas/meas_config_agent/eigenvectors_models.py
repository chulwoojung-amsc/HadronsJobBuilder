from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter, PositiveFloat, PositiveInt
from typing import Literal, Union, List, Optional, Tuple
from .hadrons_xml import HadronsXML
from femtomeas.agent_common.common import *
from femtomeas.agent_common.python_output_agent import parameterAgent

class ChebyParams(BaseModel):
    """Parameters of the Chebyshev polynomial"""
    alpha : PositiveFloat = Field(..., description="The lower bound of the eigenvalue window suppresed by the Chebyshev filter.")
    beta : PositiveFloat = Field(..., description="The upper bound of the eigenvalue window suppresed by the Chebyshev filter.")
    Npoly: int = Field(..., ge=3, description="The Chebyshev polynomial order.")

    def check(self, state):
        if self.beta <= self.alpha:
            return (False, "The upper window bound beta must be larger than the lower window bound alpha")
        elif self.Npoly % 2 == 0:
            return (False, "The polynomial order must be and odd integer")
        else:
            return (True,"")
        
    def setXML(self, chebyParams):
        HadronsXML.setValue(chebyParams, "alpha", self.alpha)
        HadronsXML.setValue(chebyParams, "beta", self.beta)
        HadronsXML.setValue(chebyParams, "Npoly", self.Npoly)
       
        
    
class LanczosEigenSolver(BaseModel):
    """Parameters of the Lanczos eigensolver"""
    type: Literal["LanczosEigenSolver"] = "LanczosEigenSolver"

    cheby: ChebyParams = Field(..., description="Parameters of the Chebyshev filter")
    Nstop: PositiveInt = Field(..., description="Stop the solver once this many eigenvectors are within the tolerance")
    Nk: PositiveInt = Field(..., description="The number of eigenvectors to keep on each restart")
    Nextra: PositiveInt = Field(..., description="The number of extra/spare eigenvectors in the Krylov space which are discarded on each restart, i.e. Nm-Nk")
    resid: PositiveFloat = Field(..., description="The eigensolver convergence tolerance")
    MaxIt: PositiveInt = Field(20, description="The maximum number of restart iterations")

    storeEvecs: bool = Field(False, description="Indicate whether the eigenvectors are to be written to disk")
    fileStem: str = Field("", description="The file stem for the eigenvectors. The trajectory index will be appended")    

    def check(self, state, action_name):
        cheby_valid, cheby_why = self.cheby.check(state)
        if not cheby_valid:
            return (False, cheby_why)
        if not state.isValidAction(action_name):
            return (False, "Provided action name is not among the list of actions in the state")
        if self.storeEvecs and self.fileStem == "":
            return (False, "If writing the eigenvectors, a valid file stem must be provided")
        if self.Nstop > self.Nk:
            return (False, "Nk must be greater than or equal to Nstop")
        
        return (True,"")
        
    
    def setXML(self,name,xml,action_name):
        #We expose only the guesser to the agents, the other crud can stay internal
        
        lanc_name = name + "_solver"
        op_name = name + "_op"

        #Operator module
        op_opt = xml.addModule(op_name, "MFermion::Operators")
        HadronsXML.setValue(op_opt, "action", action_name)

        #Guesser module (create one even if we don't intend to use it as it has no real overhead)
        guesser_opt = xml.addModule(name, "MGuesser::ExactDeflation")
        HadronsXML.setValue(guesser_opt, "eigenPack", lanc_name)
        HadronsXML.setValue(guesser_opt, "size", self.Nstop)

        #Lanczos module
        lanc_opt = xml.addModule(lanc_name,"MSolver::FermionImplicitlyRestartedLanczos")

        #LancParams
        lp = HadronsXML.createSubElement(lanc_opt, "lanczosParams")        
        cheby = HadronsXML.createSubElement(lp, "Cheby")
        self.cheby.setXML(cheby)

        HadronsXML.setValue(lp, "Nstop", self.Nstop)
        HadronsXML.setValue(lp, "Nk", self.Nk)
        HadronsXML.setValue(lp, "Nm", self.Nk + self.Nextra)
        HadronsXML.setValue(lp, "resid", self.resid)
        HadronsXML.setValue(lp, "MaxIt", self.MaxIt)
        HadronsXML.setValue(lp, "betastp", 0.) #this doesn't do anything in Grid's solver anymore, but still must be specified :(
        HadronsXML.setValue(lp, "MinRes", 0) #forces a minimum number of restarts, not sure why this would be useful...

        #Lanczos options
        HadronsXML.setValue(lanc_opt, "op", op_name + "_schur") #Use with even/odd fields
        HadronsXML.setValue(lanc_opt, "output", self.fileStem)
        HadronsXML.setValue(lanc_opt, "redBlack", True)
        HadronsXML.setValue(lanc_opt, "multiFile", False) #I think this stores the evecs to part files per node rather than to one big file, maybe make an option later

class EigenSolverConfig(BaseModel):
    name : str = Field(..., description="The name for the eigensolver instance")  #technically this is the name of the guesser but that is irrelevant to the functioning of the agents
    action : str = Field(..., description="The name of the associated action")
    solver_args: Union[LanczosEigenSolver] = Field(..., description="Parameters of the eigensolver. Each item must have a 'type' field. Valid values are: LanczosEigenSolver", discriminator='type')
    
    def check(self, state):
        return self.solver_args.check(state, self.action)

    def setXML(self,xml):
        self.solver_args.setXML(self.name, xml, self.action)
