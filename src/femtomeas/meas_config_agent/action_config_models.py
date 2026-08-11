from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
import xml.etree.ElementTree as ET
from .hadrons_xml import HadronsXML
from .agent_workflow_globals import registerInstanceModel, getInstanceModel
from .meas_agent_common import Precision

@registerInstanceModel("actions")
class DWFaction(BaseModel):
    """A Domain Wall Fermion (DWF) action instance"""
    type: Literal["DWF"] = "DWF"
    Ls: int = Field(..., description="The length/size of the fifth dimension")
    mass: float = Field(..., description="the mass parameter of the action and its associated propagators")
    M5: float = Field(..., description="the M5 parameter of the action")

    def setXML(self,name,precision,xml):
        opt = xml.addModule(name,"MAction::DWF" if precision == "Double" else "MAction::DWFF")
        HadronsXML.setValues(opt, [ ("gauge", "gauge" if precision == "Double" else "gaugef"), 
                                   ("Ls", self.Ls), ("mass", self.mass), ("M5",self.M5), ("boundary", "1 1 1 -1"), ("twist", "0. 0. 0. 0.") ] )

@registerInstanceModel("actions")       
class WilsonCloverAction(BaseModel):
    """A Wilson-Clover (aka Clover) action instance"""
    type: Literal["WilsonClover"] = "WilsonClover"
    mass: float = Field(..., description="the mass parameter of the action and its associated propagators")
    csw_r: float = Field(..., description="Clover-term coefficient c_SW^r")
    csw_t: float = Field(..., description="Clover-term coefficient c_SW^t")
                        
    def setXML(self,name,precision,xml):
        opt = xml.addModule(name,"MAction::WilsonClover" if precision == "Double" else "MAction:WilsonCloverF")
        HadronsXML.setValues(opt, [ ("gauge", "gauge" if precision == "Double" else "gaugef"), ("mass", self.mass), ("csw_r",self.csw_r), ("csw_t",self.csw_t) ] )

        ca = ET.SubElement(opt, "clover_anisotropy")
        HadronsXML.setValues(ca, [ ("isAnisotropic", "false"), ("t_direction",3), ("xi_0", "1.0"), ("nu", "1.0") ] )
                        
        HadronsXML.setValues(opt, [ ("boundary", "1 1 1 -1"), ("twist", "0. 0. 0. 0.") ] )                               
    
class ActionConfig(BaseModel):
    name : str = Field(..., description="The name/tag for the action instance")
    action: getInstanceModel("actions") = Field(..., description="Parameters of the action. Each item must have a 'type' field.",discriminator='type')
    precision: Precision = Field(..., description="The floating point precision of the action")
    
    def setXML(self,xml):
        self.action.setXML(self.name, self.precision, xml)
