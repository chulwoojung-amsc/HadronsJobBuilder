from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
from femtomeas.agent_common.common import *
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.meas_config_agent.meas_agent_common import Gammas
from femtomeas.meas_config_agent.source_config import momentumStr

mesonSpecialKeywords = Literal["pion","kaon","pseudoscalar","vector","axial-vector"]

def getMesonGammas(op : mesonSpecialKeywords)-> List[Gammas]:
    if op in ("pion","kaon","pseudoscalar"):
        return ["Gamma5"]
    elif op == "vector":
        return ["GammaX","GammaY","GammaZ"]
    elif op == "axial-vector":
        return ["GammaXGamma5","GammaYGamma5","GammaZGamma5"]
    else:
        raise Exception("Unknown meson operator")

def validateProps(prop_names, state):
    for p in prop_names:
        if not state.isValidPropagator(p):
            return (False, f"-Propagator instance {p} does not exist")
    return (True,"")

#TODO: cache and reuse these? Don't think there's much need but it makes neater XML
class ContractionSinkPoint(BaseModel):
    """A point sink with optional momentum (default zero-momentum)"""
    type: Literal["contraction_sink_point"] = "contraction_sink_point"
    momentum : Tuple[float,float,float,float] | None = Field(..., description="An optional four-momentum")

    def setXML(self,obs_name,xml):
        name = obs_name + "_sinkpoint"
        snk = xml.addModule(name, "MSink::ScalarPoint")
        HadronsXML.setValue(snk, "mom", momentumStr(self.momentum))
        return name

    def check(self, state):        
        return (True, "")
      
class ContractionSinkNone(BaseModel):
    """You *use* this sink type when using smeared propagators. Never use this source type for regular, unsmeared propagators"""
    type: Literal["contraction_sink_none"] = "contraction_sink_none"

    def setXML(self,obs_name,xml):
        return ""

    def check(self, state):        
        return (True, "")

class Meson2ptConfig(BaseModel):
    """A meson two-point function or "correlator" instance."""    
    type: Literal["meson2pt_config"] = "meson2pt_config"

    sink : Union[ContractionSinkNone, ContractionSinkPoint] = Field(..., description="The sink smearing for contracting unsmeared propagators", discriminator='type')
    propagators : Tuple[str,str] = Field(..., description="The tags of the propagators used to compute the observable")

    sink_gammas: List[Gammas] = Field(...,description="The list of Gamma-matrix combinations to use at the sink")
    source_gammas: List[Gammas] = Field(...,description="The list of Gamma-matrix combinations to use at the sink")
    
    def setXML(self, xml, name):
        gammas_snk_src = ""
        for gsnk in self.sink_gammas:
            for gsrc in self.source_gammas:
                gammas_snk_src = gammas_snk_src + f"({gsnk} {gsrc})"

        sink_nm = self.sink.setXML(name, xml)

        #NB: gamma5-hermiticity used on q2
        opt = xml.addModule(name, "MContraction::Meson")
        HadronsXML.setValues(opt, [ ("q1", self.propagators[0]), ("q2", self.propagators[1]), ("gammas", gammas_snk_src), ("sink", sink_nm), ("output",f"{name}.out") ])


    def checkProps(self, state):
        lprop_exists = state.isValidPropagator(self.propagators[0])
        rprop_exists = state.isValidPropagator(self.propagators[1])
        lsprop_exists = state.isValidSmearedPropagator(self.propagators[0])
        rsprop_exists = state.isValidSmearedPropagator(self.propagators[1])

        if not lprop_exists and not lsprop_exists:
            return (False, f"Propagator {self.propagators[0]} does not exist")
        if not rprop_exists and not rsprop_exists:
            return (False, f"Propagator {self.propagators[1]} does not exist")
        
        if (lprop_exists and not rprop_exists) or (rprop_exists and not lprop_exists) or (lsprop_exists and not rsprop_exists) or (rsprop_exists and not lsprop_exists):
            return (False, "Cannot mix smeared and unsmeared propagators in meson contractions")

        if (lsprop_exists and not isinstance(self.sink, ContractionSinkNone) ):
            return (False, "Sink argument must be ContractionSinkNone when using smeared propagators")
        if (lprop_exists and isinstance(self.sink,ContractionSinkNone) ):
            return (False, "Sink argument must be different from ContractionSinkNone when using unsmeared propagators")
        
        return (True, "")

    def check(self, state):
        result = True
        reason = ""

        if len(self.sink_gammas) == 0 or len(self.source_gammas) == 0:
            result = False
            reason += "\nBoth source and sink must have at least one Gamma-matrix combination"
        
        r = self.checkProps(state)
        if not r[0]:
            result = False
            reason = reason + "\n" + r[1] 

        return (result, reason)

class WritePropagators(BaseModel):
    """List of propagators to write to disk and their filestems (filename without .${CFG}.bin extension)"""
    type: Literal["write_propagators"] = "write_propagators"
    write_props : List[ Tuple[str,str] ] = Field(..., description="List of propagator name, local filestem pairs")

    def setXML(self, xml, name):
        for p in self.write_props:
            nm = name + "_" + p[0]
            opt = xml.addModule(nm, "MIO::SavePropagator")
            HadronsXML.setValues(opt, [ ("name", p[0]), ("fileStem", p[1]) ])

    def check(self, state):
        props = [p[0] for p in self.write_props]
        return validateProps(props)


class ObservableConfig(BaseModel):
    """An instance of an observable."""
    obs: Union[Meson2ptConfig, WritePropagators] = Field(...,description="The observation instance and configuration.", discriminator='type')
    name : str = Field(..., description="The name/tag for the observable instance")

    def setXML(self, xml):
        self.obs.setXML(xml, self.name)

    def check(self, state):
        return self.obs.check(state)     
