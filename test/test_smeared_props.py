from femtomeas.meas_config_agent.source_config import SeqGammaSource, PointSource, VolumeMomentumSource, Z2BandSource, GaussianSmearedPointSource
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.meas_config_agent.gauge import UnitGauge
from femtomeas.meas_config_agent.action_config import DWFaction
from femtomeas.meas_config_agent.propagator_config import PropagatorConfig
from femtomeas.meas_config_agent.solver_config import RBPrecCGsolver
from femtomeas.meas_config_agent.observable_config import WritePropagators, Meson2ptConfig, Meson2ptInstance, ContractionSinkPoint, ContractionSinkNone
from femtomeas.meas_config_agent.smeared_prop_config import SmearedPropagatorConfig, WallSmear

xml = HadronsXML()
xml.setRunID(0)

act = DWFaction(Ls=12, mass=0.01, M5=1.8)
act.setXML("dwfact", xml)

solv = RBPrecCGsolver(residual=1e-5, maxIteration=10000, guesser="")
solv.setXML("solv", "dwfact", xml)

src = PointSource(location=(0,0,0,0))
src.setXML("src", xml)

prop = PropagatorConfig(name="prop", source="src", solver="solv", user_info="")
prop.setXML(xml)

sprop = SmearedPropagatorConfig(name="sprop", input_prop="prop", user_info="",  smearing=WallSmear(momentum=(1,0,1,0)  ))
sprop.setXML(xml)

meson2pt = Meson2ptConfig(sink_gammas=["Gamma5"], source_gammas=["Gamma5"], instances=[ 
    Meson2ptInstance(propagators=("prop","prop"), name="prop_meson", sink=ContractionSinkPoint(momentum=(1,0,0,0)) ),
    Meson2ptInstance(propagators=("sprop","sprop"), name="sprop_meson", sink=ContractionSinkNone() )    
    ]  )
meson2pt.setXML(xml)

unit = UnitGauge() #sets trajcounter
unit.setXML(xml)

xml.write("test.xml")


