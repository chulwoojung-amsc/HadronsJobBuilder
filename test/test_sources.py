from femtomeas.meas_config_agent.source_config import SeqGammaSource, PointSource, VolumeMomentumSource, Z2BandSource, GaussianSmearedPointSource
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.meas_config_agent.gauge import UnitGauge
from femtomeas.meas_config_agent.action_config import DWFaction
from femtomeas.meas_config_agent.propagator_config import PropagatorConfig
from femtomeas.meas_config_agent.solver_config import RBPrecCGsolver
from femtomeas.meas_config_agent.observable_config import WritePropagators, Meson2ptConfig, Meson2ptInstance


xml = HadronsXML()
xml.setRunID(0)

act = DWFaction(Ls=12, mass=0.01, M5=1.8)
act.setXML("dwfact", xml)

solv = RBPrecCGsolver(residual=1e-5, maxIteration=10000, guesser="")
solv.setXML("solv", "dwfact", xml)

vmom = VolumeMomentumSource(momentum=(1,2,1,2) )
vmom.setXML("vmom",xml)

vmomprop = PropagatorConfig(name="vmomprop", source="vmom", solver="solv", user_info="")
vmomprop.setXML(xml)

z2 = Z2BandSource(tA=0, tB=1)
z2.setXML("z2",xml)

z2prop = PropagatorConfig(name="z2prop", source="z2", solver="solv", user_info="")
z2prop.setXML(xml)


gauss = GaussianSmearedPointSource(position=(0,0,0), momentum=(1,0,1,0), tA=0, tB=1, width=2.0)
gauss.setXML("gauss",xml)

gaussprop = PropagatorConfig(name="gaussprop", source="gauss", solver="solv", user_info="")
gaussprop.setXML(xml)

writeprops = WritePropagators(write_props=[ ("vmomprop", "vmomprop_filestem"), ("z2prop", "z2prop_filestem"), ("gaussprop", "gaussprop_filestem") ]  )
writeprops.setXML(xml)

# meson2pt = Meson2ptConfig(sink_gammas=["Gamma5"], source_gammas=["Gamma5"], instances=[ Meson2ptInstance(propagators=("vmomprop","vmomprop"), name="vmomprop_meson") ]  )
# meson2pt.setXML(xml)

unit = UnitGauge() #sets trajcounter
unit.setXML(xml)

snk = xml.addModule("point_sink_zerop", "MSink::ScalarPoint")
HadronsXML.setValue(snk, "mom", "0. 0. 0.")

xml.write("test.xml")


