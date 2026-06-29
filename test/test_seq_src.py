from femtomeas.meas_config_agent.source_config import SeqGammaSource, PointSource
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.meas_config_agent.gauge import UnitGauge
from femtomeas.meas_config_agent.action_config import DWFaction
from femtomeas.meas_config_agent.propagator_config import PropagatorConfig
from femtomeas.meas_config_agent.solver_config import RBPrecCGsolver

xml = HadronsXML()
xml.setTrajCounter(0,0,1)
xml.setRunID(0)

pt = PointSource(location=(0,0,0,0))
pt.setXML("pt",xml)

seq = SeqGammaSource(t_a=0, t_b=0, gamma="Gamma5", momentum=None, q="propin", q_source_name="pt", q_prop_info="")
seq.setXML("seq", xml)

act = DWFaction(Ls=12, mass=0.01, M5=1.8)
act.setXML("dwfact", xml)

solv = RBPrecCGsolver(residual=1e-5, maxIteration=10000, guesser="")
solv.setXML("solv", "dwfact", xml)

propin = PropagatorConfig(name="propin", source="pt", solver="solv", user_info="")
propin.setXML(xml)
propseq = PropagatorConfig(name="propseq", source="seq", solver="solv", user_info="")
propseq.setXML(xml)

unit = UnitGauge()
unit.setXML(xml)

xml.write("test.xml")


