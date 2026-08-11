from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple

class Meson2ptObs(BaseModel):
   """A meson two-point function or "correlator"."""
   type: Literal["meson2pt"] = "meson2pt"
   meson_type : str = Field(..., description="The type of meson, e.g. pion, kaon, rho. Also accept types described by their parity transformation properties, e.g. scalar, pseudoscalar, vector")

   def skill(self):
      return """- Meson two-point functions:
  Meson two-point functions are typically used to compute particle masses or decay constants (e.g. f_pi).

  This observable requires two propagators, that are contracted together at some timeslice-localized sink. The first propagator argument has quark flow from source to sink, and the second propagator argument has gamma^5-hermiticity applied to it such that the quark flow is from sink to source. We often refer to these propagators by the direction of quark flow into the vertex, i.e. "incoming" for the first quark and "outgoing" for the second.

  Pion and kaon correlators are both pseudoscalar two-point functions, the difference being that the pion usually has both quarks with the same (light) mass, whereas the kaon has one heavy and one light quark."""

class ObservableInfo(BaseModel):
   """Information about an observable to be computed."""
   obs_type: Union[Meson2ptObs]= Field(...,description="The observable contraction and particle types, and important knowledge.", discriminator="type")
   obs_tag: str = Field(..., description="A unique tag assigned to this observable")

class ObservablesInfo(BaseModel):
    observables: List[ObservableInfo] = Field(...,description="The list of observables")


def observableSkills(observables : ObservablesInfo):
   inc = set()
   out = """
--------------------------------------
Information about required observables
--------------------------------------
"""
   for o in observables.observables:
      if (t := type(o.obs_type)) not in inc:
         out = out + o.obs_type.skill() + "\n"
         inc.add(t)
   print("SKILLS", out, "END SKILLS")
   return out    