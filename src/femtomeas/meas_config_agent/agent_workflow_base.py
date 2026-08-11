from pydantic import BaseModel, Field, model_serializer, model_validator
from typing import Tuple, TypeVar, ClassVar, Callable, Any
from femtomeas.agent_common.callgraph import Node
from pydantic_core import core_schema

class BaseGroupHandle:
    def __init__(self, group_name : str, parent_node : Node):
        self.group_name = group_name
        self.parent_node = parent_node

class BaseGroup(BaseModel):
    """The base class of all group objects, with appropriate hooks to properly serialize the derived class type information"""
    _registry: ClassVar[dict[str, type["BaseGroup"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is not BaseGroup:
            BaseGroup._registry[cls.__name__] = cls

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        # Only the abstract base does polymorphic dispatch.        
        if cls.__qualname__ != "BaseGroup":
            return handler(source_type)

        return core_schema.no_info_plain_validator_function(cls._from_any)

    @classmethod
    def _from_any(cls, value: Any) -> "BaseGroup":
        if isinstance(value, BaseGroup):
            return value

        if not isinstance(value, dict):
            raise TypeError("Expected dict or BaseGroup instance")

        kind = value.get("__type__")
        if not kind:
            raise ValueError("Missing __type__")

        subcls = cls._registry.get(kind)
        if subcls is None:
            raise ValueError(f"Unknown group type: {kind}")

        payload = dict(value)
        payload.pop("__type__", None)
        return subcls.model_validate(payload)

    @model_serializer(mode="plain")
    def _dump(self) -> dict[str, Any]:
        return {**self.__dict__, "__type__": self.__class__.__name__}

GroupHandleTypes = TypeVar("GroupHandleTypes", bound=BaseGroupHandle)
GroupTypes = TypeVar("GroupTypes", bound=BaseGroup)