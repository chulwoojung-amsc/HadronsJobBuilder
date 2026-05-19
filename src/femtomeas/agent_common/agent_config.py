from pathlib import Path
from pydantic import BaseModel, Field, ValidationError

class I2APIKeyConfig(BaseModel):
    key: str = Field(..., description="The AmSC I2 API key")

def readI2APIkey(filename):
    try:
        config = I2APIKeyConfig.model_validate_json(Path(filename).read_text())
    except ValidationError as e:
        raise Exception(f"Could not parse I2 API key {filename}: {e}")
    return config.key
