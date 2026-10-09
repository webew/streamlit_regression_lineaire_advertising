from pydantic import BaseModel, Field

class Budget(BaseModel):
    TV: float = Field(ge=0, description="Budget TV (k$)")
    radio: float = Field(ge=0, description="Budget radio (k$)")
    newspaper: float = Field(ge=0, description="Budget newspaper (k$)")