from dataclasses import dataclass

@dataclass
class Character():
    name: str
    bio: str
    memory_decay: float
    memory_boost_factor: float
