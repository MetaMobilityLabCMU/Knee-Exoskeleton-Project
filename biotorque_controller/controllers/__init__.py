from .biotorque import biotorque


REGISTRY = {
    biotorque.name: biotorque,
}

def build_controller(name: str, **kwargs):
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown controller: {name}")
    return cls(**kwargs)
