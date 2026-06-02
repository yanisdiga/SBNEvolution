class Food:
    """
    Represents an energy source in the environment.
    """
    def __init__(self, x: float, y: float, energy: float = 50.0, active: bool = True):
        self.x = x
        self.y = y
        self.energy = energy
        self.alive = True
        self.active = active