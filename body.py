import numpy as np 
import astropy.units as u 
from astropy.units.quantity import Quantity
import astropy.constants as c 

class Body:
    def __init__(self, name: str, x_vec: Quantity, v_vec: Quantity, mass: Quantity) -> None:
        self.name = name
        self.x_vec = x_vec.to(u.AU).cgs
        self.v_vec = v_vec.to(u.AU/u.d).cgs
        self.mass = mass.to(u.Msun).cgs

    def return_vec(self) -> np.array:
        return np.concatenate([self.x_vec.value, self.v_vec.value])

    def return_mass(self) -> float:
        return self.mass.value
    