import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List
import astropy.units as u 
import astropy.constants as c 
from body import Body
from diff_eqs import nbody_diff_eqs

class Simulation:
    def __init__(self, *bodies: List[Body]) -> None:
        self.bodies = bodies
        self.quant_vec = np.concatenate([body.return_vec() for body in bodies])
        self.mass_vec = np.array([body.return_mass() for body in bodies])
    
    def set_diff_eq(self,diff_eqs,**kwargs) -> None:
        self.diff_eq_kwargs = kwargs
        self.diff_eqs = diff_eqs

    def rk4(self,t,dt):
        k1 = dt * self.diff_eqs(t,self.quant_vec,self.mass_vec,**self.diff_eq_kwargs) 
        k2 = dt * self.diff_eqs(t + 0.5*dt,self.quant_vec+0.5*k1,self.mass_vec,**self.diff_eq_kwargs)
        k3 = dt * self.diff_eqs(t + 0.5*dt,self.quant_vec+0.5*k2,self.mass_vec,**self.diff_eq_kwargs)
        k4 = dt * self.diff_eqs(t + dt,self.quant_vec + k2,self.mass_vec,**self.diff_eq_kwargs)

        y_new = self.quant_vec + ((k1 + 2*k2 + 2*k3 + k4) / 6.0)

        return y_new

    def run(self,T,dt,t0=0):
        self.T = T
        self.dt = dt
        T = T.cgs.value
        dt = dt.cgs.value

        n_steps = int((T-t0) / dt)
        self.history = np.zeros((n_steps, len(self.quant_vec)))
        for i in range(n_steps):
            y_new  = self.rk4(0, dt)
            self.history[i] = y_new
            self.quant_vec = y_new
        self.history = np.array(self.history)
        self._plot()

    def _plot(self):
        if self.history is None:
            raise AttributeError("Simulation not run(...) yet.")
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for i, body in enumerate(self.bodies):
            x = self.history[:, 6*i]
            y = self.history[:, 6*i + 1]
            z = self.history[:, 6*i + 2]
            ax.plot(x, y, z, label=body.name + " History")
            ax.scatter(x[-1], y[-1], z[-1], label=body.name)
        ax.legend()
        ax.set_title(f"Simulation over {str(self.T)} in timesteps of {str(self.dt)}")
        plt.show()

        
        


if __name__ == "__main__":
    M_moon = 7.347e22*u.kg
    v_moon = np.array([0,1.022,0])*u.km/u.s
    moon_momentum = M_moon * v_moon

    moon = Body(mass=M_moon,
            x_vec = np.array([3.84e5,0,0])*u.km,
            v_vec = v_moon,
            name='Moon')

    v_earth = - (moon_momentum / c.M_earth).to(u.km/u.s)

    earth = Body(mass=c.M_earth,
                x_vec=np.array([0,0,0])*u.km,
                v_vec=v_earth,
                name='Earth')

    simulation = Simulation(earth, moon)
    simulation.set_diff_eq(nbody_diff_eqs)

    simulation.run(72*u.day,1*u.hr)