import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
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
        
        xmin = min(min(self.history[:, i*6]) for i in range(len(self.bodies)))
        xmax = max(max(self.history[:, i*6]) for i in range(len(self.bodies)))
        ax.set_xlim((xmin, xmax))

        ymin = min(min(self.history[:, i*6 + 1]) for i in range(len(self.bodies)))
        ymax = max(max(self.history[:, i*6 + 1]) for i in range(len(self.bodies)))
        ax.set_ylim((ymin, ymax))

        zmin = min(min(self.history[:, i*6 + 2]) for i in range(len(self.bodies)))
        zmax = max(max(self.history[:, i*6 + 2]) for i in range(len(self.bodies)))
        ax.set_zlim((zmin, zmax))

        lines = [ax.plot([], [], [], lw=3, label=body.name)[0] for body in self.bodies]

        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return lines

        def animate(i):
            for idx, line in enumerate(lines):
                x = self.history[:i, idx*6]
                y = self.history[:i, idx*6 + 1]
                z = self.history[:i, idx*6 + 2]

                line.set_data(x, y)
                line.set_3d_properties(z)
            return lines

        ax.legend()
        anim = animation.FuncAnimation(fig, animate, init_func=init, interval=10, blit=True)
        anim.save("out.gif")
        plt.show()


        
        


if __name__ == "__main__":
    a_x = np.array((0,0,0))*u.m
    a_v = np.array((10000,0,0)) * u.m / u.s
    ba = Body("A", a_x, a_v, 1*u.Msun)

    b_x = np.array((-149598022960., 0., 0.)) * u.m
    b_v = np.array((0., -29290., 0.)) * u.m / u.s
    bb = Body("B", b_x, b_v, 1*u.Mearth)

    c_x = np.array((-149598022960. - 383397000., 100., 100.)) * u.m
    c_v = np.array((0., -29290. - 1023., 0.)) * u.m / u.s
    bc = Body("C", c_x, c_v, 1*u.Mearth / 81.3)

    simulation = Simulation(ba, bb, bc)
    simulation.set_diff_eq(nbody_diff_eqs)

    simulation.run(2*u.year,1*u.day)