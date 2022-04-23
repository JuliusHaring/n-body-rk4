import numpy as np
import astropy.constants as c

def nbody_diff_eqs(t,y,masses):
    N_bodies = int(len(y) / 6)
    solved_vector = np.zeros(y.size)
    for i in range(N_bodies):
        ioffset = i * 6 
        for j in range(N_bodies):
            joffset = j*6
            solved_vector[ioffset] = y[ioffset+3]
            solved_vector[ioffset+1] = y[ioffset+4]
            solved_vector[ioffset+2] = y[ioffset+5]
            if i != j:
                dx = y[ioffset] - y[joffset]
                dy = y[ioffset+1] - y[joffset+1]
                dz = y[ioffset+2] - y[joffset+2] 
                r = (dx**2+dy**2+dz**2)**0.5
                ax = (-c.G.cgs * masses[j] / r**3) * dx
                ay = (-c.G.cgs * masses[j] / r**3) * dy
                az = (-c.G.cgs * masses[j] / r**3) * dz
                solved_vector[ioffset+3] += ax.value
                solved_vector[ioffset+4] += ay.value
                solved_vector[ioffset+5] += az.value      
    return solved_vector
