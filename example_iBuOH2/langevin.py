import torch
import sys
from macecvs.calculations.calculator import ModifiedMACECalculator
from ase.io import read, Trajectory, write # input and output
from ase.constraints import FixCom         # usefull constrain for later visualisations
from ase.md.langevin import Langevin       # class for langevin dynamics
import ase.units as units                  # units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution # to sample initial velocity

calc = ModifiedMACECalculator("../MACE_ft.model", default_dtype="float32", device='cpu')
atoms = read('ini_atoms.xyz', format='extxyz')
atoms.set_constraint(FixCom())
atoms.calc = calc 
temperature_K = 500
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
dyn = Langevin(atoms,
               fixcm=True,
               timestep=1.0 * units.fs,
               temperature_K=temperature_K,
               friction=0.001 / units.fs,
               logfile=None,
               trajectory=None)
traj = dyn.closelater(Trajectory(filename='md_trajectory.traj',mode="a", atoms=atoms, properties=['energy', 'stress', 'forces']))
dyn.attach(traj.write, interval=1)
atoms.calc.calculate(atoms)

descriptors = atoms.calc.results["descriptors"][0].detach().cpu().reshape(1, 88, 256)
for i in range(1000):
    dyn.run(1)
    descriptors = torch.cat([descriptors, dyn.atoms.calc.results["descriptors"][0].detach().cpu().reshape(1, 88, 256)], dim=0)

torch.save(descriptors, 'descriptors.pt')

