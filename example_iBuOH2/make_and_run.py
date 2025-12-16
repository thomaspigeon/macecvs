import os
import shutil
import numpy as np

for i in range(100):
    dirname = "trajectory_" + str(i)
    os.mkdir(dirname)
    shutil.copy("job_langevin", dirname)
    os.system("cp TS1.xyz " + dirname + "/ini_atoms.xyz")
    current_dir = os.path.abspath('.')
    os.chdir(dirname)
    command = "sbatch job_langevin"
    os.system(command)
    os.chdir(current_dir)

for i in range(100,200):
    dirname = "trajectory_" + str(i)
    os.mkdir(dirname)
    shutil.copy("job_langevin", dirname)
    os.system("cp TS2.xyz " + dirname + "/ini_atoms.xyz")
    current_dir = os.path.abspath('.')
    os.chdir(dirname)
    command = "sbatch job_langevin"
    os.system(command)
    os.chdir(current_dir)
