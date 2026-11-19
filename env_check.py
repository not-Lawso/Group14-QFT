"""Print the software environment used for the study."""
import platform

import matplotlib
import numpy
import pandas
import qiskit
import qiskit_aer
import scipy

print("Python     :", platform.python_version(), "|", platform.system(), platform.machine())
print("Qiskit     :", qiskit.__version__)
print("Qiskit Aer :", qiskit_aer.__version__)
print("NumPy      :", numpy.__version__)
print("Pandas     :", pandas.__version__)
print("Matplotlib :", matplotlib.__version__)
print("SciPy      :", scipy.__version__)
from qiskit_aer import AerSimulator
print("Simulator  :", AerSimulator().name, "| methods:", ", ".join(AerSimulator().available_methods()[:4]), "...")
