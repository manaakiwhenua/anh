## standard library
from copy import copy,deepcopy
from pprint import pprint,pformat
import shutil
import warnings

## nonstandard library
import numpy as np
np.set_printoptions(linewidth=np.nan) 
from numpy import array,nan,arange,linspace,logspace,isnan,inf
from scipy import integrate
from scipy.constants import pi as π
from scipy.constants import Boltzmann as kB

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.pyplot import *


## try to import fortran stuff
# try:
    # from .fortran_tools import fortran_tools
# except ModuleNotFoundError:
    # fortran_tools = None
    # warnings.warn("Could not import fortran_tools.  Is it compiled?")


## import top level and submodules referenced to top level
# import spectr
# import spectr.tools
# import spectr.dataset
# import spectr.optimise
# import spectr.plotting
# import spectr.convert

## import submodules referenced to nothing
# from . import tools
# from . import quantum_numbers
# from . import dataset
# from . import optimise
# from . import plotting
# from . import convert

## import contents of submodules
from .tools import *
from .dataset import Dataset
from .convert import units
from .optimise import Optimiser,Parameter,P,Fixed
from .plotting import *

