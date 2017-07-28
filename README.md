# Dirty AstroPy

This repository comprises a range of various functions for reading, plotting, and making calculations relating primarily to astrophysical simulation data.  For example, playing around with particles from hydrodynamic simulations, or galaxies from semi-analytic models.  This is not designed to be a coherent codebase, but is rather a collection of routines that I have slowly been adding to since I started my PhD.

Note, at the moment, there are some other dependencies on others' code, which I intentionally have not added to this repository.  For now, if any routines look of interest you, rip them out and add them into your own codebase.  I will slowly make efforts to improve descriptions / comments / code efficiency throughout.

In principle, the way one would directly import these routines is to have the following code at the top of your Python script:
`
import sys  
sys.path.insert(0, '/path/Dirty-AstroPy/')  
from galprops import galplot as gp  
from galprops import galread as gr  
from galprops import galcalc as gc  
`
