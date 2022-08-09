# Adam Stevens, 2013--2021
# Functions for calculating properties of galaxies in a general, pipelined manner.

import numpy as np
import math
from . import galread as gr
from scipy import optimize as op
from scipy import signal as ss
from scipy import interpolate
from time import time


def densprof(x,y,z,mass,pt=1,r_max=40000.,z_max=40000.,Nbins=512,rsmooth=None,r_vir=None):
	# Calculate the surface (volume for DM) density profile for a galaxy based on a given particle type and produce a suggested cut-off radius for assessing other properties
	
	###  INPUTS ###
	# x,y & z = Particles' coordinates.  z must be parallel to the angular momentum vector of the galaxy (i.e. normal to a disk).
	# mass = Particles' mass
	
	# pt = particle type, either gas (0), star (1) or DM (2). Only needs to be changed from default for DM.
	# r_max = Disk radius cut to look at particles.  Default 40 kpc (arbitrary).
	# z_max = Height of disk from centre.  Default 40 kpc to ensure for non-disky galaxies.
	# Nbins = Number of radial bins for collecting mass for calculating surface density
	# rsmooth = radius over which to smooth the mass if needed.
	# r_vir = virial radius.  This actually caps the maximum values allowed for the output radii.  Note that this only need to be specified if rsmooth is used.  Otherwise it will default to be = r_max.
	
	### OUTPUTS ###
	# Sigma = surface density
	# r_vec = radius vector corresponding to Sigma
	# Sigma_lim = Sigma up to a limited radius, where Sigma reaches a factor relcut its maximum
	# r_lim = radius vector corresponding to Sigma_lim
	# r_end_rel = spherical radius reasonable for considering the enclosed mass as the total mass for a galaxy (for the prescribed particle type)
	# r_end_abs = end radius based on an absolute cutoff rather than relative
	
	if pt!=0 and pt!=1 and pt!=2:
		pt = 1 # If someone screws up the input, just default to stars
		print('Incorrect particle type index input. Assuming "pt=1" for stars was intended and continuing.\n')
	
	r_max, z_max, Nbins = float(r_max), float(z_max), int(Nbins) # If default variables changed, ensure they are the right type of object.

	if r_vir is None: r_vir = float(r_max)

	if Nbins>len(x)/2: # Pointless to have more bins than particles. Factor of two arbitrary.
		Nbins = len(x)/2 +1  # The +1 is just to avoid the potential to have a value of zero
	
	if pt==1:
		r = np.sqrt(x**2+y**2) # Disk radius coordinates
		filt = (r<r_max)*(z<z_max)*(z>-z_max) # Filter to select at least all of the galaxy
		relcut = 1e-4 # The relative surface density value at which the radius is cut
		abscut = 4e-1 # NEW ADDITION - ABSOLUTE CUT-OFF VALUE IN SOLAR MASSES PER SQUARE PARSEC
	elif pt==0:
		r = np.sqrt(x**2+y**2) # Disk radius coordinates
		filt = (r<r_max)*(z<z_max)*(z>-z_max) # Filter to select at least all of the galaxy
		relcut = 10**(-2.5) # Lower than for stars due to difference in profile.
		abscut = 1 # NEW ADDITION AS ABOVE
	elif pt==2:	# Want a volume density for dark matter, as its distribution is roughly spherical
		r = np.sqrt(x**2+y**2+z**2)
		filt=(r<r_max)
		relcut = 1e-4
	
	
	if rsmooth!=None: # If smoothing for the particles' mass is necessary, do the following
		im, xedges, yedges = np.histogram2d(x[filt], y[filt], bins=2*Nbins, weights=mass[filt], range=[[-r_max,r_max],[-r_max,r_max]]) # Build column density profile
		Lpix = r_max/Nbins # Length of each pixel in the 2d profile
		k = sphere2dk(rsmooth, Lpix, 2*rsmooth/Lpix) # Generate convolution kernel for smoothing
		im = ss.convolve2d(im,k,mode='same') # Perform the smoothing ("im" is the mass binned in 2d)
		
		cn = np.array([np.arange(2*Nbins)]*2*Nbins) # Matrix of the column numbers for im
		rn = cn.transpose() # Matrix of the row numbers for im
		rad = np.sqrt((cn-Nbins+0.5)**2 + (rn-Nbins+0.5)**2)*Lpix # Radius value of each pixel
		
		massbin, xedges = np.histogram(rad, bins=Nbins, weights=im) # Binned mass for 1d radius
	else:
		massbin, xedges = np.histogram(r[filt],bins=Nbins,weights=mass[filt]) # Bin the mass by particle position directly

	r_bin = xedges[1]-xedges[0] # Radial bin width
	r_vec = (np.arange(Nbins)+0.5)*r_bin # Vector of r corresponding to Sigma, centred on each bin
	
	if pt==0 or pt==1:
		Sigma = massbin/(np.pi*( (r_vec+0.5*r_max/Nbins)**2 - (r_vec-0.5*r_max/Nbins)**2 )) # Gives surface density in M_sun/pc^2
	elif pt==2:
		Sigma = massbin/((4./3.)*np.pi*( (r_vec+0.5*r_max/Nbins)**3 - (r_vec-0.5*r_max/Nbins)**3 )) # Gives volume density in M_sun/pc^3
	
	relfilt = (Sigma > np.max(Sigma)*relcut) # Relative-cut filter
	r_lim = r_vec[relfilt] # Radius vector for values where surface (volume for DM) density is above the relative limit.
	Sigma_lim = Sigma[relfilt] # Values of Sigma above the prescribed limit
	r_lim, Sigma_lim = cleancut(r_lim,Sigma_lim,r_bin) # Truncate data if the surface density values rise above the cutoff again
	
	try:
		r_end_rel = np.max(r_lim) + 0.5*r_bin # Radius to consider in pc for enclosing the total mass
		if r_end_rel>r_vir: r_end_rel=r_vir # Don't allow to go above r_vir
	except ValueError:
		r_end_rel = 0 # If there's nothing in the region being checked, then this manually needs to be set to zero.
	

	# Find radius using the absolute cut-off technique
	if pt!=2:
		absfilt = (Sigma > abscut)
		r_lim2, Sigma_lim2 = r_vec[absfilt], Sigma[absfilt] # Vectors under the new limit
		r_lim2, Sigma_lim2 = cleancut(r_lim2,Sigma_lim2,r_bin)
		
		try:
			r_end_abs = np.max(r_lim2) + 0.5*r_bin # Radius to consider in pc for enclosing the total mass
			if r_end_abs>r_vir: r_end_abs = r_vir # Don't allow to go above r_vir
		except ValueError:
			r_end_abs = 0 # If there's nothing in the region being checked, then this manually needs to be set to zero.
	
	return [Sigma, r_vec], [Sigma_lim, r_lim], [r_end_rel, r_end_abs]


def surfbright(x, y, z, L, d_A=1e8, redshift=None, rmax=4e4, zmax=4e4, rsmooth=None, Nbins=512, vega_L=None, band=None, AB=0):
	"""
	Calculate a surface brightness profile for a galaxy and return the radius at which it hits 25 mag/arcsec^2
	x, y, z = particle positions in pc
	L = brightness of the particles in ergs
	d_A = angular-diameter distance in pc (default 100 Mpc arbitrary)
	"""
	
	# Want less bins than particles (factor of 2 arbitrary)
	if Nbins>len(x)/2: Nbins = len(x)/2 +1  
	
	r = np.sqrt(x**2 + y**2)
	filt = (r<=rmax)*(z<=zmax)
	asec_rad = 360*60*60 / (2*np.pi) # arcseconds per radian
	r = asec_rad*np.arctan(r/d_A) # Convert radial distance to arcsec
	
	Lpix = rmax/Nbins # Length of each pixel
	#print 'rsmooth/Lpix', rsmooth/Lpix
	if rsmooth>Lpix: # No point in running this code if the smoothing kernel is smaller than a pixel
		im, xedges, yedges = np.histogram2d(x[filt], y[filt], bins=2*Nbins, weights=L[filt], range=[[-rmax,rmax],[-rmax,rmax]]) # Build column density profile
		k = sphere2dk(rsmooth, Lpix, 2*rsmooth/Lpix) # Generate convolution kernel for smoothing
		im = ss.convolve2d(im,k,mode='same') # Perform the smoothing ("im" is the luminosity binned in 2d)
		
		cn = np.array([np.arange(2*Nbins)]*2*Nbins) # Matrix of the column numbers for im
		rn = cn.transpose() # Matrix of the row numbers for im
		rad = np.sqrt((cn-Nbins+0.5)**2 + (rn-Nbins+0.5)**2)* asec_rad * np.arctan(Lpix/d_A) # Radius value of each pixel in arcsec
		
		Lbin, redges = np.histogram(rad, bins=Nbins, weights=im) # Binned luminosity for 1d radius
	else:
		Lbin, redges = np.histogram(r[filt],bins=Nbins,weights=L[filt]) # Bin the luminosity by particle position directly
	
	area = (np.pi*(redges[1:]**2 - redges[:-1]**2)) # Area of the binned rings in arcsec^2
	bright = Lbin / area # Surface brightness in erg/s/arcsec^2
	
	if vega_L==None and band!=None:
		[vega_spec, vega_wl, L_bol_vega, vega_L] = gr.readvega(band=band) # Get Vega's magnitude in the requested band
		
	d_L = d_A*(1+redshift)**2 if redshift!=None else d_A # Luminosity distance.  If redshift not specified, assume d_A = d_L
	bright = lum2mag(bright,vega_L) + 2*(10**0.4)*(np.log10(d_L)-1) # Convert to mag/arcsec^2.  First term is absolute magnitude, second is distance modulus
	bright -= AB # This is to alter the magnitudes to a different system, like AB
	rvec = redges[1:] - (redges[1]-redges[0])/2.
	
	rvec_pc = d_A*np.tan(rvec/asec_rad)
	r_25 = rvec_pc[np.argwhere(bright>=25)[0][0]] # Radius where brightness is 25 mag/arcsec^2
	#print 'Brightness at Marie\'s R25 = ', bright[np.argwhere(rvec_pc>=15100)[0][0]], 'mag/arcsec^2'
	
	return r_25, [rvec,rvec_pc,bright]


def ABcorr(band):
	# Get the Vega-AB correction for a given band
	bands = np.array(['h','ic','j','k','rc','u','v','sdss_g','sdss_i','sdss_r','sdss_u','sdss_z'])
	corrs = [1.39,0.45,0.91,1.85,0.21,0.79,0.02,-0.08,0.37,0.16,0.91,0.54]
	return corrs[np.argwhere(bands==band)[0][0]]


def surfbrightalt(x, y, z, L, rmax=4e4, zmax=4e4, Nbins=512):
	"""
	As for surfbright but through an alternative method.
	Pretty sure this doesn't work...
	"""
	
	# Want to clearly less bins than particles (factor of 2 arbitrary)
	if Nbins>len(x)/2: Nbins = len(x)/2 +1
	
	r = np.sqrt(x**2 + y**2)
	filt = (r<=rmax)*(z<=zmax)

	Lbin, redges = np.histogram(r[filt],bins=Nbins,weights=L[filt]) # Bin the luminosity by particle position directly
	area = (np.pi*(redges[1:]**2 - redges[:-1]**2)) # Area of the binned rings in pc^2
	bright = Lbin / (area*3.839e33) # Surface brightness in L_sun/pc^2
	bright = 5.45 + 21.572 - 2.5*np.log10(bright) # Convert to mag/arcsec^2 (assumes g-band)

	rvec = redges[1:] - (redges[1]-redges[0])/2.
	r_25 = rvec[np.argwhere(bright>=25)[0][0]] # Radius where brightness is 25 mag/arcsec^2
	
	return r_25, [rvec,bright]


def cleancut(x_lim,y_lim,x_bin):
	# Remove any excess data from a threshold cut that's there due to the data values rising after dropping below the threshold or vice versa (intended for densprof, but should be generally usable)
	
	### INPUTS ###
	# x_lim = x-axis data after being initially cut
	# y_lim = y-axis data after being initially cut
	# x_bin = separation between bins of x in uncut data
	
	dx = np.diff(x_lim) # Check difference between each value in x_lim.  Want to eliminate any part where the y-data rises again.
	indices = np.argwhere(dx>1.01*x_bin) # Finding the places where y_lim rises again
	
	if np.shape(indices)[0] != 0: # Without this "if" statement, when there's no cut-off to be made, errors occur.
		cutoff = np.min(indices) # Getting the cut-off point
		if cutoff!=0: # Evidently need this too
			x_lim = x_lim[:cutoff] # Removing the places where it rises again
			y_lim = y_lim[:cutoff]
	
	return x_lim, y_lim





def totmass(x,y,z,mass,r_end):
	# Calculate total mass of particles within some defined radius a' la densprof.  Equally usable for summing any property, eg. luminosity.
	r = np.sqrt(x**2+y**2+z**2)
	mass_tot = sum(mass[r<=r_end]) # Total mass
	return mass_tot



def totstellarmassalt(x,y,z,mass,r_max=40000.,Nbins=512):
	# Alternative method for finding the mass by first creating a surface density profile from a collapsed SPHERE instead of a cylinder.
	r_max, Nbins = float(r_max), int(Nbins) # If default variables changed, ensure they are the right type of object.
	
	rd = np.sqrt(x**2+y**2) # Disk radius coordinates
	rs = np.sqrt(x**2+y**2+z**2) # Sphere radius coordinates
	
	filt = (rs<r_max) # Filter to select at least all of the galaxy
	
	Sigma,xedges = np.histogram(rd[filt],bins=Nbins,weights=mass[filt]) # Sigma is surface density (volume density for DM)
	r_vec = (np.arange(Nbins) + 0.5)*r_max/Nbins # Vector of r corresponding to Sigma, centred on each bin
	
	Sigma = Sigma/(np.pi*((r_vec+r_max/Nbins)**2-r_vec**2)) # Gives surface density in M_sun/pc^2
	
	r_lim = r_vec[Sigma > np.max(Sigma)*1e-4] # Radius vector for values where surface density is above the relative limit.
	dr = np.diff(r_lim) # Check difference between each value in r_lim.  Want to eliminate any part where the surface density rises again.
	Sigma_lim = Sigma[Sigma > np.max(Sigma)*1e-4] # Values of Sigma above the prescribed limit
	for i in range(len(dr)):
		if dr[i] > 1.01*r_max/Nbins: # Check the difference.  Factor of 1.01 for safety given we're dealing with floats, and the minimum difference greater than a factor of 1 is 2.
			r_lim = r_lim[:i] # Truncate radius vector where necessary
			Sigma_lim = Sigma_lim[:i] # Keep r_lim and Sigma_lim corresponding
			break
	
	r_end = np.max(r_lim) + 0.5*r_max/Nbins # Radius to consider in pc
	tsm = sum(mass[rs<=r_end]) # Total stellar mass
	
	return tsm, r_lim, Sigma_lim, r_end





def diskfit(r,Sigma):
	# Fit a standard analytical galaxy surface density profile to a galaxy's disk.
	
	###  INPUTS ###
	# r = vector of radius values (in order)
	# Sigma = vector of surface density values corresponding to r
	
	### OUTPUTS ###
	# Sigma_fitted = vector for the best fit analytical function of the surface density
	
	### ------- ###
	
	filt = Sigma < np.max(Sigma)*10**(-1.5)
	
	Sigma_disk = Sigma[filt] # Surface density values that should be from disk-dominated regions
	r_disk = r[filt] # Corresponding radius vector
	
	#r_disk, Sigma_disk = cleancut(r_disk, Sigma_disk, np.max(r)/len(r)) # Cut away data that's retarded, assuming a constant separation between the binning of r.
	try:
		pguess = [np.max(Sigma_disk)+2.7182*np.min(r_disk)/3000, 1/3000.] # Make appropriate initial guesses for parameters to fit.
		pfit, pcov = op.curve_fit(expdecaylog, r_disk, np.log10(Sigma_disk), p0=pguess) # Extract best parameters and covariance matrix for a FIT IN LOG SPACE
		Sigma_fitted = expdecay(r,pfit[0],pfit[1])# Create the fitted curve
	except RuntimeError or ValueError: # Catch any lack of fitting
		Sigma_fitted, pfit = None, [0,0]
		print('diskfit did not converge')
	
	return Sigma_fitted, pfit





def bulgefit(r,Sigma,Sigma_disk=None):
	# Fit a Sersic profile to a surface denisty profile.  Same inputs as diskfit.  Can also input information from diskfit itself in attempt to find a better fit.
	
	if Sigma_disk is not None:
		Sigma = Sigma - Sigma_disk
	
	filt = r<5000.
	Sigma_b, r_b = Sigma[filt], r[filt]
	
	try:
		pguess = [np.max(Sigma_b), 1/50., 1.5]
		pfit, pcov = op.curve_fit(Sersiclog, r_b, np.log10(Sigma_b), p0=pguess)
		Sigma_fitted = Sersic(r,pfit[0],pfit[1],pfit[2])
	except RuntimeError or ValueError: # Catch any lack of fitting
		Sigma_fitted, pfit = None, [0,0,0]
		print('bulgefit did not converge')
	
	return Sigma_fitted, pfit





def galfit(r,Sigma):
	"""
	Fit a standard analytical galaxy surface density profile for the disk + bulge while eliminating the contribution of satellites.  All the outputs will be "None" if the fit does not converge
	Inputs and outputs identical to "diskfit" but without "r_disk" obviously, with the addition of also outputting the r and Sigma vectors to which the final fit was actually made (where the spiked regions have been removed)
	"""
	r2fit, Sigma2fit = np.array(r), np.array(Sigma) # Names for the data to which the fit will occur.  Spikes in surface density will attempt to be removed from the fit.
	Sigma2fit_new = np.zeros(2) # Give this some initial definition to begin the "while" loop
	
	Sigma_disk, p_disk = diskfit(r2fit,Sigma2fit) # Do diskfit to get the best parameter guesses
	Sigma_bulge, p_bulge = bulgefit(r2fit,Sigma2fit,Sigma_disk) # Do bulgefit to get the best parameter guesses
	
	while len(Sigma2fit_new)!=len(Sigma2fit):  # Once there's no high chi squared points fitted, the loop will break, ensuring satellites are removed from the fit.
		print('\n(Re)fitting surface density profile\n')
		
		if len(Sigma2fit_new)>2: # On the first run, this needs to not happen, otherwise keep refitting with new arrays.
			r2fit, Sigma2fit = np.array(r2fit_new), np.array(Sigma2fit_new)
		
		try:
			pguess = [p_disk[0], p_disk[1], p_bulge[0], p_bulge[1], p_bulge[2]]
			pfit, pcov = op.curve_fit(galfn, r2fit, np.log10(Sigma2fit), p0=pguess) # Extract best parameters and covariance matrix
			Sigma2fit_fitted = 10**(galfn(r2fit,pfit[0],pfit[1],pfit[2],pfit[3],pfit[4])) # Create the fitted curve at each radius-value in r2fit
		except RuntimeError or ValueError: # If the fit didn't work, capture the error and just let all the outputs be None or zeroes.
			pfit = [0,0,0,0,0]
			r2fit_new, Sigma2fit_new = None, None
			print('galfit did not converge\n')
			break
		
		chi = np.log10(Sigma2fit)-np.log10(Sigma2fit_fitted) # Find the chi vector (rather than chisqr just because it's easier to read on a graph)
		condit = (Sigma2fit<Sigma2fit_fitted)+(chi<0.15)*(Sigma2fit>Sigma2fit_fitted)+(r2fit<=5000.) # Condition for which points SHOULD be fitted to.  chi2>0.15 is arbitrary but chosen from looking at plots.  Intentionally only remove points ABOVE the fit and after a radius of 5 kpc, as the removal is permanent.
		r2fit_new, Sigma2fit_new = r2fit[condit], Sigma2fit[condit]

	if r2fit_new is not None:
		Sigma_fitted = 10**(galfn(r,pfit[0],pfit[1],pfit[2],pfit[3],pfit[4])) # Create the fitted curve to plot
	else:
		Sigma_fitted = None

	return [Sigma_fitted, pfit], [r2fit_new, Sigma2fit_new]





def SFRest(birth,mass,x,y,z,r,t):
	# Estimate the star formation rate at each time step based on the mass of stars formed in the 50 Myr prior to it
	
	###  INPUTS ###
	# birth = vector of times when stars were born
	# mass = mass of star particles
	# x,y,z = stars' positions
	# r = radius at which stars should be considered part of the galaxy
	# t = time of snapshot in Myr
	
	### OUTPUT  ###
	# sfr = star formation rate for the snapshot in solar masses per Myr
	
	starfilt = (birth<=t)*(birth>=t-50)*(np.sqrt(x**2+y**2+z**2)<=r)
	sfr = sum(mass[starfilt])/50
	return sfr





def SFRtrack(id_s,t_s,mass,id_g,x_g,y_g,z_g,r_g,t_g):
	# Calculate star formation rate by tracking stellar/gas particles between 2 frames.
	# Designed to work for simulations where stars form between each frame (but not necessarily optimal for Marie's, given I have only every tenth snapshot)
	# Equally can be used for star death rates, where star particles are loaded for the earlier frame and gas in the later one.  If so, reverse the words stars and gas in the input definitions, with the exception of r_g.
	
	###  INPUTS ###
	# id_s; id_g = vector of identifications for the star particles in later frame and gas particles in earlier frame, respectively.
	# mass = mass of the star particles in latter frame
	# x_g,y_g,z_g = positions of gas in earlier frame
	# r_g = galaxy gas radius in the earlier frame
	# t_s; t_g = time of later frame and earlier frame, respectively
	
	### OUTPUT  ###
	# sfr = star formation rate for the snapshot in solar masses per Myr
	
	gasfilt = (x_g**2+y_g**2+z_g**2<=r_g**2) # Find all gas particles within the galaxy in the previous time step
	id_ga = id_g[gasfilt] # Get the IDs for those gas particles
	
	mass_match = mass[np.in1d(id_s,id_ga)] # Obtain the masses of star particles that were gas particles in the galaxy previously
	
	sfr = divide(sum(mass_match), abs(t_s-t_g))
	return sfr



def GTRest(x,y,z,vx,vy,vz,mass,rgal,dt):
	"""
	A quick and dirty calculation of gas ejection and accretion rates.
	Shifts particles using their current velocities over time dt to see if they move out of the galaxy
		
	x, y, z = coordinates for gas particles (pc)
	vx, vy, vz = velocity components of gas particles (km/s)
	mass = mass of gas particles (solar masses)
	rgal = radius of the galaxy (pc; boundary that determines if stuff has been accreted recently or will be ejected)
	dt = time step to do the calculation (Gyr)
	"""
	
	# Convert velocities to pc/Gyr
	fac = (3.15576e16) / (3.0857e13) # (s/Gyr) / (km/pc)
	vx, vy, vz = vx*fac, vy*fac, vz*fac
	
	r = np.sqrt(x**2 + y**2 + z**2)
	
	xf, yf, zf = x+vx*dt, y+vy*dt, z+vz*dt
	rf = np.sqrt(xf**2 + yf**2 + zf**2)
	filt1 = (r<=rgal)*(rf>rgal)
	GER = 1e-9*np.sum(mass[filt1])/dt # Gas ejection rate (solar masses per year)

	xp, yp, zp = x-vx*dt, y-vy*dt, z-vz*dt
	rp = np.sqrt(xp**2 + yp**2 + zp**2)
	filt2 = (r<=rgal)*(rp>rgal)
	GAR = 1e-9*np.sum(mass[filt2])/dt # Gas accretion rate (solar masses per year)

	#print 'If everything goes...', 1e-9*np.sum(mass[r<=rgal])/dt
	
	return GAR, GER



def massgain(id1,x1,y1,z1,r1,t1,mass,id2,x2,y2,z2,r2,t2):
	# Identify what particles within the given radius were not within the galaxy's radius at a previous time, then give the rate of mass increase.
	# Designed to have, e.g. ONLY stars OR gas in the later frame, but BOTH in the earlier frame, in case formation/death occurs outside.
	# Can be used for mass loss too by making 1 indicate the later frame and 2 the earlier frame.
	
	###  INPUTS ###
	# id1; id2 = vectors of identification for the earlier and later frame
	# x1,y1,z1; x2,y2,z2 = positions of particles at the two frames
	# r1; r2 = radii of the galaxy at the two frames
	# t1; t2 = time of the two frames
	# mass = mass of the particles in the later frame
	
	r2filt = x2**2+y2**2+z2**2<=r2**2
	id2_red = id2[r2filt] # IDs of particles in the later frame's radius
	mass_red = mass[r2filt]
	
	id1_red = id1[x1**2+y1**2+z1**2<=r1**2] # IDs of particles inside the earlier frame's radius
	
	massrate = divide(sum(mass_red[True - np.in1d(id2_red,id1_red)]), abs(t2-t1)) # Rate of mass change
	
	return massrate





def recentre(x,y,z,vx,vy,vz,mass,r=0):
	# Recentre particles to have the COM at 0,0 based on mass within a prescribed radius and find frame where galaxy has no net velocity
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(z))
    assert np.all(np.isfinite(vx))
    assert np.all(np.isfinite(vy))
    assert np.all(np.isfinite(vz))
    assert np.all(np.isfinite(mass))
    if r==0:
        filt = np.array([True]*len(x))
    else:
        filt = ((x**2+y**2+z**2)<r**2)

    if len(filt[filt])==0:
        print('gc.recentre() found no particles within r =', r)
        return x, y, z, vx, vy, vz

    xcom, ycom, zcom = com(x[filt], y[filt], z[filt], mass[filt])
    vxcom, vycom, vzcom = com(vx[filt], vy[filt], vz[filt], mass[filt])

    xc = x-xcom
    yc = y-ycom
    zc = z-zcom
    vxc = vx-vxcom
    vyc = vy-vycom
    vzc = vz-vzcom

    if str(max(xc))=='nan': # Can get NaNs and not sure how else to stop this.  Just return nothing new if it occurs.
        print('Ignored results of recentre for r=',r, 'as NaNs returned')
        return x, y, z, vx, vy, vz
    else:
        return xc, yc, zc, vxc, vyc, vzc

def recentre2(pos, vel, mass, r=0):
    x, y, z, vx, vy, vz = recentre(pos[:,0], pos[:,1], pos[:,2], vel[:,0], vel[:,1], vel[:,2], mass, r)
    return x, y, z, vx, vy, vz

def com(x,y,z,mass):
    # Find centre of mass
    M = sum(mass)
    mx, my, mz = x*mass, y*mass, z*mass
    xcom, ycom, zcom = divide(sum(mx),M), divide(sum(my),M), divide(sum(mz),M)
    return xcom, ycom, zcom


def com2(pos,mass):
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    xcom, ycom, zcom = com(x,y,z,mass)
    return np.array([xcom, ycom, zcom])


def recentregal(x,y,z,vx,vy,vz,mass):
	# Run recentre multiple times to ensure the centre of coords is in fact the galaxy's COM
	x,y,z,vx,vy,vz = recentre(x,y,z,vx,vy,vz,mass,0.)
	x,y,z,vx,vy,vz = recentre(x,y,z,vx,vy,vz,mass,200000.)
	x,y,z,vx,vy,vz = recentre(x,y,z,vx,vy,vz,mass,100000.)
	x,y,z,vx,vy,vz = recentre(x,y,z,vx,vy,vz,mass,50000.)
	x,y,z,vx,vy,vz = recentre(x,y,z,vx,vy,vz,mass,30000.)
	x,y,z,vx,vy,vz = recentre(x,y,z,vx,vy,vz,mass,25000.)
	x,y,z,vx,vy,vz = recentre(x,y,z,vx,vy,vz,mass,20000.)
#	x,y,z,vx,vy,vz = recentre(x,y,z,vx,vy,vz,mass,10000.)
#	x,y,z,vx,vy,vz = recentre(x,y,z,vx,vy,vz,mass,10000.)
#	x,y,z,vx,vy,vz = recentre(x,y,z,vx,vy,vz,mass,5000.)
	return x,y,z,vx,vy,vz


def recentregal_coords(pos, mass, apertures=None):
    xcom, ycom, zcom = com(pos[:,0],pos[:,1],pos[:,2],mass)
    pos[:,0] -= xcom
    pos[:,1] -= ycom
    pos[:,2] -= zcom
    
    if apertures==None: apertures = [2e5, 1e5, 5e4, 3e4, 2.5e4, 2e4]
    for aperture in apertures:
        R = np.sqrt(np.sum(pos**2, axis=1))
        pos_in_aperture = pos[R<=aperture]
        if len(pos_in_aperture)<20: break
        xcom, ycom, zcom = com(pos_in_aperture[:,0],pos_in_aperture[:,1],pos_in_aperture[:,2],mass[R<=aperture])
        pos[:,0] -= xcom
        pos[:,1] -= ycom
        pos[:,2] -= zcom

    return pos



def recentregalall(x,y,z,vx,vy,vz,massl):
	"""
		Recentre the coordinates of ALL the particles for a galaxy/halo
		
		==INPUTS==
		x,y,z = Each should be lists where each entry in the list is an array pertaining to those coordinates for a particle type.
		The FIRST entry in each list will be taken to be the set to centre the galaxy on (probably should be stars)
		vx,vy,vz = Same deal but for velocities
		massl = List of masses as above.  Really only needs masses for 1 particle type, but might not know in advance which one.
		"""

	# Making the call that 2 particles of a given type is as useless as zero
	if len(x[0])>2:
		j = 0 # This sets the index for centring particles on the first entry in each list
		mass = massl[0]
	elif len(x[1])>2:
		j = 1 # In case there are no stars in the input, try the next particle type
		mass = massl[1] # also, set the mass array to work in this case, where it's assumed all the particles are the same mass (crude but simple)
	elif len(x[2])>2:
		j = 2 # And one more option for safety
		mass = massl[2]
	else:
		j=-1 # Use all particles at once
	if j!=-1:
		x0, y0, z0, vx0, vy0, vz0 = x[j], y[j], z[j], vx[j], vy[j], vz[j] # Store initial values
		x[j], y[j], z[j], vx[j], vy[j], vz[j] = x0-np.median(x0), y0-np.median(y0), z0-np.median(z0), vx0-np.median(vx0), vy0-np.median(vy0), vz0-np.median(vz0) # Shift by medians first, because evidently (and retardedly, if true) some subfind groups (subgroup 1066 namely) have (star) particle(s) that are ~10 Mpc away from the actual halo
		#mdx, mdy, mdz = x0*mass, y0*mass, z0*mass # Mass-distance vectors
		#x[j], y[j], z[j], vx[j], vy[j], vz[j] = x[j]-divide(np.mean(mdx),np.mean(mass)), y[j]-divide(np.mean(mdy),np.mean(mass)), z[j]-divide(np.mean(mdz),np.mean(mass)), vx[j]-np.mean(vx0), vy[j]-np.mean(vy0), vz[j]-np.mean(vz0)
#		x[j], y[j], z[j], vx[j], vy[j], vz[j] = recentre(x[j], y[j], z[j], vx[j], vy[j], vz[j], mass, 1e6)
#		x[j], y[j], z[j], vx[j], vy[j], vz[j] = recentre(x[j], y[j], z[j], vx[j], vy[j], vz[j], mass, 90000.)
#		x[j], y[j], z[j], vx[j], vy[j], vz[j] = recentre(x[j], y[j], z[j], vx[j], vy[j], vz[j], mass, 80000.)
		x[j], y[j], z[j], vx[j], vy[j], vz[j] = recentregal(x[j],y[j],z[j],vx[j],vy[j],vz[j],mass)
		for i in range(len(x)):
			if len(x[i])>0 and i!=j: # Can't do transformations to empty arrays
				x[i], y[i], z[i], vx[i], vy[i], vz[i] = x[i]+x[j][0]-x0[0], y[i]+y[j][0]-y0[0], z[i]+z[j][0]-z0[0], vx[i]+vx[j][0]-vx0[0], vy[i]+vy[j][0]-vy0[0], vz[i]+vz[j][0]-vz0[0]
		return x,y,z,vx,vy,vz
	else:
		xcat, ycat, zcat = np.array([]), np.array([]), np.array([])
		vxcat, vycat, vzcat, mcat = np.array([]), np.array([]), np.array([]), np.array([])
		li = np.array([],dtype=np.int32)
		for i in range(len(x)):
			xcat = np.append(xcat, x[i])
			ycat = np.append(ycat, y[i])
			zcat = np.append(zcat, z[i])
			vxcat = np.append(vxcat, vx[i])
			vycat = np.append(vycat, vy[i])
			vzcat = np.append(vzcat, vz[i])
			mcat = np.append(mcat, massl[i])
			li = np.append(li, np.ones(len(x[i])))
		xcat, ycat, zcat, vxcat, vycat, vzcat = recentre(xcat, ycat, zcat, vxcat, vycat, vzcat, mcat)
		xl, yl, zl = [], [], []
		vxl, vyl, vzl = [], [], []
		for i in range(len(x)):
			xl += [xcat[li==i]]
			yl += [ycat[li==i]]
			zl += [zcat[li==i]]
			vxl += [vxcat[li==i]]
			vyl += [vycat[li==i]]
			vzl += [vzcat[li==i]]
		return xl,yl,zl,vxl,vyl,vzl







def rotategalall(x,y,z,vx,vy,vz,m,r=25000.,extra=False):
	"""
		Designed similarly to recentregalall.  Should be performed after that to rotate the galaxy to position
		r = radius to filter out particles for considering angular momentum (without this, a single distant particle can screw it up)
		"""
	if len(x[0])>5:
		j = 0 # This sets the index for centring particles on the first entry in each list
	elif len(x[1])>5:
		j = 1 # In case there are no stars in the input, try the next particle type
	elif len(x[2])>5:
		j = 2 # And one more option for safety
	filt = ((x[j]**2 + y[j]**2 + z[j]**2) < r**2)
	try:
		axis, angle = compute_rotation_to_z(x[j][filt],y[j][filt],z[j][filt],vx[j][filt],vy[j][filt],vz[j][filt],m[j][filt]) # Calculate angle to rotate coordinates
		if (max(axis)>=0 or max(axis)<0) and (angle>=0 or angle<0): # Should catch the NaNs!
			for i in range(len(x)):
				if len(x[i])>0: # Can't do transformations to empty arrays
					x[i],y[i],z[i] = rotate(x[i],y[i],z[i],axis,angle)
					vx[i],vy[i],vz[i] = rotate(vx[i],vy[i],vz[i],axis,angle)
	except RuntimeWarning:
		print('Runtime Warning in rotategalall')
	if extra:
		return x,y,z,vx,vy,vz,axis,angle
	else:
		return x,y,z,vx,vy,vz



def recenrotgalall(x,y,z,vx,vy,vz,massl,r=25e3):
	# Do recentregalall followed by rotategalall
	try:
		x,y,z,vx,vy,vz = recentregalall(x,y,z,vx,vy,vz,massl)
	except:
		'Could not perform recentregalall in recenrotgalall due to error.'
	try:
		x,y,z,vx,vy,vz = rotategalall(x,y,z,vx,vy,vz,massl,r)
	except:
		'Could not perform rotategalall in recenrotgalall due to error.'
	return x,y,z,vx,vy,vz


def recenrotgalall2(pos,vel,m,r=25e3):
    x,y,z,vx,vy,vz = [],[],[],[],[],[]
    for i in range(len(m)):
        x += [pos[i][:,0]]
        y += [pos[i][:,1]]
        z += [pos[i][:,2]]
        vx += [vel[i][:,0]]
        vy += [vel[i][:,1]]
        vz += [vel[i][:,2]]
    
    x,y,z,vx,vy,vz = recentregalall(x,y,z,vx,vy,vz,m)
    x,y,z,vx,vy,vz = rotategalall(x,y,z,vx,vy,vz,m,r)
    return x,y,z,vx,vy,vz

def recentregalall2(pos,vel,m):
    x,y,z,vx,vy,vz = [],[],[],[],[],[]
    for i in range(len(m)):
        x += [pos[i][:,0]]
        y += [pos[i][:,1]]
        z += [pos[i][:,2]]
        vx += [vel[i][:,0]]
        vy += [vel[i][:,1]]
        vz += [vel[i][:,2]]
    
    x,y,z,vx,vy,vz = recentregalall(x,y,z,vx,vy,vz,m)
    return x,y,z,vx,vy,vz

def recentregal_coord(pos,vel,m):
    x,y,z,vx,vy,vz = recentregalall2(pos,vel,m)
    idx = 0
    while len(pos[idx])==0: idx += 1
    pos_move = pos[idx][0,:] - np.array([x[idx][0], y[idx][0], z[idx][0]])
    vel_move = vel[idx][0,:] - np.array([vx[idx][0], vy[idx][0], vz[idx][0]])
    return pos_move, vel_move

def rotategalall2(pos,vel,m,r=25e3):
    x,y,z,vx,vy,vz = [],[],[],[],[],[]
    for i in range(len(m)):
        x += [pos[i][:,0]]
        y += [pos[i][:,1]]
        z += [pos[i][:,2]]
        vx += [vel[i][:,0]]
        vy += [vel[i][:,1]]
        vz += [vel[i][:,2]]
    
    x,y,z,vx,vy,vz,axis,angle = rotategalall(x,y,z,vx,vy,vz,m,r,True)
    return x,y,z,vx,vy,vz,axis,angle


def centreonhalo(haloid,star,gas,dm,bh=None,use_baryons=True):
    # Recentre the coordinate system and frame of reference onto a DIFFERENT halo/galaxy.
    # Centres on stars or baryons centre of mass.
    # Each of stars, gas, dm should be lists of arrays in the order [x,y,z,mass,hid,vx,vy,vz]
    # At this stage, bh only requires [x,y,z]
    [x, y, z, mass, hid, vx, vy, vz] = star
    [x_g, y_g, z_g, mass_g, hid_g, vx_g, vy_g, vz_g] = gas
    [x_dm, y_dm, z_dm, mass_dm, hid_dm, vx_dm, vy_dm, vz_dm] = dm
    assert np.all(np.isfinite(x))
    
    if bh is not None:
        x_bh, y_bh, z_bh = bh[0], bh[1], bh[2] # Read Black Holes if they're provided
    else:
        x_bh, y_bh, z_bh = np.array([]), np.array([]), np.array([])

    # Create filter and assign new position and velocity vectors corresponding to all baryonic particles that match the input haloid
    fs, fg, fdm = (hid==haloid), (hid_g==haloid), (hid_dm==haloid)
    xf, yf, zf = np.concatenate((x[fs],x_g[fg])), np.concatenate((y[fs],y_g[fg])), np.concatenate((z[fs],z_g[fg])) # Filtered baryons' positions
    vxf,vyf,vzf = np.concatenate((vx[fs],vx_g[fg])), np.concatenate((vy[fs],vy_g[fg])), np.concatenate((vz[fs],vz_g[fg])) # Filtered baryons' velocities
    mf = np.concatenate((mass[fs],mass_g[fg])) # Filtered baryons' masses
    
    
    if (len(xf)>5 or len(xf)>len(x_dm[fdm])) and use_baryons: # set minimum number of particles, else use DM
    #		print('\nNumber of baryon particles from (sub)halo '+str(haloid)+' = '+str(len(xf)))
    
        if len(x[fs])>5: # use stars if enough for centering
            x0,y0,z0,vx0,vy0,vz0 = x[fs][0], y[fs][0], z[fs][0], vx[fs][0], vy[fs][0], vz[fs][0]
            xc,yc,zc,vxc,vyc,vzc = recentregal(x[fs],y[fs],z[fs],vx[fs],vy[fs],vz[fs],mass[fs])
            delta_x, delta_y, delta_z = xc[0]-x0, yc[0]-y0, zc[0]-z0
            delta_vx, delta_vy, delta_vz = vx[0]-vx0, vy[0]-vy0, vz[0]-vz0
        else:
            xf0,yf0,zf0,vxf0,vyf0,vzf0 = xf[0],yf[0],zf[0],vxf[0],vyf[0],vzf[0] # Store original coords/velocities of 1 particle
            # Why not recentregalall?
            xf,yf,zf,vxf,vyf,vzf = recentregal(xf,yf,zf,vxf,vyf,vzf,mf) # Properly centre on the halo
            delta_x, delta_y, delta_z = xf[0]-xf0, yf[0]-yf0, zf[0]-zf0
            delta_vx, delta_vy, delta_vz = vxf[0]-vxf0, vyf[0]-vyf0, vzf[0]-vzf0
                
        # Translate all the stars
        x += delta_x
        y += delta_y
        z += delta_z
        vx += delta_vx
        vy += delta_vy
        vz += delta_vz
        
        assert np.all(np.isfinite(x))
        
        # Translate all the gas
        x_g += delta_x
        y_g += delta_y
        z_g += delta_z
        vx_g += delta_vx
        vy_g += delta_vy
        vz_g += delta_vz
        
        # Translate all the dark matter
        x_dm += delta_x
        y_dm += delta_y
        z_dm += delta_z
        vx_dm += delta_vx
        vy_dm += delta_vy
        vz_dm += delta_vz
                
        if len(x[fs])>5: # Use stars to determine the disk
            axis, angle = compute_rotation_to_z(x[fs],y[fs],z[fs],vx[fs],vy[fs],vz[fs],mass[fs]) # Calculate angle to rotate coordinates
            rot = True
        elif len(xf)>5: # If not enough particles use all baryons
            axis, angle = compute_rotation_to_z(xf,yf,zf,vxf,vyf,vzf,mf)
            rot = True
        else: # If still not enough, then don't bother with rotating
            print('centreonhalo will not rotate particles due to lack of baryons')
            rot = False
                                            
        halo_coords = -np.array([delta_x, delta_y, delta_z])
        halo_vel = -np.array([delta_vx, delta_vy, delta_vz])
                                            
        if rot:
            x,y,z = rotate(x,y,z,axis,angle) # Rotate positions so z is normal to the disk
            vx,vy,vz = rotate(vx,vy,vz,axis,angle)
            assert np.all(np.isfinite(x))
            
            x_g, y_g, z_g = rotate(x_g,y_g,z_g,axis,angle) # Rotate gas to match coords
            vx_g,vy_g,vz_g = rotate(vx_g,vy_g,vz_g,axis,angle) # Ditto velocities
            
            x_dm, y_dm, z_dm = rotate(x_dm,y_dm,z_dm,axis,angle) # Rotate dark matter to match coords
            vx_dm,vy_dm,vz_dm = rotate(vx_dm,vy_dm,vz_dm,axis,angle) # Ditto velocities

        if bh is not None:
            x_bh, y_bh, z_bh = x_bh+delta_x, y_bh+delta_y, z_bh+delta_z # Recentre DM to match coords
            if rot: x_bh, y_bh, z_bh = rotate(x_bh,y_bh,z_bh,axis,angle) # Rotate to match coords
                
    elif len(x_dm[fdm])>0: # If no baryons to centre on, then just use DM data (won't rotate)
        halo_coords = com(x_dm[fdm], y_dm[fdm], z_dm[fdm], mass_dm[fdm])
        halo_vel = np.array([np.mean(vx_dm[fdm]), np.mean(vy_dm[fdm]), np.mean(vz_dm[fdm])])
    
        x_dm, y_dm, z_dm = x_dm-halo_coords[0], y_dm-halo_coords[1], z_dm-halo_coords[2] # Recentre DM to match coords
        vx_dm,vy_dm,vz_dm = vx_dm-halo_vel[0], vy_dm-halo_vel[1], vz_dm-halo_vel[2]
    
        x, y, z = x-halo_coords[0], y-halo_coords[1], z-halo_coords[2]
        vx, vy, vz = vx-halo_vel[0], vy-halo_vel[1], vz-halo_vel[2]
        assert np.all(np.isfinite(x))

        x_g, y_g, z_g = x_g-halo_coords[0], y_g-halo_coords[1], z_g-halo_coords[2]
        vx_g, vy_g, vz_g = vx_g-halo_vel[0], vy_g-halo_vel[1], vz_g-halo_vel[2]

        if bh is not None:
            x_bh, y_bh, z_bh = x_bh-halo_coords[0], y_bh-halo_coords[1], z_bh-halo_coords[2]
        
    else: # Just in case one gets through that doesn't actually have any particles (happened in trials)
        halo_coords, halo_vel = np.array([0,0,0]), np.array([0,0,0])
        print('centreonhalo failed due to lack of particles in the desired halo')
    
    assert np.all(np.isfinite(x))
    
    if bh is None:
        return [halo_coords,halo_vel],[x,y,z],[vx,vy,vz],[x_g,y_g,z_g],[vx_g,vy_g,vz_g],[x_dm,y_dm,z_dm],[vx_dm,vy_dm,vz_dm] # Returns the original coordinates if the halo particles can't be identified.
    else:
        return [halo_coords,halo_vel],[x,y,z],[vx,vy,vz],[x_g,y_g,z_g],[vx_g,vy_g,vz_g],[x_dm,y_dm,z_dm],[vx_dm,vy_dm,vz_dm],[x_bh,y_bh,z_bh]


def centreonhalo2(haloid, pos, vel, mass, ids, use_baryons=True):
    # Each input (except haloid) is a list, where stars are first, gas second, DM third
    
    star = [pos[0][:,0], pos[0][:,1], pos[0][:,2], mass[0], ids[0], vel[0][:,0], vel[0][:,1], vel[0][:,2]]
    gas = [pos[1][:,0], pos[1][:,1], pos[1][:,2], mass[1], ids[1], vel[1][:,0], vel[1][:,1], vel[1][:,2]]
    dm = [pos[2][:,0], pos[2][:,1], pos[2][:,2], mass[2], ids[2], vel[2][:,0], vel[2][:,1], vel[2][:,2]]
    if len(pos)==4:
        bh = [pos[3][:,0], pos[3][:,1], pos[3][:,2]]
        [halo_coords,halo_vel], posl_s, vell_s, posl_g, vell_g, posl_dm, vell_dm, posl_bh = centreonhalo(haloid,star,gas,dm,bh,use_baryons)
        return [halo_coords,halo_vel], [np.array(posl_s).T, np.array(posl_g).T, np.array(posl_dm).T, np.array(posl_bh).T], [np.array(vell_s).T, np.array(vell_g).T, np.array(vell_dm).T]
    else:
        [halo_coords,halo_vel], posl_s, vell_s, posl_g, vell_g, posl_dm, vell_dm = centreonhalo(haloid,star,gas,dm,None,use_baryons)
        return [halo_coords,halo_vel], [np.array(posl_s).T, np.array(posl_g).T, np.array(posl_dm).T], [np.array(vell_s).T, np.array(vell_g).T, np.array(vell_dm).T]




def halosearch(t,halo_id,halo_mass,halo_time):
	# Identify which halo IDs could potentially still be haloes in the current snapshot. Designed for Marie's sims.
	
	### INPUTS ###
	# t = cosmic time of snapshot
	# halo_id = vector listing halo ID numbers
	# halo_mass = corresponding vector of those haloes' masses
	# halo_time = time the halo entred the simulation (the above 3 are the outputs of gr.halomarie)
	
	### OUTPUT ###
	# pot_halo_ids = list (array) of potentially still exisiting halo IDs
	
	filt = (halo_mass>1e9)*(halo_time<=t)*(halo_id!=0) # Filter out the main galaxy, all ones that haven't entered the box and those that are small
	pot_halo_ids = halo_id[filt]
	return pot_halo_ids



def findsurf(x,y,z,mass,Lboxh=40000.,Nbins=512):
	# Create a filter based on volume density and eliminate all below a threshold.  This effectively contains the galaxy by a shape appropriate specifically to it.
	
	###  INPUTS ###
	# x,y & z = Particles' coordinates.  z must be parallel to the angular momentum vector of the galaxy (i.e. normal to a disk).
	# mass = Particles' mass
	
	# Lboxh = Half the one-dimensional size of 3D box for initial isolation.  Default box size is 80000^3 pc^3
	# Nbins = Number of bins in each dimension.  Default 512.
	
	### OUTPUT  ###
	# rhoreduce = 3D matrix of density values within the bins of the shape considered
	# step = dimensions of histogram3d bins
	
	### ------- ###
	
	filtbox = (x<Lboxh)*(x>-Lboxh)*(y<Lboxh)*(y>-Lboxh)*(z<Lboxh)*(z>-Lboxh)
	massbin, step, edges = histogram3d(x[filtbox], y[filtbox], z[filtbox], bins=Nbins, weights=mass[filtbox])
	rho = massbin/(step[0]*step[1]*step[2])
	cutoff = (rho>np.max(rho)*1e-4).astype(int) # Being consistent with my arbitrary choice of less than a factor of 10^-4 being meaningless
	rhoreduce = rho*cutoff # Turns all the factors less than the cutoff to 0.
	return rhoreduce, step





def histogram3d(x,y,z, bins=10,weights=None,minwidth=None,normed=False, xmin=None,xmax=None,ymin=None,ymax=None,zmin=None,zmax=None):
	# Because numpy only does 1D and 2D with this method.  Currently works for cells of a single size (not the same in each dimension necessarily though).
	# minwidth ensures the width of the bins meets a minimum threshold.  If "bins" is set to be too high, this number will change accordingly
	
	if type(bins)==int:
		bins = np.array([bins, bins, bins])
	elif type(bins)!=int and len(bins)!=3:
		print('ERROR: Bins input must be a single integer or an array of 3')
		return

	if xmin==None: xmin = np.min(x)
	if xmax==None: xmax = np.max(x)
	if ymin==None: ymin = np.min(y)
	if ymax==None: ymax = np.max(y)
	if zmin==None: zmin = np.min(z)
	if zmax==None: zmax = np.max(z) # Obtain max/min values if not user-specified (taking the full possible range)

	xstep, ystep, zstep = (xmax-xmin)/bins[0], (ymax-ymin)/bins[1], (zmax-zmin)/bins[2] # Gives the dimensions of each cuboidal bin

	if minwidth!=None:
		if xstep<minwidth: bins[0] = int((xmax-xmin)/minwidth) # Change number of bins if breaking the threshold width
		if ystep<minwidth: bins[1] = int((ymax-ymin)/minwidth)
		if zstep<minwidth: bins[2] = int((zmax-zmin)/minwidth)
		xstep, ystep, zstep = (xmax-xmin)/bins[0], (ymax-ymin)/bins[1], (zmax-zmin)/bins[2] # Recalculate the bin widths
	#print bins
	bindata = np.zeros((bins[0],bins[1],bins[2])) # Create empty array to fill
	for i in range(bins[2]):
		#if i/10.==int(i/10): print 'loop', i
		zfilt = (z >= zmin + zstep*i)*(z < zmin + zstep*(i+1))
		bindata[:,:,i], xedges, yedges = np.histogram2d(x[zfilt], y[zfilt], bins=[bins[0],bins[1]], weights=weights[zfilt], range=[[xmin,xmax],[ymin,ymax]])
		"""
			if i>0:
			ran0 = int(bins[0]*np.random.random())
			ran1 = int(bins[1]*np.random.random())
			if xedges[ran0]!=xedgesold[ran0] or yedges[ran1]!=yedgesold[ran1]:
			print 'YEP THIS IS TOTALLY FUCKED'
			xedgesold = np.array(xedges)
			yedgesold = np.array(yedges)
			"""
	step = [xstep, ystep, zstep]
	if normed==True: bindata /= np.product(step) # Give a density for the quantity if requested.
	return bindata, step, bins





def diskmass(a,dsr):
	# Calculate mass of the disk component of a galaxy in solar masses.
	
	### INPUTS ###
	# a = central disk surface density in solar masses per pc
	# dsr = disk scale radius in pc
	
	return 2*np.pi*a*dsr**2





def bulgemass(r,a,d,n,steps=512):
	# Calculate mass of the bulge component of a galaxy in solar masses.
	
	### INPUTS ###
	# r = radius vector or single value in pc
	# a = central bulge surface density in solar masses per pc
	# d = exponential power (same in diskfit and Sersic)
	# n = Sersic index
	# steps = number of steps for numerical integration
	
	r_vec = divide(np.arange(steps)*np.max(r), steps-1.) # Radius vector for numerical integration
	Sig_b = r_vec*Sersic(r_vec,a,d,n) # Sigma for the bulge with weighting of r
	dr = np.diff(r_vec)
	bm = np.sum(Sig_b[:-1]*dr + 0.5*np.diff(Sig_b)*dr) # Numerically integrate
	return 2*np.pi*bm # Total bulge mass






def virial_radius(mass,r,rho_crit,r_max=400000.,it_max=400,densfac=200.):
	# Find the radius of a halo with an internal average density 200 times greater than the critical density of the universe.
	
	###  INPUTS ###
	# mass = mass vector of particles (all types or DM) in solar masses
	# x,y,z = positions of particles in parsecs
	# rho_crit = critical density of the Universe in solar masses per cubic parsec
	# it_max = maximum number of iterations to find the virial radius
	# r_max = maximum radius to search within
	# densfac = factor of critical density to find within the radius
	### ------- ###
	#print('\nEntering virrad')
	start = time()

	rho_av = lambda r_sph : np.sum(mass[r<=r_sph])/((4./3.)*np.pi*r_sph**3) # r_sph is the radius of a sphere to find the internal average density
	
	tol = 1e-3 # This is the tolerance, i.e. must find r_200 to within 0.1%
	left = 0. # Left bound for selecting new radius
	right = r_max # Right bound for selecting new radius in pc.  Will try radius in between the bounds
	
	for i in range(it_max):
		r_try = (left+right)/2. # Try this radius
		dif = rho_av(r_try)/(rho_crit*densfac) - 1 # Difference between the average density internal of r_try and the value sought.
		if abs(dif)<=tol:
			break # When difference below the tolerance, we have a virial radius
		
		if dif>0:
			left = r_try
		else:
			right = r_try
	#print('Time on virrad = '+str(time()-start)+' s after '+str(i+1)+' iterations\n')
	if i==it_max-1: print('virrad hit max no. of iterations after', time()-start, 's')
	return r_try


def virrad(mass,x,y,z,rho_crit,r_max=400000.,it_max=400,densfac=200.):
    r = np.sqrt(x**2+y**2+z**2) # Actual radii of the particles
    return virial_radius(mass,r,rho_crit,r_max,it_max,densfac)

def z2tL(z, h=0.6774, Omega_M=0.3089, Omega_Lambda=0.6911, Omega_R=0, nele=100000):
    # Convert redshift z to look-backtime time tL.
    # nele is the number if elements used in the numerical integration

    if z<=0: return 0. # not designed for blueshifts!

    H_0 = 100*h
    Omega_k = 1. - Omega_R - Omega_M - Omega_Lambda # Curvature density as found from the radiation, matter and dark energy energy densities.
    Mpc_km = 3.08567758e19 # Number of km in 1 Mpc
    yr_s = 60*60*24*365.24 # Number of seconds in a year

    # Create the z-array that is to be integrated over and matching integrand
    zprime = np.linspace(0, z, nele)
    integrand = 1./((1+zprime)*np.sqrt(Omega_R*(1+zprime)**4 + Omega_M*(1+zprime)**3 + Omega_k*(1+zprime)**2 + Omega_Lambda))

    # Numerically integrate trapezoidally
    integrated = 0.5 * np.sum(np.diff(zprime)*(integrand[:-1] + integrand[1:]))
    tL = np.divide(integrated*Mpc_km, H_0*yr_s*1e9)

    return tL # Gives look-back time in Gyr


def z2t(z, H_0=67.74, Omega_R=0, Omega_M=0.3089, Omega_Lambda=0.6911, nele=100000):
	# Convert redshift z to cosmic time t.  See ztlookup for other parameter definitions.
	return z2tL(2000, H_0*0.01, Omega_M, Omega_Lambda, Omega_R) - z2tL(z, H_0*0.01, Omega_M, Omega_Lambda, Omega_R, nele)


def z2dA(z, H_0=67.74, Omega_R=0, Omega_M=0.3089, Omega_Lambda=0.6911, nele=100000):
	# Convert redshift to an angular-diameter distance
	c = 299792.458 # Speed of light in km/s
	Omega_k = 1. - Omega_R - Omega_M - Omega_Lambda
	zprime = np.linspace(0,z,nele)
	integrand = 1./np.sqrt(Omega_R*(1+zprime)**4 + Omega_M*(1+zprime)**3 + Omega_k*(1+zprime)**2 + Omega_Lambda)
	intval = 0.5*np.sum(np.diff(zprime)*(integrand[:-1] + integrand[1:]))
	d_A = 1e6*c*intval / (H_0*(1+z)) # Angular-diameter distance in pc
	return d_A


def ztlookup(zmin=0, zmax=8, H_0=67.74, Omega_R=0, Omega_M=0.3089, Omega_Lambda=0.6911, nele=100000):
	# Produce a look-up table for converting cosmic time to redshift in spaces of 0.001 in redshift

    ###  INPUTS ###
    # zmax = Maximum redshift to convert to time
    # Omega_R = Density parameter for radiation at z=0
    # Omega_M = Density parameter for mass at z=0
    # Omega_Lambda = Denisty parameter for dark energy at z=0
    # H_0 = Hubble parameter at z=0

    ### OUTPUT  ###
    # tarr = Array of time values corresponding to increments in z of 0.001 from 0 to zmax

    if type(zmax) != 'int':
        zmax = int(zmax+1) # Round zmax up to the nearest integer

    tarr = np.zeros((zmax-zmin)*1000) # Produce empty array for time
    zarr = np.arange(zmin,zmax,0.001)
    for i in range(len(zarr)):
        tarr[i] = z2tL(zarr[i], H_0*0.01, Omega_M, Omega_Lambda, Omega_R, nele)
    tarr = z2tL(2000, H_0*0.01, Omega_M, Omega_Lambda, Omega_R) - tarr
    return tarr, zarr





def t2z(t,tarr,zarr='None'):
	# Convert time, t, (array of single value) in Gyr after Big Bang to redshift, z, by using a look-up table, tarr, produced from ztlookup with default parameters.
	if type(zarr)==str:
		zarr = np.arange(len(tarr))/1000.
	
	tarr = tarr[::-1]
	zarr = zarr[::-1] # Reverse the arrays so tarr is increasing.
	
	z = np.interp(t,tarr,zarr)
	return z


def Hubble(z,h=0.6774,Omega_R=0,Omega_M=0.3089,Omega_Lambda=0.6911):
    H_0 = 100*h
    Omega_k = 1.0 - Omega_R - Omega_M - Omega_Lambda
    return H_0*np.sqrt(Omega_R*(1+z)**4 + Omega_M*(1+z)**3 + Omega_k + Omega_Lambda)

def critdens(z,H_0=67.3,Omega_R=0,Omega_M=0.315,Omega_Lambda=0.685):
	# Find critical density at redshift z.  Will probably need alteration as well
	G = 6.67384e-11 # Gravitational constant
	Mpc_km = 3.08567758e19 # Number of km in 1 Mpc
	pc = 3.08567758e16 # Number of m in 1 pc
	M_sun = 1.9891e30 # Number of kg in 1 solar mass
	
	Hsqr = (H_0**2)*(Omega_R*(1+z)**4 + Omega_M*(1+z)**3 + Omega_Lambda) # Square of Hubble parameter at redshift z
	#print 'H(z)', np.sqrt(Hsqr)
	
	Hsqr = Hsqr/(Mpc_km**2) # Convert Hubble parameter to seconds
	rho_crit = 3*Hsqr/(8*np.pi*G) # Critical density in kg/m^3
	return (rho_crit*pc**3)/M_sun # Critical density in M_sun/pc^3





def critmatterdens(z,H_0=67.74,Omega_R=0,Omega_M=0.3089,Omega_Lambda=0.6911):
	# Same as critdens except find the critical value specifically for matter (assuming a universe where radiation density is negligible)
	# Now of the opinion this isn't particularly useful
	rho_crit = critdens(z,H_0=H_0,Omega_M=Omega_M,Omega_Lambda=Omega_Lambda)
	rho_crit_0 = critdens(0,H_0=H_0,Omega_M=Omega_M,Omega_Lambda=Omega_Lambda)
	rho_M_crit = rho_crit - Omega_Lambda*rho_crit_0
	return rho_M_crit




def expdecay(x,a,b):
	return a*np.exp(-b*x)

def expdecaylog(x,a,b):
	return np.log10(a*np.exp(-b*x))

def linear(x,a,b):
	return a*x + b

def Sersic(x,c,d,n):
	return c*np.exp(-d*x**(divide(1, n)))

def Sersiclog(x,c,d,n):
	# For fitting a Sersic profile in log space.
	return np.log10(c*np.exp(-d*x**(1./n)) + 1e-15)


def galfn(x,a,b,c,d,n):
	# Combination of exponential decay and a Sersic profile, but to be fitted in log space. Safety included to prevent log of zero.
#	if a>0 and b>0 and c>0 and d>0 and n>0:
	y = np.log10(a*np.exp(-b*x) + c*np.exp(-d*x**(divide(1, n))) + 1e-15)
#	else:
#		y = 1e10
	return y

def baryonmp(x,b,m):
	m = abs(m) # Ensure m is positive
	c = 1./(1. - np.exp(-b)) - m # Ensure it crosses (1,1)
	return (1. - np.exp(-x*b)) * (m*x + c)

def baryonmpderic(x,b,m):
	m = abs(m)
	c = 1./(1. - np.exp(-b)) - m
	return b*np.exp(-b*x)*(m*x+c) + m*(1-np.exp(-b*x))

def Gaussian(x,x0,sigma,A):
	return A*np.exp(-((x-x0)**2)/(2*sigma**2))

def Poisson(x,lam):
	n = len(x) # Number of points
	P = np.zeros(n)
	dx = x[1]-x[0]
	karr = np.array(x/dx, dtype='i4')
	for k in karr:
		P[k] = (lam**k) * (np.exp(-lam)) / (math.factorial(k))
	return P

def PolyLog(s,z,eps=1e-7):
    z = float(z)
    pl = z/2. # Setting randomly to get loop going
    pl_new = z
    k = 2
    while abs(pl_new-pl) >= abs(eps*pl_new):
        pl = pl_new
        try:
            pl_new = pl + (z**k)/(k**s)
        except OverflowError:
            print('Overflow Error arose')
            break
        k += 1
    print(k)
    return pl_new

def piecewise_linear_parabola(x, xbreak, l0, l1, p0, p1):
    y = l0*x + l1
    p2 = l0*xbreak + l1 - p0*xbreak**2 - p1*xbreak
    fbreak = (x>xbreak)
    y[fbreak] = p0*x[fbreak]**2 + p1*x[fbreak] + p2
    return y

def piecewise_parabola_linear(x, xbreak, l0, l1, p0, p1):
    y = l0*x + l1
    p2 = l0*xbreak + l1 - p0*xbreak**2 - p1*xbreak
    fbreak = (x<xbreak)
    y[fbreak] = p0*x[fbreak]**2 + p1*x[fbreak] + p2
    return y



def tophatoptfilt(L,wl):
	# Apply a top-hat filter over the optical range to obtain a mock-observed luminosity from a spectrum.
	
	### INPUTS ###
	# L = 3d array of luminosity per wavelength values.  Expects the second index to correspond to wavelength
	# wl = wavelength vector in Angstroms corresponding to this data
	### ------ ###
	
	dim1, dim2, dim3 = len(L[:,0,0]), len(L[0,:,0]), len(L[0,0,:])
	L_filt = np.zeros((dim1,dim3))
	
	wl_filt = (wl>4000.)*(wl<8000.) # Wavelength range of filter
	
	for i in range(dim1):
		for j in range(dim3):
			L_filt[i,j] = np.sum(L[i,wl_filt,j]) # Might be more efficient to use the histogram2d function
	# THIS ISN'T RIGHT. NEED TO INTEGRATE, NOT JUST SUM, I.E. NEED TO MULTIPLY BE WAVELENGTH SOMEWHERE
	return L_filt



def lumfilt(spec,wl,filt_resp,filt_wl):
	# Apply an input filter to return the luminosity (power) in the filter.
	
	### INPUTS ###
	# spec = spectrum intensity (although actually luminosity per unit wavelength)
	# wl = wavelength vector in Angstroms corresponding to this spectrum
	# filt_resp = vector of the filter's response
	# filt_wl = wavelength vector corresponding to the filter's response.
	### ------ ###
	
	spec_itp = np.interp(filt_wl,wl,spec) # Interpolate the spectrum data points to match with the filter's wavelength vector
	#spec_filtd = spec_itp*filt_resp/np.max(filt_resp) # Multiply the functions, ensuring the response function has its maximum normalised to 1.
	spec_filtd = spec_itp*filt_resp
	dwl = np.diff(filt_wl)
	return np.sum(spec_filtd[:-1]*dwl + 0.5*np.diff(spec_filtd)*dwl) # Numerically integrate to obtain luminosity.



def lumfiltarray(L,wl,filt_resp,filt_wl):
	# Perform lumfilt for spectra in an array, (i.e. the type of array one gets from the stellar population model)
	
	### INPUTS ###
	# L = 3D array of luminosity per wavelength output from gr.starpopmodel
	# wl = wavelength vector corresponding to the second index of L
	# filt_resp = response of the filter
	# filt_wl = wavelength vector corresponding to the filter's response
	
	### OUTPUT ###
	# L_filt = 2D array of power emittied in the filter band
	### ------ ###
	
	dim1, dim2, dim3 = len(L[:,0,0]), len(L[0,:,0]), len(L[0,0,:])
	L_filt = np.zeros((dim1,dim3))
	
	for i in range(dim1):
		for j in range(dim3):
			L_filt[i,j] = lumfilt(L[i,:,j],wl,filt_resp,filt_wl)

	return L_filt




def bollum(L,wl):
	# Turn the spectral luminosity data cube into a 2D bolometric luminosity data array.
	
	dim1, dim2, dim3 = len(L[:,0,0]), len(L[0,:,0]), len(L[0,0,:])
	L_bol = np.zeros((dim1,dim3))
	dwl = np.diff(wl) # Differential of wavelength
	
	for i in range(dim1):
		for j in range(dim3):
			L_bol[i,j] = np.sum(L[i,:-1,j]*dwl + 0.5*np.diff(L[i,:,j])*dwl) # Numerically integrate

	return L_bol


def mass2lum(mass,birth,time,met=None,band=None,starpopmodel=None):
	"""
		Calculate star particles' luminosities based on their birth time.
		Assumes solar metallicity for all particles if not specified
		Birth and time need to be in Gyr
		Giving a band will also return the luminosity in that band, i.e. band='v'
		Band can also be a list of bands
		"""
	if starpopmodel==None:
		Larr, ages, wl, metals = gr.starpopmodel() # Read in stellar population model
	else:
		Larr, ages, wl, metals = starpopmodel[0], starpopmodel[1], starpopmodel[2], starpopmodel[3]
	
	age_sp = time-birth # Age of the stellar particles from the sim in Gyr
	
	if band is not None:
		if type(band)==str: band = [band] # turn into list form for next steps
		L_filt = [] # Initialise
		for i in range(len(band)):
			filt_wl, filt_resp = gr.readfilter(band[i]) # Read in filter
			Larr_filt = lumfiltarray(Larr,wl,filt_resp,filt_wl) # Get the luminosity array for the filter
			if met==None: # If metallicities aren't specified, assume one of the columns
				L_filt += [mass*np.interp(age_sp,ages,Larr_filt[:,2])] # Luminosity of star particles in erg/s for the band (perhaps rather a power than a luminosity).  Arbitrarily choosing solar metallicity.
			else:
				interp_fn = interpolate.RectBivariateSpline(ages,metals,Larr_filt)
				L_filt += [mass*interp_fn.ev(age_sp,met)]
		return L_filt
	else: # CODE NO LONGER RETURNS L_bol BY DEFAULT
		Larr_bol = bollum(Larr,wl) # Get bolometric luminosity array
		if met==None:
			L_bol = mass*np.interp(age_sp,ages,Larr_bol[:,2]) # Bolometric (actual) luminosity
		else:
			interp_fn = interpolate.RectBivariateSpline(ages,metals,Larr_bol) # 2D interpolation over the grid
			L_bol = mass*interp_fn.ev(age_sp,met)
		return L_bol




def lum2mag(L, vega_L=None, band=None):
	# Convert luminosity/power to magnitude via the vega system.  L and vega_L need to be the same units.
	
	if vega_L==None and band!=None:
		[vega_spec, vega_wl, L_bol_vega, vega_L] = gr.readvega(band=band) # Get Vega's magnitude in the requested band
	
	DF = (7.68/10)**2 # Distance Factor.  Vega system based on Vega's apparent magnitude (it lies at a distance of 7.68 pc), but absolute magntiude is for when the object is at 10 pc.
	dMag = -(10.**0.4)*np.log10(divide(L*DF,vega_L))
	return dMag


def lum2ABmag(L, d_L, band=None, respint=None):
	""" Convert luminosity into an AB magntiude
		L = lumonosity of particle (erg/s)
		d_L = luminosity distance (pc)
	"""
	#L_Jy = L * 1e19 # Convert luminosity to units of 1e-26 W
	area = 4 * np.pi * (d_L*3.0857e18)**2 # Area the emission is spread over when observed in cm
	flux = L / area

	# Calculate the absolute flux to normalise to, appropriate for the band
	if respint==None and band!=None:
		wl, resp = gr.readfilter(band)
		dwl = np.diff(wl)
		respint = np.sum(resp[:-1]*dwl + np.diff(resp)*dwl)
	else:
		print('Please specify a band')
	mag = -2.5*np.log10(flux/respint) - 48.6
	return mag
	


def divide(num,den):
	# This is a multi-purpose function, not at all specific to anything.  It takes a 2 arrays of the same size (or 2 numbers, 2 vectors) and returns the first divided by the second (num=numerator, den=denominator).  If the denominiator includes any zeroes, rather than returning an infinity or an error, it will just return a value of zero.
	
	if type(den)==np.ndarray:
		size = den.shape # Number of entries in the input vectors
		ans = np.zeros(size) # Initialize output
		w = np.where((den!=0)*np.isfinite(den))
		if len(size)==1:
			ans[w[0]] = (1.0*num[w[0]])/den[w[0]]
		else:
			ans[w[0],w[1]] = (1.0*num[w[0],w[1]])/den[w[0],w[1]]
	else:
		if den!=0:
			den = float(den) # Ensure at least one of the inputs is a float to return a float
			ans = num/den
		else:
			ans = 0.

	return ans




def cleansample(var, val=0, mode='g'):
	# Input "var" can be a single array of list of arrays
	# val is the value at which to clean the sample from.
	# mode: g = greater than, ge = greater than or equal to, etc...
	
	if type(var)==list:
		
		out = [] # Initialize output
		
		for i in range(len(var)):
			arr = var[i] # Extract array from the list
			if mode=='g':
				arr = arr[arr>val] # Remove all values of zero or less
			elif mode=='ge':
				arr = arr[arr>=val]
			elif mode=='l':
				arr = arr[arr<val]
			elif mode=='le':
				arr = arr[arr<=val]
			
			if len(arr)==0:
				arr = np.array([0]) # Don't let the output be of length zero

			out += [arr] # Put back into a list of the same form it was input as
	else:
		
		if mode=='g':
			out = var[var>val] # Remove all values of zero or less
		elif mode=='ge':
			out = var[var>=val]
		elif mode=='l':
			out = var[var<val]
		elif mode=='le':
			out = var[var<=val]

		if len(out)==0:
			out = np.array([0])

	return out


"""
	def nozeros(v1,v2):
	# Remove all entries in arrays v1 and v2 where v1 is equal to zero.  Similar to cleansample in that regard.
	
	
	
	def pcile(var,pc):
	# Choose a percent (pc) to return the values in variable (var) for the prescribed percentile around the median.
	lowin = len(var)*(0.5-pc) # Low index
	intfrac = lowin - int(lowin) # Interpolation fraction (clearly the index won't be an integer necessarily)
	low = var[int(lowin)]*(1-intfrac) + var[int(lowin)+1]*intfrac
	return low # DON'T KNOW WHAT I WAS DOING WITH THIS ROUTINE!
	"""

def sphere2dk(R,Lbin,Nbin):
    # Make a square 2d kernel of a collapsed sphere.  Very basic attempt.
    """
    R = radius of the sphere
    Lbin = length of the bins (in equivalent units)
    Nbin = number of bins on a side of the square
    """
    Nbin = int(Nbin) # Ensure an integer number of bins
    if Nbin%2==0: Nbin+=1 # make sure it's an odd number of bins
    k = np.zeros((Nbin,Nbin)) # k is the convolution kernel to be output
    ii = np.arange(Nbin)[np.newaxis].T * np.ones(Nbin)
    jj = np.arange(Nbin) * np.ones(Nbin)[np.newaxis].T
    rr = Lbin*np.sqrt((ii - 0.5*(Nbin-1))**2 + (jj - 0.5*(Nbin-1))**2)
    inside = (rr<R)
    k[inside] = np.sqrt(R**2 - rr[inside]**2)

    k /= np.sum(k) # Make it normalised
    return k

def recenbox(x,y,z):
	# Recentre the coordinates of particles in a simulation box, such that the centre is actually at the centre.
	"""
		INPUT
		x,y,z = list of arrays, or a single array, containing the x-, y-, and z-coordinates of particles
		
		OUTPUT
		x,y,z = lists in same format as input, giving the recentred coords
		xmax = the distance of the furthest particle projected onto the axis that gives the highest value.  Informs the box size to use when plotting.
		"""
	n = len(x) # number of arrays within the lists
	
	if type(x)==list:
		xr, yr, zr = np.array([]), np.array([]), np.array([])
		for i in range(n-1): # doing n-1 now because BHs can be annoying, so put those last
			xr = np.append(xr,x[i])
			yr = np.append(yr,y[i])
			zr = np.append(zr,z[i])
	else:
		xr, yr, zr = np.array(x), np.array(y), np.array(z)

	xc, yc, zc = (np.max(xr)+np.min(xr))/2., (np.max(yr)+np.min(yr))/2., (np.max(zr)+np.min(zr))/2. # Centres in each coord
	xd, yd, zd = (np.max(xr)-np.min(xr))/2., (np.max(yr)-np.min(yr))/2., (np.max(zr)-np.min(zr))/2. # The half-range over each dimension
	xmax = np.max([xd,yd,zd])

	if type(x)==list:
		#xout, yout, zout = [np.array([])]*len(x), [np.array([])]*len(y), [np.array([])]*len(z) # Initialise
		for i in range(n):
			x[i] -= xc # Subtract off the value at the centre so coords are recentred
			y[i] -= yc
			z[i] -= zc
	else: # If a single array, don't need to loop
		x -= xc
		y -= yc
		z -= zc

	return x, y, z, xmax


def smhm(N=0.0351, M1=11.590, beta=1.376, gamma=0.608):
	# Produce a stellar mass to halo mass relationship from Moster et al (2010,2013)
	logM = np.linspace(8,15,200) # Range of halo masses (in log solar masses)
	M = 10**logM
	M1 = 10**M1 # Was in log units
	mM = 2*N / ( (M/M1)**(-beta) + (M/M1)**gamma ) # stellar-to-halo mass fraction
	return mM, logM

def bhmf(Phistar=7.7e-3,Mstar=6.4e7,alpha=-1.11,beta=0.49):
	# Produce a black hole mass function using the expression from Shankar et al 2004
	logM = np.linspace(6,10,200) # Range of black hole masses in log space
	M = 10**logM
	Phi = Phistar * (M/Mstar)**(alpha+1) * np.exp(-(M/Mstar)**beta)
	return Phi, logM

def bhmf2(phistar,Mstar,alpha,beta,perr,Merr,aerr,berr):
	Ms = [Mstar-Merr, Mstar, Mstar+Merr]
	p = [phistar-perr, phistar, phistar+perr]
	a = [alpha-aerr, alpha, alpha+aerr]
	b = [beta-berr, beta, beta+berr]
	PA = np.zeros((200,3**4))
	i = 0
	for q in range(3):
		for w in range(3):
			for r in range(3):
				for t in range(3):
					PA[:,i], logM = bhmf(p[w],Ms[q],a[r],b[t])
					i += 1
	Phi_min = np.array([min(PA[j,:]) for j in range(200)])
	Phi_max = np.array([max(PA[j,:]) for j in range(200)])
	return Phi_min, Phi_max, logM

def integrate_schechter(phistar, Mstar, alpha, Mlow, Mhigh, Mlog=True, Npoints=10000):
    Phi, logM = schechter(phistar, Mstar, alpha, Mlog, [Mlow, Mhigh], Npoints)
    return np.sum((Phi[1:]+Phi[:-1])/2.*np.diff(logM))



def schechter(phistar, Mstar, alpha, Mlog=False, range=[7,12], Npoints=2000, logM=None):
    # Create a Schechter function (see my PDF on this; it follows eq. 6)
    if Mlog: Mstar = 10**Mstar
    if logM is None: logM = np.linspace(range[0],range[1],Npoints)
    M = 10**logM
    Phi = np.log(10.) * (phistar) * (M/Mstar)**(alpha+1) * np.exp(-M/Mstar)
    return Phi, logM

def schechter2(phistar, Mstar, alpha, perr, Merr, aerr, Mlog=False):
	Ms = [Mstar-Merr[0], Mstar, Mstar+Merr[1]] if type(Merr)==list else [Mstar-Merr, Mstar, Mstar+Merr]
	p = [phistar-perr[0], phistar, phistar+perr[1]] if type(perr)==list else [phistar-perr, phistar, phistar+perr]
	a = [alpha-aerr[0], alpha, alpha+aerr[1]] if type(aerr)==list else [alpha-aerr, alpha, alpha+aerr]
	PA = np.zeros((2000,3**3))
	i=0
	for q in range(3):
		for w in range(3):
			for r in range(3):
				PA[:,i], logM = schechter(p[w],Ms[q],a[r],Mlog)
				i += 1
	Phi_min = np.min(PA, axis=1)
	Phi_max = np.max(PA, axis=1)
	return Phi_min, Phi_max, logM


def doubleschechter(Mstar,phistar1,alpha1,phistar2,alpha2,h=0.7):
	# Create a double Schechter function a la Baldry et al. (2008) (see my PDF, eq. 8)
	logM = np.linspace(7,12,200)
	M = (10**logM) 
	Mstar = 10**Mstar
	phistar1 *= 1e-3
	phistar2 *= 1e-3 # If put in in the same units quoted by Eq. 3 in Baldry+.
	Phi = np.exp(-M/Mstar) * np.log(10.) * (phistar1*(M/Mstar)**(alpha1+1) + phistar2*(M/Mstar)**(alpha2+1))
	Phi = Phi * (h/0.7)**3 # Baldry used h=0.7; need to account for if different
	return Phi, np.log10(M*(0.7/h)**2) 

def doubleschechter2(Mstar,phistar1,alpha1,phistar2,alpha2,Merr,p1err,a1err,p2err,a2err,h=0.7):
	# Do double Schechter with errors and return span to plot
	Ms = [Mstar-Merr, Mstar, Mstar+Merr]
	p1 = [phistar1-p1err, phistar1, phistar1+p1err]
	a1 = [alpha1-a1err, alpha1, alpha1+a1err]
	p2 = [phistar2-p2err, phistar2, phistar2+p2err]
	a2 = [alpha2-a2err, alpha2, alpha2+a2err]
	PA = np.zeros((200,243)) # Initialise Phi Array
	i = 0
	for q in range(3):
		for w in range(3):
			for r in range(3):
				for t in range(3):
					for y in range(3):
						PA[:,i], logM = doubleschechter(Ms[q],p1[w],a1[r],p2[t],a2[y],h=h)
						i += 1
	Phi_min = np.array([min(PA[j,:]) for j in range(200)])
	Phi_max = np.array([max(PA[j,:]) for j in range(200)])
	return Phi_min, Phi_max, logM


def groupparticles(x,y,z,mass,soft, Nbins=100,xmax=None,ymax=None,zmax=None,r2max=None,r3max=None):
	"""
		Group particles by building a 3D grid, projecting positions onto the grid and calculating a density for each cell.
		Each particle will be shifted to one of the neighbouring 6 cells with the highest density and continue to do so until it stops moving.
		Particles will then be grouped by which cell they end up in.
		
		====INPUTS====
		x, y, z = particles' positions
		mass = particles' masses
		Nbins = number of bins for the grouping
		soft = gravitational softening scale of the simulation
		xmax, ymax, zmax, rmax = limits on the maximum values allowed to be considered for either x, y, z, cylindrical radius (x,y), or spherical radius (x,y,z)
		"""
	N = len(x) # Number of particles
	
	groupid = -np.ones(N) # Initialise array to hold the final group IDs
	
	# Filter out any particles as requested
	filt = np.array([True]*N)
	if xmax != None: filt *= (abs(x)<xmax)
	if ymax != None: filt *= (abs(y)<ymax)
	if zmax != None: filt *= (abs(z)<zmax)
	if r2max != None: filt *= (np.sqrt(x**2+y**2)<r2max)
	if r3max != None: filt *= (np.sqrt(x**2+y**2+z**2)<r3max)
	fi = np.argwhere(filt==True) # Filter Indices
	xf, yf, zf, mf = x[filt], y[filt], z[filt], mass[filt]
	
	# Bin the mass.  Widths must be at least the softening scale. Values in binmass are densities if normed=True in histogram3d
	binmass, step, Nbins = histogram3d(xf, yf, zf, bins=Nbins, weights=mf, minwidth=soft, normed=True)
	print(Nbins)
	#print binmass
	print('Step is', step)
	
	# Initial cell IDs.  Says which cell each particle is actually in.
	xid, yid, zid = np.array((xf-np.min(xf))/step[0], dtype='i8'), np.array((yf-np.min(yf))/step[1], dtype='i8'), np.array((zf-np.min(zf))/step[2], dtype='i8')
	xid[np.argwhere(xid==Nbins[0])] = Nbins[0]-1 # Need this to stop the particles at the final boundary having an out-of-bounds index
	yid[np.argwhere(yid==Nbins[1])] = Nbins[1]-1
	zid[np.argwhere(zid==Nbins[2])] = Nbins[2]-1
	
	maxima = [] # Initialise list containing lists of coordinates for the cells that are local maxima
	for xi in range(Nbins[0]):
		for yi in range(Nbins[1]):
			for zi in range(Nbins[2]):
				cellmass = binmass[xi,yi,zi] # Mass of the cell considered
				adj = [] # Initialise object that holds mass of adjacent cells
				adj += [binmass[xi-1,yi,zi]] if xi!=0 else [0] # If at an edge, count the cell that would be there are zero
				adj += [binmass[xi+1,yi,zi]] if xi!=Nbins[0]-1 else [0]
				adj += [binmass[xi,yi-1,zi]] if yi!=0 else [0]
				adj += [binmass[xi,yi+1,zi]] if yi!=Nbins[1]-1 else [0]
				adj += [binmass[xi,yi,zi-1]] if zi!=0 else [0]
				adj += [binmass[xi,yi,zi+1]] if zi!=Nbins[2]-1 else [0]
				if max(adj)<cellmass: maxima += [[xi,yi,zi]] # Obtain all the local maxima
				print(cellmass, adj)

	Nmax = len(maxima) # Number of maxima
	k = 0 # Just an index
	maxid = -np.ones(Nmax)
	# Determine which maxima should be considered the same group (if any)
	for i in range(Nmax):
		for j in range(Nmax):
			if j>i:
				sep = np.sqrt( ((maxima[i][0]-maxima[j][0])*step[0])**2 + ((maxima[i][1]-maxima[j][1])*step[1])**2 + ((maxima[i][2]-maxima[j][2])*step[2])**2 )
				if sep < 3*max(step): # Let thrice the bin width be the maximum separation for maxima to be considered the same group
					if maxid[i]==-1 and maxid[j]==-1:
						maxid[i], maxid[j] = k, k
						k += 1
					elif maxid[i]!=-1 and maxid[j]==-1:
						maxid[j] = maxid[i]
					elif maxid[i]==-1 and maxid[j]!=-1:
						maxid[i] = maxid[j]
		if maxid[i]==-1:
			maxid[i] = k
			k += 1

	Ngroup = k # Total number of groups

	binid = -np.ones((Nbins[0],Nbins[1],Nbins[2])) # Initialise arrays that will give the IDs for particles inside each bin (number indicates which local maximum they end up at)
	for i in range(Nmax): binid[maxima[i]] = maxid[i] # Set the IDs at the maxima in the 3d mass field


	for i in range(Nbins[0]):
		for j in range(Nbins[1]):
			for k in range(Nbins[2]):
				
				xi, yi, zi = i, j, k # Indices for each dimension
				inlist = [] # Initialise index list
				if binid[xi,yi,zi] == -1: print('Next cell to check', [i,j,k],', id', binid[xi,yi,zi])
				
				while binid[xi,yi,zi] == -1: # If a cell has an ID, move on to the next cell
					
					inlist += [[xi,yi,zi]]
					cellmass = binmass[xi,yi,zi]
					adj = [] # Initialise object that holds mass of adjacent cells
					adj += [binmass[xi-1,yi,zi]] if xi!=0 else [0] # If at an edge, count the cell that would be there are zero
					adj += [binmass[xi+1,yi,zi]] if xi!=Nbins[0]-1 else [0]
					adj += [binmass[xi,yi-1,zi]] if yi!=0 else [0]
					adj += [binmass[xi,yi+1,zi]] if yi!=Nbins[1]-1 else [0]
					adj += [binmass[xi,yi,zi-1]] if zi!=0 else [0]
					adj += [binmass[xi,yi,zi+1]] if zi!=Nbins[2]-1 else [0]
					adjarr = np.array(arr) # convert to array
					
					grad = -np.diff(adjarr)[0,2,4] # Finds the gradients in the positive x, y and z directions (based on the neighbouring particles' masses)
					img = np.argwhere(abs(grad)==np.max(abs(grad))) # index for maximum gradient
					sign = int(np.max(grad)/np.max(abs(grad)))
					if img==0: xi += sign
					if img==1: yi += sign
					if img==2: zi += sign
					"""
						next = np.argwhere(adjarr=np.max(adjarr)) # Find which of the adjacent cells is the heaviest
						while len(next)>1: next = np.argwhere(adjarr=np.max(adjarr[adjarr>np.max(adjarr)])) # If 2 cells are the heaviest, then move to the third
						
						if next==0:
						xi -= 1
						elif next==1:
						xi += 1
						elif next==2:
						yi -= 1
						elif next==3:
						yi += 1
						elif next==4:
						zi -= 1
						elif next==5:
						zi += 1
						"""
					idcell = binid[xi,yi,zi] # When this is not -1, the loop will be broken and all cells considered will be given the same ID as this cell
					print('Moving to cell', [xi,yi,zi])
				for m in range(len(inlist)): binid[inlist[m]] = idcell # Add the appropriate ID to each cell in the sequence

	groupid[fi] = binid[xid,yid,zid]
	group_Nparts = np.zeros(Ngroup)
	for i in range(Ngroup): group_Nparts[i] = len(groupid[groupid==i]) # Give the number of particles in each group
	return Ngroup, groupid, group_Nparts, maxima, maxid, step




def partdens(x,y,z,mass,soft, Nbins=128,minpart=5,filt=None,xmax=None,ymax=None,zmax=None,r2max=None,r3max=None,xmin=None,ymin=None,zmin=None,r2min=None,r3min=None,masson=False,printon=False):
	"""
		Find the densities for gas particles.
		Works by building a 3D grid and assigning the average density in each cell as the density for each particle in the cell.
		See groupparticles for info on inputs.
		Masson = True means the masses from each bin will also be output
		minpart = effective minimum number of particles in a cell to trust density
		"""
	start = time()
	#print '\nEntering partdens'
	
	N = len(x) # Number of particles
	
	rho = np.zeros(N) # Initialise density array.  Will hold zeroes for any particle not considered.
	
	# Filter out any particles as requested
	if filt==None: filt = np.array([True]*N)
	if xmax != None: filt *= (x<xmax)
	if ymax != None: filt *= (y<ymax)
	if zmax != None: filt *= (z<zmax)
	if r2max != None: filt *= (np.sqrt(x**2+y**2)<r2max)
	if r3max != None: filt *= (np.sqrt(x**2+y**2+z**2)<r3max)
	if xmin != None: filt *= (x>=xmin)
	if ymin != None: filt *= (y>=ymin)
	if zmin != None: filt *= (z>=zmin)
	if r2min != None: filt *= (np.sqrt(x**2+y**2)>=r2min)
	if r3min != None: filt *= (np.sqrt(x**2+y**2+z**2)>=r3min)
	fi = np.argwhere(filt==True) # Filter Indices
	xf, yf, zf, mf = x[filt], y[filt], z[filt], mass[filt]
	
	# Bin the mass.  Widths must be at least the softening scale. Values in binmass are densities if normed=True in histogram3d
	density, step, Nbins = histogram3d(xf, yf, zf, bins=Nbins, weights=mf, minwidth=soft, normed=True)
	#print 'Cell size used (pc) = ', step
	
	# Identify which particle belongs to which cell
	xid, yid, zid = np.array((xf-np.min(xf))/step[0], dtype='i8'), np.array((yf-np.min(yf))/step[1], dtype='i8'), np.array((zf-np.min(zf))/step[2], dtype='i8')
	xid[np.argwhere(xid==Nbins[0])] = Nbins[0]-1 # Need this to stop the particles at the final boundary having an out-of-bounds index
	yid[np.argwhere(yid==Nbins[1])] = Nbins[1]-1
	zid[np.argwhere(zid==Nbins[2])] = Nbins[2]-1
	
	del xf,yf,zf
	
	rho[fi] = density[xid,yid,zid] # Give the density of each particle as the density of the cell it resides in.
	
	mindens = np.min(mf) / np.product(step) # Minimum density for a cell (if all masses of particles are equal.  Otherwise, technically would need to use np.min instead of np.max, but this is more conservative)
	rho[np.argwhere(rho<minpart*mindens)] = 0 # Set the untrustworthy values to zero
	rhofi = rho[fi]
	
	#print 'Calculated gas particles densities in ', time()-start, 's'
	if printon==True:
		print('Number of untrustworthy particle densities', len(rhofi[rhofi==0]), 'from', len(rhofi))
	#print 'Exiting partdens'
	if masson==False:
		return rho
	else:
		return rho, rho*np.product(step)

def dens2temp(rho):
	"""
		Convert gas densities to temperatures based on the equations in the RAMSES PDF
		Input units M_sun pc^-3
		Outputs: Temp (K), internal energy (J/kg), density (H/cc)
		"""
	M_sun = 1.989e30 # Mass of sun in kg
	M_H = 1.673e-27 # Mass of hydrogen in kg
	pc_cm = 3.08567758e18 # Centimetres in a parsec
	k_B = 1.3806488e-23 # Boltzmann constant (J/K)
	
	rho = rho*M_sun/(M_H*pc_cm**3) # Convert density to units H/cm^3
	#print 'rho', rho[:100]
	
	arg0 = np.argwhere(rho==0)
	filt1 = (rho>0)*(rho<1e-3)
	arg1 = np.argwhere(filt1==True) # Arguments for where gas is assumed diffuse
	filt2 = (rho>=1e-3)*(rho<0.3)
	arg2 = np.argwhere(filt2==True) # Arguments where gas is assumed in the disk and optically thin
	arg3 = np.argwhere(rho>=0.3) # Arguments where gas is assumed in the disk and unstable
	
	T = np.zeros(len(rho)) # Initialise array for Temperature
	T[arg1] = 4e6 * (rho[arg1]*1e3)**((5./3)-1)
	T[arg0] = np.min(T[arg1]) if len(T[arg1])>0 else 4e6 # Can only associate the temperature with an upper limit
	T[arg2] = 1e4
	T[arg3] = 1e4 * (rho[arg3]/0.3)**(-0.5) # Temps are in K, assuming the vast majority of the gas is HI
	
	u = 1.5*k_B*T/M_H # Total internal energy per unit mass (J/kg).  Units equivalent to (m/s)^2.
	
	return T, u, rho


def u2temp(u, gamma=5./3, mu=1.0):
	# Convert gas energy per unit mass to temperature.  Assumes input of J/kg = (m/s)^2
	M_H = 1.673e-27 # Mass of hydrogen in kg
	k_B = 1.3806488e-23 # Boltzmann constant (J/K)
	temp = u * M_H * mu * (gamma-1.) / k_B
	return temp

def temp2u(temp, gamma=5./3, mu=1.0):
    # Convert temperature to gas energy per unit mass.  Output in J/kg = (m/s)^2
    M_H = 1.673e-27 # Mass of hydrogen in kg
    k_B = 1.3806488e-23 # Boltzmann constant (J/K)
    u = temp * k_B / (M_H * mu * (gamma-1.))
    return u


def partdensstep(x,y,z,mass,r_gal=5e4,r_vir=1e5,Nbins=128):
	"""
		Calculate gas particle densities using different resolutions in appropriate places (specifically for Marie's simulations)
		r_gal is the galaxy size, and I believe the best option is the GAS ABSOLUTE radius
		r_vir is the virial radius
		"""
	soft = 150. # Gravitational softening (pc)
	
	z_gal = max(min(r_gal, Nbins*soft/2), 2e4) # Want to make sure the z-direction is binned to the highest possible resolution while also not going too high
	r_gal = max(r_gal, Nbins*soft/2) # May as well.  Also, ensures that if a value of zero for the radius is input, reasonable results will still be returned.
	if r_vir<r_gal: r_vir = r_gal*1.5 # This also catches any funny business with the virial radius
	
	# Central disk
	rho = partdens(x,y,z,mass,soft,Nbins=Nbins, xmin=-r_gal,xmax=r_gal,ymin=-r_gal,ymax=r_gal,zmin=-z_gal,zmax=z_gal)
	filt_cen = (x**2+y**2 <= r_gal**2)*(z>=-z_gal)*(z<z_gal)
	argcen = np.argwhere(filt_cen==False)
	rho[argcen] = 0 # Although I've used cuboidal grid, I want to only use information within a cylindrical cut-out of that, to not favour any horizontal directions
	
	# Remainder of halo
	rhohalo = partdens(x,y,z,mass,soft,Nbins=Nbins, xmin=-r_vir,xmax=r_vir,ymin=-r_vir,ymax=r_vir,zmin=-r_vir,zmax=r_vir)
	filt_halo = (x**2 + y**2 + z**2 <= r_vir**2)
	arghalo = np.argwhere(filt_halo==False)
	rhohalo[arghalo] = 0 # Same as before, but now using a spherical cut-out of the near cubic region to avoid favouring any direction
	args = np.argwhere(rho==0)
	rho[args] = rhohalo[args]
	
	# Remainder of simulation
	rhosim = partdens(x,y,z,mass,soft,Nbins=Nbins)
	argsim = np.argwhere(rho==0)
	rho[argsim] = rhosim[argsim]
	
	return rho


def partdensnn(x,y,z,mass,nn=5,ids=None,id1=None,arg=None):
	"""
		Calculate the density of a particle by considering the nn nearest neighbours
		Either input and array of ids and the desired particle id OR the argument of the particle in the x,y,z,mass arrays
		"""
	
	if ids!=None and id1!=None:
		arg = np.argwhere(ids==id1)
	elif ids==None and id1==None and arg==None:
		print('partdensnn ERROR: Need to specify the particle')

	x, y, z = x-x[arg], y-y[arg], z-z[arg] # Centre coords on the particle
	r = np.sqrt(x**2 + y**2 + z**2) # Distance of each particle from the one concerned
	sargs = np.argsort(r) # Sorted arguments (i.e. rank orders of the particles based on their distance)
	dist = r[np.argwhere(sargs==nn)[0]] # Find distance to the nn`th nearest neighbour
	filt = r<=dist # Filter for all particles that are less than this distance
	rho = sum(mass[filt]) / ((4/3.)*np.pi*dist**3) # Average density of sphere of radius dist around the particle
	return rho


def partdensmarie(x,y,z,mass,r_gal=5e4,r_vir=1e5):
	# Calculate particle densities for Marie's simulations (see partdensstep)
	
	# Measure density for different grid resolutions
	rho1 = partdensstep(x,y,z,mass,r_gal,r_vir,512) # Densities of gas particles in solar masses per cubic parsec
	rho2 = partdensstep(x,y,z,mass,r_gal,r_vir,256)
	rho3 = partdensstep(x,y,z,mass,r_gal,r_vir,128)
	rho4 = partdensstep(x,y,z,mass,r_gal,r_vir,64)
	rho5 = partdensstep(x,y,z,mass,r_gal,r_vir,32)
	rho6 = partdensstep(x,y,z,mass,r_gal,r_vir,16)
	#rho7 = partdensstep(x,y,z,mass,r_gal,r_vir,8)
	
	rho = np.array([rho1,rho2,rho3,rho4,rho5,rho6]).max(axis=0) # Take the maximum value from each resolution
	
	zeroargs = np.argwhere(rho==0)
	print('\nNumber of subthreshold particles =', len(zeroargs))
	preloop = time()
	for arg in zeroargs:
		rho[arg] = partdensnn(x, y, z, mass, 5, arg=arg)
	print('Loop time for them = ', time()-preloop, 's')
	
	return rho


def partdensMBII(x,y,z,mass,h=0.702):
	# Calculate the (gas) particle densities from a MassiveBlack-II subhalo (only input gas data for particles in that halo)
	soft = 1850/h # Softening scale of the simulation
	Npart = len(x) # number of particles
	maxNcells = int(max(np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z))/soft)
	print('maxNcells', maxNcells)
	if maxNcells>256:
		try:
			rho1,mass1 = partdens(x,y,z,mass,soft,512,masson=True,minpart=4)
		except:
			rho1,mass1 = np.zeros(Npart), np.zeros(Npart)
	else:
		rho1,mass1 = np.zeros(Npart), np.zeros(Npart)

	if maxNcells>128:
		try:
			rho2,mass2 = partdens(x,y,z,mass,soft,256,masson=True,minpart=4)
		except:
			rho2,mass2 = np.zeros(Npart), np.zeros(Npart)
	else:
		rho2,mass2 = np.zeros(Npart), np.zeros(Npart)

	if maxNcells>64:
		try:
			rho3,mass3 = partdens(x,y,z,mass,soft,128,masson=True,minpart=4)
		except:
			rho3,mass3 = np.zeros(Npart), np.zeros(Npart)
	else:
		rho3,mass3 = np.zeros(Npart), np.zeros(Npart)

	try:
		rho4,mass4 = partdens(x,y,z,mass,soft,64,masson=True,minpart=4)
	except:
		rho4,mass4 = np.zeros(Npart), np.zeros(Npart)

	try:
		rho5,mass5 = partdens(x,y,z,mass,soft,32,masson=True,minpart=4)
	except:
		rho5,mass5 = np.zeros(Npart), np.zeros(Npart)

	try:
		rho6,mass6 = partdens(x,y,z,mass,soft,16,masson=True,minpart=4,printon=True)
	except:
		rho6,mass6 = np.zeros(Npart), np.zeros(Npart)

	# Want to find the densities associated with the lowest mass bins that were above the threshold.  Those at the threshold were given zero, so make these large instead to find the true minima.
	massarr = np.array([mass1,mass2,mass3,mass4,mass5,mass6])
	zargs = np.argwhere(massarr==0)
	massarr[zargs[:,0],zargs[:,1]] = np.max(massarr)
	minargs = massarr.argmin(axis=0) # Arguments for mass minima for each particle

	#for i in range(6): masslist[i][np.argwhere(masslist[i]==0)] = np.max(masslist[i])
	#print np.array(masslist)

	# Create a mask to know which of the 6 densites to use for each particle
	arg1 = np.argwhere(minargs==0)
	arg2 = np.argwhere(minargs==1)
	arg3 = np.argwhere(minargs==2)
	arg4 = np.argwhere(minargs==3)
	arg5 = np.argwhere(minargs==4)
	arg6 = np.argwhere(minargs==5)
	
	rho = np.zeros(len(x)) # Initialise
	
	# Use densities from the appropriate calculations
	rho[arg1] = rho1[arg1]
	rho[arg2] = rho2[arg2]
	rho[arg3] = rho3[arg3]
	rho[arg4] = rho4[arg4]
	rho[arg5] = rho5[arg5]
	rho[arg6] = rho6[arg6]

	print('Number non-zero rho-values', len(rho1[rho1>0]), len(rho2[rho2>0]), len(rho3[rho3>0]), len(rho4[rho4>0]), len(rho5[rho5>0]), len(rho6[rho6>0]), len(rho[rho>0]))
	return rho


def chisqr(ydata, yfit):
	# Calculate the chi squared value for some fit.  Return the full chi2 and the average chi for a single element.
	dif2 = (ydata-yfit)**2 # difference squared
	chi2 = sum(dif2) # full chi squared
	chiav = np.sqrt(np.mean(dif2)) # Average deviation from the fit.  Basically, this is the rms.
	return chi2, chiav

def reduced_chisqr(y_data, y_expect, uncert, Nparam=0):
    chi2 = np.sum((y_data-y_expect)**2 / uncert**2)
    return chi2 / (len(y_data) - Nparam)


def BaryMP(x,y,eps=0.01,grad=1):
	"""
	Find the radius for a galaxy from the BaryMP method
	x = r/r_200
	y = cumulative baryonic mass profile
	eps = epsilon, if data 
	"""
	dydx = np.diff(y)/np.diff(x)
	
	maxarg = np.argwhere(dydx==np.max(dydx))[0][0] # Find where the gradient peaks
	xind = np.argwhere(dydx[maxarg:]<=grad)[0][0] + maxarg # The index where the gradient reaches 1
	
	x2fit_new, y2fit_new = x[xind:], y[xind:] # Should read as, e.g., "x to fit".
	x2fit, y2fit = np.array([]), np.array([]) # Gets the while-loop going
	
	while len(y2fit)!=len(y2fit_new):
		x2fit, y2fit = np.array(x2fit_new), np.array(y2fit_new)
		p = np.polyfit(x2fit, y2fit, 1)
		yfit = p[0]*x2fit + p[1]
		chi = abs(yfit-y2fit) # Separation in the y-direction for the fit from the data
		chif = (chi<eps) # Filter for what chi-values are acceptable
		x2fit_new, y2fit_new = x2fit[chif], y2fit[chif]
	
	r_bmp = x2fit[0] # Radius from the baryonic-mass-profile technique, returned as a fraction of the virial radius!
	Nfit = len(x2fit) # Number of points on the profile fitted to in the end

	return r_bmp, Nfit




def diststats(x):
	# Return median, mean, and the 68% boundaries (one sigma for a Gaussian) for an array (distribution) of values
	N = len(x)
	x = np.sort(x)
	mean = np.mean(x)
	median = np.median(x)
	low = x[int(N*0.16)]
	high = x[int(N*0.84)]
	return mean, median, low, high


def logten(x):
	if type(x)==np.ndarray:
		logx = np.zeros(len(x))
		logx[x>0] = np.log10(x[x>0])
		logx[x==0] = -np.max(abs(logx)) - 10 # Makes the zero values arbitrarily large (10 orders of magnitude)
	elif x>0:
		logx = np.log10(x)
	else:
		print('logten(0) value set to -100')
		logx = -100
	return logx


def hernquist_zintegral(r_in,z_in,a):
    
    #print r_in
    #print z_in
    #print a
    
    if type(r_in)==np.ndarray:
        ans = np.zeros(len(r_in))
        r = r_in[r_in**2>a**2]
        z = z_in[r_in**2>a**2] if type(z_in)==np.ndarray else z_in
        
        #print 'r_in', r_in
        #print 'arctan', np.arctan(z*(r**2-a**2)**-0.5)
        
        ans[r_in**2>a**2] = (0.5*(r**2-a**2)**-2.5) * ( (2*a**2+r**2)*np.arctan(z*(r**2-a**2)**-0.5) + (r**2+z**2-a**2)**-2 * ( z*(r**2-a**2)**0.5 * (-4*a**4 + 5*a**3*(r**2+z**2)**0.5 + a**2*(3*r**2+2*z**2) - a*(r**2+z**2)**0.5*(5*r**2+3*z**2) +r**2*(r**2+z**2)) - (2*a**2+r**2)*(r**2+z**2-a**2)*np.arctan(a*z*(r**2-a**2)**-0.5*(r**2+z**2)**-0.5) ) )
        r = r_in[r_in**2<a**2]
        z = z_in[r_in**2<a**2] if type(z_in)==np.ndarray else z_in
        ans[r_in**2<a**2] = (0.5*(a**2-r**2)**-2.5) * ( (2*a**2+r**2)*np.arctanh(z*(a**2-r**2)**-0.5) + (r**2+z**2-a**2)**-2 * ( z*(a**2-r**2)**0.5 * (-4*a**4 + 5*a**3*(r**2+z**2)**0.5 + a**2*(3*r**2+2*z**2) - a*(r**2+z**2)**0.5*(5*r**2+3*z**2) +r**2*(r**2+z**2)) - (2*a**2+r**2)*(r**2+z**2-a**2)*np.arctanh(a*z*(a**2-r**2)**-0.5*(r**2+z**2)**-0.5) ) )
        
    elif r_in**2>a**2:
        r = r_in
        z = z_in
        ans = (0.5*(r**2-a**2)**-2.5) * ( (2*a**2+r**2)*np.arctan(z*(r**2-a**2)**-0.5) + (r**2+z**2-a**2)**-2 * ( z*(r**2-a**2)**0.5 * (-4*a**4 + 5*a**3*(r**2+z**2)**0.5 + a**2*(3*r**2+2*z**2) - a*(r**2+z**2)**0.5*(5*r**2+3*z**2) +r**2*(r**2+z**2)) - (2*a**2+r**2)*(r**2+z**2-a**2)*np.arctan(a*z*(r**2-a**2)**-0.5*(r**2+z**2)**-0.5) ) )
        
    elif r_in**2<a**2:
        r = r_in
        z = z_in
        ans = (0.5*(a**2-r**2)**-2.5) * ( (2*a**2+r**2)*np.arctanh(z*(a**2-r**2)**-0.5) + (r**2+z**2-a**2)**-2 * ( z*(a**2-r**2)**0.5 * (-4*a**4 + 5*a**3*(r**2+z**2)**0.5 + a**2*(3*r**2+2*z**2) - a*(r**2+z**2)**0.5*(5*r**2+3*z**2) +r**2*(r**2+z**2)) - (2*a**2+r**2)*(r**2+z**2-a**2)*np.arctanh(a*z*(a**2-r**2)**-0.5*(r**2+z**2)**-0.5) ) )
    else:
        ans = 0.0
    
    #print 'ans', ans, '\n'
    return ans


def hernquist_2Dannulus_mass(r_2D_bins, a, r_trunc, mass_trunc, N=1e6):
    pmass = mass_trunc / (1.0*N)
    mterm = np.sqrt(np.random.rand(N)) * r_trunc / (r_trunc + a)
    r = a * mterm / (1 - mterm)
    phi = 2.0*np.pi*np.random.rand(N)
    theta = np.arcsin(2.0*np.random.rand(N)-1.0)
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    r_2D = np.sqrt(x**2 + y**2)
    N_bin, r_2D_bins = np.histogram(r_2D, bins=r_2D_bins)
    return N_bin * pmass



def compute_rotation_to_z(x,y,z,vx,vy,vz,m):
    # Modified version of Marie's function
    Lxtot = sum(m*y*vz-m*z*vy)
    Lytot = sum(m*z*vx-m*x*vz)
    Lztot = sum(m*x*vy-m*y*vx)
    
    axis = np.zeros(3)
    Lnorm = np.sqrt(Lxtot**2+Lytot**2+Lztot**2)
    Lnorm2 = np.sqrt(Lxtot**2+Lytot**2)
    
    if Lnorm2>0:
        axis[0]=Lytot/Lnorm2
        axis[1]=-Lxtot/Lnorm2
    else:
        print('No appropriate rotation axis found in compute_rotation_to_z')
        axis[0] = 0.
        axis[1] = 0.
    axis[2]=0.

    if Lnorm>0:
        angle = np.arccos(Lztot/Lnorm)
    else:
        print('No appropriate angle found in compute_rotation_to_z')
        angle = 0.
    
    return axis, angle

def compute_rotation_to_z2(pos,vel,m,r=0):
    if r>0:
        f = np.sqrt(np.sum(pos**2, axis=1)) <= r
        pos = pos[f]
        vel = vel[f]
        m = m[f]
    return compute_rotation_to_z(pos[:,0], pos[:,1], pos[:,2], vel[:,0], vel[:,1], vel[:,2], m)



def rotate(x,y,z,axis,angle):
    # Copied from Marie
    x_rot=x
    y_rot=y
    z_rot=z
    axisx=axis[0]*np.ones(len(x))
    axisy=axis[1]*np.ones(len(x))
    axisz=axis[2]*np.ones(len(x))
    
    dot=x*axisx+y*axisy+z*axisz
    
    crossx=axisy*z-axisz*y
    crossy=axisz*x-axisx*z
    crossz=axisx*y-axisy*x
    
    cosa=np.cos(angle)
    sina=np.sin(angle)
    
    x_rot=x*cosa+crossx*sina+axis[0]*dot*(1-cosa)
    y_rot=y*cosa+crossy*sina+axis[1]*dot*(1-cosa)
    z_rot=z*cosa+crossz*sina+axis[2]*dot*(1-cosa)
    
    
    return x_rot,y_rot,z_rot

def rotate2(xyz, axis, angle):
    x, y, z = rotate(xyz[:,0], xyz[:,1], xyz[:,2], axis, angle)
    return np.concatenate((np.array([x]).T,np.array([y]).T,np.array([z]).T), axis=1)


def hist_Nmin(x, bins, Nmin, hard_bins=np.array([])):
    Nhist, bins = np.histogram(x, bins)
    ij = 0
    while len(Nhist[Nhist<Nmin])>ij:
        ii = np.where(Nhist<Nmin)[0][ij]
#        print ii==0 or (ii!=len(Nhist)-1 and Nhist[ii+1]<Nhist[ii-1])
#        print np.all(~(bins[ii+1] <= 1.01*hard_bins and bins[ii+1] >= 0.99*hard_bins))
        if (ii==0 or (ii!=len(Nhist)-1 and Nhist[ii+1]<Nhist[ii-1])) and np.all(~((bins[ii+1] <= 1.01*hard_bins) * (bins[ii+1] >= 0.99*hard_bins))):
            bins = np.delete(bins,ii+1)
        elif np.all(~((bins[ii] <= 1.01*hard_bins) * (bins[ii] >= 0.99*hard_bins))):
            bins = np.delete(bins,ii)
        else:
            print('hard_bins prevented gc.hist_Nmin() from enforcing Nmin.  Try using wider input bins.')
            ij += 1
#            Nhist, bins = np.histogram(x, bins)
#            break
        Nhist, bins = np.histogram(x, bins)
    if bins[0]<np.min(x): 
        del_bins = np.where(bins<np.min(x))[0]
        if len(del_bins)>1: 
            bins = np.delete(bins, del_bins[1:])
            Nhist, bins = np.histogram(x, bins)
        bins[0] = 0.99*np.min(x)
    if bins[-1]>np.max(x): 
        del_bins = np.where(bins>np.max(x))[0]
        if len(del_bins)>1: 
            bins = np.delete(bins, del_bins[:-1])
            Nhist, bins = np.histogram(x, bins)
        bins[-1] = 1.01*np.max(x)

    return Nhist, bins

def percentiles(x, y, low=0.16, med=0.5, high=0.84, bins=20, addMean=False, xrange=None, yrange=None, Nmin=10, weights=None, hard_bins=np.array([]), outBins=False, bootstrap=False, logout=False):
    """
    Calculate running percentiles for a set of two-dimensional data
    
    Input:
    x = array of values for the dependent variable. This dimension defines the bins
    y = array of values for the independent variable.  This dimension defines the percentiles within each bin
    low, mid, high = lowest, middle, and highest of three percentiles to output (each expressed as a decimal, rather than an actual percentage -- i.e. 0.16 returns the 16th percentile, 0.5 is the median)
    bins = either specify a number of bins (int) or feed in an array of floats for pre-defined bins.  If an int, will start with equal-sized bins across the full range of x-values.  Note that these bins are not strict -- they can be adjusted within the code (see below)
    addMean = if True, adds an output for the running mean
    xrange, yrange = list of 2 values each, used to cut any data whose x or y values respectively fall outside the specified range
    Nmin = int, minimum number of data needed in each bin.  Where bins do not have data in them, their widths are expanded to meet this minimum
    weights = array of float of len(x); specifies weights for each datum when calculating percentiles
    hard_bins = array of floats that also exist in the array fed into bins.  These bin edges will not be changed in the function.  This overwrites the minimum number of data requirement for the adjacent bins
    outBins = set to True if you want to output what the final bins in x were after adapting to meet minimum requirements
    bootstrap = set to True if you want to output bootstrapping errors on the percentiles of interest.  Returns the 16th, 50th and 84th percentile OF EACH PERCENTILE OF THE DATA after resampling
    logout = set to True if you want the outputs to be in log-space
    """
    # Given some values to go on x and y axes, bin them along x and return the percentile ranges
    f = np.isfinite(x)*np.isfinite(y)
    if xrange is not None: f = (x>=xrange[0])*(x<=xrange[1])*f
    if yrange is not None: f = (y>=yrange[0])*(y<=yrange[1])*f
    x, y = x[f], y[f]
    if type(bins)==int:
        if len(x)/bins < Nmin: bins = len(x)/Nmin
        indices = np.array(np.linspace(0,len(x)-1,bins+1), dtype=int)
        bins = np.sort(x)[indices]
    elif Nmin>0: # Ensure a minimum number of data in each bin
        Nhist, bins = hist_Nmin(x, bins, Nmin, hard_bins)
    Nbins = len(bins)-1
    y_low, y_med, y_high = np.zeros(Nbins), np.zeros(Nbins), np.zeros(Nbins)
    if bootstrap: y_low_wci, y_med_wci, y_high_wci = np.zeros((Nbins,3)), np.zeros((Nbins,3)), np.zeros((Nbins,3)) # wci means 'with confidence interval'
    x_av, N = np.zeros(Nbins), np.zeros(Nbins)
    if addMean: y_mean = np.zeros(Nbins)
    for i in range(Nbins):
        f = (x>=bins[i])*(x<bins[i+1]) if i<Nbins-1 else (x>=bins[i])*(x<=bins[i+1])
        if len(f[f])>2:
            if weights is None:
                [y_low[i], y_med[i], y_high[i]] = np.percentile(y[f], [100*low, 100*med, 100*high], interpolation='linear')
            else:
                [y_low[i], y_med[i], y_high[i]] = weighted_percentile(y[f], [low, med, high], weights[f])
            x_av[i] = np.mean(x[f])
            N[i] = len(x[f])
            if addMean: y_mean[i] = np.mean(y[f]) if weights is None else np.sum(y[f]*weights[f])/np.sum(weights[f])
            if bootstrap:
                pcile_pciles = bootstrap_percentiles(y[f])
                y_low_wci[i,[0,2]] = pcile_pciles[:,0]
                y_med_wci[i,[0,2]] = pcile_pciles[:,1]
                y_high_wci[i,[0,2]] = pcile_pciles[:,2]
    fN = (N>0) if Nmin>0 else np.array([True]*Nbins)
    if len(fN[~fN])>0:
        print('\npercentiles: fN =', fN)
        print('percentiles: bins =', bins)
        print('percentiles: N =', N)
        print('percentiles: max(x) =', np.max(x))

    if bootstrap:
        y_low_wci[:,1] = 1.0*y_low
        y_med_wci[:,1] = 1.0*y_med
        y_high_wci[:,1] = 1.0*y_high
        
        if weights is not None: # bootstrapped percentiles don't properly account for possibility of weights.  This is the simplest way to deal with it -- just extend the confidence intervals to encompass the 'true' weighted percentile if need be
            y_low_wci[:,0] = np.min(y_low_wci[:,:2], axis=1)
            y_med_wci[:,0] = np.min(y_med_wci[:,:2], axis=1)
            y_high_wci[:,0] = np.min(y_high_wci[:,:2], axis=1)
            
            y_low_wci[:,2] = np.max(y_low_wci[:,1:], axis=1)
            y_med_wci[:,2] = np.max(y_med_wci[:,1:], axis=1)
            y_high_wci[:,2] = np.max(y_high_wci[:,1:], axis=1)

                
        y_low = 1.0*y_low_wci # replace the original array to include the bootstrapped confidence intervals
        y_med = 1.0*y_med_wci
        y_high = 1.0*y_high_wci
    
    if logout:
        x_av, y_high, y_med, y_low, bins = np.log10(x_av), np.log10(y_high), np.log10(y_med), np.log10(y_low), np.log10(bins)
        try:
            floor = np.min(y_low[np.isfinite(y_low)]) - 10
        except ValueError:
            floor = np.min(y_med[np.isfinite(y_med)]) - 10
        y_low[~np.isfinite(y_low)] = floor
        y_med[~np.isfinite(y_med)] = floor
        y_high[~np.isfinite(y_high)] = floor
        if addMean: 
            y_mean = np.log10(y_mean)
            y_mean[~np.isfinite(y_mean)] = floor

    if not addMean and not outBins:
        return x_av[fN], y_high[fN], y_med[fN], y_low[fN]
    elif not addMean and outBins:
        return x_av[fN], y_high[fN], y_med[fN], y_low[fN], bins
    elif addMean and  not outBins:
        return x_av[fN], y_high[fN], y_med[fN], y_low[fN], y_mean[fN]
    else:
        return x_av[fN], y_high[fN], y_med[fN], y_low[fN], y_mean[fN], bins

def weighted_percentile(data, percentile, weights):
    if len(data)==0: return 0
    arg = np.argsort(data)
    data = data[arg]
    weights = weights[arg]
    weights /= np.sum(weights)
    wsum = np.cumsum(weights)
    return np.interp(percentile, wsum, data)


def meantrend(x, y, bins=20, xrange=None, yrange=None):
    f = np.isfinite(x)*np.isfinite(y) 
    if xrange is not None: f = (x>=xrange[0])*(x<=xrange[1])*f
    if yrange is not None: f = (y>=yrange[0])*(y<=yrange[1])*f
    x, y = x[f], y[f]
    if type(bins)==int:
        indices = np.array(np.linspace(0,len(x)-1,bins+1), dtype=int)
        bins = np.sort(x)[indices]
    Nbins = len(bins)-1
    N, bins = np.histogram(x, bins=bins)
    xx, bins = np.histogram(x, bins=bins, weights=x)
    yy, bins = np.histogram(x, bins=bins, weights=y)
    return xx/N, yy/N
        

def percentiles1(data, low=0.16, med=0.5, high=0.84, del0=False):
    # Feed a 1D array and find percentile values
    #data[np.where(np.isfinite(data)==False)] = 0
    data = data[np.isfinite(data)]
    if del0: data = data[data>0]
    data = np.sort(data)
    N = len(data)
    i_low, frac_low = int(N*low)-1, N*low-int(N*low)
    i_med, frac_med = int(N*med)-1, N*med-int(N*med)
    i_high, frac_high = int(N*high)-1, N*high-int(N*high)
    #print i_low, i_med, i_high, N
    val_low = data[i_low]*(1-frac_low) + data[i_low+1]*frac_low if i_low>0 else data[0]
    val_med = data[i_med]*(1-frac_med) + data[i_med+1]*frac_med if i_med>0 else data[0]
    val_high = data[i_high]*(1-frac_high) + data[i_high+1]*frac_high if i_high<N-1 else data[-1]
    return val_low, val_med, val_high

def percentiles2(data, low=0.16, med=0.5, high=0.84, del0=False):
    # Feed a 2D array where the columns correspond to some fixed quantity.  Find the percentile values in each column.
    data[np.where(np.isfinite(data)==False)] = 0
    data = data[~np.all(np.equal(data, 0), axis=1)]
    row, col = data.shape
    if not del0:
        data = np.sort(data, axis=0)
        i_low = max(0,int(row*low)-1)
        i_med = int(row*med)-1
        i_high = int(row*high)-1
        return data[i_low,:], data[i_med,:], data[i_high,:]
    else:
        alow, amed, ahigh = np.zeros(col), np.zeros(col), np.zeros(col)
        for i in range(col):
            alow[i], amed[i], ahigh[i] = percentiles1(data[:,i],low,med,high,True)
        return alow, amed, ahigh


def clean2(data):
    # Clean a 2D array by setting nans/infs to 0 and removing rows of 0
    data[np.where(np.isfinite(data)==False)] = 0
    data = data[~np.all(np.equal(data, 0), axis=1)]
    return data


def coolrate(logTemp, met, arrs=None):
    # logTemp is log10(temperature/Kelvin)
    # met is log10(mass_metals/mass)
    # logLambda is log10 of cgs units.  Factor of 10^13 to get to SI.
    if arrs==None:
        logLambda_arr, logTemp_arr, met_arr = gr.cooltables()
    else:
        [logLambda_arr, logTemp_arr, met_arr] = arrs

    if logTemp>np.min(logTemp_arr) and logTemp<np.max(logTemp_arr):
        row = np.where(logTemp_arr>=logTemp)[0][0]
        logLambda_1d = logLambda_arr[row,:] - (logLambda_arr[row,:]-logLambda_arr[row-1,:])*(logTemp_arr[row]-logTemp)/(logTemp_arr[row]-logTemp_arr[row-1])
    elif logTemp<=np.min(logTemp_arr) or not np.isfinite(logTemp):
        logLambda_1d = logLambda_arr[0,:]
    else:
        logLambda_1d = logLambda_arr[-1,:]

    if met>np.min(met_arr) and met<np.max(met_arr):
        col = np.where(met_arr>=met)[0][0]
        logLambda = logLambda_1d[col] - (logLambda_1d[col]-logLambda_1d[col-1])*(met_arr[col]-met)/(met_arr[col]-met_arr[col-1])
    elif met<=np.min(met_arr) or not np.isfinite(met):
        logLambda = logLambda_1d[0]
    else:
        logLambda = logLambda_1d[-1]

    return logLambda


def softbins(r, soft, Nbins, Nmin=1):
    # Find bins where there are an almost equal no. of particles in them, while respecting a minimum separation
    rsort = np.sort(r)
    bins = np.zeros(Nbins+1)
    for b in range(Nbins):
        Ninbin = int(round((1.0*len(rsort))/(Nbins-b)))
        if Ninbin<Nmin:
            Ninbin=Nmin if len(rsort)>Nmin else len(rsort)
#        print '1', Ninbin, len(rsort), Nbins-b
        if Ninbin==0:
            bins[b+1:] = np.nan
#            print 'break1'
            break
        dr = rsort[Ninbin-1]-bins[b]
        if dr>=soft:
            bins[b+1] = rsort[Ninbin-1]
        else:
            try:
                Ninbin = np.where(rsort-rsort[0]>=soft)[0][0]+1
                bins[b+1] = rsort[Ninbin-1]
            except: # Assuming there aren't enough particles left to have a bin wider than the softening scale
                bins[b+1] = rsort[-1]
                if b!=len(bins):
                    bins[b+2:] = np.nan
#                print 'break2'
                break
#        print '2', Ninbin, len(rsort), Nbins-b
        rsort = np.delete(rsort, range(Ninbin))
    return bins


def meanbins(x, y, xmeans, tol=0.02, itmax=100):
    # Find bins in some dataset (x,y) which will have mean values of x matching xmeans
    #print xmeans
    fnan = np.isfinite(x)*np.isfinite(y)
    x, y = x[fnan], y[fnan]
    N = len(xmeans)
    bins = np.zeros(N+1)
    mean_x, mean_y = np.zeros(N), np.zeros(N)
    bins[0] = np.min(x)
    bins[-1] = np.max(x)
    for i in range(1,N+1):
        xleft = np.sort(x[x>=bins[i-1]])
        cumav = np.cumsum(xleft) / np.arange(1,len(xleft)+1)
        diff = abs(xmeans[i-1]-cumav)
        arg = np.where(diff==np.min(diff))[0][0]
#        if abs(cumav[arg]-xmeans[i-1])>=abs(cumav[arg-1]-xmeans[i-1]): arg -= 1
        if i!=N and arg<len(diff)-1: arg += 1
        bins[i] = xleft[arg]
        m = cumav[arg]
        mean_x[i-1] = m
        f = (xleft<bins[i]) if i!=N else (xleft<=bins[i])
        mean_y[i-1] = np.mean(y[np.in1d(x,xleft[f])])
        
#        binleft = 0
#        binright = len(xleft)
#        #print 'xleft', xleft
#        #bins[i] = np.min(xleft) + (np.max(xleft)-np.min(xleft)) / (N-i+1)
#        #bins[i] = xleft[int(1.0*len(xleft)/(N-i+1))-1]
#        bins[i] = xleft[int((binleft+binright)/2.)]
#
#        m = np.mean(xleft[f])
#        count = 0
#        while abs(m-xmeans[i-1])/xmeans[i-1]>tol:
#            if xmeans[i-1]>m:
#                bins[i] = np.median(xleft[xleft>bins[i]])
#            elif xmeans[i-1]<m:
#                bins[i] = np.median(xleft[xleft<bins[i]])
#            f = (xleft<bins[i]) if i!=N-1 else (xleft<=bins[i])
#            m = np.mean(xleft[f])
#            count += 1
#            if count==itmax:
#                print 'gc.meanbins: itmax reached'
#                break
#        #print bins
    return bins, mean_x, mean_y



def hist_with_xcoords(xdata, bins, ydata):
    Nhist, bins = np.histogram(xdata, bins=bins)
    xhist, bins = np.histogram(xdata, bins=bins, weights=xdata)
    xav = (1.*xhist) / (1.*Nhist)
    yhist, bins = np.histogram(xdata, bins=bins, weights=ydata)
    return xav, yhist

def density_profile(R, Rbins, mass):
    # Build a density profile of particles given their position and mass
    Rav, mhist = hist_with_xcoords(R, Rbins, mass)
    vol = 4./3. * np.pi * (Rbins[1:]**3.-Rbins[:-1]**3.)
    return Rav, mhist/vol

def surface_density_profile(r, rbins, mass):
    rav, mhist = hist_with_xcoords(r, rbins, mass)
    area = np.pi * (rbins[1:]**2.-rbins[:-1]**2.)
    return rav, mhist/area


def hydrogen_decomp_exp_disc(mass_gas, j_disc, v_rot, R_vir, mass_star, Z_gas, P_0=5.93e-13, chi_0=0.92, f_He=0.75, f_warm=1/1.3, sig_gas=11., elements=1000):
    # Calculate the HI and H2 content of a disc given its integrated properties, assuming it follows an exponential surface density profile

    # masses from solar masses to kg
    mass_gas = np.array(mass_gas, dtype=np.float64)*1.989e30
    mass_star = np.array(mass_star, dtype=np.float64)*1.989e30
    
    # j from kpc*km/s to m^2/s
    j_disc = np.array(j_disc*1e3*3.0857e19, dtype=np.float64)
    
    # velocities from km/s to m/s
    v_rot = np.array(v_rot*1e3, dtype=np.float64)
    sig_gas *= 1e3
    
    # distances from kpc to m
    R_vir = np.array(R_vir*3.0857e19, dtype=np.float64)
    
    G = 6.67408e-11
    
    r = (np.ones((len(mass_gas),elements)) * np.linspace(0, 1, elements)).T * R_vir
    r_eff = j_disc / (2*v_rot)
    Sigma_gas = np.exp(-r/r_eff) * mass_gas / (2*np.pi*r_eff**2)
    Sigma_star = np.exp(-r/r_eff) * mass_star / (2*np.pi*r_eff**2)
    sig_star = np.sqrt(np.pi*G/7.3 * Sigma_star * r_eff) # velocity dispersion
    P_ext = np.pi*G/2 * Sigma_gas * (Sigma_gas + sig_gas/sig_star * Sigma_star)
    R_H2 = (P_ext / P_0)**chi_0
    f_H2 = f_He*f_warm/(1./R_H2 + 1) * (1-Z_gas)
    integrand = 2*np.pi*f_H2*Sigma_gas*r
    m_H2 = np.sum(np.diff(r, axis=0) * (integrand[1:,:]+integrand[:-1,:])/2., axis=0)
    m_HI = mass_gas*f_He*f_warm*(1-Z_gas) - m_H2
    m_HI[mass_gas==0] = 0
    m_H2[mass_gas==0] = 0
    m_HI /= 1.989e30
    m_H2 /= 1.989e30
    assert len(m_H2) == len(R_vir)
    return m_HI, m_H2


def hydrogen_decomp_exp_disc2(mass_gas, r_eff, mass_star, Z_gas, P_0=5.93e-13, chi_0=0.92, f_He=0.75, f_warm=1/1.3, sig_gas=11., elements=1000):
    # Calculate the HI and H2 content of a disc given its integrated properties, assuming it follows an exponential surface density profile
    
    # masses from solar masses to kg
    mass_gas = np.array(mass_gas, dtype=np.float64)*1.989e30
    mass_star = np.array(mass_star, dtype=np.float64)*1.989e30
    # velocities from km/s to m/s
    sig_gas *= 1e3
    
    # distances from kpc to m
    r_eff = np.array(r_eff, dtype=np.float64)*3.0857e19
    
    G = 6.67408e-11

    r = (np.ones((len(mass_gas),elements)) * np.linspace(0, 1, elements)).T * (20.*r_eff)
    Sigma_gas = np.exp(-r/r_eff) * mass_gas / (2*np.pi*r_eff**2)
    Sigma_star = np.exp(-r/r_eff) * mass_star / (2*np.pi*r_eff**2)
    sig_star = np.sqrt(np.pi*G/7.3 * Sigma_star * r_eff) # velocity dispersion
    P_ext = np.pi*G/2 * Sigma_gas * (Sigma_gas + sig_gas/sig_star * Sigma_star)
    R_H2 = (P_ext / P_0)**chi_0
    f_H2 = f_He*f_warm/(1./R_H2 + 1) * (1-Z_gas)
    integrand = 2*np.pi*f_H2*Sigma_gas*r
    m_H2 = np.sum(np.diff(r, axis=0) * (integrand[1:,:]+integrand[:-1,:])/2., axis=0)
    m_HI = mass_gas*f_He*f_warm*(1-Z_gas) - m_H2
    m_HI[mass_gas==0] = 0
    m_H2[mass_gas==0] = 0
    m_HI /= 1.989e30
    m_H2 /= 1.989e30
    assert len(m_H2) == len(mass_gas)
    return m_HI, m_H2












def rahmati2013_neutral_frac(redshift, nH, T, onlyA1=True,noCol = False,onlyCol = False,SSH_Thresh = 1,extrapolate=False,local=False, UVB='FG09-Dec11'):
    """
        WARNING: I DO NOT PERSONALLY CONDONE THE USE OF THIS FUNCTION.  IT IS HERE PRIMARILY AS A MEANS OF COMPARING MY OWN METHODOLOGY WITH THOSE USED IN PREVIOUS PUBLISHED WORKS.
        """
    
#; --------------------------------------------------------------------------------------------
#;+
#; NAME:
#;       rahmati2013_neutral_frac
#;
#; PURPOSE:
#;       Computes particle neutral fractions based on the fitting functions of
#;       Rahmati et al. (2013a). By default, it uses the parameters of Table A1
#;       (based on small cosmological volumes) for z > 1, and of Table A2 (based
#;       on a 50Mpc volume) for z < 1, to better account for the effects of
#;       collisional ionisation on the self-shielding density.
#;
#; CATEGORY:
#;       I/O, HDF5, EAGLE, HI
#;
#; CALLING SEQUENCE:
#;       NeutralFraction = rahmati2013_neutral_frac(redshift,nH,Temperature_Type0)
#;       To compute neutral (HI+H_2) mass of particle, multiply NeutralFraction by
#;       xH and ParticleMass
#;
#; INPUTS:
#;       redshift:        Redshift of snapshot.
#;       nH:  hydrogen number density of the gas
#;       T:     temperature of the gas
#;
#; KEYWORD PARAMETERS:
#;       onlyA1:          routine will use Table A1 parameters for z < 0.5
#;       noCol:           the contribution of collisional ionisation to
#;       the overall ionisation rate is neglected
#;       onlyCol:         the contribution of photoionisation to
#;       the overall ionisation rate is neglected
#;       SSH_Thresh:      all particles above this density are assumed
#;       to be fully shielded (f_neutral=1)
#;
#; OUTPUTS:
#;       Array containing neutral fractions
#;
#; DEVELOPMENT:
#;       Written by Rob Crain, Leiden, March 2014, with input from Ali
#;       Rahmati. Based on Rahmati et al. (2013).
#;       Converted to python by Michelle Furlong, Dec 2014.
#;      Further edits by Adam Stevens, Jan 2018
#; --------------------------------------------------------------------------------------------
    if redshift>5:
        print('Using the z=5 relation for rahmati2013_neutral_frac when really z=',redshift)
        redshift = 5.0
    if redshift < 1.0:
        dlogz = (np.log10(1+redshift) - 0.0)/np.log10(2.)
        if onlyA1:
            lg_n0_lo     = -2.94
            gamma_uvb_lo =  8.34e-14 if UVB=='HM12' else 3.99e-14
            alpha1_lo    = -3.98
            alpha2_lo    = -1.09
            beta_lo      =  1.29
            f_lo         =  0.01
        else:
            lg_n0_lo     = -2.56
            gamma_uvb_lo =  8.34e-14 if UVB=='HM12' else 3.99e-14
            alpha1_lo    = -1.86
            alpha2_lo    = -0.51
            beta_lo      =  2.83
            f_lo         =  0.01
        lg_n0_hi     = -2.29
        gamma_uvb_hi =  7.39e-13 if UVB=='HM12' else 3.03e-13
        alpha1_hi    = -2.94
        alpha2_hi    = -0.90
        beta_hi      =  1.21
        f_hi         =  0.03
    elif (redshift >= 1.0 and redshift < 2.0):
        dlogz = (np.log10(1+redshift) - np.log10(2.))/(np.log10(3.)-np.log10(2.))
        lg_n0_lo     = -2.29
        gamma_uvb_lo =  7.39e-13 if UVB=='HM12' else 3.03e-13
        alpha1_lo    = -2.94
        alpha2_lo    = -0.90
        beta_lo      =  1.21
        f_lo         =  0.03
        
        lg_n0_hi     = -2.06
        gamma_uvb_hi =  1.50e-12 if UVB=='HM12' else 6.00e-13
        alpha1_hi    = -2.22
        alpha2_hi    = -1.09
        beta_hi      =  1.75
        f_hi         =  0.03
    elif (redshift >= 2.0 and redshift < 3.0):
        dlogz = (np.log10(1+redshift) - np.log10(3.))/(np.log10(4.)-np.log10(3.))
        lg_n0_lo     = -2.06
        gamma_uvb_lo =  1.50e-12 if UVB=='HM12' else 6.00e-13
        alpha1_lo    = -2.22
        alpha2_lo    = -1.09
        beta_lo      =  1.75
        f_lo         =  0.03
        
        lg_n0_hi     = -2.13
        gamma_uvb_hi =  1.16e-12 if UVB=='HM12' else 5.53e-13
        alpha1_hi    = -1.99
        alpha2_hi    = -0.88
        beta_hi      =  1.72
        f_hi         =  0.04
    elif (redshift >= 3.0 and redshift < 4.0):
        dlogz = (np.log10(1+redshift) - np.log10(4.))/(np.log10(5.)-np.log10(4.))
        lg_n0_lo     = -2.13
        gamma_uvb_lo =  1.16e-12 if UVB=='HM12' else 5.53e-13
        alpha1_lo    = -1.99
        alpha2_lo    = -0.88
        beta_lo      =  1.72
        f_lo         =  0.04
        
        lg_n0_hi     = -2.23
        gamma_uvb_hi =  7.91e-13 if UVB=='HM12' else 4.31e-13
        alpha1_hi    = -2.05
        alpha2_hi    = -0.75
        beta_hi      =  1.93
        f_hi         =  0.02
    elif (redshift >= 4.0 and redshift <= 5.0):
        dlogz = (np.log10(1+redshift) - np.log10(5.))/(np.log10(6.)-np.log10(5.))
        lg_n0_lo     = -2.23
        gamma_uvb_lo =  7.91e-13 if UVB=='HM12' else 4.31e-13
        alpha1_lo    = -2.05
        alpha2_lo    = -0.75
        beta_lo      =  1.93
        f_lo         =  0.02
        
        lg_n0_hi     = -2.35
        gamma_uvb_hi =  5.43e-13 if UVB=='HM12' else 3.52e-13
        alpha1_hi    = -2.63
        alpha2_hi    = -0.57
        beta_hi      =  1.77
        f_hi         =  0.01
    else:
        print('[rahmati2013_neutral_frac] ERROR: parameters only valid for z < 5, you asked for z = ', redshift)
        exit()

    # [Adam] All of this code could be massively reduced by just putting the hi/low values into a table and using np.interp....
    lg_n0     = lg_n0_lo     + dlogz*(lg_n0_hi     - lg_n0_lo)
    n0        = 10.**lg_n0
    lg_gamma_uvb_lo, lg_gamma_uvb_hi = np.log10(gamma_uvb_lo), np.log10(gamma_uvb_lo)
    gamma_uvb = 10**(lg_gamma_uvb_lo + dlogz*(lg_gamma_uvb_hi - lg_gamma_uvb_lo))
    alpha1    = alpha1_lo    + dlogz*(alpha1_hi    - alpha1_lo)
    alpha2    = alpha2_lo    + dlogz*(alpha2_hi    - alpha2_lo)
    beta      = beta_lo      + dlogz*(beta_hi      - beta_lo)
    f         = f_lo         + dlogz*(f_hi         - f_lo)
    
#    if onlyA1:
#        print '[rahmati2013_neutral_frac] using Table A1 parameters for all redshifts'
#    else:
#        print '[rahmati2013_neutral_frac] using Table A1 parameters for z > 1 and Table A2 parameters for z < 1'
#    if noCol:
#        print '[rahmati2013_neutral_frac] neglecting collisional ionisation'
#    if onlyCol:
#        print '[rahmati2013_neutral_frac] neglecting photoionisation'
#    print '[rahmati2013_neutral_frac] adopting SSH_Thresh/cm^-3 = ',SSH_Thresh
#    print '[rahmati2013_neutral_frac] using Rahmati et al. 2013 parameters for z = ',redshift
#    print ' lg_n0/cm^-3    = ',lg_n0
#    print ' gamma_uvb/s^-1 = ',gamma_uvb
#    print ' alpha1         = ',alpha1
#    print ' alpha2         = ',alpha2
#    print ' beta           = ',beta
#    print ' f              = ',f

    # Use fitting function as per Rahmati, Pawlik, Raicevic & Schaye 2013
#    print ' nH range       = ', min(nH), max(nH), np.median(nH)
    gamma_ratio = (1.-f) * (1. + (nH / n0)**beta)**alpha1 + f*(1. + (nH / n0))**alpha2
    gamma_phot  = gamma_uvb * gamma_ratio
    
    # if "local" is set, we include an estimate of the local
    # photoionisation rate from local sources, as per Ali's paper
    if local:
        # from Rahmati et al. (2013), equation (7)
        # assuming unity gas fraction and P_tot = P_thermal
        gamma_local = 1.3e-13 * nH**0.2 * (T/1.0e4)**0.2
        gamma_phot += gamma_local

    lambda_T  = 315614.0 / T
    AlphaA    = 1.269e-13 * (lambda_T)**(1.503) / ((1. + (lambda_T / 0.522)**0.470)**1.923)
    LambdaT   = 1.17e-10*(np.sqrt(T)*np.exp(-157809.0/T)/(1.0 + np.sqrt(T/1.0e5)))
    
    if noCol: LambdaT    = 0.0
    if onlyCol: gamma_phot = 0.0
    
    A = AlphaA + LambdaT
    B = 2.0*AlphaA + (gamma_phot/nH) + LambdaT
    sqrt_term = np.array([np.sqrt(B[i]*B[i] - 4.0*A[i]*AlphaA[i]) if (B[i]*B[i] - 4.0*A[i]*AlphaA[i])>0 else 0.0 for i in range(len(B))])
    f_neutral = (B - sqrt_term) / (2.0*A)
    f_neutral[f_neutral <= 0] = 1e-30 #negative values seem to arise from rounding errors - AlphaA and A are both positive, so B-sqrt_term should be positive!
    
    if SSH_Thresh:
#        print '[rahmati2013_neutral_frac] setting the eta = 1 for densities higher than: ', SSH_Thresh
        ind = np.where(nH > SSH_Thresh)[0]
        if(len(ind) > 0): f_neutral[ind] = 1.0

    return f_neutral



""" WARNING: THE FOLLOWING FUNCTIONS WERE INHERETED AND IS NOT USED FOR SCIENCE.  ITS PRESENCE IS MERELY TO COMPARE TO OLDER METHODS.  SEE HI_H2_MASSES() BELOW FOR THE PROPER FUNCTION """
def fH2_Gnedin_ParticleBasis(SFR,m,Zgas,nH,Density_Tot,T,fneutral,redshift):
    #;Formalism described in Gnedin and Kravtsov (2011) to calculate H2-to-total H ratio applied to EAGLE.
    #
    #;List of inputs:
    #;- SFR: SFR of the gas particle in units of gr/s. In the case of non star-forming particles SFR=0 and we calculate the photoionisation background from the
    #;Haardt & Madau (2012) backgroung
    #;- Zgas: gas metallicity (from the smoothed calculated values) in units of the solar metallicity.
    #;- nH: density of neutral hydrogen from Rahmati+13 in units of cm^-3.
    #;- T: temperature of the gas particles.
    #;- redshift: at the time of the calculation (need for the photoionisation background).
    #;
    #;Outputs:
    #;Fh2_SFR: the H2 to total neutral gas fraction.
    #;
    #;Calling sequence:
    #;fH2_Gnedin_ParticleBasis,SFR,Zgas,nH,T,redshift,Fh2_SFR
    #;Writen by Claudia Lagos, converted to python by Michelle Furlong
    # Further editing by me
    print('WARNING: YOU ARE CALLING AN OLD FUNCTION FOR THE GK11 HI/H2 PRESCRIPTION THAT PROBABLY SHOULD NOT BE USED FOR SCIENCE!')

    
    SFRMW        = 6.62e-21 #; in units of gr/s/cm^2. SFR density of the local neighbourhood (Bonatto+11)
    
    Zfloor       = 1e-5                 #;Arbitrary floor applied to metallicities.
    FH2Floor     = 1e-15                #Arbitrary floor applied to H2 fractions.
    UVsolarNeigh = 2.2e-12              #;in units of eV/s.
    
    #;Lecture of Haardt & Madau Uv background (units are in eV/s)
    data = gr.HaardtMadau12()
    redshiftHM12 = data[0]
    UVbackground = data[1]/UVsolarNeigh
    
    #;Find UV background at the redshift
    highz=np.where(redshiftHM12 >= redshift)[0]
    SFRfloor = UVbackground[highz[0]] #;take first object that has a redshift greater or equal the one of this snapshot.
    #;Applied to non star-forming particles. This corresponds to the Haardt & Madau (2012) UV background in units of the background in the solar neighbourhood (Allen+04).
    
#    print 'Applying the Gnedin & Kravtsov 11 formalism.'
#    print 'For all particles at redshift ',redshift
#    print 'UV background in units of the UV field in the local neighbourhood ',SFRfloor

    SigmaGas = 4.54e7 * np.sqrt(T*Density_Tot) #;from ntot*Jeans length. The units here are SigmaGas in gr/cm^2, T in K and Density_Tot in gr/cm^3.
    
    SF_UMW        = SFR/m*SigmaGas/SFRMW #;equivalent to doing rho_SFR * Jeans length divided by SFRMW.
    
    lowSFR = np.where(SF_UMW <= 0.0)[0]
    if (len(lowSFR) > 0):
        SF_UMW[lowSFR]=SFRfloor
        print('Interstellar Radiation Field hit floor')

    SF_DMW = Zgas                       #;it's already in units of solar metallicities.
    LowZ=np.where(SF_DMW < Zfloor)[0]    #;apply floor to metallicities.
    if (len(LowZ) > 0):
        SF_DMW[LowZ]=Zfloor
        print('Dust ratio hit floor')

    SF_Dstar      = 1.5e-3*np.log(1.0+(3.0*SF_UMW)**(1.7))
    SF_alpha      = 5.0*(SF_UMW/2.0)/(1.0+(SF_UMW/2.0)**2.0)
    SF_s          = 0.04/(SF_Dstar+SF_DMW)
    SF_g          = (1.0+SF_alpha*SF_s+SF_s**2.0)/(1.0+SF_s)
    SF_Lambda     = np.log(1.0+SF_g*SF_DMW**(3.0/7.0)*(SF_UMW/15.0)**(4.0/7.0))
    
    Y = 0.26
    X = 1-Y
    mu = (1.0-3.0*Y)/(2.0-fneutral)
    Sigma0        = fneutral*0.279*np.sqrt(T*nH*X/mu) #;from nH*Jeans length. The units here are Sigma0 in Msun/pc2, T in K and nH in cm^-3. (total gas surface density).
    
    Sigmastar     = 20.0*SF_Lambda**(4.0/7.0)/SF_DMW/np.sqrt(1.0+SF_UMW*SF_DMW**2.0)
    
    Fh2_SFR       = 1.0/(1.0+Sigmastar/Sigma0)**2.0
    ZeroSigma0 = np.where(Sigma0 <= 0.0)[0]
    if (len(ZeroSigma0) > 0):
        Fh2_SFR[ZeroSigma0] = FH2Floor
    
    return Fh2_SFR


""" WARNING: THE FOLLOWING FUNCTIONS WERE INHERETED AND IS NOT USED FOR SCIENCE.  ITS PRESENCE IS MERELY TO COMPARE TO OLDER METHODS.  SEE HI_H2_MASSES() BELOW FOR THE PROPER FUNCTION """
def fH2_Krumholz_ParticleBasis(SFR,m,Zgas,nH,Density_Tot,T,fneutral,redshift):
    #;Formalism described in Krumholz (2013) to calculate H2-to-total H ratio applied to EAGLE.
    #
    #;List of inputs:
    #;- SFR: SFR of the gas particle in units of gr/s. In the case of non star-forming particles SFR=0 and we calculate the photoionisation background from the
    #;Haardt & Madau (2012) backgroung
    #;- Zgas: gas metallicity (from the smoothed calculated values) in units of the solar metallicity.
    #;- nH: density of neutral hydrogen from Rahmati+13 in units of cm^-3.
    #;- T: temperature of the gas particles.
    #;- redshift: at the time of the calculation (need for the photoionisation background).
    #;
    #;Outputs:
    #;Fh2_SFR: the H2 to total neutral gas fraction.
    #;
    #;Calling sequence:
    #;fH2_Krumholz_ParticleBasis,SFR,Zgas,nH,T,redshift,Fh2_SFR
    #;Writen by Claudia Lagos, converted to python by Michelle Furlong
    # Further editing by me
    
    print('WARNING: YOU ARE CALLING AN OLD FUNCTION FOR THE K13 HI/H2 PRESCRIPTION THAT PROBABLY SHOULD NOT BE USED FOR SCIENCE!')
    
    SFRMW        = 6.62e-21          #;in units of gr/s/cm^2. SFR density of the local neighbourhood (Bonatto+11)
    Zfloor       = 1e-5              #;Arbitrary floor applied to metallicities.
    UVsolarNeigh = 2.2e-12           #;in units of eV/s.
    
    #;Lecture of Haardt & Madau Uv background (units are in eV/s)
    data = gr.HaardtMadau12()
    redshiftHM12 = data[0]
    UVbackground = data[1]/UVsolarNeigh
    
    #;Find UV background at the redshift
    highz=np.where(redshiftHM12 >= redshift)[0]
    SFRfloor = UVbackground[highz[0]] #;take first object that has a redshift greater or equal the one of this snapshot.
    #;Applied to non star-forming particles. This corresponds to the Haardt & Madau (2012) UV background in units of the background in the solar neighbourhood (Allen+04).
    
#    print 'Applying the Krumholz+13 formalism.'
#    print 'For all particles at redshift ',redshift
#    print 'UV background in units of the UV field in the local neighbourhood ',SFRfloor

    fc           = 5.0  #;clumping factor.
    #;Parameters below are defined in case we decide to use the hydrodynamic cold neutral medium density as a lower limit to the CNM density.
    alpha        = 5.0  #;factor that contains how much additional support is provided by turbulence, magnetic fields and cosmic rays. (Pth=Pmp/alpha)
    Cdshape      = 0.33 #;specfic value may change depending on the shape of the gas density profile.
    fw           = 0.5
    cw           = 8.0e5  #;in cm/s: sound speed of the warm neutral medium
    rhosd        = 0.01*(1.989e33)/(3.0857e18)**3 #; in unts of g/cm^3: volume density of dark matter plus stars. This value is typical of galaxies in the local universe.
    
    G_cgs = 2.1069818 # Gravtiational constant in cgs units
    
    SF_DMW =Zgas #;It's already in units of solar metallicities
    lowz=np.where(SF_DMW < Zfloor)[0]
    if (len(lowz) > 0):
        SF_DMW[lowz] = Zfloor
    
    SigmaGas = 4.54e7 * np.sqrt(T*Density_Tot) #;from ntot*Jeans length. The units here are SigmaGas in gr/cm^2, T in K and Density_Tot in gr/cm^3.
    G0            = SFR/m*SigmaGas/SFRMW #;equivalent to doing rho_SFR * Jeans length divided by SFRMW.
    
    lowG0=np.where(G0 <= 0.0)[0] #;this selects non star-forming galaxies.
    if (len(lowG0) > 0):
        G0[lowG0] = SFRfloor
    
    Sigma0     = fneutral*0.279*np.sqrt(T*nH) #;from nH*Jeans length. The units here are Sigma0 in Msun/pc2, T in K and nH in cm^-3.
    Sigma0_cgs = SigmaGas * 1.989e33 / (3.0857e18)**2
    ncnm_2p    = 23.0*G0*((1.0+3.1*SF_DMW**(0.365))/4.1)**(-1)/10.0 #; in units of 10xcm^-3
    ncnm_hydro = np.pi * G_cgs * Sigma0_cgs**2 / (4*alpha) * (1 + np.sqrt(1+ 32 * Cdshape * alpha * fw * cw*cw * rhosd / (np.pi * G_cgs * Sigma0_cgs**2))) / 10.0 #; in units of 10xcm^-3
    ncnm       = ncnm_2p
    
    lowncm = np.where(ncnm_hydro > ncnm_2p)[0]
    if (len(lowncm) > 0):
        ncnm[lowncm]=ncnm_hydro[lowncm]
    
    Chi        = 7.2*G0/ncnm
    
    Tauc       = 0.066*fc*SF_DMW*Sigma0
    sfac       = np.log(1.0+0.6*Chi+0.01*Chi**2.0)/(0.6*Tauc)
    
    Fh2_SFR    = 1.0-0.75*sfac/(1.0+0.25*sfac)
    lows=np.where(sfac >= 2)[0]
    if (len(lows) > 0):
        Fh2_SFR[lows] = 0.0
    
    return Fh2_SFR
### =================================================================================================================================================================================== ###






def HI_H2_masses(mass, SFR, Z, rho, temp, fneutral, redshift, method=4, mode='T', UVB='FG09-Dec11', U_MW_z0=None, rho_sd=0.01, col=2, gamma_fixed=None, mu_fixed=None, S_Jeans=True, T_CNMmax=243., Pth_Lagos=False, Jeans_cold=False, Sigma_SFR0=1e-9, UV_MW=None, X=None, UV_pos=None, f_esc=0.15, f_ISM=None):
    """
        This is my own version of calculating the atomic- and molecular-hydrogen masses of gas particles/cells from simulations.  This was originally adapted from the Python scripts written by Claudia Lagos and Michelle Furlong, and followed the basis of Appendix A of Lagos et al (2015b).  This has since been vastly modified and is still being developed further.  This has been developed in tandem with Benedikt Diemer's code for Illustris-TNG, and has been tested to produce the same results.  Please cite Stevens et al. (2019, MNRAS, 483, 5334) if you use this function for a publication (also note the erratum to that paper: 2019, MNRAS, 484, 5499).
        Expects each non-default input as an array, except for reshift.  Input definitions and units are as follows:
        mass = total mass of gas particles/cells [M_sun]
        SFR = star formation rate of particles/cells [M_sun/yr]
        Z = ratio of metallic mass to total particle/cell mass
        rho = density of particles/cells [M_sun/pc^3]
        temp = temperature of particles [K] OR specific thermal energy [(m/s)^2] (see mode)
        fneutral = fraction of particle/cell mass that is not ionized.  If given as None, it will automatically be calculated using the Rahmati+13 prescription.
        redshift = single float for the redshift being considered
        method = 0 - Return results for methods 2, 3, and 4
                 1 - Gnedin & Kravtsov (2011) eq 6
                 2 - Gnedin & Kravtsov (2011) eq 10
                 3 - Gnedin & Draine (2014) eq 6
                 4 - Krumholz (2013) eq 10
                 5 - Gnedin & Draine (2014) eq 8
          all of these also use Schaye and Dalla Vecchia (2008) for the Jeans length
        mode = 'T' - temp is actually temperature
               'u' - have fed in internal energy per unit mass instead of temperature
        UVB = 'HM12' - Haardt & Madau (2012)
              'FG09' - Faucher-Giguere et al. (2009)
              'FG09-Dec11' - Updated FG09 table.  Uses pre-built values normalised by the Draine (1978) field, tabulated by Benedikt Diemer
        U_MW_z0 = strength of UV background at z=0 in units of the Milky Way's interstellar radiation field.  Has a default value if set to None.
        rho_sd = local density of dark matter and stars. Used in method 4. [Msun/pc^3]
        col = only used for UVB='FG09-Dec11', decides on column to use in table
        gamma_fixed = set gamma to be a fixed value if not None, even if that breaks self-consistency (there for testing)
        mu_fixed = as above but for mu
        S_Jeans = use Jeans scale for S variable in GD14, else use the cube root of cell volume (former makes more sense for SPH, but not cells)
        T_CNMmax = Maximum temperature of cold neutral medium for K13
        Pth_Lagos = calculate P_th for K13 as in eq.A15 of Lagos+15b rather than eq.6 of K13.  Not advised for science.  Here for comparison purposes.
        Jeans_cold = use cold clouds of SF gas cells only for calculating Jeans length [NOT YET IMPLEMENTED]
        Sigma_SFR0 = local star formation rate surface density of the solar neighbourhood [Msun/yr/pc^2]
        UV_MW = pre-computed UV fluxes (normed by Milky Way) for each cell
        X = pre-computed hydrogen fractions for particles/cells (or a chosen constant)
        UV_pos = positions of particles/cells, used to approximate the UV field of non-SF particles/cells based on nearby SF particles/cells.  Using this will definitely slow the code but should return more realistic HI/H2 fractions for non-star-forming cells/particles.  Is redundant when UV_MW is provided. [pc]
        f_esc = fudge factor for escape fraction in the approximate calculation for UV
        f_ISM = boolean array for particles/cells stating whether they should be considered 'ISM' for the sake of the K13 prescription.  Those that are not will not have the nCNMhydro floor applied.  This will also inform the mean Sigma_SFR for the UV calculation.
    """
    
    
    kg_per_Msun = 1.989e30
    m_per_pc = 3.0857e16
    s_per_yr = 60*60*24*365.24

    # Convert physical constants to internal units
    m_p = 1.6726219e-27 / kg_per_Msun
    G = 6.67408e-11 * kg_per_Msun * s_per_yr**2 / m_per_pc**3
    k_B = 1.38064852e-23 * s_per_yr**2 / m_per_pc**2 / kg_per_Msun
    const_ratio = k_B / (m_p * G)
    f_th = 1.0 # Assuming all gas is thermal

    Z[Z<1e-5] = 1e-5 # Floor on metallicity (BBN means there must be a tiny bit)
    if X is None: # Approximate hydrogen fraction from pre-determined fitting function (very simplistic)
        p = [0.0166, -1.3452, 0.75167, 31.193, -2.38233]
        X = piecewise_parabola_linear(Z, *p)
        X[X>0.76] = 0.76 # safety
    Y = 1. - X - Z
    
    # Protons per cm^3 in all forms of hydrogen
    denom = m_p * (m_per_pc*100)**3
    n_H = X * rho / denom
    
    
    if gamma_fixed is not None:
        gamma = 1.0*gamma_fixed
    else:
        gamma = 5./3. # initialise

    if mode=='u':
        u = 1.0*temp
        temp = u2temp(u, gamma, 0.59) # initialise

    if fneutral is None:
        calc_fneutral = True
    else:
        calc_fneutral = False

    # Calculate (initialise in the case of mode='u') neutral fraction if it wasn't already provided
    if calc_fneutral:
        fneutral = rahmati2013_neutral_frac(redshift, rho/denom, temp, UVB=UVB)

    fzero = (fneutral <= 0)
    fneutral[fzero] = 1e-6 # Floor on neutral fraction.  Prevents division by zero below

    if mu_fixed is not None:
        mu = 1.0*mu_fixed
    else:
        mu = (X + 4*Y)/((2-fneutral)*(X+Y)) # Initialise mean molecular weight


    # Initialise lists if all methods wanted
    mHI_list, mH2_list = [], []

    # Set floor of interstellar radiation field from UV background, in units of Milky Way field
    if UVB not in ['HM12', 'FG09', 'FG09-Dec11']:
        print('Could not interpret input for UVB.  UVB should be set to either HM12, FG09, or FG09-Dec11.  Defaulting to FG09-Dec11.')
        UVB = 'FG09-Dec11'
    
    if UVB=='FG09-Dec11':
        if col not in [1,2,3]: col = 3
        data = gr.U_MW_FG09_Dec11()
        redshift_UVB = data[:,0]
        UVbackground = data[:,col]
    else:
        if UVB=='HM12':
            data = gr.HaardtMadau12()
        elif UVB=='FG09':
            data = gr.FaucherGiguere09()
        redshift_UVB = data[0,:]
        if U_MW_z0 is None:
            UVbackground = data[1,:]/2.2e-12 # Divides through by local MW value (that number is in eV/s).  Original reference is unknown.  This probably should be avoided.
        else:
            UVbackground = data[1,:] / data[1,0] * U_MW_z0
    ISRF_floor = 10**np.interp(redshift, redshift_UVB, np.log10(UVbackground)) * np.ones(len(mass))
    
    # Leaving this commented as I have been feeding in values that already have this floor applied.  Note that in future if values fed in don't have the floor applied, these lines should be uncommented.
#    if UV_MW is not None:
#        UV_MW[UV_MW<ISRF_floor] = ISRF_floor

    # Approximate UV field based on average SF density
    sf = (SFR>0) if f_ISM is None else (SFR>0) * f_ISM
    if UV_pos is not None and UV_MW is None and len(sf[sf])>0:
        CoSF = np.sum(mass[sf] * UV_pos[sf].T, axis=1) / np.sum(mass[sf]) # centre of star formation
        Rsqr = np.sum((UV_pos - CoSF)**2, axis=1)
        Rsqr_area = np.max((Rsqr + 0.5*(mass/rho)**(2./3.))[sf])
        Sigma_SFR_cen = np.sum(SFR) / Rsqr_area / np.pi
        ISRF_floor = np.maximum(ISRF_floor, f_esc*Sigma_SFR_cen/Sigma_SFR0/Rsqr* Rsqr_area)


        
    # Dust to gas ratio relative to MW
    D_MW = Z / 0.0127
    
    it_max = 300 # Maximum iterations for calculating f_H2 (arbitrary)
    f_H2_old = np.zeros(len(mass)) # Initialise before iterating
    fneutral_old = np.zeros(len(mass))
    
    if method==1: # GK11, eq6
        for it in range(it_max):
            if it==it_max-1: print('iterations hit maximum for GK11, eq6 in HI_H2_masses()')
            f_mol = X*fneutral*f_H2_old /  (X+Y)
            if gamma_fixed is None: gamma = (5./3.)*(1-f_mol) + 1.4*f_mol
            if mu_fixed is None: mu = (X + 4*Y) * (1.+ (1.-fneutral)/fneutral) / ((X+Y) * (1.+ 2*(1.-fneutral)/fneutral - f_H2_old/2.))
            if mode=='u':
                temp = u2temp(u, gamma, mu)
                if calc_fneutral and not np.allclose(fneutral, fneutral_old, rtol=5e-3): fneutral = rahmati2013_neutral_frac(redshift, rho/denom, temp, UVB=UVB) # too slow to do every time (hopefully this converges faster)
            Sigma = np.sqrt(gamma * const_ratio * f_th * rho * temp / mu) # Approximate surface density as true density * Jeans length (see eq. 7 of Schaye and Dalla Vecchia 2008)
            area = mass / Sigma # Effective area covered by particle
            Sigma_SFR = SFR / area
            G0 = np.maximum(ISRF_floor, f_esc * Sigma_SFR / Sigma_SFR0) if UV_MW is None else 1.0*UV_MW # Calculte interstellar radiation field, assuming it's proportional to SFR density, normalised by local Sigma_SFR of solar neighbourhood.
            D_star = 1.5e-3 * np.log(1. + (3.*G0)**1.7)
            alpha = 2.5*G0 / (1.+(0.5*G0)**2.)
            s = 0.04 / (D_star + D_MW)
            g = (1. + alpha*s + s*s) / (1.+s)
            Lambda = np.log(1. + g * D_MW**(3./7.) * (G0/15.)**(4./7.))
            x = Lambda**(3./7.) * np.log(D_MW * fneutral*n_H/(Lambda*25.))
            f_H2 = 1./ (1.+ np.exp(-4.*x - 3.*x**3)) # H2/(HI+H2)
            if np.allclose(f_H2[~fzero], f_H2_old[~fzero], rtol=5e-3): break
            f_H2_old = 1.*f_H2
            fneutral_old = 1.*fneutral

        mass_H2 = f_H2 * fneutral * X * mass
        mass_HI = (1.-f_H2) * fneutral * X * mass
        mass_H2[fzero] = 0.
        mass_HI[fzero] = 0.

    if method==2 or method==0: # GK11, eq10 (entry 0 for method==0)
        for it in range(it_max):
            if it==it_max-1: print('iterations hit maximum for GK11, eq11 in HI_H2_masses()')
            f_mol = X*fneutral*f_H2_old /  (X+Y)
            if gamma_fixed is None: gamma = (5./3.)*(1-f_mol) + 1.4*f_mol
            if mu_fixed is None: mu = (X + 4*Y) * (1.+ (1.-fneutral)/fneutral) / ((X+Y) * (1.+ 2*(1.-fneutral)/fneutral - f_H2_old/2.))
            if mode=='u':
                temp = u2temp(u, gamma, mu)
                if calc_fneutral and not np.allclose(fneutral, fneutral_old, rtol=5e-3): fneutral = rahmati2013_neutral_frac(redshift, rho/denom, temp, UVB=UVB)
            Sigma = np.sqrt(gamma * const_ratio * f_th * rho * temp / mu)
            Sigma_n = fneutral * X * Sigma # neutral hydrogen density
            area = mass / Sigma
            Sigma_SFR = SFR / area
            G0 = np.maximum(ISRF_floor, f_esc * Sigma_SFR / Sigma_SFR0) if UV_MW is None else 1.0*UV_MW
            D_star = 1.5e-3 * np.log(1. + (3.*G0)**1.7)
            alpha = 2.5*G0 / (1.+(0.5*G0)**2.)
            s = 0.04 / (D_star + D_MW)
            g = (1. + alpha*s + s*s) / (1.+s)
            Lambda = np.log(1. + g * D_MW**(3./7.) * (G0/15.)**(4./7.))
            Sigma_c = 20. * Lambda**(4./7.) / (D_MW * np.sqrt(1.+G0*D_MW**2.))
            f_H2 = (1.+Sigma_c/Sigma_n)**-2. # H2/(HI+H2)
            if np.allclose(f_H2[~fzero], f_H2_old[~fzero], rtol=5e-3): break
            f_H2_old = 1.*f_H2
            fneutral_old = 1.*fneutral

        mass_H2 = f_H2 * fneutral * X * mass
        mass_HI = (1.-f_H2) * fneutral * X * mass
        mass_H2[fzero] = 0.
        mass_HI[fzero] = 0.

    if method==0:
        mHI_list += [mass_HI]
        mH2_list += [mass_H2]

    if method==3: # GD14, eq6
        for it in range(it_max):
            if it==it_max-1: print('iterations hit maximum for GD14, eq6 in HI_H2_masses()')
            f_mol = X*fneutral*f_H2_old /  (X+Y)
            if gamma_fixed is None: gamma = (5./3.)*(1-f_mol) + 1.4*f_mol
            if mu_fixed is None: mu = (X + 4*Y) * (1.+ (1.-fneutral)/fneutral) / ((X+Y) * (1.+ 2*(1.-fneutral)/fneutral - f_H2_old/2.))
            if mode=='u':
                temp = u2temp(u, gamma, mu)
                if calc_fneutral and not np.allclose(fneutral, fneutral_old, rtol=5e-3): fneutral = rahmati2013_neutral_frac(redshift, rho/denom, temp, UVB=UVB)
            Sigma = np.sqrt(gamma * const_ratio * f_th * rho * temp / mu)
            S = Sigma / rho * 0.01 if S_Jeans else (mass/rho)**(1./3) * 0.01 # Spatial scale: either the Jeans length or approx cell length (per 100 pc)
            D_star = 0.17*(2.+S**5.)/(1.+S**5.)
            U_star = 9.*D_star/S
            g = np.sqrt(D_MW*D_MW + D_star*D_star)
            G0 = np.maximum(ISRF_floor, f_esc * SFR / mass * Sigma / Sigma_SFR0) if UV_MW is None else 1.0*UV_MW # Instellar radiation field in units of MW's local field (assumed to be proportional to local SFR density).  Reduced from several lines in other methods.
            Lambda = np.log(1.+ (0.05/g+G0)**(2./3)*g**(1./3)/U_star)
            n_half = 14. * np.sqrt(D_star) * Lambda / (g*S)
            x = (0.8 + np.sqrt(Lambda)/S**(1./3)) * np.log(fneutral*n_H/n_half)
            f_H2 = 1./ (1 + np.exp(-x*(1-0.02*x+0.001*x*x))) # H2/(HI+H2)
            if np.allclose(f_H2[~fzero], f_H2_old[~fzero], rtol=5e-3): break
            f_H2_old = 1.*f_H2
            fneutral_old = 1.*fneutral

        mass_H2 = f_H2 * fneutral * X * mass
        mass_HI = (1.-f_H2) * fneutral * X * mass
        mass_H2[fzero] = 0.
        mass_HI[fzero] = 0.


    if method==5 or method==0: # GD14, eq8 (entry 1 for method==0)
        # This has now replaced eq6 for the same output position when method=0
        for it in range(it_max):
            if it==it_max-1: print('iterations hit maximum for GD14, eq8 in HI_H2_masses()')
            f_mol = X*fneutral*f_H2_old /  (X+Y)
            if gamma_fixed is None: gamma = (5./3.)*(1-f_mol) + 1.4*f_mol
            if mu_fixed is None: mu = (X + 4*Y) * (1.+ (1.-fneutral)/fneutral) / ((X+Y) * (1.+ 2*(1.-fneutral)/fneutral - f_H2_old/2.))
            if mode=='u':
                temp = u2temp(u, gamma, mu)
                if calc_fneutral and not np.allclose(fneutral, fneutral_old, rtol=5e-3): fneutral = rahmati2013_neutral_frac(redshift, rho/denom, temp, UVB=UVB)
            Sigma = np.sqrt(gamma * const_ratio * f_th * rho * temp / mu)
            S = Sigma / rho * 0.01 if S_Jeans else (mass/rho)**(1./3) * 0.01 # Spatial scale: either the Jeans length or approx cell length (per 100 pc)
            D_star = 0.17*(2.+S**5.)/(1.+S**5.)
            U_star = 9.*D_star/S
            g = np.sqrt(D_MW*D_MW + D_star*D_star)
            G0 = np.maximum(ISRF_floor, f_esc * SFR / mass * Sigma / Sigma_SFR0) if UV_MW is None else 1.0*UV_MW # Instellar radiation field in units of MW's local field (assumed to be proportional to local SFR density).  Reduced from several lines in other methods.
            alpha = 0.5 + 1./(1. + np.sqrt(G0*D_MW*D_MW/600.))
            Sigma_R1 = 50./g * np.sqrt(0.001+0.1*G0) / (1. + 1.69*np.sqrt(0.001+0.1*G0)) # Note the erratum on the paper for this equation!
            R = (Sigma * fneutral * X / Sigma_R1)**alpha
            f_H2 = R / (R + 1.)
            if np.allclose(f_H2[~fzero], f_H2_old[~fzero], rtol=5e-3): break
            f_H2_old = 1.*f_H2
            fneutral_old = 1.*fneutral
        
        mass_H2 = f_H2 * fneutral * X * mass
        mass_HI = (1.-f_H2) * fneutral * X * mass
        mass_H2[fzero] = 0.
        mass_HI[fzero] = 0.
    
    if method==0:
        mHI_list += [mass_HI]
        mH2_list += [mass_H2]

    
    if method==4 or method==0: # K13, eq10 (entry 2 for method==0)
        f_c = 5.0 # clumping factor
        alpha = 5.0 # relative pressure of turbulence, magnetic fields vs thermal
        zeta_d = 0.33
        f_w = 0.5
        c_w = 8e3 / m_per_pc * s_per_yr # sound speed of warm medium -- could calculate this better
        for it in range(it_max):
            if it==it_max-1: print('iterations hit maximum for K13, eq10 in HI_H2_masses()')
            f_mol = X*fneutral*f_H2_old /  (X+Y)
            if gamma_fixed is None: gamma = (5./3.)*(1-f_mol) + 1.4*f_mol
            if mu_fixed is None: mu = (X + 4*Y) * (1.+ (1.-fneutral)/fneutral) / ((X+Y) * (1.+ 2*(1.-fneutral)/fneutral - f_H2_old/2.))
            if mode=='u':
                temp = u2temp(u, gamma, mu)
                if calc_fneutral and not np.allclose(fneutral, fneutral_old, rtol=5e-3): fneutral = rahmati2013_neutral_frac(redshift, rho/denom, temp, UVB=UVB)
            Sigma = np.sqrt(gamma * const_ratio * f_th * rho * temp / mu)
            Sigma_n = fneutral * X * Sigma # neutral hydrogen density
            G0 = np.maximum(ISRF_floor, f_esc * SFR / mass * Sigma / Sigma_SFR0) if UV_MW is None else 1.0*UV_MW
            #
            n_CNM2p = 23.*G0 * 4.1 / (1. + 3.1*D_MW**0.365)
            #
            if not Pth_Lagos:
                R_H2 = f_H2_old / (1. - f_H2_old)
                Sigma_HI = (1 - f_H2_old) * Sigma_n
                frac = 32. * zeta_d * alpha * f_w * c_w*c_w * rho_sd / (np.pi*G*Sigma_HI**2.)
                P_th = np.pi*G*Sigma_HI**2./(4.*alpha) * (1. + 2*R_H2 + np.sqrt((1.+ 2*R_H2)**2 + frac))  
            else:
                frac = 32. * zeta_d * alpha * f_w * c_w*c_w * rho_sd / (np.pi*G*Sigma_n**2.)
                P_th = np.pi*G*Sigma_n**2./(4.*alpha) * (1. + np.sqrt(1 + frac))
            n_CNMhydro = P_th / (1.1 * k_B * T_CNMmax) / (m_per_pc*100.)**3.
            #
            n_CNM = np.max(np.array([n_CNM2p,n_CNMhydro]),axis=0)
            if f_ISM is not None: n_CNM[~f_ISM] = n_CNM2p[~f_ISM] # floor doesn't apply to diffuse halo gas
            chi = 7.2*G0 / (0.1*n_CNM)
            tau_c = 0.066 * f_c * D_MW * Sigma_n
            s = np.log(1.+ 0.6*chi + 0.01*chi*chi) / (0.6*tau_c)
            #
            f_H2 = np.zeros(len(mu))
            f_H2[s<2] = 1. - 0.75*s[s<2]/(1. + 0.25*s[s<2])
            if np.allclose(f_H2[~fzero], f_H2_old[~fzero], rtol=5e-3): break
            f_H2_old = 1.*f_H2
            fneutral_old = 1.*fneutral

        mass_H2 = f_H2 * fneutral * X * mass
        mass_HI = (1.-f_H2) * fneutral * X * mass
        mass_H2[fzero] = 0.
        mass_HI[fzero] = 0.

    if method==0:
        mHI_list += [mass_HI]
        mH2_list += [mass_H2]

    if method==0:
        if calc_fneutral:
            return mHI_list, mH2_list, fneutral
        else:
            return mHI_list, mH2_list
    else:
        if calc_fneutral:
            return mass_HI, mass_H2, fneutral
        else:
            return mass_HI, mass_H2



def fatm_q(G, sigma=11.0, f_He=0.75, DiscBinEdge=None, h=0.73, single=False):
    # Calculate fatm and q for Dark Sage galaxies.  Expects more than one galaxy in G by default.  Can be told to assume 1 by setting single=True
    if DiscBinEdge is None: DiscBinEdge = np.append(0, np.array([1.0*1.4**i for i in range(30)])) / h
    j_bins = (DiscBinEdge[1:]+DiscBinEdge[:-1])/2.
    Grav = 6.67408e-11 * 1.989e30 / 3.0857e19 / 1e6
    f_HeZ = f_He - G.MetalsColdGas/G.ColdGas
    
    if not single:
        f_HeZ[G.ColdGas<=0] = f_He
        HImass = np.sum(G.DiscHI, axis=1)*1e10/h
        H2mass = np.sum(G.DiscH2, axis=1)*1e10/h
        DiscMetallicity = G.DiscGasMetals/G.DiscGas
        DiscMetallicity[G.DiscGas<=0] = 0.0
        DiskMass = (G.StellarMass - G.InstabilityBulgeMass - G.MergerBulgeMass)*1e10/h + (HImass+H2mass)/f_HeZ
        j_disk = np.sum((G.DiscStars+(G.DiscHI+G.DiscH2)/(f_He-DiscMetallicity))*1e10/h*j_bins, axis=1)/DiskMass
    else:
        if G.ColdGas<=0: f_HeZ = f_He
        HImass = np.sum(G.DiscHI)*1e10/h
        H2mass = np.sum(G.DiscH2)*1e10/h
        DiscMetallicity = G.DiscGasMetals/G.DiscGas
        DiscMetallicity[G.DiscGas<=0] = 0.0
        DiskMass = (G.StellarMass - G.InstabilityBulgeMass - G.MergerBulgeMass)*1e10/h + (HImass+H2mass)/f_HeZ
        j_disk = np.sum((G.DiscStars+(G.DiscHI+G.DiscH2)/(f_He-DiscMetallicity))*1e10/h*j_bins)/DiskMass

    q = j_disk * sigma / (Grav * DiskMass)
    f_atm =  HImass/f_HeZ / DiskMass
    return f_atm, q


def neutralFraction_SFcells(u, n_H, n_H_th=0.13, T_cold=1000, T_SN=5.73e7, A0=573.0, f_H=0.76):
    """
        Calculate neutral fraction for star-forming cells in Illustris-type simulations.  This takes the neutral fraction as the particle mass fraction in the 'cold' phase, from Springel & Hernquist's (2003) two-phase model.  Code is adapted from that privately sent by Benedikt Diemer.
        Input definitions:
        u = Internal specific energy of gas particles [m/s]^2
        n_H = Density of gas particles in proton masses / cm^3
    """
    u_cold = temp2u(T_cold, mu=4./(1.+3*f_H))
    u_SN = temp2u(T_SN, mu=4./(8.-5*(1-f_H)))
    u_4 = temp2u(1e4, mu=4./(8.-5*(1-f_H)))
    A = A0 * (n_H / n_H_th)**-0.8
    u_hot = u_SN / (1.+A) + u_cold
    fneutral = (u_hot - u) / (u_hot - u_cold)
    fneutral[fneutral>1.0] = 1.0 # numerical errors could give answer just above 1.0 otherwise
    fneutral[fneutral<0.0] = 0.0
    return fneutral

def neutralFraction_SFcells_SH03(u, n_H, beta=0.1, T_c=1000, T_SN=1e8, A0=1e3, f_H=0.76, tstar0=2.1, Lambda_tab=None):
    # As above, but properly going through the Springel & Hernquist (2003) methodology.
    if Lambda_tab is None: Lambda_tab = np.loadtxt('/Users/adam/Illustris/Lambda.txt')
    u_c = temp2u(T_c, mu=4./(1.+3*f_H))
    u_SN = temp2u(T_SN, mu=4./(8.-5*(1-f_H)))
    u_4 = temp2u(1e4, mu=4./(8.-5*(1-f_H)))
    x_th = 1. + (A0+1.)*(u_c-u_4)/u_SN
    T_cool = u2temp(u_SN/A0, mu=4./(8.-5*(1-f_H)))
    Lambda = -np.interp(T_cool, Lambda_tab[:,0], Lambda_tab[:,1]) * (f_H**2)  # The f_H^2 factor takes care of the convention in Katz+96, which is where the Lambda table comes from. 
    tstar0 *= (1e9 * 60*60*24*365.25) # input units are Gyr.  Convert to s.
    n_H_th = x_th/(1.-x_th)**2. * (beta*u_SN - (1-beta)*u_c)/(tstar0*Lambda/1.6726219e-24) * 1e4 # factor 1e4 for energy conversion to cgs.  Proton mass used here to go from cgs density to [m_p cm^-3]
#    print 'SH03 critical density', n_H_th, 'm_proton / cm^3'
    A = A0 * (n_H / n_H_th)**-0.8
    u_h = u_SN / (1.+A) + u_c
    fneutral = (u_h - u) / (u_h - u_c)
    fneutral[fneutral>1.0] = 1.0 # numerical errors could give answer just above 1.0 otherwise
    fneutral[fneutral<0.0] = 0.0
    return fneutral

def neutralFraction_from_electronFraction(u, EA, SFR):
    """
        Approximate the neutral fraction of non-star-forming cells using the ElectronAbundance and the InternalEnergy fields.  Based on a pre-determined fitting function
    """
    lp  = [0.9856, -0.9066, 0.9947, 1.8542, -4.5616]
    lp2 = [0.9840, -0.9962, 0.9980, 0.8588, -1.9429]
    main = (SFR<=0) * (u < 10**8.7)
    fneutral = np.zeros(len(u))
    fneutral[main] = piecewise_linear_parabola(EA[main], *lp)
    fneutral[~main] = piecewise_linear_parabola(EA[~main], *lp2)
    fneutral[fneutral<0] = 0.
    return fneutral



def interp_polytonic(x, xp, yp):
    """
        Similar to np.interp, except one doesn't require yp to be a monotonic function of xp.  Will find the largest value of yp that could correspond to x.  This can be used for finding the HI radii of galaxies, for example.  Currently just works for a single value x.  Intend to update once demand is present.  Assumes yp is in increasing order.
    """
    w = np.where(xp>x)[0]
    if len(w)==0:
        return 0. # Zero retunred if x > all xp, given one can't really extrapolate something polytonic
    if np.max(yp)==np.max(yp[w]):
        return np.max(yp)
    else:
        ymax = np.max(yp[w])
        i = np.where(yp==ymax)[0][0]
        dx = xp[i+1] - xp[i]
        return yp[i]*(xp[i+1]-x)/dx + yp[i+1]*(x-xp[i])/dx


def histogram(x, bins, weights=None):
    Nbin = len(bins)-1
    hist = np.zeros(Nbin, dtype=np.int32) if weights==None else np.zeros(Nbin, dtype=np.float32)
    for i in range(Nbin):
        f = (x>=bins[i]) * (x<bins[i]) if i!=Nbin-1 else (x>=bins[i]) * (x<=bins[i])
        hist[i] = len(x[f]) if weights==None else np.sum(weights[f])
    return hist


def comoving_distance(z, H_0=67.74, Omega_R=0, Omega_M=0.3089, Omega_L=0.6911):
    # calculate co-moving distance from redshift [Mpc]
    zprime = np.linspace(0,z,10000)
    E = np.sqrt(Omega_R*(1+zprime)**4 + Omega_M*(1+zprime)**3 + Omega_L)
    integrand = 1/E
    integral = np.sum(0.5*(integrand[1:]+integrand[:-1])*np.diff(zprime))
    c = 299792.458
    return c/H_0 * integral
    
    
def survey_volume(RA, dec, z, H_0=67.74, Omega_R=0, Omega_M=0.3089, Omega_L=0.6911):
    # calculate comoving survey volume from RA, dec, z ranges.  Each input is a list with 2 entries.
    dmin = comoving_distance(z[0], H_0, Omega_R, Omega_M ,Omega_L) * (1+z[0])**2
    dmax = comoving_distance(z[1], H_0, Omega_R, Omega_M ,Omega_L) * (1+z[0])**2
    vol = 1./3. * abs(RA[1]-RA[0])*np.pi/180 * abs(np.sin(dec[1]*np.pi/180)-np.sin(dec[0]*np.pi/180)) * abs(dmax**3 - dmin**3)
    return vol
    
def Mvir2Rvir(mass, crit_fac=200., z=0, H_0=67.74, Omega_R=0, Omega_M=0.3089, Omega_L=0.6911):
    # convert virial mass [Msun] to virial radius [kpc]
    vol = mass / critdens(z,H_0,Omega_R,Omega_M,Omega_L) / crit_fac
    return (3.*vol/4./np.pi)**(1./3.) * 1e-3
    
def Mvir2Vvir(mass, crit_fac=200., z=0, H_0=67.74, Omega_R=0, Omega_M=0.3089, Omega_L=0.6911):
    # convert virial mass [Msun] to virial velocity [km/s]
    Rvir = Mvir2Rvir(mass, crit_fac, z, H_0, Omega_R, Omega_M, Omega_L)
    return np.sqrt(6.67408e-11 * mass*1.989e30 / (Rvir*3.0857e19))*1e-3
    
def Mvir2tdyn(mass, crit_fac=200., z=0, H_0=67.74, Omega_R=0, Omega_M=0.3089, Omega_L=0.6911):
    # convert virial mass [Msun] to dynamical time [Gyr]
    t = Mvir2Rvir(mass, crit_fac, z, H_0, Omega_R, Omega_M, Omega_L) * 3.0857e16 / Mvir2Vvir(mass, crit_fac, z, H_0, Omega_R, Omega_M, Omega_L) / (60**2 * 24 * 365.24 * 1e9)
    
def Rvir2Mvir(R, crit_fac=200., z=0, H_0=67.74, Omega_R=0, Omega_M=0.3089, Omega_L=0.6911):
    # converts virial radius [kpc] to virial mass [Msun]
    vol = 4./3. * np.pi * R*R*R
    density = critdens(z,H_0,Omega_R,Omega_M,Omega_L)*1e9 * crit_fac
    return density * vol
    
def Rvir2Vvir(R, crit_fac=200., z=0, H_0=67.74, Omega_R=0, Omega_M=0.3089, Omega_L=0.6911):
    # converts virial radius [kpc] to virial velocity [km/s]
    Mvir = Rvir2Mvir(R, crit_fac, z, H_0, Omega_R, Omega_M, Omega_L)
    return np.sqrt(6.67408e-11 * Mvir*1.989e30 / (R*3.0857e19))*1e-3

def integrand_HIprof_model1(r_norm, rb, Sigma_0):
    # These HI profiles refer to my size--mass paper of 2019
    Sigma_c = 1.0 # Msun/pc^2
    Sigma_HI = np.ones(len(r_norm))*Sigma_0
    rs =  (1-rb) / np.log(Sigma_0/Sigma_c) if rb<1 else 0.
    fexp = np.where(r_norm>rb)[0]
    Sigma_HI[fexp] *= np.exp(-(r_norm[fexp]-rb)/rs)
    return r_norm * Sigma_HI
    
def integrand_HIprof_model2(r_norm, rb, Sigma_0):
    Sigma_c = 1.0 # Msun/pc^2
    Sigma_HI = np.ones(len(r_norm))*Sigma_0
    rS =  (1-rb) / np.sqrt(np.log(Sigma_0/Sigma_c)) if rb<1 else 0.
    fexp = np.where(r_norm>rb)[0]
    Sigma_HI[fexp] *= np.exp(-((r_norm[fexp]-rb)/rS)**2)
    return r_norm * Sigma_HI

def logHIprof_model3(r_norm, rd, delta_logSigma):
    rd, delta_logSigma = limits_model3(rd, delta_logSigma)
    logSigma_0H = delta_logSigma + np.log10(np.exp(1./rd))
    Sigma_0H = 10**logSigma_0H
    Sigma_HI = Sigma_0H * np.exp(-r_norm / rd) / (1 + Sigma_0H*np.exp((0.6-1.6*r_norm)/rd) - np.exp(1.6*(1-r_norm)/rd))
    return np.log10(Sigma_HI)

def integrand_HIprof_model3(r_norm, rd, delta_logSigma):
    return r_norm * 10**logHIprof_model3(r_norm, rd, delta_logSigma)

def limits_model3(rd_in, delta_logSigma_in):
    Sigma_0H = 10**delta_logSigma_in * np.exp(1./rd_in)
#    if Sigma_0H<4.2193: print 'Sigma_0H reset from', Sigma_0H, 'to 4.2193'
    Sigma_0H = max(Sigma_0H, 4.2193) # enforce limit
    rd = max(rd_in, 1./np.log(Sigma_0H))# enforce limit
    rd = min(rd, -1./np.log((2.6 + Sigma_0H**0.6)/(1.6*Sigma_0H - Sigma_0H**0.4))) # enforce limit
    delta_logSigma = np.log10(Sigma_0H * np.exp(-1./rd))
    if not np.isfinite(delta_logSigma): delta_logSigma = min(1e-10, delta_logSigma_in)  # set to small number in case the exponential blows up
#    if not np.allclose(rd_in, rd): 
#        print '\nrd changed from', rd_in, 'to', rd
#    if not np.allclose(delta_logSigma_in, delta_logSigma): 
#        print '\ndelta_logSigma changed from', delta_logSigma_in, 'to', delta_logSigma
#        print 'rd_in, rd, exp(-1/rd) =', rd_in, rd, np.exp(-1./rd)
    return rd, delta_logSigma
    
def mHI_model1(rb, Sigma_0, rHI, Sigma_c=1.):
    rs =  (1-rb) / np.log(Sigma_0/Sigma_c)
    if type(rs)==np.ndarray: rs[rb==1] = 0
    return np.pi * Sigma_0 * (rb**2 + 2*rs*(rs+rb)) * rHI**2

def mHI_model2(rb, Sigma_0, rHI, Sigma_c=1.):
    rS =  (1-rb) / np.sqrt(np.log(Sigma_0/Sigma_c))
    return np.pi * Sigma_0 * (rb**2 + rS*(rS+np.sqrt(np.pi)*rb)) * rHI**2

def mHI_model3(rd, deltaLogSigma_0, rHI):
    # Assuming Sigma_0 is normalised by Msun/pc^2
    import mpmath as mm
    logSigma_0H = deltaLogSigma_0 + np.log10(np.exp(1./rd))
    Sigma_0 = 10**logSigma_0H
    a1, a3 = 0.625, 0.625
    a2 = 1.0
    b1, b2 = 1.625, 1.625
    if type(rd)==np.ndarray and type(Sigma_0)==np.ndarray:
        hyper = np.zeros(len(rd))
        for i in range(len(rd)):
            c = np.exp(1.6/rd[i]) - Sigma_0[i]*np.exp(0.6/rd[i])
            if np.isfinite(c): hyper[i] = mm.hyper([a1,a2,a3],[b1,b2],c)
    elif type(rd)==np.ndarray:
        hyper = np.zeros(len(rd))
        for i in range(len(rd)):
            c = np.exp(1.6/rd[i]) - Sigma_0*np.exp(0.6/rd[i])
            if np.isfinite(c): hyper[i] = mm.hyper([a1,a2,a3],[b1,b2],c)
    elif type(Sigma_0)==np.ndarray:
        hyper = np.zeros(len(Sigma_0))
        for i in range(len(Sigma_0)):
            c = np.exp(1.6/rd) - Sigma_0[i]*np.exp(0.6/rd)
            if np.isfinite(c): hyper[i] = mm.hyper([a1,a2,a3],[b1,b2],c)
    else:
        c = np.exp(1.6/rd) - Sigma_0*np.exp(0.6/rd)
        hyper = mm.hyper([a1,a2,a3],[b1,b2],c)
    
    return 1.60769 * np.pi * Sigma_0 * rd**2 * hyper * rHI**2
    
def interp_2Darray(xval, xarr, yarr, rising=True):
    # Linearly interpolate across each row of a 2D array with a corresponding 2D array for the same value
    # Currently assumes the function is always increasing at the point where it is to be interpolated
    (nrow, ncol) = xarr.shape
    (row, col) = np.where(xarr<xval) if rising else np.where(xarr>xval)
    filt = np.append(np.diff(row)>0, True)
    row, col = row[filt], col[filt]
    yout = np.zeros(nrow)
    row, col = row[col<ncol-1], col[col<ncol-1]
    gradient = (yarr[row,col+1]-yarr[row,col]) / (xarr[row,col+1]-xarr[row,col])
    yout[row] = yarr[row,col] + gradient * (xval - xarr[row,col])
    return yout


def interp_2Darray_with_1Darray(xinterp, xarr, yarr, rising=True):
    # as interp_2Darray but feeding in an array of x values to interpolate, rather than just 1
    (nrow, ncol) = xarr.shape
    yout = np.zeros(( nrow, len(xinterp) ))
    for i, xval in enumerate(xinterp):
        yout[:,i] = interp_2Darray(xval, xarr, yarr, rising=rising)
    return yout
    
def alpha_CO(logOH, logSFR, logMstar, z, h):
    # Calcuate the conversion factor for CO(1-0) luminosity to H2 mass, based on Accurso et al. (2017, eq 25).  Uncertainty introduced is 0.165 dex.
    
    z_MS, slope_MS, norm_MS = gr.Pearson_MS(h) # Evolution of main sequence according to Person+18
    slopes = np.interp(z, z_MS, slope_MS)
    norms = np.interp(z, z_MS, norm_MS)
    delta_MS = logSFR - (slopes*(logMstar-10.5) + norms)

    log_alphaCO = 14.752 - 1.623*logOH + 0.062*delta_MS
    alphaCO = 10**log_alphaCO * 0.76 # get rid of helium contribution
    return alphaCO
    
    
def H2_from_CO(LCO, logSFR, logMstar, z, h, uncerts_logLCO=[0,0], uncerts_logMstar=[0,0]):
    # covert CO luminosity to H2 mass, provided other properties are supplied.  Assumes LCO is for the 1->0 transition and is in units of K*km*pc^2/s.  Uses gas fraction and redshift to approximate metallicity using IllustrisTNG, then applies the conversion factor of Accurso et al. (2017, eq 25).  Uncerts should be lists of 2 arrays/values.
    
    z_MS, slope_MS, norm_MS = gr.Pearson_MS(h) # Evolution of main sequence according to Person+18
    i_z = np.searchsorted(z_MS, z)
    
    MS_low = slope_MS[i_z]*(logMstar-10.5) + norm_MS[i_z]
    MS_high = slope_MS[i_z+1]*(logMstar-10.5) + norm_MS[i_z+1]
    MS_interp = MS_low + (z-z_MS[i_z]) * (MS_high-MS_low)/(z_MS[i_z+1]-z_MS[i_z])
    delta_MS = logSFR - MS_interp

    # momentarily hard-coding this read-in for ease
    Zz_array = np.loadtxt('/Users/adam/Illustris/zhigh_gasprops/MassMetFits.txt')[::-1,:]

#    RHS1 = 14.752 + 0.062*delta_MS
#    coeff_gf = 1.2
#    RHS2 = 8.1 + 0.2*(logMstar-8.) + 1.5*np.log10(0.5*(1+z)) - coeff_gf*(np.log10(LCO)-logMstar)
#    logOH = (RHS1 - RHS2/coeff_gf) / (1.623 - 1./coeff_gf)

    i_z = np.searchsorted(Zz_array[:,0], z)
#    print i_z
    try:
        i_z[i_z==len(Zz_array[:,0])] -= 1
    except TypeError:
        if i_z==len(Zz_array[:,0]): i_z -= 1
    b0, b1, b2 = Zz_array[i_z,1], Zz_array[i_z,2], Zz_array[i_z,3]
    logOH_lowz = ( b1*(14.752 + 0.062*delta_MS) +
                 (b2 + b0*logMstar + b1*(np.log10(LCO)-logMstar)) ) / (1.623*b1+1.)
    uncert_low_lowz = np.sqrt((((b0-b1)**2 + (0.062*b1)**2)*uncerts_logMstar[0]**2 + (b1*uncerts_logLCO[0])**2)/(1+1.623*b1)**2 + Zz_array[i_z,6]**2)
    uncert_upp_lowz = np.sqrt((((b0-b1)**2 + (0.062*b1)**2)*uncerts_logMstar[1]**2 + (b1*uncerts_logLCO[1])**2)/(1+1.623*b1)**2 + Zz_array[i_z,6]**2)
    
#    uncert_MH2_low_lowz = np.sqrt(((b0-b1)**2 + (0.062*b1)**2)*uncerts_logMstar[0]**2/(1+1.623*b1)**2 + (b1/(1+1.623*b1)+1)**2*uncerts_logLCO[0]**2 + Zz_array[i_z,6]**2)
#    uncert_MH2_upp_lowz = np.sqrt(((b0-b1)**2 + (0.062*b1)**2)*uncerts_logMstar[1]**2/(1+1.623*b1)**2 + (b1/(1+1.623*b1)+1)**2*uncerts_logLCO[1]**2 + Zz_array[i_z,6]**2)
    
    try:
        b0, b1, b2 = Zz_array[i_z+1,1], Zz_array[i_z+1,2], Zz_array[i_z+1,3]
        logOH_highz = ( b1*(14.752 + 0.062*delta_MS) + 
                      (b2 + b0*logMstar + b1*(np.log10(LCO)-logMstar)) ) / (1.623*b1+1.)
        uncert_low_highz = np.sqrt((((b0-b1)**2 + (0.062*b1)**2)*uncerts_logMstar[0]**2 + (b1*uncerts_logLCO[0])**2)/(1+1.623*b1)**2 + Zz_array[i_z+1,6]**2)
        uncert_upp_highz = np.sqrt((((b0-b1)**2 + (0.062*b1)**2)*uncerts_logMstar[1]**2 + (b1*uncerts_logLCO[1])**2)/(1+1.623*b1)**2 + Zz_array[i_z+1,6]**2)
        zratio = (z-Zz_array[i_z,0])/(Zz_array[i_z+1,0]-Zz_array[i_z,0])
        logOH = logOH_lowz + zratio * (logOH_highz-logOH_lowz)
        logOH_uncert_low = np.maximum(uncert_low_highz, uncert_low_lowz) # conservatively take the max of the uncerts for the 2 redshifts
        logOH_uncert_upp = np.maximum(uncert_upp_highz, uncert_upp_lowz)

        # repeat for uncert in MH2 -- note this has to be done like this to not incorporate the uncert in LCO twice like it's independent each time
#        uncert_MH2_low_highz = np.sqrt(((b0-b1)**2 + (0.062*b1)**2)*uncerts_logMstar[0]**2/(1+1.623*b1)**2 + (b1/(1+1.623*b1)+1)**2*uncerts_logLCO[0]**2 + Zz_array[i_z+1,6]**2)
#        uncert_MH2_upp_highz = np.sqrt(((b0-b1)**2 + (0.062*b1)**2)*uncerts_logMstar[1]**2/(1+1.623*b1)**2 + (b1/(1+1.623*b1)+1)**2*uncerts_logLCO[1]**2 + Zz_array[i_z+1,6]**2)

#        logMH2_uncert_low = np.maximum(uncert_MH2_low_highz, uncert_MH2_low_lowz)
#        logMH2_uncert_upp = np.maximum(uncert_MH2_upp_highz, uncert_MH2_upp_lowz)


    except IndexError:
        logOH = 1.0*logOH_lowz
        logOH_uncert_low = 1.0*uncert_low_lowz
        logOH_uncert_upp = 1.0*uncert_upp_lowz
#        logMH2_uncert_low = 1.0*uncert_MH2_low_lowz
#        logMH2_uncert_upp = 1.0*uncert_MH2_upp_lowz

    # Get the approximate metallicity if they're on the other track (i.e. quiescent systems that have metallicity independent from gas fraction)
    c1, c2 = Zz_array[i_z,4], Zz_array[i_z,5]
    logOH_lowz = c1*logMstar + c2
    uncert_low_lowz = np.sqrt((c1*uncerts_logMstar[0])**2 + Zz_array[i_z,7]**2)
    uncert_upp_lowz = np.sqrt((c1*uncerts_logMstar[1])**2 + Zz_array[i_z,7]**2)
    try:
        c1, c2 = Zz_array[i_z+1,4], Zz_array[i_z+1,5]
        logOH_highz = c1*logMstar + c2
        logOH_alt = logOH_lowz + (z-Zz_array[i_z,0]) * (logOH_highz-logOH_lowz)/(Zz_array[i_z+1,0]-Zz_array[i_z,0])
        uncert_low_highz = np.sqrt((c1*uncerts_logMstar[0])**2 + Zz_array[i_z+1,7]**2)
        uncert_upp_highz = np.sqrt((c1*uncerts_logMstar[1])**2 + Zz_array[i_z+1,7]**2)
        logOH_alt_uncert_low = np.maximum(uncert_low_lowz, uncert_low_highz)
        logOH_alt_uncert_upp = np.maximum(uncert_upp_lowz, uncert_upp_highz)
    except IndexError:
        logOH_alt = 1.0*logOH_lowz
        logOH_alt_uncert_low = 1.0*uncert_low_lowz
        logOH_alt_uncert_upp = 1.0*uncert_upp_lowz

#    try:
#        logOH[logOH>9.2] = 9.2
#        logOH[logOH<7.2] = 7.2
#    except TypeError:
#        logOH = min(max(7.2, logOH), 9.2)
    
    log_alphaCO = 14.752 - 1.623*logOH + 0.062*delta_MS + np.log10(0.76) # no helium contribution
    log_MH2 = log_alphaCO + np.log10(LCO)

    log_alphaCO_alt = 14.752 - 1.623*logOH_alt + 0.062*delta_MS + np.log10(0.76) # no helium contribution
    log_MH2_alt = log_alphaCO_alt + np.log10(LCO)


    # If gas fraction is outside the nominal fitted range for metallicity, use the independent estimate
    logGF = log_MH2 - logMstar
    try:
        f = (logGF>0.5) + (logGF<-2)
        log_MH2[f] = log_MH2_alt[f]
        logOH[f] = logOH_alt[f]
        logOH_uncert_low[f] = logOH_alt_uncert_low[f]
        logOH_uncert_upp[f] = logOH_alt_uncert_upp[f]
    except TypeError:
        if logGF>0.5 or logGF<-2:
            log_MH2 = 1.0*log_MH2_alt
            logOH = 1.0*logOH_alt
            logOH_uncert_low = 1.0*logOH_alt_uncert_low
            logOH_uncert_upp = 1.0*logOH_alt_uncert_upp

    log_MH2_uncerts = [np.sqrt((1.623*logOH_uncert_low)**2 + 0.165**2 + uncerts_logLCO[0]**2),
                       np.sqrt((1.623*logOH_uncert_upp)**2 + 0.165**2 + uncerts_logLCO[1]**2)]

#    log_MH2_sanity = -6.865 - 0.0995*delta_MS + 3.126*logMstar + 3.908*np.log10(0.5*(1+z)) - 1.605*np.log10(LCO)
#    print 'sanity check', log_MH2, log_MH2_sanity
    
#    return log_MH2, logOH, log_MH2_alt, logOH_alt
#    return log_MH2_alt, logOH_alt

    try:
        assert uncerts_logLCO==[0,0] 
        assert uncerts_logMstar==[0,0]
        return log_MH2, logOH
    except ValueError or AssertionError:
        return log_MH2, logOH, log_MH2_uncerts
        
        
def fit_divide_SFMS(mass_stars, SFR, sSFR_init = 10**-10.5):
    # Find and fit a star-forming main sequence, then find a dividing line for star-forming and quiescent galaxies
    logSM = np.log10(mass_stars)

    # fit SF main sequence
    bins = np.arange(np.min(logSM), np.max(logSM)+0.2, 0.2)
    f = (SFR >= sSFR_init * mass_stars) * (np.isfinite(SFR)) * (logSM<10.5)
    f2 = ~f * np.isfinite(SFR) * (SFR>0)
    p = np.polyfit(np.log10(mass_stars[f]), np.log10(SFR[f]), 1)
    p2 = np.polyfit(np.log10(mass_stars[f2]), np.log10(SFR[f2]), 1)

    nbins = len(bins)-1

    p3 = np.array([2.,2.])
    p3_old = np.array([1.,1.])

    its = 0
    while not np.allclose(p3, p3_old, rtol=1e-2, atol=1e-5):
        its += 1
        p3_old = p3
        minima = -40*np.ones(nbins)
        for i in range(nbins):
            f = (mass_stars>=10**bins[i]) * (mass_stars<10**bins[i+1]) * (np.log10(SFR)<= p[0]*np.log10(mass_stars)+p[1]) * (np.log10(SFR)>= p2[0]*np.log10(mass_stars)+p2[1])
            counts, edges = np.histogram(np.log10(SFR[f]), bins=12)
            grad = abs(np.diff(counts))
            gradgrad = np.diff(np.diff(counts))
            try:
                w = np.where(counts==np.min(counts))[0][0]
                minima[i] = 0.5*(edges[w] + edges[w+1])
            except:
                pass

        bincen = 0.5*(bins[1:]+bins[:-1])
        p3 = np.polyfit(bincen[minima>-40], minima[minima>-40], 1)
        sf = (np.log10(SFR) >= p3[0]*np.log10(mass_stars) + p3[1])
        if its>990: print(its, np.allclose(p3, p3_old, rtol=1e-2, atol=1e-5), p3, p3_old, chisqr(minima[minima>-40], p3[0]*bincen[minima>-40]+p3[1]))
        if its==1000: break
        
        f = (SFR >= sSFR_init * mass_stars) * (np.isfinite(SFR)) * (logSM<10.5)
        f2 = ~f * np.isfinite(SFR) * (SFR>0)
        p = np.polyfit(np.log10(mass_stars[f]), np.log10(SFR[f]), 1)
        p2 = np.polyfit(np.log10(mass_stars[f2]), np.log10(SFR[f2]), 1)

    # parameters for main sequence, red sequence, division line
    return p, p2, p3
    
def reset_periodic_positions(pos, Lbox):
    # Shift all particles in a periodic simuation so they are all contained within +/- half the box length
    while len(pos[pos<-0.5*Lbox])>0:
        pos[pos<-0.5*Lbox] += Lbox
    while len(pos[pos>0.5*Lbox])>0:
        pos[pos>0.5*Lbox] -= Lbox
    return pos
    
    
def timestring(hours):
    # convert a float of total hours into a string of 'hh:mm:ss'
    int_hours = int(hours)
    str_hours = str(int_hours)
    if len(str_hours)==1: str_hours = '0'+str_hours
    residual_minutes = (hours-int_hours)*60.
    int_minutes = int(residual_minutes)
    str_minutes = str(int_minutes)
    if len(str_minutes)==1: str_minutes = '0'+str_minutes
    residual_seconds = (residual_minutes-int_minutes)*60.
    str_seconds = str(int(residual_seconds))
    if len(str_seconds)==1: str_seconds = '0'+str_seconds
    str_time = str_hours+':'+str_minutes+':'+str_seconds
    return str_time
    
    
def bootstrap_percentiles(sample, Nboot=100000, sample_pciles=[16,50,84], boot_pciles=[16,84]):
    # Calculate bootstrap uncertainties of specific percentiles (sample_pciles) of a sample of data.  The uncertainties will be the percentiles (boot_pciles) of the bootstrapped sample percentiles.
    Nsample = len(sample)
    resample_array = sample[np.random.randint(Nsample, size=(Nsample,Nboot))]
    resampled_pciles = np.percentile(resample_array, sample_pciles, axis=0)
    pcile_pciles = np.percentile(resampled_pciles, boot_pciles, axis=1) # columns refer to the sample percentile of interest.  Rows refer to the bootstrapped percentile of that percentile.
    return pcile_pciles


def ks_weighted(data1, data2, wei1, wei2):
    # Shamelessly copied from https://stackoverflow.com/questions/40044375/how-to-calculate-the-kolmogorov-smirnov-statistic-between-two-weighted-samples
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1)/sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2)/sum(wei2)])
    cdf1we = cwei1[[np.searchsorted(data1, data, side='right')]]
    cdf2we = cwei2[[np.searchsorted(data2, data, side='right')]]
    return np.max(np.abs(cdf1we - cdf2we))


def cross_product(a, b):
    # Perform a cross product on a row-by-row basis for arrays a and b.  E.g. an array of Nx3 positions and Nx3 velocities (reprenting N particles) could be fed as a and b to return the specific angular momentum in each dimension of each particle.
    Npart = len(a)
    assert a.shape == (Npart, 3)
    assert b.shape == (Npart, 3)
    cross = np.zeros((Npart,3))
    cross[:,0] = a[:,1]*b[:,2] - a[:,2]-b[:,1]
    cross[:,1] = a[:,2]*b[:,0] - a[:,0]-b[:,2]
    cross[:,2] = a[:,0]*b[:,1] - a[:,1]-b[:,0]
    return cross


def build_data_cube(x, y, vz, m, rad, vtherm, xmax, vmax, pixel_size, vel_res, beam=0., Nstd=3):
    """
    # Build a data cube from simulation particle data
    
    Inputs:
    x = array of particle x-coordinates
    y = array of particle y-coordinates
    vz = array of particle velocities in the z-direction
    m = array of particle masses
    rad = radius of each particle
    vtherm = thermal velocity of each particle
    xmax = cube will cover x and y space from -xmax to +xmax
    vmax = cube will cover vz space from -vmax to +vmax
    pixel_size = square xy dimensions of each spaxel
    vel_res = velocity channel width in vz dimension
    beam = beam size to further smooth cube (optional), can be a list of 2 numbers for different smoothing in each x and y dimension) -- STANDARD DEVIATION, NOT FWHM
    Nstd = number of standard deviations where smoothing Gaussians are truncated
    """
    
    from scipy.ndimage import gaussian_filter
    
    # make sure 'beam' is a length-2 list (or equivalent)
    try:
        len(beam)
    except TypeError:
        if beam>0:
            beam = [beam, beam]
            
    xedge = np.arange(-xmax, xmax+pixel_size, pixel_size)
    vedge = np.arange(-vmax, vmax+vel_res, vel_res)
    Nchan = len(vedge)-1 # number of velocity channels
    Npix = len(xedge)-1 # number of pixels in each dimension

    # Initialise data cube
    data_cube = np.zeros((Npix,Npix,Nchan))
    
    # Filter out any particles that definitely won't contribute to the cube
    f = (abs(x) - rad - Nstd*beam[0] < xmax) * (abs(y) - rad - Nstd*beam[1] < xmax) * (abs(vz) - Nstd*vtherm < vmax) * (m>0)
    x = x[f]
    y = y[f]
    vz = vz[f]
    m = m[f]
    rad = rad[f]
    vtherm = vtherm[f]
    
    # Cell indices that each element will end up in
    ii = ((x+xmax)/pixel_size).astype(np.int32)
    ii[x+xmax<0] -= 1 # Any float in (-1,1) will be cast to 0 when converted to int, which is not what we want.
    ij = ((y+xmax)/pixel_size).astype(np.int32)
    ij[y+xmax<0] -= 1
    ik = ((vz+vmax)/vel_res).astype(np.int32)
    ik[vz+vmax<0] -= 1

    # Particles whose entirety is within the cube boundaries
    f_main = (abs(x) + rad + Nstd*beam[0] < xmax) * (abs(y) + rad + Nstd*beam[1] < xmax) * (abs(vz) + Nstd*vtherm < vmax)
    plist_main = np.where(f_main)[0]
    
    # Particles whose mass is partially within and partially outside the cube boundaries
    plist_part = np.where(~f_main)[0]
    print('gc.build_data_cube(): Number of clean cells to process', len(plist_main))
    print('gc.build_data_cube(): Number of boundary cells to process', len(plist_part))
    
    beam_in_pix = [beam[0]/pixel_size, beam[1]/pixel_size]
    
    # Processing the "main" particles first
    for p in plist_main:
        TwoD = np.zeros((Npix,Npix)) # Spatial distribution
        OneD = np.zeros(Nchan) # Velocity distribution
        i, j, k = ii[p], ij[p], ik[p] # get the indices where the particle ended up in the cube
        
        OneD[k] = 1.0
        OneD = gaussian_filter(OneD, vtherm[p]/vel_res, truncate=Nstd)
        
        kernel = sphere2dk(rad[p], pixel_size, 2*rad[p]/pixel_size)
        kr = (len(kernel)-1)/2 # kernel "radius" in pixels

        TwoD[i-kr:i+kr+1, j-kr:j+kr+1] = m[p] * kernel
        TwoD = gaussian_filter(TwoD, beam_in_pix, truncate=Nstd) # add beam-smearing
        
        single_cube = (OneD[np.newaxis][np.newaxis].T * TwoD.T).T
        data_cube += single_cube

    for p in plist_part:
        # guaranteed odd number of pixels in each dimension for cublet
        xcen = int((rad[p] + Nstd*beam[0])/pixel_size)
        ycen = int((rad[p] + Nstd*beam[1])/pixel_size)
        vcen = int(Nstd*vtherm[p]/vel_res)
        xpix = 2*xcen + 1
        ypix = 2*ycen + 1
        vpix = 2*vcen + 1
        
        TwoD = np.zeros((xpix,ypix)) # Spatial distribution
        OneD = np.zeros(vpix) # Velocity distribution
        
        OneD[vcen] = 1.0
        OneD = gaussian_filter(OneD, vtherm[p]/vel_res, truncate=Nstd)
        
        kernel = sphere2dk(rad[p], pixel_size, 2*rad[p]/pixel_size)
        kr = (len(kernel)-1)/2 # kernel "radius" in pixels
        
        TwoD[xcen-kr:xcen+kr+1, ycen-kr:ycen+kr+1] = m[p] * kernel
        TwoD = gaussian_filter(TwoD, beam_in_pix, truncate=Nstd) # add beam-smearing
        
        cubelet = (OneD[np.newaxis][np.newaxis].T * TwoD.T).T
        
        # which elements in the full cube does this particle contribute to?
        i, j, k = ii[p], ij[p], ik[p]
        ii_min, ii_max = i-xcen, i+xcen+1
        ij_min, ij_max = j-ycen, j+ycen+1
        ik_min, ik_max = k-vcen, k+vcen+1
        
        # which elements in the cubelet get passed into the full cube?
        ki_min, kj_min, kk_min = max(0, -ii_min), max(0, -ij_min), max(0, -ik_min)
        ki_max, kj_max, kk_max = min(xpix-(ii_max-Npix), xpix), min(ypix-(ij_max-Npix), ypix), min(vpix-(ik_max-Nchan), vpix)

        # close logic
        ii_min, ii_max = max(0, ii_min), min(Npix, ii_max)
        ij_min, ij_max = max(0, ij_min), min(Npix, ij_max)
        ik_min, ik_max = max(0, ik_min), min(Nchan, ik_max)

        data_cube[ii_min:ii_max, ij_min:ij_max, ik_min:ik_max] += cubelet[ki_min:ki_max, kj_min:kj_max, kk_min:kk_max] # note, not normalised by area or channel width!
        
    return data_cube


def integrate(x, integrand, xmin, xmax):
    # numerically integrate integrand as a function of x over the range xmin and xmax
    if xmin<np.min(x):
        xmin = np.min(x)
        print('gc.integrate(): xmin too small, reseting to', np.min(x))
    if xmax>np.max(x):
        xmax = np.max(x)
        print('gc.integrate(): xmax too large, reseting to', np.max(x))
   
    wmin = np.where(x==xmin)[0]
    if len(wmin)>0: 
        wmin = wmin[0]
        x = x[wmin:]
        integrand[wmin:]
    else:
        integrand_min = np.interp(xmin, x, integrand)
        wmin = np.where(x>xmin)[0][0]
        x = np.append(xmin, x[wmin:])
        integrand = np.append(integrand_min, integrand[wmin:])
        
    wmax = np.where(x==xmax)[0]
    if len(wmax)>0: 
        wmax = wmax[-1]
        x = x[:wmax+1]
        integrand = integrand[:wmax+1]
    else:
        integrand_max = np.interp(xmax, x, integrand)
        wmax = np.where(x<xmax)[0][-1]
        x = np.append(x[:wmax+1], xmax)
        integrand = np.append(integrand[:wmax+1], integrand_max)
        
    dx = np.diff(x)
    return 0.5 * np.sum((integrand[1:] + integrand[:-1]) * dx)


def Bardeen_radiative_efficiency(a):
    # calculate the radiative efficiency for a black spin with spin a
    Z1 = 1 + np.cbrt(1-a*a) * (np.cbrt(1+a) + np.cbrt(1-a))
    Z2 = np.sqrt(3*a*a + Z1*Z1)
    term = np.sqrt( (3 - Z1) * (3 + Z1 + 2*Z2) )
    rrat = 3 + Z2 - term if a>=0 else 3 + Z2 + term
    return 1 - np.sqrt(1 - 2./(3*rrat))


def return_fraction_and_SN_ChabrierIMF(m_min=0.1, m_max=100.0, A=0.84342328, k=0.23837777, m_c=0.08, sigma=0.69, ratio_Ia_II=0.2):
    # array of mass values covering the full range that stars are assumed to fall within
    m = np.linspace(m_min, m_max, 10001) # solar masses

    # Chabrier IMF
    IMF = k * m**(-2.3)
    flow = (m<=1)
    IMF[flow] = A/m[flow] * np.exp(-np.log10(m[flow]/m_c)**2/(2*sigma**2))

    # lifetimes of stars based on simple scaling + slope of the main sequence of the HR diagram, with the Sun assumed to have a lifetime of 10 Gyr
    lifetime = 10 * m**(-2.5) # Gyr

    # mass of remnants at the end of evolution
    m_remnant = 1.0*m
    f1, f2, f3 = (m>=1)*(m<=7), (m>7)*(m<8), (m>=8)*(m<=50)
    m_remnant[f1] = 0.444 + 0.084*m[f1]
    m_remnant[f2] = 0.419 + 0.109*m[f2]
    m_remnant[f3] = 1.4
    m_returned = m - m_remnant

    # fraction of returned mass, integrating from the highest mass down
    dm = m[1]-m[0]
    integrand = (m_returned*IMF)[::-1]
    returned_mass_fraction_integrated = 0.5*dm*np.cumsum(integrand[:-1] + integrand[1:])[::-1]
    returned_mass_fraction_integrated = np.append(returned_mass_fraction_integrated, 0)

    # integrate the IMF from the highest mass down to get the cumulative number density of stars
    int_IMF = k/1.3 * m**(-1.3)
    int_IMF -= np.interp(50, m, int_IMF)

    # cumulative number density of supernovae in a stellar population (integrating from the highest mass down)
    ncum_SN = np.zeros(len(m))
    ncum_SN[f3] = 1.0*int_IMF[f3]
    ncum_SN[f1+f2] = ncum_SN[f3][0] + ratio_Ia_II/15.3454 * (int_IMF[f1+f2] - int_IMF[f3][0])
    ncum_SN[m<1] = np.max(ncum_SN)

    return m, lifetime, returned_mass_fraction_integrated, ncum_SN


def NFW_potential(r, Mvir, Mstellar, z, Rvir=None, H_0=67.74, Omega_R=0, Omega_M=0.3089, Omega_L=0.6911):
    # input Mvir and Mstellar in units of [10^10 h^-1 Msun]
    a = 0.520 + (0.905-0.520)*np.exp(-0.617*z**1.21)
    b = -0.101 + 0.026*z
    c_DM = 10.0**(a+b*log10(Mvir*0.01))
    X = np.log10(Mstellar/Mvir)
    c = c_DM * (1.0 + 3e-5 * np.exp(3.4*(X+4.5)))
    h = 0.01 * H_0
    if Rvir is None:
        Rvir = Mvir2Rvir(Mvir*1e10/h, 200., z, H_0, Omega_R, Omega_M, Omega_L) * 1e-3 * h # converts to units of Mpc/h
    r_2 = Rvir / c

    UnitLength_in_cm = 3.08568e+24
    UnitVelocity_in_cm_per_s = 100000.0
    SEC_PER_MEGAYEAR = 3.1556736e13
    UnitMass_in_g = 1.989e+43
    UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
    UnitTime_in_Megayears = UnitTime_in_s / SEC_PER_MEGAYEAR
    GRAVITY = 6.672e-8
    G = GRAVITY / UnitLength_in_cm**3 * UnitMass_in_g * UnitTime_in_s**2

    pot_energy = - G * Mvir / r * log(1.0 + r/r_2) / (log(1.0+Rvir/r_2) - Rvir/(Rvir+r_2))
    return pot_energy
