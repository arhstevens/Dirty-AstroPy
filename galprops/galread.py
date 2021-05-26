# Adam Stevens, 2013-2020
# Functions for reading in data files from simulations and models

import numpy as np
import math
import sys
import galcalc as gc
import os

sys.path.insert(0, '..')
from read import read_marie
from read import particles
from time import time


# ========== Marie's simulations ==========

def readstars(filename,Lbox=800000.):
	# Read in stellar data from one of Marie's simulation snapshots
	file = read_marie.MarieParticleFile(filename)
	data = file.read()
	time = file.time+1368  # Time since beginning of the Universe in the snapshot in Myr
	
	particles = data[1]
	x, y, z = particles["pos"][:,0]-Lbox/2., particles["pos"][:,1]-Lbox/2., particles["pos"][:,2]-Lbox/2. # Exract positions
	vx, vy, vz = particles["vel"][:,0], particles["vel"][:,1], particles["vel"][:,2] # Extract velocities
	mass = particles["mass"]*1e9 # Extract masses
	birth=particles["birth"]+1368
	
	x,y,z,vx,vy,vz = recentregal(x,y,z,vx,vy,vz,mass) # Computes centre of mass for particles within a given radius and shifts all particles so the COM is at 0,0
	
	filter_rad = (np.sqrt(x**2+y**2+z**2)<20000.) # Creates a logic matrix based on the set condition
	axis, angle = gc.compute_rotation_to_z(x[filter_rad],y[filter_rad],z[filter_rad],vx[filter_rad],vy[filter_rad],vz[filter_rad],mass[filter_rad]) # Axis and angle of rotation to bring galaxy in the xy plane
	
	x,y,z = rotate(x,y,z,axis,angle) # Rotate positions so z is normal to the disk
	vx,vy,vz = rotate(vx,vy,vz,axis,angle) # Ditto velocities
	
	return x,y,z,vx,vy,vz,mass,birth,time




def readall(filename,Lbox=800000.,band=''):
	# Expansion of readstars to read stars, gas and dm all at once.  L must be a string that is the name of a filter
	file = read_marie.MarieParticleFile(filename)
	data = file.read()
	time = file.time+1368-237.85  # Time since beginning of the Universe in the snapshot in Myr
	
	particles = data[1]  # Stellar stuff
	x_s, y_s, z_s = particles["pos"][:,0]-Lbox/2., particles["pos"][:,1]-Lbox/2., particles["pos"][:,2]-Lbox/2. # Exract positions
	vx_s, vy_s, vz_s = particles["vel"][:,0], particles["vel"][:,1], particles["vel"][:,2] # Extract velocities
	mass = particles["mass"]*1e9 # Extract masses
	birth = particles["birth"]+1368-237.85
	id_s = particles["id"]
	hid_s = particles["app"] # This is the id that says which halo in the simulation the particle originates from
	
	x,y,z,vx,vy,vz = gc.recentregal(x_s,y_s,z_s,vx_s,vy_s,vz_s,mass) # Computes centre of mass for particles within a given radius and shifts all particles so the COM is at 0,0
	
	filter_rad = (np.sqrt(x**2+y**2+z**2)<20000.) # Creates a logic matrix based on the set condition
	axis, angle = gc.compute_rotation_to_z(x[filter_rad],y[filter_rad],z[filter_rad],vx[filter_rad],vy[filter_rad],vz[filter_rad],mass[filter_rad]) # Axis and angle of rotation to bring galaxy in the xy plane
	
	xrot,yrot,zrot = rotate(x,y,z,axis,angle) # Rotate positions so z is normal to the disk (note, it's important I change the variable name for translating the other particles)
	vxrot,vyrot,vzrot = rotate(vx,vy,vz,axis,angle) # Ditto velocities
	
	gas = data[0]
	x_g, y_g, z_g = gas["pos"][:,0]-Lbox/2., gas["pos"][:,1]-Lbox/2., gas["pos"][:,2]-Lbox/2. # Extract gas positions
	x_g, y_g, z_g = x_g-(x_s[0]-x[0]), y_g-(y_s[0]-y[0]), z_g-(z_s[0]-z[0]) # Recentre to match coords
	x_g, y_g, z_g = rotate(x_g,y_g,z_g,axis,angle) # Rotate to match coords
	vx_g, vy_g, vz_g = gas["vel"][:,0], gas["vel"][:,1], gas["vel"][:,2] # Extract gas velocities
	vx_g, vy_g, vz_g = vx_g-(vx_s[0]-vx[0]), vy_g-(vy_s[0]-vy[0]), vz_g-(vz_s[0]-vz[0])
	vx_g, vy_g, vz_g = rotate(vx_g,vy_g,vz_g,axis,angle)
	mass_g = gas["mass"]*1e9
	id_g = gas["id"]
	hid_g = gas["app"]
	
	dm = data[2]
	x_dm, y_dm, z_dm = dm["pos"][:,0]-Lbox/2., dm["pos"][:,1]-Lbox/2., dm["pos"][:,2]-Lbox/2.
	x_dm, y_dm, z_dm = x_dm-(x_s[0]-x[0]), y_dm-(y_s[0]-y[0]), z_dm-(z_s[0]-z[0]) # Recentre to match coords
	x_dm, y_dm, z_dm = rotate(x_dm,y_dm,z_dm,axis,angle) # Rotate to match coords
	vx_dm, vy_dm, vz_dm = dm["vel"][:,0], dm["vel"][:,1], dm["vel"][:,2] # Extract gas velocities
	vx_dm, vy_dm, vz_dm = vx_dm-(vx_s[0]-vx[0]), vy_dm-(vy_s[0]-vy[0]), vz_dm-(vz_s[0]-vz[0])
	vx_dm, vy_dm, vz_dm = rotate(vx_dm,vy_dm,vz_dm,axis,angle)
	mass_dm = dm["mass"]*1e9
	id_dm = dm["id"]
	hid_dm = dm["app"]
	
	# Add the ability to include luminosity calculations for particles off the bat
	if len(band)>0:
		Larr, ages, wl, metals = starpopmodel() # Read in stellar population model
		filt_wl, filt_resp = readfilter(band) # Read in filter
		Larr_bol = gc.bollum(Larr,wl) # Get bolometric luminosity array
		Larr_filt = gc.lumfiltarray(Larr,wl,filt_resp,filt_wl) # Get the luminosity array for the filter
		
		age_sp = (time-birth)*1e-3 # Age of the stellar particles from the sim in Gyr
		
		L_filt = mass*np.interp(age_sp,ages,Larr_filt[:,5]) # Luminosity of star particles in ergs for the band (perhaps rather a power than a luminosity).  Arbitrarily choosing solar metallicity.
		L_bol = mass*np.interp(age_sp,ages,Larr_bol[:,5]) # Bolometric (actual) luminosity
		
		return time,[xrot,yrot,zrot,mass,id_s,hid_s,birth,L_bol,L_filt],[vxrot,vyrot,vzrot],[x_g,y_g,z_g,mass_g,id_g,hid_g],[vx_g,vy_g,vz_g],[x_dm,y_dm,z_dm,mass_dm,id_dm,hid_dm]
	
	else:
		return time,[xrot,yrot,zrot,mass,id_s,hid_s,birth],[vxrot,vyrot,vzrot],[x_g,y_g,z_g,mass_g,id_g,hid_g],[vx_g,vy_g,vz_g],[x_dm,y_dm,z_dm,mass_dm,id_dm,hid_dm],[vx_dm,vy_dm,vz_dm]




def halomarie():
	# Read in the data file for tracking haloes in Marie's sims
	
	with open('cat_halo_106.dat','r') as f:
		data = f.read()
		ds = data.split()
	
	with open('CI_106.txt','r') as f2:
		data2 = f2.read()
		ds2 = data2.split()
    
	halo_id = np.array(ds2[0::11]+ds[0::12],dtype='int64') # The tag that particles that originated in this halo will have)
	halo_mass = np.array(ds2[1::11]+ds[8::12],dtype='float64') # Initial masses of those haloes
	halo_time = np.array([1368]*21+ds[1::12],dtype='float64') # Time the halo enters the simulation
	
	return halo_id, halo_mass, halo_time




def starmasses():
	# This is specifically to just read in stellar masses at various times as measured from my routines on Marie's sims, so I can quickly overplot with initial SAGE run on Mini Millennium.
	SM = np.array([[1.36500000e-03, 1.89225000e-01, 2.46090000e-01, 2.95305000e-01,5.24860500e+00, 7.61194500e+00, 1.14788250e+01, 1.26047400e+01, 1.36371000e+01, 1.43198100e+01, 1.57353750e+01, 1.59327450e+01, 1.66953000e+01, 1.82805150e+01, 2.19048900e+01, 2.26825800e+01, 2.35719300e+01, 2.58971550e+01, 2.85720000e+01, 2.98312350e+01, 3.10293300e+01, 3.18434850e+01, 3.32736750e+01, 3.49259850e+01, 3.68909850e+01, 3.85485750e+01, 3.98652300e+01, 4.01841150e+01, 4.14936750e+01, 4.20164400e+01, 4.24074600e+01, 4.34534850e+01, 4.40414100e+01, 4.44820950e+01], [1.29000000e-03, 0.00000000e+00, 5.16000000e-03, 1.41060000e-01, 2.23618500e+00, 4.07304000e+00, 1.08771150e+01, 1.24934400e+01, 1.35673350e+01, 1.42357200e+01, 1.56500400e+01, 1.58961000e+01, 1.65738450e+01, 1.82425350e+01, 2.18853300e+01, 2.26221150e+01, 2.34948900e+01, 2.57975850e+01, 2.85128250e+01, 2.97637050e+01, 3.09915300e+01, 3.18215550e+01, 3.32608800e+01, 3.49132800e+01, 3.68909850e+01, 3.86000550e+01, 3.98652300e+01, 4.01990250e+01, 4.15089300e+01, 4.20449850e+01, 4.25214450e+01, 4.35375450e+01, 4.42990800e+01, 4.45722150e+01], [1.44000000e-03, 2.39565000e-01, 3.01650000e-01, 5.58420000e-01, 5.43420000e+00, 8.04978000e+00, 1.20752700e+01, 1.37830800e+01, 1.47596850e+01, 1.83183900e+01, 1.95954450e+01, 1.73804550e+01, 2.12935200e+01, 2.29672050e+01, 2.35707750e+01, 2.40619950e+01, 2.51572950e+01, 2.72990850e+01, 2.91978750e+01, 3.08887800e+01, 3.21947400e+01, 3.31042950e+01, 3.45831300e+01, 3.56451900e+01, 3.85461000e+01, 3.97102800e+01, 4.03403550e+01, 4.09227450e+01, 4.22874450e+01, 4.25019000e+01, 4.32506100e+01, 4.46540250e+01, 4.46887050e+01, 4.49325300e+01], [1.36500000e-03, 9.59250000e-02, 1.37520000e-01, 4.26435000e-01, 5.23101000e+00, 7.84557000e+00, 1.14476700e+01, 1.35689400e+01, 1.39885800e+01, 1.76025600e+01, 1.61974800e+01, 1.65515850e+01, 2.04745200e+01, 2.20443450e+01, 2.24232450e+01, 2.36921550e+01, 2.41521900e+01, 2.64525300e+01, 2.88824850e+01, 3.02156700e+01, 3.13375950e+01, 3.24053400e+01, 3.35921550e+01, 3.52225050e+01, 3.70274250e+01, 3.85284750e+01, 3.98384100e+01, 4.00103850e+01, 4.13353950e+01, 4.19574750e+01, 4.21932000e+01, 4.31654550e+01, 4.38831900e+01, 4.38709950e+01]])
	t = np.array([1.368+0.375*i for i in xrange(33)]+[13.7055]) # Values of time at each step
	return np.transpose(SM), t



def readtestout(dir='testout/run16/'):
	# This should read in the data I output from test.py.
	
	with open(dir+'data.txt','r') as f:
		fred = f.read() # Read in file (called fred...)
		ds = fred.split() # Turns all the info into a list of strings
		head = ds[:70] # Header
		data = np.array(ds[70:],dtype='float64') # Actual data
	
	time = data[0::70]
	redshift = data[1::70]
	r_vir = data[2::70]
	tdmm = data[3::70]
	dmar = data[4::70]
	dmer = data[5::70]
	dsm = data[6::70]
	bsm = data[7::70]
	pfit0 = data[8::70]
	dsr = data[9::70]
	pfit2 = data[10::70]
	pfit3 = data[11::70]
	n = data[12::70]
	btt = data[13::70]
	r_rel = data[14::70]
	r_abs = data[15::70]
	r_rel_g = data[16::70]
	r_abs_g = data[17::70]
	
	tsm = np.transpose(np.array([data[18::70],data[19::70],data[20::70],data[21::70]]))
	sfr = np.transpose(np.array([data[22::70],data[23::70],data[24::70],data[25::70]]))
	sdr = np.transpose(np.array([data[26::70],data[27::70],data[28::70],data[29::70]]))
	sar = np.transpose(np.array([data[30::70],data[31::70],data[32::70],data[33::70]]))
	ser = np.transpose(np.array([data[34::70],data[35::70],data[36::70],data[37::70]]))
	tgm = np.transpose(np.array([data[38::70],data[39::70],data[40::70],data[41::70]]))
	gar = np.transpose(np.array([data[42::70],data[43::70],data[44::70],data[45::70]]))
	ger = np.transpose(np.array([data[46::70],data[47::70],data[48::70],data[49::70]]))
	tbl = np.transpose(np.array([data[50::70],data[51::70],data[52::70],data[53::70]]))
	mag_bol = np.transpose(np.array([data[54::70],data[55::70],data[56::70],data[57::70]]))
	tvl = np.transpose(np.array([data[58::70],data[59::70],data[60::70],data[61::70]]))
	mag_v = np.transpose(np.array([data[62::70],data[63::70],data[64::70],data[65::70]]))
	sfe = np.transpose(np.array([data[66::70],data[67::70],data[68::70],data[69::70]]))
	
	return [time, redshift, r_vir, tdmm, dmar, dmer, dsm, bsm, [pfit0,dsr,pfit2,pfit3,n], btt, r_rel, r_abs, r_rel_g, r_abs_g, tsm, sfr, sdr, sar, ser, tgm, gar, ger, tbl, mag_bol, tvl, mag_v, sfe]


def r25(hno):
	# Read in the R25 radius values that Marie has provided for the galaxies
	fname = 'R25/R25_evol_'+str(hno)+'.txt'
	f = open(fname)
	data = f.read()
	ds = data.split()
	dsa = np.array(ds, dtype='float64')
	R25 = dsa[1::2]
	f.close()
	return R25


def test3out(hno,run=3):
	fname = 'testout/3run'+str(int(run))+'/halo_'+str(int(hno))+'/t3out'+str(int(hno))
	f = open(fname, 'rb')
	
	Ntech = 7 if run<3 else 17 # Number of techniques was increased in later runs
	
	time = np.fromfile(f, 'f8', 34) # Time of each snapshot in Gyr
	redshift = np.fromfile(f, 'f8', 34) # Redshift of each snapshot assuming cosmological parameters used in run of test3.py
	Radii = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # The radius for each entry in the following 2D arrays (from the 7 techniques)
	DMM = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Dark matter mass
	DMAR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Dark matter accretion rate
	DMER = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Dark matter ejection rate
	SM = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Stellar mass
	GM = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Gas mass
	SFR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Star formation rate
	SAR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Stellar accretion rate
	GAR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Gas accretion rate
	SDR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Star death rate
	SER = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Stellar ejection rate
	GER = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Gas ejection rate
	TBL = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Total bolometric luminosity
	TvL = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Total v-band luminosity
	mag_bol = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Bolometric magnitude
	mag_v = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # v-band magnitude
	f.close()
	return [time, redshift, Radii, DMM, DMAR, DMER, SM, GM, SFR, SAR, GAR, SDR, SER, GER, TBL, TvL, mag_bol, mag_v]


def test4out(hno,run=4):
	fname = 'testout/4out_'+str(hno)+'_'+str(run)
	f = open(fname, 'rb')
	
	if run<=2:
		Ntech = 17
		time = np.fromfile(f, 'f8', 34) # Time of each snapshot in Gyr
		redshift = np.fromfile(f, 'f8', 34) # Redshift of each snapshot assuming cosmological parameters used in run of test3.py
		Radii = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # The radius for each entry in the following 2D arrays (from the 7 techniques)
		DMM = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Dark matter mass
		DMAR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Dark matter accretion rate
		DMER = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Dark matter ejection rate
		SM = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Stellar mass
		CGM = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Cold gas mass
		HGM = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Hot mass
		SFR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Star formation rate
		SDR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Star death rate
		SAR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Stellar accretion rate
		CGAR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Cold gas accretion rate
		HGAR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Hot gas accretion rate
		SER = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Stellar ejection rate
		CGER = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Cold gas ejection rate
		HGER = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Hot gas ejection rate
		TBL = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Total bolometric luminosity
		TvL = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Total v-band luminosity
		mag_bol = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Bolometric magnitude
		mag_v = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # v-band magnitude
		f.close()
		return [time, redshift, Radii, DMM, DMAR, DMER, SM, CGM, HGM, SFR, SDR, SAR, CGAR, HGAR, SER, CGER, HGER, TBL, TvL, mag_bol, mag_v]
	else:
		Ntech = 18
		time = np.fromfile(f, 'f8', 34) # Time of each snapshot in Gyr
		redshift = np.fromfile(f, 'f8', 34) # Redshift of each snapshot assuming cosmological parameters used in run of test3.py
		Radii = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # The radius for each entry in the following 2D arrays (from the 7 techniques)
		DMM = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Dark matter mass
		DMAR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Dark matter accretion rate
		DMER = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Dark matter ejection rate
		SM = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Stellar mass
		CGM = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Cold gas mass
		HGM = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Hot mass
		SFR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Star formation rate
		SDR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Star death rate
		SAR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Stellar accretion rate
		CGAR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Cold gas accretion rate
		HGAR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Hot gas accretion rate
		SER = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Stellar ejection rate
		CGER = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Cold gas ejection rate
		HGER = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Hot gas ejection rate
		TBL = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Total bolometric luminosity
		TvL = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Total v-band luminosity
		TuL = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Total u-band luminosity
		TjL = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Total j-band luminosity
		TkL = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Total k-band luminosity
		TicL = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Total ic-band luminosity
		f.close()
		return [time, redshift, Radii, DMM, DMAR, DMER, SM, CGM, HGM, SFR, SDR, SAR, CGAR, HGAR, SER, CGER, HGER, TBL, TvL, TuL, TjL, TkL, TicL]


def test5out(hno,run=13,dir='testout/'):
	fname = dir+'5out_'+str(hno)+'_'+str(run)
	f = open(fname, 'rb')
	
	if run<4:
		Ntech = 15
	else:
		Ntech = np.fromfile(f, 'f8', 1)[0]
	Ntech = int(Ntech) # I'm a noob for not storing this as an integer in the first place -.-

	time = np.fromfile(f, 'f8', 34) # Time of each snapshot in Gyr
	redshift = np.fromfile(f, 'f8', 34) # Redshift of each snapshot assuming cosmological parameters used in run of test3.py
	Radii = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # The radius for each entry in the following 2D arrays (from the 7 techniques)
	DMM = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Dark matter mass
	DMAR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Dark matter accretion rate
	DMER = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Dark matter ejection rate
	SM = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Stellar mass
	CGM = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Cold gas mass
	HGM = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Hot mass
	SFR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Star formation rate
	SDR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Star death rate
	SAR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Stellar accretion rate
	CGAR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Cold gas accretion rate
	HGAR = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Hot gas accretion rate
	SER = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Stellar ejection rate
	CGER = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Cold gas ejection rate
	HGER = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Hot gas ejection rate
	TBL = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Total bolometric luminosity
	TvL = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Total v-band luminosity
	TuL = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Total u-band luminosity
	TjL = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Total j-band luminosity
	TkL = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Total k-band luminosity
	TicL = np.fromfile(f, np.dtype(('f8',Ntech)), 34) # Total ic-band luminosity
	
	if run<4:
		return [time, redshift, Radii, DMM, DMAR, DMER, SM, CGM, HGM, SFR, SDR, SAR, CGAR, HGAR, SER, CGER, HGER, TBL, TvL, TuL, TjL, TkL, TicL]
	elif run==5:
		Radii2 = np.fromfile(f, np.dtype(('f8',40)), 34)
		DMM2 = np.fromfile(f, np.dtype(('f8',40)), 34)
		SM2 = np.fromfile(f, np.dtype(('f8',40)), 34)
		CGM2 = np.fromfile(f, np.dtype(('f8',40)), 34)
		HGM2 = np.fromfile(f, np.dtype(('f8',40)), 34)
		f.close()
		return [time, redshift, Radii, DMM, DMAR, DMER, SM, CGM, HGM, SFR, SDR, SAR, CGAR, HGAR, SER, CGER, HGER, TBL, TvL, TuL, TjL, TkL, TicL], [Radii2, DMM2, SM2, CGM2, HGM2]

	else:
		Radii2 = np.fromfile(f, np.dtype(('f8',1000)), 34)
		DMM2 = np.fromfile(f, np.dtype(('f8',1000)), 34)
		SM2 = np.fromfile(f, np.dtype(('f8',1000)), 34)
		CGM2 = np.fromfile(f, np.dtype(('f8',1000)), 34)
		HGM2 = np.fromfile(f, np.dtype(('f8',1000)), 34)
		SFR2 = np.fromfile(f, np.dtype(('f8',1000)), 34)

		Radii3 = np.fromfile(f, np.dtype(('f8',1000)), 34)
		DMM3 = np.fromfile(f, np.dtype(('f8',1000)), 34)
		SM3 = np.fromfile(f, np.dtype(('f8',1000)), 34)
		CGM3 = np.fromfile(f, np.dtype(('f8',1000)), 34)
		HGM3 = np.fromfile(f, np.dtype(('f8',1000)), 34)
		SFR3 = np.fromfile(f, np.dtype(('f8',1000)), 34)
		return [time, redshift, Radii, DMM, DMAR, DMER, SM, CGM, HGM, SFR, SDR, SAR, CGAR, HGAR, SER, CGER, HGER, TBL, TvL, TuL, TjL, TkL, TicL], [Radii2, DMM2, SM2, CGM2, HGM2, SFR2], [Radii3, DMM3, SM3, CGM3, HGM3, SFR3]


def ahfhaloid(fpre, ids):
    """
    Read in the halo IDs output from AHF
    fpre is everything before the final .
    ids = either an array of particle IDs or a list of arrays (should be a superset of the IDs read in in this routine)
    Output hid will be returned as per the order of the input ids
    """
#    f = open(fpre+'.AHF_halos')
#    d = f.read()
#    ds = d.split()
#    Npart = np.array(ds[87::83],dtype='i8') # Obtain number of particles in each halo
#    f.close()
    #print Npart
    Npart = np.loadtxt(fpre+'.AHF_halos', skiprows=1, usecols=4, dtype=np.int64)
        
#    try:
#        f = open(fpre+'.AHF_particles')
#        d = f.read()
#    except IOError:
#        f = open(fpre+'.AHF_particles.gz')
#        d = f.read()
#    ds = np.array(d.split(),dtype='i8')
    try:
        ds = np.loadtxt(fpre+'.AHF_particles', skiprows=1, usecols=0, dtype=np.int64)
    except IOError:
        ds = np.loadtxt(fpre+'.AHF_particles.gz', skiprows=1, usecols=0, dtype=np.int64)

#    Nhalo = int(ds[0]) # Number of haloes for the file
    accum = 1 # Value used to keep track of reading the AHF file
    pid, hid = np.array([],dtype='i8'), np.array([],dtype='i8') # Initialise particle and halo ID arrays
        
    for i in xrange(len(Npart)):
        hid = np.append(hid, np.ones(Npart[i], dtype='i8')*i)
        
        args = np.arange(Npart[i]) + accum # Arguments for the halo's particle IDs
        pid = np.append(pid, np.array(ds[args]))
        accum += (1 + Npart[i])

    
        
    if type(ids)==list: # Put/ensure all input IDs in one array
        idarr = np.array([])
        for i in xrange(len(ids)): idarr = np.append(idarr,ids[i])
    else:
        idarr = np.array(ids)
    
    argorder = np.argsort(idarr) # Order for increasing values in pid
    argreorder = np.argsort(np.arange(len(idarr))[argorder]) # Arguments to reorder everything back
    hid_out = -np.ones(len(idarr), dtype='i8') # Negative ones initially as -1 implies no halo/galaxy for that particle
    
    idargs = np.searchsorted(idarr[argorder], pid) # Find the arguments where the IDs match (in order)
    hid_out[idargs] = hid # Fill the matching entries with halo IDs
    hid_out = hid_out[argreorder] # Return to the same order as the input
    #print hid_out
        
    if type(ids)==list:
        acc = 0
        listout = []
        for i in xrange(len(ids)):
            #print len(ids[i])
            listout += [hid_out[acc:acc+len(ids[i])]]
            acc += len(ids[i])
        return listout
    else:
        return hid_out



def ahflargest(fname):
	# Just read in the information for the largest halo in an AHF output (no substructure stripped)
	
	if fname[-4:]!='cles': fname+='.AHF_particles' # Allows for a prefix of the filename to be input
	
	f = open(fname)
	d = f.read()
	ds = d.split()
	print 'No. of subhaloes', ds[0]
	Npart = int(ds[1])
	print 'No. of particles in largest', Npart
	pid = np.array(ds[3::2][:Npart])
	return pid



def ahf1halo(fpre, hid, h=0.7):
	"""
		Read in a single AHF halo specified by its halo ID.  Does not remove any substructure (unlike ahfhaloid)
		hid = the specific halo ID desired
		Returns the particle IDs for that halo and the position of the halo
		"""
	f = open(fpre+'.AHF_halos')
	d = f.read()
	ds = d.split()
	Npart = np.array(ds[87::83],dtype='i8') # Obtain number of particles in each halo
	xc = (float(ds[88+83*hid]) - float(ds[88]))*1e3/h # Obtain halo position in pc, translated to the coords the simulations are actually in
	yc = (float(ds[89+83*hid]) - float(ds[89]))*1e3/h
	zc = (float(ds[90+83*hid]) - float(ds[90]))*1e3/h
	f.close()
	
	Nskip = 3 + 2*(sum(Npart[:hid]) + hid)
	f = open(fpre+'.AHF_particles')
	d = f.read()
	ds = np.array(d.split(), dtype='i8')
	args = np.arange(Npart[hid])*2 + Nskip
	pid = ds[args]
	f.close()
	return pid, [xc,yc,zc]


def ahfsubhalos(fpre, hid):
	"""
		Find the subhalos belonging to halo number hid
		Returns the halo IDs and the number of subtstructures each of them has.
		"""
	f = open(fpre+'.AHF_halos')
	d = f.read()
	ds = d.split()
	
	HHid = np.array(ds[84::83],dtype='i8') # Host Halo IDs for the haloes
	shid = np.argwhere(HHid==hid) # Gives IDs of subhaloes of the requested halo
	Nsub = np.array(ds[85::83],dtype='i8')[shid] # Number of substructures for each subhalo
	return shid, Nsub


def ahfcentral(fpre, ids, pos_s, mass_s, h=0.7):
	"""
		Find the central (i.e. the main) AHF subhalo.  Will have all substructure stripped.
		It is ensured that this main halo includes stars in something remotely galaxy-like in distribution
		ids = list of arrays of IDs, with stars the first entry
		pos_s = positions of star particles, i.e. a list with [x_s,y_s,z_s]
		mass_s = mass of star particles
		"""
	hid = ahfhaloid(fpre, ids)
	hid_s = hid[0] # Halo IDs for stars
	id_s = ids[0] # Particle IDs for stars
	
	hid_ws = np.array(list(set(hid_s))) # Halo IDs for which there are stars are part of the haloes
	hid_ws = np.sort(hid_ws[hid_ws!=-1]) # Get rid of the -1 entries and sort in increasing order
	
	Nhalo = len(hid_ws)
	
	for i in hid_ws:
		f = (hid_s==i) # Filter
		x_h, y_h, z_h, mass_h = pos_s[0][f], pos_s[1][f], pos_s[2][f], mass_s[f] # Positions+mass of halo-identified stars
		xc, yc, zc = gc.com(x_h,y_h,z_h,mass_h) # Find centre of mass
		x_h, y_h, z_h = x_h-xc, y_h-yc, z_h-zc # Recentre coords
		[Sigma, r_vec], [Sigma_lim, r_lim], [r_end_rel, r_end_abs] = gc.densprof(x_h, y_h, z_h, mass_h)
		if r_end_rel>0 or r_end_abs>0: break # Probably the legit subhalo if one of these radii is found
	
	if Nhalo==0:
		i=-2 # Flags if there are no subhaloes (don't use "-1" as this would suggest all particles not in haloes are actually one big halo)
		xc, yc, zc, r_end_rel, r_end_abs = 0, 0, 0, 0, 0
	"""
		# Find the Virial Radius of the halo according to AHF
		f = open(fpre+'.AHF_halos')
		d = f.read()
		ds = d.split()
		Rvir = float(ds[94+83*i])*1e3/h
		f.close()
		"""
	
	return hid, i, [xc,yc,zc], [r_end_rel, r_end_abs] # returns the array of halo IDs and the ID of the one desired, plus its COM and stellar radii





def mariebinary(hno='106',sno='339',lustre=False):
	# This reads in the gadget-format binary files I created of Marie's sims.
	if lustre==False:
		fname = 'halo_'+hno+'/gadgetformat'+sno+'.dat'
	else:
		fname = '/lustre/projects/p004_swin/astevens/6-MonthProject/halo_'+hno+'/gadgetformat'+sno+'.dat'  

	f = open(fname, 'rb')
	
	Nbytes1 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the header uses
	N = np.fromfile(f, 'u4', 6) # Number of particles for each particle type in this file
	Nsum = sum(N) # Total number of particles in the file
	mass_pt = np.fromfile(f, 'f8', 6) # Mass of each particle type.  If 0 then it varies for each particle of that type
	Nmass = sum(N[np.argwhere(mass_pt==0.0)]) # Number of particles in the file with individual masses to be read in
	a = np.fromfile(f, 'f8', 1)[0] # Expansion factor (normalised to 1 at z=0)
	z = np.fromfile(f, 'f8', 1)[0] # Redshift of snapshot
	flag_sfr = np.fromfile(f, 'i4', 1)[0] # Flag for star formation rate
	flag_feedback = np.fromfile(f, 'i4', 1)[0] # Flag for feedback
	Ntot = np.fromfile(f, 'u4', 6) # Total number of particles for each particle type in the entire simulation
	flag_cool = np.fromfile(f, 'i4', 1)[0] # Flag for cooling
	Nfiles = np.fromfile(f, 'i4', 1)[0] # Number of files for each snapshot
	boxsize = np.fromfile(f, 'f8', 1)[0] # Size of box if periodic boundary conditions are used
	Omega_M = np.fromfile(f, 'f8', 1)[0] # Omega Matter
	Omega_L = np.fromfile(f, 'f8', 1)[0] # Omega (Lambda) Dark Energy
	h = np.fromfile(f, 'f8', 1)[0] # Little Hubble h
	flag_StarAge = np.fromfile(f, 'i4', 1)[0] # Flag for the creation times of stars
	flag_metals = np.fromfile(f, 'i4', 1)[0] # Ignore
	NallHW = np.fromfile(f, 'u4', 6) # Ignore
	flag_entropy = np.fromfile(f, 'i4', 1)[0] # Ignore
	flag_double = np.fromfile(f, 'i4', 1)[0] # Ignore
	flag_ic_info = np.fromfile(f, 'i4', 1)[0] # Ignore
	flag_scale = np.fromfile(f, 'i4', 1)[0] # Ignore
	unused = np.fromfile(f, 'u4', 12) # Ignore
	Nbytes1 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes2 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (particle positions)
	pos = np.fromfile(f, np.dtype(('f8',3)), Nsum)*a*1e3/h # Convert to pc
	Nbytes2 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes3 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (particle velocities)
	vel = np.fromfile(f, np.dtype(('f8',3)), Nsum) # Already in km/s
	Nbytes3 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes4 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (particle IDs)
	ids = np.fromfile(f, 'u8', Nsum) # Particle IDs
	Nbytes4 = np.fromfile(f, 'i4', 1)[0]
	
	if Nmass>0: # Block won't exist if there are no particles with non-standard masses
		Nbytes5 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (particle masses for those that vary)
		mass = np.fromfile(f, 'f8', Nmass)*1e10/h # Convert to M_sun
		Nbytes5 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes6 = np.fromfile(f, 'i4', 1)[0]
	U = np.fromfile(f, 'f8', N[0]) # Internal energy per unit mass in (km/s)^2
	Nbytes6 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes7 = np.fromfile(f, 'i4', 1)[0]
	rho = np.fromfile(f, 'f8', N[0])*10*h*h/(a**3) # Density in solar masses per cubic parsec
	Nbytes7 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes8 = np.fromfile(f, 'i4', 1)[0]
	f.seek(Nbytes1+Nbytes2+Nbytes3+Nbytes4+Nbytes5+Nbytes6+Nbytes7+Nbytes8 + 8*4*2)
	
	Nbytes9 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (star formation time)
	birth_a = np.fromfile(f, 'f8', N[4]) # Expansion factor when each star particle was born
	Nbytes9 = np.fromfile(f, 'i4', 1)[0]
	
	f.close()
	
	x_g, y_g, z_g = pos[:N[0],0], pos[:N[0],1], pos[:N[0],2]
	x_dm, y_dm, z_dm = pos[N[0]:N[0]+N[1],0], pos[N[0]:N[0]+N[1],1], pos[N[0]:N[0]+N[1],2]
	x_s, y_s, z_s = pos[sum(N[:4]):sum(N),0], pos[sum(N[:4]):sum(N),1], pos[sum(N[:4]):sum(N),2]
	
	vx_g, vy_g, vz_g = vel[:N[0],0], vel[:N[0],1], vel[:N[0],2]
	vx_dm, vy_dm, vz_dm = vel[N[0]:N[0]+N[1],0], vel[N[0]:N[0]+N[1],1], vel[N[0]:N[0]+N[1],2]
	vx_s, vy_s, vz_s = vel[sum(N[:4]):sum(N),0], vel[sum(N[:4]):sum(N),1], vel[sum(N[:4]):sum(N),2]
	
	id_g, id_dm, id_s = ids[:N[0]], ids[N[0]:N[0]+N[1]], ids[sum(N[:4]):sum(N)]
	
	mass_g, mass_dm, mass_s = np.ones(N[0],dtype='f8')*mass_pt[0]*1e10/h, np.ones(N[1],dtype='f8')*mass_pt[1]*1e10/h, mass
	
	tarr,zarr = gc.ztlookup(H_0=100*h,Omega_M=Omega_M,Omega_Lambda=Omega_L)
	birth_z = (1./birth_a)-1
	birth = np.interp(birth_z,zarr,tarr)
	
	return z, [x_s,y_s,z_s,vx_s,vy_s,vz_s,id_s,mass_s,birth], [x_g,y_g,z_g,vx_g,vy_g,vz_g,id_g,mass_g,rho], [x_dm,y_dm,z_dm,vx_dm,vy_dm,vz_dm,id_dm,mass_dm]


def mariebinarytrick(hno='106',sno='339'):
	# This reads the "trick format" binary version.  I.e. what was processed through AHF for the baryon-only analysis.
	fname = 'halo_'+hno+'/trickformat'+sno+'.dat'
	
	f = open(fname, 'rb')
	
	Nbytes1 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the header uses
	N = np.fromfile(f, 'u4', 6) # Number of particles for each particle type in this file
	print 'N', N
	Nsum = sum(N) # Total number of particles in the file
	mass_pt = np.fromfile(f, 'f8', 6) # Mass of each particle type.  If 0 then it varies for each particle of that type
	Nmass = sum(N[np.argwhere(mass_pt==0.0)]) # Number of particles in the file with individual masses to be read in
	a = np.fromfile(f, 'f8', 1)[0] # Expansion factor (normalised to 1 at z=0)
	z = np.fromfile(f, 'f8', 1)[0] # Redshift of snapshot
	flag_sfr = np.fromfile(f, 'i4', 1)[0] # Flag for star formation rate
	flag_feedback = np.fromfile(f, 'i4', 1)[0] # Flag for feedback
	Ntot = np.fromfile(f, 'u4', 6) # Total number of particles for each particle type in the entire simulation
	flag_cool = np.fromfile(f, 'i4', 1)[0] # Flag for cooling
	Nfiles = np.fromfile(f, 'i4', 1)[0] # Number of files for each snapshot
	boxsize = np.fromfile(f, 'f8', 1)[0] # Size of box if periodic boundary conditions are used
	Omega_M = np.fromfile(f, 'f8', 1)[0] # Omega Matter
	Omega_L = np.fromfile(f, 'f8', 1)[0] # Omega (Lambda) Dark Energy
	h = np.fromfile(f, 'f8', 1)[0] # Little Hubble h
	flag_StarAge = np.fromfile(f, 'i4', 1)[0] # Flag for the creation times of stars
	flag_metals = np.fromfile(f, 'i4', 1)[0] # Ignore
	NallHW = np.fromfile(f, 'u4', 6) # Ignore
	flag_entropy = np.fromfile(f, 'i4', 1)[0] # Ignore
	flag_double = np.fromfile(f, 'i4', 1)[0] # Ignore
	flag_ic_info = np.fromfile(f, 'i4', 1)[0] # Ignore
	flag_scale = np.fromfile(f, 'i4', 1)[0] # Ignore
	unused = np.fromfile(f, 'u4', 12) # Ignore
	Nbytes1 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes2 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (particle positions)
	pos = np.fromfile(f, np.dtype(('f8',3)), Nsum)*a*1e3/h # Convert to pc
	Nbytes2 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes3 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (particle velocities)
	vel = np.fromfile(f, np.dtype(('f8',3)), Nsum) # Already in km/s
	Nbytes3 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes4 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (particle IDs)
	ids = np.fromfile(f, 'u8', Nsum) # Particle IDs
	Nbytes4 = np.fromfile(f, 'i4', 1)[0]
	
	if Nmass>0: # Block won't exist if there are no particles with non-standard masses
		Nbytes5 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (particle masses for those that vary)
		mass = np.fromfile(f, 'f8', Nmass)*1e10/h # Convert to M_sun
		Nbytes5 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes6 = np.fromfile(f, 'i4', 1)[0]
	U = np.fromfile(f, 'f8', N[0]) # Internal energy per unit mass in (km/s)^2
	Nbytes6 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes7 = np.fromfile(f, 'i4', 1)[0]
	rho = np.fromfile(f, 'f8', N[0])*10*h*h/(a**3) # Density in solar masses per cubic parsec
	Nbytes7 = np.fromfile(f, 'i4', 1)[0]
	
	f.close()
	
	x_g, y_g, z_g = pos[:N[0],0], pos[:N[0],1], pos[:N[0],2]
	x_s, y_s, z_s = pos[N[0]:N[0]+N[1],0], pos[N[0]:N[0]+N[1],1], pos[N[0]:N[0]+N[1],2]
	
	vx_g, vy_g, vz_g = vel[:N[0],0], vel[:N[0],1], vel[:N[0],2]
	vx_s, vy_s, vz_s = vel[N[0]:N[0]+N[1],0], vel[N[0]:N[0]+N[1],1], vel[N[0]:N[0]+N[1],2]
	
	id_g, id_s = ids[:N[0]], ids[N[0]:N[0]+N[1]]
	
	mass_g, mass_s = np.ones(N[0],dtype='f8')*mass_pt[0]*1e10/h, mass

	return z, [x_s,y_s,z_s,vx_s,vy_s,vz_s,id_s,mass_s], [x_g,y_g,z_g,vx_g,vy_g,vz_g,id_g,mass_g,rho]




def ahfprops(hno):
	# Read in file with the already-measured integrated galaxy properties from AHF (output from ahfnext.py)
	f = open('ahfnextout/ahfzero'+str(hno), 'rb')
	SM = np.fromfile(f, 'f8', 34)
	GM = np.fromfile(f, 'f8', 34)
	SFR = np.fromfile(f, 'f8', 34)
	SDR = np.fromfile(f, 'f8', 34)
	SAR = np.fromfile(f, 'f8', 34)
	SER = np.fromfile(f, 'f8', 34)
	GAR = np.fromfile(f, 'f8', 34)
	GER = np.fromfile(f, 'f8', 34)
	r_od = np.fromfile(f, 'f8', 34)
	f.close()
	return [SM, GM, SFR, SDR, SAR, SER, GAR, GER, r_od]




# ========== Photometry-related code ===========

def starpopmodel():
	# Load the Maratson (2005) stellar population model for ability to do spectroscopy and photometry.
	
	with open('photospec/ssp.ssz','r') as f1:
		data1 = f1.read() # Reads it in as a single string
		ds1 = data1.split() # Breaks into a list of strings (1 per entry)
		dsa1 = np.array(ds1,dtype='float64') # Converts to an array of integers
		arr2d = np.array(np.split(dsa1,len(dsa1)/7)) # Converts to the 2D array it was meant to be.  Entries are luminosities in erg/(s*Angstrom*M_sun).  Column (second index) indicates metallicity.  Row (first index) indicates wavelength.  Blocks of rows are age bins.
		L = np.array(np.split(arr2d,67)) # Produces luminosity data in a 3d array.  Index 1 denotes age, index 2 denotes wavelength and index 3 denotes metallicity
	
	# There are entries of zeroes in L which are generally unhelpful, which I'm filling with adjacent information
	L[:51,:,0] = L[:51,:,1]
	L[:51,:,6] = L[:51,:,5]
	
	with open('photospec/agegrid.dati','r') as f2:
		data2 = f2.read()
		ds2 = data2.split()[1:] # First entry is just the length (=67)
		ages = np.array(ds2,dtype='float64') # Vector of ages for the age bins in Gyr
	
	with open('photospec/lambda.dati','r') as f3:
		data3 = f3.read()
		ds3 = data3.split()
		wl = np.array(ds3,dtype='float64') # Vector of wavelengths in Angstroms
	
	metals = np.array([0.0001,0.001,0.004,0.01,0.02,0.04,0.07]) # Ratio of metal mass to gas mass in the models.
	
	return L, ages, wl, metals



def starpopmodel_Kroupa():
	# Load the Maratson (2005) stellar population model for a Kroupa IMF
	# Still some zeroes in the highest metallicity, as doesn't have all the ages.
	metals = np.array([0.001,0.01,0.02,0.04,0.07])
	mstr = ['0001','001','002','004','007'] # Strings for files of different metallicity
	L = np.zeros((67,1221,5), dtype='f8') # Initialise array to hold luminosities
	
	for i, m in enumerate(mstr):
		f = open('photospec/sed.krz'+m+'.rhb')
		ds = f.read().split()

		if i==0:
			ages = np.array(ds[0::1221*4], dtype='f8') # Get array for ages of SSPs
			wl = np.array(ds[:1221*4][2::4], dtype='f8') # Array of wavelengths for the spectra

		#age_file = np.array(ds[0::1221*4], dtype='f8') # Get ages considered in this specific file
		age_first = np.float64(ds[0]) # First age considered
		init = np.argwhere(ages==age_first)[0][0]
		
		for j in xrange(67-init):
			k = j+init
			L[k,:,i] = np.array(ds[j*4*1221:(j+1)*4*1221][3::4], dtype='f8')

	return L, ages, wl, metals




def readfilter(fname):
	# Read in a filter from data files.  fname is the name of the filter and should be a string, eg. 'u'
	
	with open('photospec/'+fname+'.dat','r') as f:
		data = f.read()
		ds = data.split()
	
	dsa = np.array(ds,dtype='float64')
	wl = dsa[1::2] # Extract the wavelength vector for the filter.  Note the first entry in dsa is the length of wl and resp, hence ignored.
	resp = dsa[2::2] # Extract the response function
	
	nozero = (resp>0) # Want to remove all the entries where the response function is zero... seems silly that they're there in the first place.
	return wl[nozero], resp[nozero]





def readvega(band=''):
	# Read in the data file for Vega's spectrum
	# If band = name of a filter (string) then this will return bolometric luminosity and band power as well
	
	with open('photospec/A0V_KUR_BB.SED','r') as f:
		data = f.read()
		ds = data.split()
	
	dsa = np.array(ds,dtype='float64')
	wl = dsa[0::2] # Extract wavelength vector
	spec = dsa[1::2]*(2e-17)*2*np.pi*(3.08567758e19)**2 # Extract the spectrum data in erg/s/A
	
	dwl = np.diff(wl) # Difference vector of Vega's wl vector
	L_bol = np.sum(spec[:-1]*dwl + 0.5*np.diff(spec)*dwl) # Vega's luminosity
	
	if len(band)>0:
		wl_filt, resp_filt = readfilter(band) # Read in the filter
		L_filt = gc.lumfilt(spec,wl,resp_filt,wl_filt) # Vega's power in the selected band
		return [spec, wl, L_bol, L_filt]
	else:
		return [spec, wl, L_bol]




def epsexpt():
	# Read in data output from epsexpt2.py
	f = open('epsexpt/2/output','rb')
	smin = np.fromfile(f, 'i4', 1)[0]
	#smin = 4
	Nprof = (34-smin)*16
	epsilon = np.fromfile(f, 'f8', 198)
	Nfit = np.fromfile(f, np.dtype(('f8',198)), Nprof)
	r_bmp = np.fromfile(f, np.dtype(('f8',198)), Nprof)
	BMint = np.fromfile(f, np.dtype(('f8',198)), Nprof)
	f.close()
	return epsilon, Nfit, r_bmp, BMint


def epsexptmeans():
	# Read in data output from epsexpt2.py
	f = open('epsexpt/2/output_means','rb')
	Neps = np.fromfile(f, 'i8', 1)[0]
	#Neps = 198
	epsilon = np.fromfile(f, 'f8', Neps)
	Nfit = np.fromfile(f, 'f8', Neps)
	r_bmp = np.fromfile(f, 'f8', Neps)
	BMint = np.fromfile(f, 'f8', Neps)
	f.close()
	return epsilon, Nfit, r_bmp, BMint

def epsexptstd():
	# Read in data output from epsexpt2.py
	f = open('epsexpt/2/output_std','rb')
	Neps = np.fromfile(f, 'i8', 1)[0]
	Nfit_std = np.fromfile(f, 'f8', Neps)
	r_bmp_std = np.fromfile(f, 'f8', Neps)
	BMint_std = np.fromfile(f, 'f8', Neps)
	f.close()
	return Nfit_std, r_bmp_std, BMint_std

# ========== SAGE-related code ==========

def galdtype():
    # Define the data type for reading galaxy data from SAGE outputs
	Galdesc_full = [
                    ('Type'                         , np.int32),
                    ('GalaxyIndex'                  , np.int64),
                    ('HaloIndex'                    , np.int32),
                    ('FOFHaloIdx'                   , np.int32),
                    ('TreeIdx'                      , np.int32),
                    ('SnapNum'                      , np.int32),
                    ('CentralGal'                   , np.int32),
                    ('CentralMvir'                  , np.float32),
                    ('mergeType'                    , np.int32),
                    ('mergeIntoID'                  , np.int32),
                    ('mergeIntoSnapNum'             , np.int32),
                    ('dT'                           , np.float32),
                    ('Pos'                          , (np.float32, 3)),
                    ('Vel'                          , (np.float32, 3)),
                    ('Spin'                         , (np.float32, 3)),
                    ('Len'                          , np.int32),
                    ('Mvir'                         , np.float32),
                    ('Rvir'                         , np.float32),
                    ('Vvir'                         , np.float32),
                    ('Vmax'                         , np.float32),
                    ('VelDisp'                      , np.float32),
                    ('ColdGas'                      , np.float32),
                    ('StellarMass'                  , np.float32),
                    ('ClassicalBulgeMass'           , np.float32),
                    ('SecularBulgeMass'             , np.float32),
                    ('HotGas'                       , np.float32),
                    ('EjectedMass'                  , np.float32),
                    ('BlackHoleMass'                , np.float32),
                    ('IntraClusterStars'            , np.float32),
                    ('MetalsColdGas'                , np.float32),
                    ('MetalsStellarMass'            , np.float32),
                    ('ClassicalMetalsBulgeMass'     , np.float32),
                    ('SecularMetalsBulgeMass'       , np.float32),
                    ('MetalsHotGas'                 , np.float32),
                    ('MetalsEjectedMass'            , np.float32),
                    ('MetalsIntraClusterStars'      , np.float32),
                    ('SfrDisk'                      , np.float32),
                    ('SfrBulge'                     , np.float32),
                    ('SfrDiskZ'                     , np.float32),
                    ('SfrBulgeZ'                    , np.float32),
                    ('DiskRadius'                   , np.float32),
                    ('BulgeRadius'                  , np.float32),
                    ('Cooling'                      , np.float32),
                    ('Heating'                      , np.float32),
                    ('LastMajorMerger'              , np.float32),
                    ('OutflowRate'                  , np.float32),
                    ('infallMvir'                   , np.float32),
                    ('infallVvir'                   , np.float32),
                    ('infallVmax'                   , np.float32)
                    ]
	names = [Galdesc_full[i][0] for i in xrange(len(Galdesc_full))]
	formats = [Galdesc_full[i][1] for i in xrange(len(Galdesc_full))]
	Galdesc = np.dtype({'names':names, 'formats':formats}, align=True)
	return Galdesc

def galdtype_DarkSage2016():
    # Specific for the version used in the 2016 paper.
    Nannuli=30
    floattype = np.float32
    Galdesc_full = [
                    ('Type'                         , np.int32),
                    ('GalaxyIndex'                  , np.int64),
                    ('HaloIndex'                    , np.int32),
                    ('SimulationHaloIndex'          , np.int32),
                    ('TreeIndex'                    , np.int32),
                    ('SnapNum'                      , np.int32),
                    ('CentralGalaxyIndex'           , np.int64),
                    ('CentralMvir'                  , floattype),
                    ('mergeType'                    , np.int32),
                    ('mergeIntoID'                  , np.int32),
                    ('mergeIntoSnapNum'             , np.int32),
                    ('dT'                           , floattype),
                    ('Pos'                          , (floattype, 3)),
                    ('Vel'                          , (floattype, 3)),
                    ('Spin'                         , (floattype, 3)),
                    ('Len'                          , np.int32),
                    ('Mvir'                         , floattype),
                    ('Rvir'                         , floattype),
                    ('Vvir'                         , floattype),
                    ('Vmax'                         , floattype),
                    ('VelDisp'                      , floattype),
                    ('DiscRadii'                    , (floattype, Nannuli+1)),
                    ('ColdGas'                      , floattype),
                    ('StellarMass'                  , floattype),
                    ('MergerBulgeMass'              , floattype),
                    ('InstabilityBulgeMass'          , floattype),
                    ('HotGas'                       , floattype),
                    ('EjectedMass'                  , floattype),
                    ('BlackHoleMass'                , floattype),
                    ('IntraClusterStars'            , floattype),
                    ('DiscGas'                      , (floattype, Nannuli)),
                    ('DiscStars'                    , (floattype, Nannuli)),
                    ('SpinStars'                    , (floattype, 3)),
                    ('SpinGas'                      , (floattype, 3)),
                    ('SpinSecularBulge'             , (floattype, 3)),
                    ('SpinClassicalBulge'           , (floattype, 3)),
                    ('StarsInSitu'                  , floattype),
                    ('StarsInstability'             , floattype),
                    ('StarsMergeBurst'              , floattype),
                    ('DiscHI'                       , (floattype, Nannuli)),
                    ('DiscH2'                       , (floattype, Nannuli)),
                    ('DiscSFR'                      , (floattype, Nannuli)),
                    ('TotInstabEvents'              , np.int32),
                    ('TotInstabEventsGas'           , np.int32),
                    ('TotInstabEventsStar'          , np.int32),
                    ('TotInstabAnnuliGas'           , np.int32),
                    ('TotInstabAnnuliStar'          , np.int32),
                    ('FirstUnstableAvGas'           , floattype),
                    ('FirstUnstableAvStar'          , floattype),
                    ('TotSinkGas'                   , (floattype, 30)),
                    ('TotSinkStar'                  , (floattype, 30)),
                    ('MetalsColdGas'                , floattype),
                    ('MetalsStellarMass'            , floattype),
                    ('ClassicalMetalsBulgeMass'     , floattype),
                    ('SecularMetalsBulgeMass'       , floattype),
                    ('MetalsHotGas'                 , floattype),
                    ('MetalsEjectedMass'            , floattype),
                    ('MetalsIntraClusterStars'      , floattype),
                    ('DiscGasMetals'                , (floattype, Nannuli)),
                    ('DiscStarsMetals'              , (floattype, Nannuli)),
                    ('SfrDisk'                      , floattype),
                    ('SfrBulge'                     , floattype),
                    ('SfrDiskZ'                     , floattype),
                    ('SfrBulgeZ'                    , floattype),
                    ('DiskScaleRadius'              , floattype),
                    ('BulgeRadius'                  , floattype),
                    ('Cooling'                      , floattype),
                    ('Heating'                      , floattype),
                    ('LastMajorMerger'              , floattype),
                    ('OutflowRate'                  , floattype),
                    ('infallMvir'                   , floattype),
                    ('infallVvir'                   , floattype),
                    ('infallVmax'                   , floattype)
                    ]
    names = [Galdesc_full[i][0] for i in xrange(len(Galdesc_full))]
    formats = [Galdesc_full[i][1] for i in xrange(len(Galdesc_full))]
    Galdesc = np.dtype({'names':names, 'formats':formats}, align=True)
    return Galdesc


def galdtype_adam(Nannuli=30):
	# Give data type for SAGE output that includes my edits for the disc project
	floattype = np.float32 # Run 160 onward uses 32, was 64 prior
	Galdesc_full = [
                    ('Type'                         , np.int32),
                    ('GalaxyIndex'                  , np.int64),
                    ('HaloIndex'                    , np.int32),
                    ('SimulationHaloIndex'          , np.int32),
                    ('TreeIndex'                    , np.int32),
                    ('SnapNum'                      , np.int32),
                    ('CentralGalaxyIndex'           , np.int64), # Changed to 64-bit at run 96
                    ('CentralMvir'                  , floattype),
                    ('mergeType'                    , np.int32),
                    ('mergeIntoID'                  , np.int32),
                    ('mergeIntoSnapNum'             , np.int32),
                    ('dT'                           , floattype),
                    ('Pos'                          , (floattype, 3)),
                    ('Vel'                          , (floattype, 3)),
                    ('Spin'                         , (floattype, 3)),
                    ('Len'                          , np.int32),
                    ('LenMax'                       , np.int32), # Added at run 409
                    ('Mvir'                         , floattype),
                    ('Rvir'                         , floattype),
                    ('Vvir'                         , floattype),
                    ('Vmax'                         , floattype),
                    ('VelDisp'                      , floattype),
                    ('DiscRadii'                    , (floattype, Nannuli+1)), # Added at run 145
                    ('ColdGas'                      , floattype),
                    ('StellarMass'                  , floattype),
                    ('MergerBulgeMass'              , floattype),
                    ('InstabilityBulgeMass'          , floattype),
                    ('HotGas'                       , floattype),
                    ('EjectedMass'                  , floattype),
                    ('BlackHoleMass'                , floattype),
                    ('IntraClusterStars'            , floattype),
                    ('DiscGas'                      , (floattype, Nannuli)),
                    ('DiscStars'                    , (floattype, Nannuli)),
                    ('SpinStars'                    , (floattype, 3)),
                    ('SpinGas'                      , (floattype, 3)),
#                    ('SpinSecularBulge'             , (floattype, 3)), #=# This and next added at run 154. This one (only) removed at 432
                    ('SpinClassicalBulge'           , (floattype, 3)), #=#
                    ('StarsInSitu'                  , floattype), # This and next 2 introduced at run 66, removed 432, re-added run 436
                    ('StarsInstability'             , floattype), #
                    ('StarsMergeBurst'              , floattype), #
                    ('DiscHI'                       , (floattype, Nannuli)), ## This and next introduced at run 69
                    ('DiscH2'                       , (floattype, Nannuli)), ##
                    ('DiscSFR'                      , (floattype, Nannuli)), #====# Added run 186
#                    ('AccretedGasMass'              , floattype), ###### Added run 324, removed run 432
#                    ('EjectedSNGasMass'             , floattype),######
#                    ('EjectedQuasarGasMass'         , floattype),######
#                    ('TotInstabEvents'              , np.int32), ### This and next few introduced at run 144, removed 432
#                    ('TotInstabEventsGas'           , np.int32), ###
#                    ('TotInstabEventsStar'          , np.int32), ###
#                    ('TotInstabAnnuliGas'           , np.int32), ###
#                    ('TotInstabAnnuliStar'          , np.int32), ###
#                    ('FirstUnstableAvGas'           , floattype), ###
#                    ('FirstUnstableAvStar'          , floattype), ###
#                    ('TotSinkGas'                   , (floattype, 30)), ###
#                    ('TotSinkStar'                  , (floattype, 30)), ###
                    ('MetalsColdGas'                , floattype),
                    ('MetalsStellarMass'            , floattype),
                    ('ClassicalMetalsBulgeMass'     , floattype),
                    ('SecularMetalsBulgeMass'       , floattype),
                    ('MetalsHotGas'                 , floattype),
                    ('MetalsEjectedMass'            , floattype),
                    ('MetalsIntraClusterStars'      , floattype),
                    ('DiscGasMetals'                , (floattype, Nannuli)),
                    ('DiscStarsMetals'              , (floattype, Nannuli)),
                    ('SfrDisk'                      , floattype), #--# Removed run 554
                    ('SfrBulge'                     , floattype), #--#
#                    ('SfrFromH2'                    , floattype), #----# Added run 554
#                    ('SfrInstab'                    , floattype), #----# 
#                    ('SfrMergeBurst'                , floattype), #----# 
                    ('SfrDiskZ'                     , floattype),
                    ('SfrBulgeZ'                    , floattype),
                    ('DiskScaleRadius'              , floattype),
                    ('CoolScaleRadius'              , floattype), # Added run 516
                    ('StellarDiscScaleRadius'       , floattype), # Added run 476
#                    ('BulgeRadius'                  , floattype), # Removed run 455 (accidentally there for 457-464 because I hadn't re-made code -.-)
                    ('Cooling'                      , floattype),
                    ('Heating'                      , floattype),
                    ('LastMajorMerger'              , floattype),
                    ('LastMinorMerger'              , floattype), # Added run 433
                    ('OutflowRate'                  , floattype),
                    ('infallMvir'                   , floattype),
                    ('infallVvir'                   , floattype),
                    ('infallVmax'                   , floattype)
                    ]
	names = [Galdesc_full[i][0] for i in xrange(len(Galdesc_full))]
	formats = [Galdesc_full[i][1] for i in xrange(len(Galdesc_full))]
	Galdesc = np.dtype({'names':names, 'formats':formats}, align=True)
	return Galdesc


def galdtype_public():
    Galdesc_full = [
                ('SnapNum'                      , np.int32),
                ('Type'                         , np.int32),
                ('GalaxyIndex'                  , np.int64),
                ('CentralGalaxyIndex'           , np.int64),
                ('SAGEHaloIndex'                , np.int32),
                ('SAGETreeIndex'                , np.int32),
                ('SimulationFOFHaloIndex'       , np.int32),
                ('mergeType'                    , np.int32),
                ('mergeIntoID'                  , np.int32),
                ('mergeIntoSnapNum'             , np.int32),
                ('dT'                           , np.float32),
                ('Pos'                          , (np.float32, 3)),
                ('Vel'                          , (np.float32, 3)),
                ('Spin'                         , (np.float32, 3)),
                ('Len'                          , np.int32),
                ('Mvir'                         , np.float32),
                ('CentralMvir'                  , np.float32),
                ('Rvir'                         , np.float32),
                ('Vvir'                         , np.float32),
                ('Vmax'                         , np.float32),
                ('VelDisp'                      , np.float32),
                ('ColdGas'                      , np.float32),
                ('StellarMass'                  , np.float32),
                ('BulgeMass'                    , np.float32),
                ('HotGas'                       , np.float32),
                ('EjectedMass'                  , np.float32),
                ('BlackHoleMass'                , np.float32),
                ('IntraClusterStars'            , np.float32),
                ('MetalsColdGas'                , np.float32),
                ('MetalsStellarMass'            , np.float32),
                ('MetalsBulgeMass'              , np.float32),
                ('MetalsHotGas'                 , np.float32),
                ('MetalsEjectedMass'            , np.float32),
                ('MetalsIntraClusterStars'      , np.float32),
                ('SfrDisk'                      , np.float32),
                ('SfrBulge'                     , np.float32),
                ('SfrDiskZ'                     , np.float32),
                ('SfrBulgeZ'                    , np.float32),
                ('DiskRadius'                   , np.float32),
                ('Cooling'                      , np.float32),
                ('Heating'                      , np.float32),
                ('QuasarModeBHaccretionMass'    , np.float32),
                ('TimeSinceMajorMerger'         , np.float32),
                ('TimeSinceMinorMerger'         , np.float32),
                ('OutflowRate'                  , np.float32),
                ('infallMvir'                   , np.float32),
                ('infallVvir'                   , np.float32),
                ('infallVmax'                   , np.float32)
                ]
    names = [Galdesc_full[i][0] for i in xrange(len(Galdesc_full))]
    formats = [Galdesc_full[i][1] for i in xrange(len(Galdesc_full))]
    Galdesc = np.dtype({'names':names, 'formats':formats}, align=True)
    return Galdesc

def galdtype_public_old0():
    Galdesc_full = [
                ('SnapNum'                      , np.int32),
                ('Type'                         , np.int32),
                ('GalaxyIndex'                  , np.int64),
                ('CentralGalaxyIndex'           , np.int64),
                ('SAGEHaloIndex'                , np.int32),
                ('SAGETreeIndex'                , np.int32),
                ('SimulationFOFHaloIndex'       , np.int32),
                ('mergeType'                    , np.int32),
                ('mergeIntoID'                  , np.int32),
                ('mergeIntoSnapNum'             , np.int32),
                ('dT'                           , np.float32),
                ('Pos'                          , (np.float32, 3)),
                ('Vel'                          , (np.float32, 3)),
                ('Spin'                         , (np.float32, 3)),
                ('Len'                          , np.int32),
                ('Mvir'                         , np.float32),
                ('CentralMvir'                  , np.float32),
                ('Rvir'                         , np.float32),
                ('Vvir'                         , np.float32),
                ('Vmax'                         , np.float32),
                ('VelDisp'                      , np.float32),
                ('ColdGas'                      , np.float32),
                ('StellarMass'                  , np.float32),
                ('BulgeMass'                    , np.float32),
                ('HotGas'                       , np.float32),
                ('EjectedMass'                  , np.float32),
                ('BlackHoleMass'                , np.float32),
                ('IntraClusterStars'            , np.float32),
                ('MetalsColdGas'                , np.float32),
                ('MetalsStellarMass'            , np.float32),
                ('MetalsBulgeMass'              , np.float32),
                ('MetalsHotGas'                 , np.float32),
                ('MetalsEjectedMass'            , np.float32),
                ('MetalsIntraClusterStars'      , np.float32),
                ('SfrDisk'                      , np.float32),
                ('SfrBulge'                     , np.float32),
                ('SfrDiskZ'                     , np.float32),
                ('SfrBulgeZ'                    , np.float32),
                ('DiskRadius'                   , np.float32),
                ('Cooling'                      , np.float32),
                ('Heating'                      , np.float32),
                ('LastMajorMerger'              , np.float32),
                ('OutflowRate'                  , np.float32),
                ('infallMvir'                   , np.float32),
                ('infallVvir'                   , np.float32),
                ('infallVmax'                   , np.float32)
                ]
    names = [Galdesc_full[i][0] for i in xrange(len(Galdesc_full))]
    formats = [Galdesc_full[i][1] for i in xrange(len(Galdesc_full))]
    Galdesc = np.dtype({'names':names, 'formats':formats}, align=True)
    return Galdesc


def galdtype_public_old1(fof64=False):
	# Relevant for reading in public-version SAGE data
	foftype = np.int64 if fof64 else np.int32
	Galdesc_full = [
	            ('Type'                         , np.int32),                    
	            ('GalaxyIndex'                  , np.int64),                    
	            ('HaloIndex'                    , np.int32),                    
	            ('FOFHaloIdx'                   , foftype),
	            ('TreeIdx'                      , np.int32),                    
	            ('SnapNum'                      , np.int32),                    
	            ('CentralGal'                   , np.int32),                    
	            ('CentralMvir'                  , np.float32),                  
	            ('mergeType'                    , np.int32),                    
	            ('mergeIntoID'                  , np.int32),                    
	            ('mergeIntoSnapNum'             , np.int32),                    
	            ('dT'                           , np.float32),                    
	            ('Pos'                          , (np.float32, 3)),             
	            ('Vel'                          , (np.float32, 3)),             
	            ('Spin'                         , (np.float32, 3)),             
	            ('Len'                          , np.int32),                    
	            ('Mvir'                         , np.float32),                  
	            ('Rvir'                         , np.float32),                  
	            ('Vvir'                         , np.float32),                  
	            ('Vmax'                         , np.float32),                  
	            ('VelDisp'                      , np.float32),                  
	            ('ColdGas'                      , np.float32),                  
	            ('StellarMass'                  , np.float32),                  
	            ('BulgeMass'                    , np.float32),                  
	            ('HotGas'                       , np.float32),                  
	            ('EjectedMass'                  , np.float32),                  
	            ('BlackHoleMass'                , np.float32),                  
	            ('IntraClusterStars'            , np.float32),                  
	            ('MetalsColdGas'                , np.float32),                  
	            ('MetalsStellarMass'            , np.float32),                  
	            ('MetalsBulgeMass'              , np.float32),                  
	            ('MetalsHotGas'                 , np.float32),                  
	            ('MetalsEjectedMass'            , np.float32),                  
	            ('MetalsIntraClusterStars'      , np.float32),                  
	            ('SfrDisk'                      , np.float32),                  
	            ('SfrBulge'                     , np.float32),                  
	            ('SfrDiskZ'                     , np.float32),                  
	            ('SfrBulgeZ'                    , np.float32),                  
	            ('DiskRadius'                   , np.float32),                  
	            ('Cooling'                      , np.float32),                  
	            ('Heating'                      , np.float32),
	            ('LastMajorMerger'              , np.float32),
	            ('OutflowRate'                  , np.float32),
	            ('infallMvir'                   , np.float32),
	            ('infallVvir'                   , np.float32),
	            ('infallVmax'                   , np.float32)
	            ]
	names = [Galdesc_full[i][0] for i in xrange(len(Galdesc_full))]
	formats = [Galdesc_full[i][1] for i in xrange(len(Galdesc_full))]
	Galdesc = np.dtype({'names':names, 'formats':formats}, align=True)
	return Galdesc


def galdtype_public_old2():
	# Relevant for reading in public-version SAGE data
	Galdesc_full = [
	            ('Type'                         , np.int32),                    
	            ('GalaxyIndex'                  , np.int64),                    
	            ('HaloIndex'                    , np.int32),                    
	            ('FOFHaloIdx'                   , np.int32),                    
	            ('TreeIdx'                      , np.int32),                    
	            ('SnapNum'                      , np.int32),                    
	            ('CentralGal'                   , np.int32),                    
	            ('CentralMvir'                  , np.float32),                  
	            ('mergeType'                    , np.int32),                    
	            ('mergeIntoID'                  , np.int32),                    
	            ('mergeIntoSnapNum'             , np.int32),                    
	            ('dT'                           , np.float32),                    
	            ('Pos'                          , (np.float32, 3)),             
	            ('Vel'                          , (np.float32, 3)),             
	            ('Spin'                         , (np.float32, 3)),             
	            ('Len'                          , np.int32),                    
	            ('Mvir'                         , np.float32),                  
	            ('Rvir'                         , np.float32),                  
	            ('Vvir'                         , np.float32),                  
	            ('Vmax'                         , np.float32),                  
	            ('VelDisp'                      , np.float32),                  
	            ('ColdGas'                      , np.float32),                  
	            ('StellarMass'                  , np.float32),                  
	            ('BulgeMass'                    , np.float32),                  
	            ('HotGas'                       , np.float32),                  
	            ('EjectedMass'                  , np.float32),                  
	            ('BlackHoleMass'                , np.float32),                  
	            ('IntraClusterStars'            , np.float32),                  
	            ('MetalsColdGas'                , np.float32),                  
	            ('MetalsStellarMass'            , np.float32),                  
	            ('MetalsBulgeMass'              , np.float32),                  
	            ('MetalsHotGas'                 , np.float32),                  
	            ('MetalsEjectedMass'            , np.float32),                  
	            ('MetalsIntraClusterStars'      , np.float32),                  
	            ('SfrDisk'                      , np.float32),                  
	            ('SfrBulge'                     , np.float32),                  
	            ('SfrDiskZ'                     , np.float32),                  
	            ('SfrBulgeZ'                    , np.float32),                  
	            ('DiskRadius'                   , np.float32),
	            ('BulgeRadius'                   , np.float32),
	            ('Cooling'                      , np.float32),                  
	            ('Heating'                      , np.float32),
	            ('LastMajorMerger'              , np.float32),
	            ('OutflowRate'                  , np.float32),
	            ('infallMvir'                   , np.float32),
	            ('infallVvir'                   , np.float32),
	            ('infallVmax'                   , np.float32)
	            ]
	names = [Galdesc_full[i][0] for i in xrange(len(Galdesc_full))]
	formats = [Galdesc_full[i][1] for i in xrange(len(Galdesc_full))]
	Galdesc = np.dtype({'names':names, 'formats':formats}, align=True)
	return Galdesc


def galdtype_carnage_workshop():
    Galdesc_full = [
                    ('Type'                         , np.int32),
                    ('GalaxyIndex'                  , np.int64),
                    ('HaloIndex'                    , np.int32),
                    ('SAGEnIFTyID'                  , np.int64),
                    ('SAGEnIFTyCentralID'           , np.int64),
                    ('TreeIdx'                      , np.int32),
                    ('SnapNum'                      , np.int32),
                    ('CentralGal'                   , np.int32),
                    ('CentralMvir'                  , np.float32),
                    ('mergeType'                    , np.int32),
                    ('mergeIntoID'                  , np.int32),
                    ('mergeIntoSnapNum'             , np.int32),
                    ('dT'                           , np.float32),
                    ('Pos'                          , (np.float32, 3)),
                    ('Vel'                          , (np.float32, 3)),
                    ('Spin'                         , (np.float32, 3)),
                    ('Len'                          , np.int32),
                    ('Mvir'                         , np.float32),
                    ('Rvir'                         , np.float32),
                    ('Vvir'                         , np.float32),
                    ('Vmax'                         , np.float32),
                    ('VelDisp'                      , np.float32),
                    ('ColdGas'                      , np.float32),
                    ('StellarMass'                  , np.float32),
                    ('BulgeMass'                    , np.float32),
                    ('HotGas'                       , np.float32),
                    ('EjectedMass'                  , np.float32),
                    ('BlackHoleMass'                , np.float32),
                    ('IntraClusterStars'            , np.float32),
                    ('MetalsColdGas'                , np.float32),
                    ('MetalsStellarMass'            , np.float32),
                    ('MetalsBulgeMass'              , np.float32),
                    ('MetalsHotGas'                 , np.float32),
                    ('MetalsEjectedMass'            , np.float32),
                    ('MetalsIntraClusterStars'      , np.float32),
                    ('SfrDisk'                      , np.float32),
                    ('SfrBulge'                     , np.float32),
                    ('SfrDiskZ'                     , np.float32),
                    ('SfrBulgeZ'                    , np.float32),
                    ('DiskRadius'                   , np.float32),
                    ('Cooling'                      , np.float32),
                    ('Heating'                      , np.float32),
                    ('LastMajorMerger'              , np.float32),
                    ('OutflowRate'                  , np.float32),
                    ('infallMvir'                   , np.float32),
                    ('infallVvir'                   , np.float32),
                    ('infallVmax'                   , np.float32)
                    ]
    names = [Galdesc_full[i][0] for i in xrange(len(Galdesc_full))]
    formats = [Galdesc_full[i][1] for i in xrange(len(Galdesc_full))]
    Galdesc = np.dtype({'names':names, 'formats':formats}, align=True)
    return Galdesc

def galdtype_multidark():
    Galdesc_full = [
                    ('SnapNum'                      , np.int32),
                    ('Type'                         , np.int32),
                    ('GalaxyIndex'                  , np.int64),
                    ('CentralGalaxyIndex'           , np.int64),
                    ('CtreesHaloID'                 , np.int64),
                    ('TreeIndex'                    , np.int32),
                    ('CtreesCentralID'              , np.int64),
                    ('mergeType'                    , np.int32),
                    ('mergeIntoID'                  , np.int32),
                    ('mergeIntoSnapNum'             , np.int32),
                    ('dT'                           , np.float32),
                    ('Pos'                          , (np.float32, 3)),
                    ('Vel'                          , (np.float32, 3)),
                    ('Spin'                         , (np.float32, 3)),
                    ('Len'                          , np.int32),
                    ('Mvir'                         , np.float32),
                    ('CentralMvir'                  , np.float32),
                    ('Rvir'                         , np.float32),
                    ('Vvir'                         , np.float32),
                    ('Vmax'                         , np.float32),
                    ('VelDisp'                      , np.float32),
#                    ('M_otherMvir'                      , np.float32), #temp
                    ('ColdGas'                      , np.float32),
                    ('StellarMass'                  , np.float32),
                    ('BulgeMass'                    , np.float32),
                    ('HotGas'                       , np.float32),
                    ('EjectedMass'                  , np.float32),
                    ('BlackHoleMass'                , np.float32),
                    ('IntraClusterStars'            , np.float32),
                    ('MetalsColdGas'                , np.float32),
                    ('MetalsStellarMass'            , np.float32),
                    ('MetalsBulgeMass'              , np.float32),
                    ('MetalsHotGas'                 , np.float32),
                    ('MetalsEjectedMass'            , np.float32),
                    ('MetalsIntraClusterStars'      , np.float32),
                    ('SfrDisk'                      , np.float32),
                    ('SfrBulge'                     , np.float32),
                    ('SfrDiskZ'                     , np.float32),
                    ('SfrBulgeZ'                    , np.float32),
                    ('DiskRadius'                   , np.float32),
                    ('Cooling'                      , np.float32),
                    ('Heating'                      , np.float32),
                    ('QuasarModeBHaccretionMass'    , np.float32),
                    ('TimeOfLastMajorMerger'         , np.float32),
                    ('TimeOfLastMinorMerger'         , np.float32),
                    ('OutflowRate'                  , np.float32),
                    ('MeanStarAge'                  , np.float32),
                    ('infallMvir'                   , np.float32),
                    ('infallVvir'                   , np.float32),
                    ('infallVmax'                   , np.float32)
                    ]
    names = [Galdesc_full[i][0] for i in xrange(len(Galdesc_full))]
    formats = [Galdesc_full[i][1] for i in xrange(len(Galdesc_full))]
    Galdesc = np.dtype({'names':names, 'formats':formats}, align=True)
    return Galdesc



def sageoutsingle(fname, dir=0, suff='', new=False, disc=False, public=False, old=0, carnage=False, fof64=False, extra_output=False, multidark=False, Nannuli=30):
    # Read a single SAGE output file, returning all the galaxy data in a record array
    
    if type(dir) != str:
        dir = '/Users/astevens/Documents/SAGE/darrencroton-sage-adc90ce8bbcc/output/results/millennium/millennium_mini/'
        
    if len(suff)>0:
        dir = dir[:-1]+suff+'/' # suff is the suffix to the default directory name, eg. suff='_alt'
    
    # Note that although not all the read-in information is used, it's needed to read things in correctly.
    if disc:
        Galdesc = galdtype_adam(Nannuli=Nannuli) if not old else galdtype_DarkSage2016()
    elif public:
        if extra_output:
            Galdesc = galdtype_public_extra()
        elif old==0:
            Galdesc = galdtype_public_old0()
        elif old==1:
            Galdesc = galdtype_public_old1(fof64)
        elif old==2:
            Galdesc = galdtype_public_old2()
        else:
            Galdesc = galdtype_public()
    elif carnage:
        Galdesc = galdtype_carnage_workshop()
    elif multidark:
        Galdesc = galdtype_multidark()
    else:
        Galdesc = galdtype()
    fin = open(dir+fname, 'rb')  # Open the file
    Ntrees = np.fromfile(fin,np.dtype(np.int32),1)  # Read number of trees in file
    NtotGals = np.fromfile(fin,np.dtype(np.int32),1)[0]  # Read number of gals in file.
    GalsPerTree = np.fromfile(fin, np.dtype((np.int32, Ntrees)),1) # Read the number of gals in each tree
    G = np.fromfile(fin, Galdesc, NtotGals) # Read all the galaxy data
    if NtotGals!=len(G): 
        print 'Ngals in sageoutsingle seems contradictory', NtotGals, len(G)
        NtotGals = len(G)
    if new: return G, NtotGals # Literally new feature for easy combining (Feb 2015)
    G = G.view(np.recarray) # Convert into a record array
    fin.close()
    return G

def sagesnap(fpre, firstfile=0, lastfile=7, dir='./', suff='', disc=False, public=False, old=0, carnage=False, extra_output=False, multidark=False, SMfilt=None, Nannuli=30):
    # Read full SAGE snapshot, going through each file and compiling into 1 array
    if disc:
        Galdesc = galdtype_adam(Nannuli=Nannuli) if not old else galdtype_DarkSage2016()
    elif public:
        if extra_output:
            Galdesc = galdtype_public_extra()
        elif old==0:
            Galdesc = galdtype_public_old0()
        elif old==1:
            Galdesc = galdtype_public_old1()
        elif old==2:
            Galdesc = galdtype_public_old2()
        else:
            Galdesc = galdtype_public()
    elif carnage:
        Galdesc = galdtype_carnage_workshop()
    elif multidark:
        Galdesc = galdtype_multidark()
    else:
        Galdesc = galdtype()
    Glist = []
    Ngal = np.array([],dtype=np.int32)
    for i in range(firstfile,lastfile+1):
        G1, N1 = sageoutsingle(fpre+'_'+str(i), dir, suff, True, disc, public, old, extra_output=extra_output, multidark=multidark, Nannuli=Nannuli)
        if SMfilt is not None:
            G1 = G1[G1.view(np.recarray).StellarMass>=SMfilt]
            N1 = len(G1)
        Glist += [G1]
        Ngal = np.append(Ngal,N1)
    G = np.empty(sum(Ngal), dtype=Galdesc)
    for i in range(firstfile,lastfile+1):
        j = i-firstfile
        G[sum(Ngal[:j]):sum(Ngal[:j+1])] = Glist[j][0:Ngal[j]].copy()
    G = G.view(np.recarray)
    return G


def reduced_DarkSage2016_catalogue(indir):
    min_num_list = [0,50,100,150,200]
    max_num_list = [49,99,149,199,255]
    G = {}

    fields_and_dtype = [
    'GalaxyIndex', 'i8',
    'Len', 'i4',
    'StellarMass', 'f4', # all masses are in [10^10 h^-1 M_sun]
    'StellarDiscMass', 'f4',
    'ColdGas',  'f4',
    'HotGas',  'f4',
    'HIMass',  'f4',
    'H2Mass',  'f4',
    'Type', 'i1',
    'StellarHalfMassRadius',  'f4', # all sizes and positions are in [h^-1 Mpc]
    'Stellar90PercentRadius',  'f4',
    'SFR',  'f4',
    'TimeSinceLastMajorMerger',  'f4',
    'j_stars',  'f4', # disc specific angular momenta in [kpc km s^-1]
    'j_HI',  'f4',
    'Mvir',  'f4',
    'Rvir',  'f4',
    'Vvir',  'f4',
    'jX_halo',  'f4', # halo specific angular momenta components [h^-1 Mpc km s^-1]
    'jY_halo',  'f4',
    'jZ_halo',  'f4',
    'X',  'f4',
    'Y',  'f4',
    'Z', 'f4']
    fields = fields_and_dtype[0::2]
    dtypes = fields_and_dtype[1::2]
    
    for field, dt in zip(fields, dtypes): G[field] = np.array([], dtype=dt)

    for minfileno, maxfileno in zip(min_num_list, max_num_list):
        f = open(indir+'DarkSage2016_z0_reduced_'+str(minfileno)+'_'+str(maxfileno)+'.bin', 'rb')
        Ngal = np.fromfile(f, 'i8', 1)[0]
        for field, dt in zip(fields, dtypes): G[field] = np.append(G[field], np.fromfile(f, dt, Ngal))
        f.close()
        
    return G
    


def zlistmm():
	# Call the list of strings which correspond to the redshift values of the output of the mini millennium files (i.e. so one can call to read the files in a list).  This is specific to details of how SAGE was run (i.e. in the desired_outputsnaps file)
	
	return ['0.000','0.020','0.041','0.064','0.089','0.116','0.144','0.175','0.208','0.242','0.280','0.320','0.362','0.408','0.457','0.509','0.564','0.624','0.687','0.755','0.828','0.905','0.989','1.078','1.173','1.276','1.386','1.504','1.630','1.766','1.913','2.070','2.239','2.422','2.619','2.831','3.060','3.308','3.576','3.866','4.179','4.520','4.888','5.289','5.724','6.197','6.712','7.272','7.883','8.550','9.278','10.073', '10.944', '11.897', '12.941', '14.086', '15.343', '16.725', '18.244', '19.916', '30.000', '50.000', '79.998', '127.000']


def zlist_gigglezHR():
    return ['0.000', '0.020', '0.042', '0.064', '0.089', '0.115', '0.144', '0.175', '0.207', '0.243', '0.279', '0.320', '0.362', '0.408', '0.457', '0.508', '0.564', '0.623', '0.687', '0.756', '0.827', '0.905', '0.990', '1.080', '1.174', '1.278', '1.388', '1.505', '1.629', '1.768', '1.917', '2.066', '2.234', '2.427', '2.620', '2.824', '3.059', '3.310', '3.577', '3.858', '4.186', '4.532', '4.892', '5.317', '5.691', '6.203', '6.729', '7.248', '7.863', '8.608', '9.328', '10.201', '11.284', '12.674', '14.538', '17.200', '21.403', '29.343', '52.709']

def zlist_gigglezMR():
    return ['0.000', '0.020', '0.041', '0.064', '0.089', '0.116', '0.144', '0.175', '0.208', '0.242', '0.280', '0.320', '0.362', '0.408', '0.457', '0.509', '0.564', '0.624', '0.687', '0.755', '0.828', '0.905', '0.989', '1.078', '1.173', '1.276', '1.386', '1.504', '1.630', '1.766', '1.913', '2.070', '2.239', '2.422', '2.619', '2.831', '3.060', '3.308', '3.576', '3.866', '4.179', '4.520', '4.888', '5.289', '5.724', '6.197', '6.712', '7.272', '7.883', '8.550', '9.278', '10.073', '10.994']

def zlist_MDPL():
    return ['0.000', '0.022', '0.045', '0.069', '0.093', '0.117', '0.142', '0.168', '0.194', '0.221', '0.248', '0.276', '0.304', '0.364', '0.394', '0.425', '0.457', '0.490', '0.523', '0.557', '0.592', '0.628', '0.664', '0.702', '0.740', '0.779', '0.819', '0.859', '0.901', '0.944', '0.987', '1.032', '1.077', '1.124', '1.172', '1.220', '1.270', '1.321', '1.372', '1.425', '1.480', '1.535', '1.593', '1.650', '1.710', '1.771', '1.833', '1.896', '1.961', '2.028', '2.095', '2.165', '2.235', '2.308', '2.382', '2.458', '2.535', '2.614', '2.695', '2.778', '2.862', '2.949', '3.037', '3.127', '3.221', '3.314', '3.411', '3.511', '3.610', '3.715', '3.819', '3.929', '4.038', '4.152', '4.266', '4.385', '4.507', '4.627', '4.754', '4.882', '5.017', '5.150', '5.289', '5.431', '5.575', '5.720', '5.873', '6.022', '6.184', '6.342', '6.508', '6.675', '6.849', '7.026', '7.203', '7.389', '7.576', '7.764', '7.961', '8.166', '8.372', '8.579', '8.794', '9.010', '9.235', '9.471', '9.707', '9.941', '10.186', '10.442', '10.696', '10.962', '11.225', '11.500', '11.771', '12.055', '12.351', '12.661', '12.966', '13.265', '13.599', '13.925', '14.244', '14.601', '14.949']

def scalefactors_TNG():
    return np.array([0.0476, 0.0625, 0.0769, 0.0833, 0.0909, 0.0964, 0.1, 0.1058, 0.1111, 0.1161, 0.121, 0.125, 0.1334, 0.1429, 0.1464, 0.1533, 0.1606, 0.1667, 0.1763, 0.1847, 0.1935, 0.2, 0.2124, 0.2226, 0.2332, 0.25, 0.2561, 0.2677, 0.279, 0.2902, 0.3012, 0.3121, 0.3228, 0.3333, 0.3439, 0.3543, 0.3645, 0.3747, 0.3848, 0.3948, 0.4, 0.4147, 0.4246, 0.4344, 0.4441, 0.4538, 0.4635, 0.4731, 0.4827, 0.4923, 0.5, 0.5115, 0.521, 0.5306, 0.5401, 0.5496, 0.5591, 0.5687, 0.5782, 0.5882, 0.5973, 0.6069, 0.6164, 0.626, 0.6357, 0.6453, 0.655, 0.6667, 0.6744, 0.6841, 0.6939, 0.7037, 0.7143, 0.7225, 0.7334, 0.7434, 0.7534, 0.7635, 0.7692, 0.7838, 0.794, 0.8043, 0.8146, 0.825, 0.8333, 0.8459, 0.8564, 0.8671, 0.8778, 0.8885, 0.8993, 0.9091, 0.9212, 0.9322, 0.9433, 0.9545, 0.9657, 0.9771, 0.9885, 1.0])

def zarr_IllustrisTNG():
    a = scalefactors_TNG()
    return 1./a - 1.


# Reading of TAO data
def tao_csv(fname, keylist=None):
    with open(fname, 'r') as f: line = f.readline()
    keys = line.split(',')
    keys[-1] = keys[-1][:-1] # gets rid of \n at the end
    if keylist==None: keylist = keys
    print 'Number of properties =', len(keys)
    dict = {}
    for i, key in enumerate(keys):
        
        if key in ['Total_Particles', 'Maximum_Number_of_Particles_over_History', 'Snapshot_Number', 'Galaxy_Classification']:
            datatype = np.int32
        elif key in ['Galaxy_ID', 'Central_Galaxy_ID', 'Simulation_Halo_ID']:
            datatype = np.int64
        else:
            datatype = np.float32
        if key in keylist:
            print 'Reading', i, key
            dict[key] = np.loadtxt(fname, skiprows=1, usecols=(i,), dtype=datatype, delimiter=', ')
    return dict


# ====== MASSIVEBLACK-related code ======

def e5snaphead(fname, dir=0):
	# Read in the important information regarding a snapshot and the simulation in general from e5 from the header of a file
	
	if type(dir) != str:
		dir = '/Users/astevens/Documents/6-MonthProject/e5/' # Default directory for files
	
#    if fname[-2:-1] != '.': # Don't need to specify a file to read a header in for the function, but one needs to be internally specified if one hasn't been already.
#        fname += '.0'
	
	f = open(dir+fname, 'rb')
	Nbytes1 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the header uses
	N = np.fromfile(f, 'u4', 6) # Number of particles for each particle type in this file
	mass_pt = np.fromfile(f, 'f8', 6) # Mass of each particle type.  If 0 then it varies for each particle of that type
	a = np.fromfile(f, 'f8', 1)[0] # Expansion factor (normalised to 1 at z=0)
	z = np.fromfile(f, 'f8', 1)[0] # Redshift of snapshot
	flag_sfr = np.fromfile(f, 'i4', 1)[0] # Flag for star formation rate
	flag_feedback = np.fromfile(f, 'i4', 1)[0] # Flag for feedback
	Ntot = np.fromfile(f, 'u4', 6) # Total number of particles for each particle type in the entire simulation
	flag_cool = np.fromfile(f, 'i4', 1)[0] # Flag for cooling
	Nfiles = np.fromfile(f, 'i4', 1)[0] # Number of files for each snapshot
	boxsize = np.fromfile(f, 'f8', 1)[0] # Size of box if periodic boundary conditions are used
	Omega_M = np.fromfile(f, 'f8', 1)[0] # Omega Matter
	Omega_L = np.fromfile(f, 'f8', 1)[0] # Omega (Lambda) Dark Energy
	h = np.fromfile(f, 'f8', 1)[0] # Little Hubble h
	flag_StarAge = np.fromfile(f, 'i4', 1)[0] # Flag for the creation times of stars
	flag_metals = np.fromfile(f, 'i4', 1)[0] # Flag for metallicity values
	f.close()
	
	return z, Ntot, Nfiles, boxsize, Omega_M, Omega_L, h


def e5snapids(fpre, dir=0):
	# Read in just the IDs from a snapshot
	
	if type(dir) != str:
		dir = '/Users/astevens/Documents/6-MonthProject/e5/' # Default directory for files
	
	z, Ntot, Nfiles, boxsize, Omega_M, Omega_L, h = e5snaphead(fpre)
	id_g, id_dm, id_s, id_bh = np.zeros(Ntot[0],dtype='u8'), np.zeros(Ntot[1],dtype='u8'), np.zeros(Ntot[4],dtype='u8'), np.zeros(Ntot[5],dtype='u8') # Initialize the ID arrays
	
	Naccum = np.zeros(6) # Accumulative vector for N
	
	for i in xrange(Nfiles):
		fname = fpre + '.' + str(i)
		f = open(dir+fname, 'rb')
		Nbytes = np.fromfile(f, 'i4', 1)[0] # Number of bytes for seeking
		N = np.fromfile(f, 'u4', 6) # Number of particles for each particle type in this file
		Nsum = sum(N) # Total number of particles in the file
		
		f.seek(Nbytes+8) # Skip to the end of the header block
		Nbytes += np.fromfile(f, 'i4', 1)[0] # Bytes for next block
		f.seek(Nbytes+16)
		Nbytes += np.fromfile(f, 'i4', 1)[0]
		f.seek(Nbytes+24)
		
		print np.fromfile(f, 'i4', 1)[0]
		ids = np.fromfile(f, 'u8', Nsum)
		print np.fromfile(f, 'i4', 1)[0]
		
		id_g[Naccum[0]:Naccum[0]+N[0]] = ids[:N[0]]
		id_dm[Naccum[1]:Naccum[1]+N[1]] = ids[N[0]:N[0]+N[1]]
		id_s[Naccum[4]:Naccum[4]+N[4]] = ids[sum(N[:4]):sum(N[:5])]
		id_bh[Naccum[5]:Naccum[5]+N[5]] = ids[sum(N[:5]):]
		
		f.close()
		Naccum += N
	
	pid = np.concatenate((id_g,id_dm,id_s,id_bh))
	return pid



def e5snapfile(fname, dir=0, t=1366, n=0):
	# Read in a single file from a snapshot of the e5 simulation
	
	if fname==None:
		fname = 'snapshot_'+str(int(t))+'.'+str(int(n))
	
	if type(dir) != str:
		dir = '/Users/astevens/Documents/6-MonthProject/e5/' # Default directory for files
	
	f = open(dir+fname, 'rb')
	
	Nbytes1 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the header uses
	N = np.fromfile(f, 'u4', 6) # Number of particles for each particle type in this file
	Nsum = sum(N) # Total number of particles in the file
	mass_pt = np.fromfile(f, 'f8', 6) # Mass of each particle type.  If 0 then it varies for each particle of that type
	Nmass = sum(N[np.argwhere(mass_pt==0.0)]) # Number of particles in the file with individual masses to be read in
	a = np.fromfile(f, 'f8', 1)[0] # Expansion factor (normalised to 1 at z=0)
	z = np.fromfile(f, 'f8', 1)[0] # Redshift of snapshot
	flag_sfr = np.fromfile(f, 'i4', 1)[0] # Flag for star formation rate
	flag_feedback = np.fromfile(f, 'i4', 1)[0] # Flag for feedback
	Ntot = np.fromfile(f, 'u4', 6) # Total number of particles for each particle type in the entire simulation
	flag_cool = np.fromfile(f, 'i4', 1)[0] # Flag for cooling
	Nfiles = np.fromfile(f, 'i4', 1)[0] # Number of files for each snapshot
	boxsize = np.fromfile(f, 'f8', 1)[0] # Size of box if periodic boundary conditions are used
	Omega_M = np.fromfile(f, 'f8', 1)[0] # Omega Matter
	Omega_L = np.fromfile(f, 'f8', 1)[0] # Omega (Lambda) Dark Energy
	h = np.fromfile(f, 'f8', 1)[0] # Little Hubble h
	flag_StarAge = np.fromfile(f, 'i4', 1)[0] # Flag for the creation times of stars
	flag_metals = np.fromfile(f, 'i4', 1)[0] # Flag for metallicity values
	NallHW = np.fromfile(f, 'u4', 6) # Don't fully understand this.  To do with simulations that use >2^32 particles
	flag_entropy = np.fromfile(f, 'i4', 1)[0] # Flag that the initial conditions contain entropy instead of thermal energy in the thermal energy block
	flag_double = np.fromfile(f, 'i4', 1)[0] # Don't know what this flag is for
	flag_ic_info = np.fromfile(f, 'i4', 1)[0] # Don't know what this flag is for
	flag_scale = np.fromfile(f, 'i4', 1)[0] # Don't know what this flag is for
	unused = np.fromfile(f, 'u4', 12) # Apparently unused stuff
	Nbytes1 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the header uses (again)
	
	Nbytes2 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (particle positions)
	pos = np.fromfile(f, np.dtype(('f8',3)), Nsum) # Particle positions (divide by h to have in comoving kpc)
	Nbytes2 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes3 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (particle velocities)
	vel = np.fromfile(f, np.dtype(('f8',3)), Nsum) # Particle velocities (should be in km/s)
	Nbytes3 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes4 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (particle IDs)
	ids = np.fromfile(f, 'u8', Nsum) # Particle IDs
	Nbytes4 = np.fromfile(f, 'i4', 1)[0]
	
	if Nmass>0: # Block won't exist if there are no particles with non-standard masses
		Nbytes5 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (particle masses for those that vary)
		mass = np.fromfile(f, 'f8', Nmass) # The masses of the particles without a standard mass (divide by h to have in 10^10 solar masses)
		Nbytes5 = np.fromfile(f, 'i4', 1)[0]
	
	if N[0]>0: # Block won't exist if there aren't any gas particles (this logically will probably apply to the next block too, but it's only explicity stated in the User Guide for this block.  Mind you, this doesn't fully follow the User Guide in any case.)
		Nbytes6 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (internal energy of the gas particles)
		U = np.fromfile(f, 'f8', N[0]) # Internal energy per unit mass of the gas particles ((km/s)^2)
		Nbytes6 = np.fromfile(f, 'i4', 1)[0]
	else:
		print 'NO GAS PARTICLES.  FREAK OUT!'
	
	Nbytes7 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (local density of the gas particles)
	rho = np.fromfile(f, 'f8', N[0]) # Local density of the gas particles (multiply by h^2 to have in 10^10 solar masses per cubic kiloparsec)
	Nbytes7 = np.fromfile(f, 'i4', 1)[0]
	
	# This is where the data blocks become different to the User Guide and follow what's listed under "class schema" in Yu's code
	if flag_cool==1:
		Nbytes8 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses
		ye = np.fromfile(f, 'f8', N[0]) # Don't know what this property is but it's related to cooling!
		Nbytes8 = np.fromfile(f, 'i4', 1)[0]
		
		Nbytes9 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses
		xHI = np.fromfile(f, 'f8', N[0]) # Don't know what this property is either but it's also related to cooling!
		Nbytes9 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes10 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (smmothing length of gas particles)
	sml = np.fromfile(f, 'f8', N[0]) # Smoothing length of gas particles
	Nbytes10 = np.fromfile(f, 'i4', 1)[0]
	
	if flag_sfr==1:
		Nbytes11 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (star formation rate of gas particles)
		sfr = np.fromfile(f, 'f8', N[0]) # Star formation rate of gas particles
		Nbytes11 = np.fromfile(f, 'i4', 1)[0]
	
	if flag_StarAge==1:
		Nbytes12 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (star formation time)
		birth = np.fromfile(f, 'f8', N[4]) # Time at which each star particle was born
		Nbytes12 = np.fromfile(f, 'i4', 1)[0]
	
	if flag_metals==1:
		Nbytes13 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (metallicity of star and gas particles)
		met = np.fromfile(f, 'f8', N[0]+N[4]) # Metallicity of gas and star particles (presumably in that order)
		Nbytes13 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes14 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the next block of data uses (black hole masses)
	bhm = np.fromfile(f, 'f8', N[5]) # Accretional black hole mass (integrated from bhmdot - not the actual particle masses but rather a theoretical value)
	Nbytes14 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes15 = np.fromfile(f, 'i4', 1)[0]
	bhmdot = np.fromfile(f, 'f8', N[5]) # Black hole mass rate of change - accretion rate
	Nbytes15 = np.fromfile(f, 'i4', 1)[0]
	
	Nbytes16 = np.fromfile(f, 'i4', 1)[0]
	bhnprog = np.fromfile(f, 'i8', N[5]) # Presumably this is the number of progenitors for the black holes
	Nbytes16 = np.fromfile(f, 'i4', 1)[0]
	
	# Format the masses
	newmass = [] # Initialize new mass variable which will house the masses of each particle
	N_vm = np.zeros(6) # Same as N but will have entries for zero where mass_pt is non-zero.  It hence lists the number of each particle type with variable mass.
	for i in xrange(6):
		if mass_pt[i]==0:
			N_vm[i] = N[i]
			ent = sum(N_vm[:i]) # Entry for where to find masses
			newmass += list(mass[ent:ent+N[i]]) # Extract the masses of the particles
		else:
			newmass += [mass_pt[i]]*N[i] # All particles of that type have the same mass
	mass = np.array(newmass)
	
	# Change units and format to match how Marie's sims were read in, so that the info may use my routines directly
	pos = pos*a*1e3 / h # Convert positions to physical parsecs
	sml = sml*1e3 / h # Convert smoothing length to parsecs
	mass = mass*1e10 / h # Convert mass to solar masses
	rho = rho*10*h*h / (a**3) # Convert density to solar masses per cubic parsec
	birth_z = 1/birth - 1 # Convert birth to redshift
	U *= 1e6 # Convert specific energy to (m/s)^2 = J/kg
	
	f.close()
	
	return N, pos, vel, ids, mass, met, sml, rho, U, sfr, birth_z, bhm, bhmdot


def e5snapfullold(fpre, dir=0):
	"""
		THIS IS AN OLD VERSION OF CODE THAT DIDN'T FUNCTION WELL AT ALL.  DON'T USE IT.
		Read in a full snapshot from e5 (should be [but not necessary to be] 8 individual files)
		fpre = start of the filename (a full filename will follow this with a dot and a number, e.g. ".0")
		dir = directory where the files are held
		"""
	print 'Running e5snapfull'
	start = time()
	
	if type(dir) != str:
		dir = '/Users/astevens/Documents/6-MonthProject/e5/' # Default directory for files
	
	print 'Reading in header, t=', time()-start, ' s'
	z, Ntot, Nfiles, boxsize, Omega_M, Omega_L, h = e5snaphead(fpre,dir)
	
	if Ntot[2]!=0 or Ntot[3]!=0:
		print "ERROR: There are bulge and disk particles in this snapshot.  e5snapfull needs to be updated to accommodate this."
	
	print 'Initializing variables, t=', time()-start, ' s'
	# Initialize variables
	x_g, y_g, z_g, x_dm, y_dm, z_dm, x_s, y_s, z_s, vx_g, vy_g, vz_g, vx_dm, vy_dm, vz_dm, vx_s, vy_s, vz_s, id_g, id_dm, id_s, mass_g, mass_dm, mass_s, met_g, met_s, sml, rho, U, sfr, birth_z = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
	
	for i in xrange(Nfiles):
		fname = fpre + '.' + str(i)
		
		print '\nReading in snapfile ', i, ' t=', time()-start, ' s'
		N, pos, vel, ids, mass, met, sml_f, rho_f, U_f, sfr_f, birth_z_f = e5snapfile(fname,dir) # The "_f" indicates for a single file.  Could clearly have written for each other variable too, but I don't want to use the others' names without "_f" for this function's output.
		
		print 'Splitting coordinates, t=', time()-start, ' s'
		# Split coordinates of different particles and slot info of each subsequent file at the bottom
		x_g += list(pos[:N[0],0])
		y_g += list(pos[:N[0],1])
		z_g += list(pos[:N[0],2]) # "_g" indicates gas particles
		x_dm += list(pos[N[0]:N[0]+N[1],0])
		y_dm += list(pos[N[0]:N[0]+N[1],1])
		z_dm += list(pos[N[0]:N[0]+N[1],2]) # "_dm" indicates dark matter particles
		x_s += list(pos[sum(N[:4]):sum(N[:5]),0])
		y_s += list(pos[sum(N[:4]):sum(N[:5]),1])
		z_s += list(pos[sum(N[:4]):sum(N[:5]),2]) # "_s" indicates star particles
		del pos # Delete the old array to avoid clogging RAM unnecessarily
		
		print 'Splitting velocities, t=', time()-start, ' s'
		# Ditto velocities
		vx_g += list(vel[:N[0],0])
		vy_g += list(vel[:N[0],1])
		vz_g += list(vel[:N[0],2])
		vx_dm += list(vel[N[0]:N[0]+N[1],0])
		vy_dm += list(vel[N[0]:N[0]+N[1],1])
		vz_dm += list(vel[N[0]:N[0]+N[1],2])
		vx_s += list(vel[sum(N[:4]):sum(N[:5]),0])
		vy_s += list(vel[sum(N[:4]):sum(N[:5]),1])
		vz_s += list(vel[sum(N[:4]):sum(N[:5]),2])
		del vel
		
		print 'Splitting IDs, t=', time()-start, ' s'
		# Ditto IDs
		id_g += list(ids[:N[0]])
		id_dm += list(ids[N[0]:N[0]+N[1]])
		id_s += list(ids[sum(N[:4]):sum(N[:5])])
		del ids
		
		print 'Splitting masses, t=', time()-start, ' s'
		# Ditto masses
		mass_g += list(mass[:N[0]])
		mass_dm += list(mass[N[0]:N[0]+N[1]])
		mass_s += list(mass[sum(N[:4]):sum(N[:5])])
		del mass
		
		print 'Splitting metallicities, t=', time()-start, ' s'
		# Ditto metallicities
		met_g += list(met[:N[0]])
		met_s += list(met[N[0]:N[0]+N[4]])
		del met
		
		print 'Splitting other properties, t=', time()-start, ' s'
		# Add the remaining properties
		sml += list(sml_f)
		rho += list(rho_f)
		U += list(U_f)
		sfr += list(sfr_f)
		birth_z += list(birth_z_f)
	
	print 'Converting to numpy arrays, t=', time()-start, ' s'
	# Convert everything to numpy arrays before outputting
	x_g, y_g, z_g, x_dm, y_dm, z_dm, x_s, y_s, z_s, vx_g, vy_g, vz_g, vx_dm, vy_dm, vz_dm, vx_s, vy_s, vz_s, id_g, id_dm, id_s, mass_g, mass_dm, mass_s, met_g, met_s, sml, rho, U, sfr, birth_z = np.array(x_g), np.array(y_g), np.array(z_g), np.array(x_dm), np.array(y_dm), np.array(z_dm), np.array(x_s), np.array(y_s), np.array(z_s), np.array(vx_g), np.array(vy_g), np.array(vz_g), np.array(vx_dm), np.array(vy_dm), np.array(vz_dm), np.array(vx_s), np.array(vy_s), np.array(vz_s), np.array(id_g), np.array(id_dm), np.array(id_s), np.array(mass_g), np.array(mass_dm), np.array(mass_s), np.array(met_g), np.array(met_s), np.array(sml), np.array(rho), np.array(U), np.array(sfr), np.array(birth_z)
	
	return [x_g, y_g, z_g, vx_g, vy_g, vz_g, id_g, mass_g, met_g, sml, rho, U, sfr], [x_dm, y_dm, z_dm, vx_dm, vy_dm, vz_dm, id_dm, mass_dm], [x_s, y_s, z_s, vx_s, vy_s, vz_s, id_s, mass_s, met_s, birth_z]


def e5snapfull(fpre=None, dir=0, t=1366, haloids=True):
    """
    Read in a full snapshot from e5 (should be [but not necessary to be] 8 individual files)
    fpre = start of the filename (a full filename will follow this with a dot and a number, e.g. ".0")
    dir = directory where the files are held
    t = time/redshift indicator (i.e. which snapshot).  Might already be part of fpre if defined
    """
    print 'Running e5snapfull'
    start = time()

    if fpre==None:
        fpre = 'snapshot_'+str(int(t))

    if type(dir) != str:
        dir = '/Users/astevens/Documents/6-MonthProject/e5/' # Default directory for files

    print 'Reading in header, t=', time()-start, ' s'
    z, Ntot, Nfiles, boxsize, Omega_M, Omega_L, h = e5snaphead(fpre,dir)

    if Ntot[2]!=0 or Ntot[3]!=0:
        print "ERROR: There are bulge and disk particles in this snapshot.  e5snapfull needs to be updated to accommodate this."

    # Initialize variables
    print 'Initializing gas variables, t=', time()-start, ' s'
    x_g, y_g, z_g, vx_g, vy_g, vz_g, id_g, mass_g, met_g, sml, rho, U, sfr = np.zeros(Ntot[0]), np.zeros(Ntot[0]), np.zeros(Ntot[0]), np.zeros(Ntot[0]), np.zeros(Ntot[0]), np.zeros(Ntot[0]), np.zeros(Ntot[0],dtype='u8'), np.zeros(Ntot[0]), np.zeros(Ntot[0]), np.zeros(Ntot[0]), np.zeros(Ntot[0]), np.zeros(Ntot[0]), np.zeros(Ntot[0])
    print 'Initializing DM variables, t=', time()-start, ' s'
    x_dm, y_dm, z_dm, vx_dm, vy_dm, vz_dm, id_dm, mass_dm = np.zeros(Ntot[1]), np.zeros(Ntot[1]), np.zeros(Ntot[1]), np.zeros(Ntot[1]), np.zeros(Ntot[1]), np.zeros(Ntot[1]), np.zeros(Ntot[1],dtype='u8'), np.zeros(Ntot[1])
    print 'Initializing star variables, t=', time()-start, ' s'
    x_s, y_s, z_s, vx_s, vy_s, vz_s, id_s, mass_s, met_s, birth_z = np.zeros(Ntot[4]), np.zeros(Ntot[4]), np.zeros(Ntot[4]), np.zeros(Ntot[4]), np.zeros(Ntot[4]), np.zeros(Ntot[4]), np.zeros(Ntot[4],dtype='u8'), np.zeros(Ntot[4]), np.zeros(Ntot[4]), np.zeros(Ntot[4])
    print 'Initializing black hole variables, t=', time()-start, ' s'
    x_bh, y_bh, z_bh, vx_bh, vy_bh, vz_bh, id_bh, mass_bh, amass_bh, amdot_bh = np.zeros(Ntot[5]), np.zeros(Ntot[5]), np.zeros(Ntot[5]), np.zeros(Ntot[5]), np.zeros(Ntot[5]), np.zeros(Ntot[5]), np.zeros(Ntot[5],dtype='u8'), np.zeros(Ntot[5]), np.zeros(Ntot[5]), np.zeros(Ntot[5])
    Naccum = np.zeros(6) # Accumulative vector for N

    print 'reading', Nfiles, 'files for snapshot'
    for i in xrange(Nfiles):
        fname = fpre + '.' + str(i) if Nfiles>1 else fpre

        print '\nReading in snapfile ', i, ' t=', time()-start, ' s'
        N, pos, vel, ids, mass, met, sml_f, rho_f, U_f, sfr_f, birth_z_f, bhm, bhmdot = e5snapfile(fname,dir) # The "_f" indicates for a single file.  Could clearly have written for each other variable too, but I don't want to use the others' names without "_f" for this function's output.

        print 'Splitting coordinates, t=', time()-start, ' s'
        # Split coordinates of different particles and slot info of each subsequent file at the bottom
        x_g[Naccum[0]:Naccum[0]+N[0]] = pos[:N[0],0]
        y_g[Naccum[0]:Naccum[0]+N[0]] = pos[:N[0],1]
        z_g[Naccum[0]:Naccum[0]+N[0]] = pos[:N[0],2] # "_g" indicates gas particles
        x_dm[Naccum[1]:Naccum[1]+N[1]] = pos[N[0]:N[0]+N[1],0]
        y_dm[Naccum[1]:Naccum[1]+N[1]] = pos[N[0]:N[0]+N[1],1]
        z_dm[Naccum[1]:Naccum[1]+N[1]] = pos[N[0]:N[0]+N[1],2] # "_dm" indicates dark matter particles
        x_s[Naccum[4]:Naccum[4]+N[4]] = pos[sum(N[:4]):sum(N[:5]),0]
        y_s[Naccum[4]:Naccum[4]+N[4]] = pos[sum(N[:4]):sum(N[:5]),1]
        z_s[Naccum[4]:Naccum[4]+N[4]] = pos[sum(N[:4]):sum(N[:5]),2] # "_s" indicates star particles
        x_bh[Naccum[5]:Naccum[5]+N[5]] = pos[sum(N[:5]):,0]
        y_bh[Naccum[5]:Naccum[5]+N[5]] = pos[sum(N[:5]):,1]
        z_bh[Naccum[5]:Naccum[5]+N[5]] = pos[sum(N[:5]):,2] # "_bh" indicates black hole particles
        print 'Deleting pos, t=', time()-start, ' s'
        del pos # Delete the old array to avoid clogging RAM unnecessarily

        print 'Splitting velocities, t=', time()-start, ' s'
        # Ditto velocities
        vx_g[Naccum[0]:Naccum[0]+N[0]] = vel[:N[0],0]
        vy_g[Naccum[0]:Naccum[0]+N[0]] = vel[:N[0],1]
        vz_g[Naccum[0]:Naccum[0]+N[0]] = vel[:N[0],2]
        vx_dm[Naccum[1]:Naccum[1]+N[1]] = vel[N[0]:N[0]+N[1],0]
        vy_dm[Naccum[1]:Naccum[1]+N[1]] = vel[N[0]:N[0]+N[1],1]
        vz_dm[Naccum[1]:Naccum[1]+N[1]] = vel[N[0]:N[0]+N[1],2]
        vx_s[Naccum[4]:Naccum[4]+N[4]] = vel[sum(N[:4]):sum(N[:5]),0]
        vy_s[Naccum[4]:Naccum[4]+N[4]] = vel[sum(N[:4]):sum(N[:5]),1]
        vz_s[Naccum[4]:Naccum[4]+N[4]] = vel[sum(N[:4]):sum(N[:5]),2]
        vx_bh[Naccum[5]:Naccum[5]+N[5]] = vel[sum(N[:5]):,0]
        vy_bh[Naccum[5]:Naccum[5]+N[5]] = vel[sum(N[:5]):,1]
        vz_bh[Naccum[5]:Naccum[5]+N[5]] = vel[sum(N[:5]):,2]
        del vel

        print 'Splitting IDs, t=', time()-start, ' s'
        # Ditto IDs
        id_g[Naccum[0]:Naccum[0]+N[0]] = ids[:N[0]]
        id_dm[Naccum[1]:Naccum[1]+N[1]] = ids[N[0]:N[0]+N[1]]
        id_s[Naccum[4]:Naccum[4]+N[4]] = ids[sum(N[:4]):sum(N[:5])]
        id_bh[Naccum[5]:Naccum[5]+N[5]] = ids[sum(N[:5]):]
        del ids

        print 'Splitting masses, t=', time()-start, ' s'
        # Ditto masses
        mass_g[Naccum[0]:Naccum[0]+N[0]] = mass[:N[0]]
        mass_dm[Naccum[1]:Naccum[1]+N[1]] = mass[N[0]:N[0]+N[1]]
        mass_s[Naccum[4]:Naccum[4]+N[4]] = mass[sum(N[:4]):sum(N[:5])]
        mass_bh[Naccum[5]:Naccum[5]+N[5]] = mass[sum(N[:5]):]
        del mass

        print 'Splitting metallicities, t=', time()-start, ' s'
        # Ditto metallicities
        met_g[Naccum[0]:Naccum[0]+N[0]] = met[:N[0]]
        met_s[Naccum[4]:Naccum[4]+N[4]] = met[N[0]:N[0]+N[4]]
        del met

        print 'Splitting other properties, t=', time()-start, ' s'
        # Add the remaining properties
        sml[Naccum[0]:Naccum[0]+N[0]] = sml_f
        rho[Naccum[0]:Naccum[0]+N[0]] = rho_f
        U[Naccum[0]:Naccum[0]+N[0]] = U_f
        sfr[Naccum[0]:Naccum[0]+N[0]] = sfr_f
        birth_z[Naccum[4]:Naccum[4]+N[4]] = birth_z_f
        amass_bh[Naccum[5]:Naccum[5]+N[5]] = bhm
        amdot_bh[Naccum[5]:Naccum[5]+N[5]] = bhmdot

        Naccum += N # Accumulate the values of N thus far to know where to start the next dumping into arrays
                                
    if haloids:
        # Import group and subhalo ID information
        f = open('e5/snapshotids'+fpre[8:],'rb')
        gid = np.fromfile(f, 'i4', sum(Ntot)) # Group IDs
        shid = np.fromfile(f, 'i4', sum(Ntot)) # Subhalo IDs
        f.close()
        
        # Split group and subhalo IDs into particle species
        gid_g, gid_dm, gid_s, gid_bh = gid[:Ntot[0]], gid[Ntot[0]:sum(Ntot[:2])], gid[sum(Ntot[:2]):sum(Ntot[:5])], gid[sum(Ntot[:5]):sum(Ntot)]
        shid_g, shid_dm, shid_s, shid_bh = shid[:Ntot[0]], shid[Ntot[0]:sum(Ntot[:2])], shid[sum(Ntot[:2]):sum(Ntot[:5])], shid[sum(Ntot[:5]):sum(Ntot)]
        
        return [x_g, y_g, z_g, vx_g, vy_g, vz_g, id_g, mass_g, met_g, sml, rho, U, sfr, gid_g, shid_g], [x_dm, y_dm, z_dm, vx_dm, vy_dm, vz_dm, id_dm, mass_dm, gid_dm, shid_dm], [x_s, y_s, z_s, vx_s, vy_s, vz_s, id_s, mass_s, met_s, birth_z, gid_s, shid_s], [x_bh, y_bh, z_bh, vx_bh, vy_bh, vz_bh, id_bh, mass_bh, amass_bh, amdot_bh, gid_bh, shid_bh]
    else:
        return [x_g, y_g, z_g, vx_g, vy_g, vz_g, id_g, mass_g, met_g, sml, rho, U, sfr], [x_dm, y_dm, z_dm, vx_dm, vy_dm, vz_dm, id_dm, mass_dm], [x_s, y_s, z_s, vx_s, vy_s, vz_s, id_s, mass_s, met_s, birth_z], [x_bh, y_bh, z_bh, vx_bh, vy_bh, vz_bh, id_bh, mass_bh, amass_bh, amdot_bh]


def e5snapfullalt(fpre,dir=0,haloids=True):
    #Trying an alternative way to read in all the data for a snapshot at once
    print 'Running e5snapfull'
    start = time()
        
    if type(dir) != str:
        dir = '/Users/astevens/Documents/6-MonthProject/e5/' # Default directory for files
                
    print 'Reading in header, t=', time()-start, ' s'
    z, Ntot, Nfiles, boxsize, Omega_M, Omega_L, h = e5snaphead(fpre,dir)
    
    if Ntot[2]!=0 or Ntot[3]!=0:
        print "ERROR: There are bulge and disk particles in this snapshot.  e5snapfull needs to be updated to accommodate this."
            
    # Initialize variables
    print 'Initializing gas variables, t=', time()-start, ' s'
    x_g, y_g, z_g, vx_g, vy_g, vz_g, id_g, mass_g, met_g, sml, rho, U, sfr = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.zeros(Ntot[0],dtype='u8'), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    print 'Initializing DM variables, t=', time()-start, ' s'
    x_dm, y_dm, z_dm, vx_dm, vy_dm, vz_dm, id_dm, mass_dm = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.zeros(Ntot[1],dtype='u8'), np.array([])
    print 'Initializing star variables, t=', time()-start, ' s'
    x_s, y_s, z_s, vx_s, vy_s, vz_s, id_s, mass_s, met_s, birth_z = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.zeros(Ntot[4],dtype='u8'), np.array([]), np.array([]), np.array([])
    print 'Initializing black hole variables, t=', time()-start, ' s'
    x_bh, y_bh, z_bh, vx_bh, vy_bh, vz_bh, id_bh, mass_bh, amass_bh, amdot_bh = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.zeros(Ntot[5],dtype='u8'), np.array([]), np.array([]), np.array([])
    
    for i in xrange(Nfiles):
        fname = fpre + '.' + str(i)

        print '\nReading in snapfile ', i, ' t=', time()-start, ' s'
        N, pos, vel, ids, mass, met, sml_f, rho_f, U_f, sfr_f, birth_z_f, bhm, bhmdot = e5snapfile(fname,dir) # The "_f" indicates for a single file.  Could clearly have written for each other variable too, but I don't want to use the others' names without "_f" for this function's output.

        N0, N1, N4, N5 = N[0], N[0]+N[1], sum(N[:4]), sum(N[:5])

        print 'Appending coordinates, t=', time()-start, ' s'
        # Split coordinates of different particles and slot info of each subsequent file at the bottom
        x_g = np.append(x_g, pos[:N0,0])
        y_g = np.append(y_g, pos[:N0,1])
        z_g = np.append(z_g, pos[:N0,2])
        x_dm = np.append(x_dm, pos[N0:N1,0])
        y_dm = np.append(y_dm, pos[N0:N1,1])
        z_dm = np.append(z_dm, pos[N0:N1,2])
        x_s = np.append(x_s, pos[N4:N5,0])
        y_s = np.append(y_s, pos[N4:N5,1])
        z_s = np.append(z_s, pos[N4:N5,2])
        x_bh = np.append(x_bh, pos[N5:,0])
        y_bh = np.append(y_bh, pos[N5:,1])
        z_bh = np.append(z_bh, pos[N5:,2])
        del pos # Delete the old array to avoid clogging RAM unnecessarily

        print 'Appending velocities, t=', time()-start, ' s'
        # Ditto velocities
        vx_g = np.append(vx_g, vel[:N0,0])
        vy_g = np.append(vy_g, vel[:N0,1])
        vz_g = np.append(vz_g, vel[:N0,2])
        vx_dm = np.append(vx_dm, vel[N0:N1,0])
        vy_dm = np.append(vy_dm, vel[N0:N1,1])
        vz_dm = np.append(vz_dm, vel[N0:N1,2])
        vx_s = np.append(vx_s, vel[N4:N5,0])
        vy_s = np.append(vy_s, vel[N4:N5,1])
        vz_s = np.append(vz_s, vel[N4:N5,2])
        vx_bh = np.append(vx_bh, vel[N5:,0])
        vy_bh = np.append(vy_bh, vel[N5:,1])
        vz_bh = np.append(vz_bh, vel[N5:,2])
        del vel

        print 'Appending IDs, t=', time()-start, ' s'
        # Ditto IDs
        id_g = np.append(id_g, ids[:N0])
        id_dm = np.append(id_dm, ids[N0:N1])
        id_s = np.append(id_s, ids[N4:N5])
        id_bh = np.append(id_bh, ids[N5:])
        del ids

        print 'Appending masses, t=', time()-start, ' s'
        # Ditto masses
        mass_g = np.append(mass_g, mass[:N0])
        mass_dm = np.append(mass_dm, mass[N0:N1])
        mass_s = np.append(mass_s, mass[N4:N5])
        mass_bh = np.append(mass_bh, mass[N5:])
        del mass

        print 'Appending metallicities, t=', time()-start, ' s'
        # Ditto metallicities
        met_g = np.append(met_g, met[:N0])
        met_s = np.append(met_s, met[N0:N0+N[4]])
        del met

        print 'Appending other properties, t=', time()-start, ' s'
        # Add the remaining properties
        sml = np.append(sml, sml_f)
        rho = np.append(rho, rho_f)
        U = np.append(U, U_f)
        sfr = np.append(sfr, sfr_f)
        birth_z = np.append(birth_z, birth_z_f)
        amass_bh = np.append(amass_bh, bhm)
        amdot_bh = np.append(amdot_bh, bhmdot)

    if haloids:
        # Import group and subhalo ID information
        f = open('e5/snapshotids'+fpre[8:],'rb')
        gid = np.fromfile(f, 'i4', sum(Ntot)) # Group IDs
        shid = np.fromfile(f, 'i4', sum(Ntot)) # Subhalo IDs
        f.close()
        
        # Split group and subhalo IDs into particle species
        gid_g, gid_dm, gid_s, gid_bh = gid[:Ntot[0]], gid[Ntot[0]:sum(Ntot[:2])], gid[sum(Ntot[:2]):sum(Ntot[:5])], gid[sum(Ntot[:5]):sum(Ntot)]
        shid_g, shid_dm, shid_s, shid_bh = shid[:Ntot[0]], shid[Ntot[0]:sum(Ntot[:2])], shid[sum(Ntot[:2]):sum(Ntot[:5])], shid[sum(Ntot[:5]):sum(Ntot)]
        
        return [x_g, y_g, z_g, vx_g, vy_g, vz_g, id_g, mass_g, met_g, sml, rho, U, sfr, gid_g, shid_g], [x_dm, y_dm, z_dm, vx_dm, vy_dm, vz_dm, id_dm, mass_dm, gid_dm, shid_dm], [x_s, y_s, z_s, vx_s, vy_s, vz_s, id_s, mass_s, met_s, birth_z, gid_s, shid_s], [x_bh, y_bh, z_bh, vx_bh, vy_bh, vz_bh, id_bh, mass_bh, amass_bh, amdot_bh, gid_bh, shid_bh]
    
    else:
        return [x_g, y_g, z_g, vx_g, vy_g, vz_g, id_g, mass_g, met_g, sml, rho, U, sfr], [x_dm, y_dm, z_dm, vx_dm, vy_dm, vz_dm, id_dm, mass_dm], [x_s, y_s, z_s, vx_s, vy_s, vz_s, id_s, mass_s, met_s, birth_z], [x_bh, y_bh, z_bh, vx_bh, vy_bh, vz_bh, id_bh, mass_bh, amass_bh, amdot_bh]



def e5snappmi(t,dir=0):
	# Read in the important information from an entire E5 snapshot for doing plotting, halo identifying in the entire box etc.
	# Designed to be efficient.  pmi = position, mass, IDs
	# t = snapshot number (eg 1366)
	start = time()
	print 'Starting e5snappmi (printed times are internal to this)'
	
	if type(dir) != str:
		dir = '/Users/astevens/Documents/6-MonthProject/e5/' # Default directory for files
	
	fpre = 'snapshot_'+str(t)
	z, Ntot, Nfiles, boxsize, Omega_M, Omega_L, h = e5snaphead(fpre,dir) # Read header info
	a = 1./(1+z)
	
	# Initialise arrays.  Mass for DM should be the same for every particle so don't need to have an array for this
	x_g, y_g, z_g, mass_g, id_g = np.zeros(Ntot[0]), np.zeros(Ntot[0]), np.zeros(Ntot[0]), np.zeros(Ntot[0]), np.zeros(Ntot[0])
	x_dm, y_dm, z_dm, id_dm = np.zeros(Ntot[1]), np.zeros(Ntot[1]), np.zeros(Ntot[1]), np.zeros(Ntot[1])
	x_s, y_s, z_s, mass_s, id_s = np.zeros(Ntot[4]), np.zeros(Ntot[4]), np.zeros(Ntot[4]), np.zeros(Ntot[4]), np.zeros(Ntot[4])
	x_bh, y_bh, z_bh, mass_bh, id_bh = np.zeros(Ntot[5]), np.zeros(Ntot[5]), np.zeros(Ntot[5]), np.zeros(Ntot[5]), np.zeros(Ntot[5])
	
	Naccum = np.zeros(6)
	
	for i in xrange(8):
		print 'Starting snapfile', i, 'at', time()-start, 's'
		Nbytes = 0 # Number of bytes to skip to get to data wanted
		fname = fpre + '.' + str(i)
		f = open(dir+fname, 'rb')
		
		Nbytes += np.fromfile(f, 'i4', 1)[0]
		N = np.fromfile(f, 'u4', 6) # Number of particles for each particle type in this file
		Nsum = sum(N) # Total number of particles in the file
		mass_pt = np.fromfile(f, 'f8', 6)
		
		# Safety net in case the standard particle masses aren't what I expect them to be.
		for j in xrange(6):
			if j==1:
				if mass_pt[j]==0:
					print 'DM PARTICLE MASSES ARE NOT THE SAME. ALTER e5snappmi CODE'
					return 'ERROR'
			else:
				if mass_pt[j]!=0:
					print 'PARTICLE TYPE', j, 'HAS A STANDARD MASS.  e5snappmi CODE COULD BE MORE EFFICIENT'
					return 'ERROR'
		
		f.seek(Nbytes + 4*3)
		pos_g = np.fromfile(f, np.dtype(('f8',3)), N[0])*1e3*a/h
		x_g[Naccum[0]:Naccum[0]+N[0]], y_g[Naccum[0]:Naccum[0]+N[0]], z_g[Naccum[0]:Naccum[0]+N[0]] = pos_g[:,0], pos_g[:,1], pos_g[:,2]
		pos_dm = np.fromfile(f, np.dtype(('f8',3)), N[1])*1e3*a/h
		x_dm[Naccum[1]:Naccum[1]+N[1]], y_dm[Naccum[1]:Naccum[1]+N[1]], z_dm[Naccum[1]:Naccum[1]+N[1]] = pos_dm[:,0], pos_dm[:,1], pos_dm[:,2]
		pos_s = np.fromfile(f, np.dtype(('f8',3)), N[4])*1e3*a/h
		x_s[Naccum[4]:Naccum[4]+N[4]], y_s[Naccum[4]:Naccum[4]+N[4]], z_s[Naccum[4]:Naccum[4]+N[4]] = pos_s[:,0], pos_s[:,1], pos_s[:,2]
		pos_bh = np.fromfile(f, np.dtype(('f8',3)), N[5])*1e3*a/h
		x_bh[Naccum[5]:Naccum[5]+N[5]], y_bh[Naccum[5]:Naccum[5]+N[5]], z_bh[Naccum[5]:Naccum[5]+N[5]] = pos_bh[:,0], pos_bh[:,1], pos_bh[:,2]
		Nbytes += np.fromfile(f, 'i4', 1)[0]
		
		Nbytes += np.fromfile(f, 'i4', 1)[0] # Skip past the velocity data
		f.seek(Nbytes + 4*7)
		id_g[Naccum[0]:Naccum[0]+N[0]] = np.fromfile(f, 'u8', N[0])
		id_dm[Naccum[1]:Naccum[1]+N[1]] = np.fromfile(f, 'u8', N[1])
		id_s[Naccum[4]:Naccum[4]+N[4]] = np.fromfile(f, 'u8', N[4])
		id_bh[Naccum[5]:Naccum[5]+N[5]] = np.fromfile(f, 'u8', N[5])
		Nbytes += np.fromfile(f, 'i4', 1)[0]
		
		f.seek(Nbytes + 4*9)
		mass_g[Naccum[0]:Naccum[0]+N[0]] = np.fromfile(f, 'f8', N[0])*1e10/h
		mass_s[Naccum[4]:Naccum[4]+N[4]] = np.fromfile(f, 'f8', N[4])*1e10/h
		mass_bh[Naccum[5]:Naccum[5]+N[5]] = np.fromfile(f, 'f8', N[5])*1e10/h
		
		Naccum += N
		f.close()
	
	mass_dm = (mass_pt[1]*1e10/h)*np.ones(Ntot[1])
	print 'Finished e5snappmi at', time()-start, 's'
	return [x_g, y_g, z_g, mass_g, id_g], [x_dm, y_dm, z_dm, mass_dm, id_dm], [x_s, y_s, z_s, mass_s, id_s], [x_bh, y_bh, z_bh, mass_bh, id_bh]

def e5halohead(fname, dir=0):
	# Read header info from the group/subhalo files
	
	if type(dir) != str:
		dir = '/Users/astevens/Documents/6-MonthProject/e5/' # Default directory for files
	
	if fname[-2:-1] != '.': # Don't need to specify a file to read a header in for the function, but one needs to be internally specified if one hasn't been already.
		fname += '.0'
	
	f = open(dir+fname, 'rb')
	Nhalo = np.fromfile(f, 'i4', 1)[0] # Number of groups (larger haloes) in this file
	Nhalotot = np.fromfile(f, 'i4', 1)[0] # Total number of groups for the snapshot
	N = np.fromfile(f, 'i4', 1)[0] # Number of particles IDs in this file
	Ntot = np.fromfile(f, 'i8', 1)[0] # Total number of particles IDs for all files for this redshift
	Nfiles = np.fromfile(f, 'i4', 1)[0] # Number of subfind files for this snapshot (same for all _ids files)
	Nprevsum = np.fromfile(f, 'i4', 1)[0] # Sum of the values of N for all the _ids_????. files where the last number is smaller than this file
	f.close()
	
	return Nhalo, Nhalotot, Ntot, Nfiles


def e5haloidfile(fname, dir=0):
	# Read in a single subhalo_id or group_id file
	
	if type(dir) != str:
		dir = '/Users/astevens/Documents/6-MonthProject/e5/' # Default directory for files
	
	f = open(dir+fname, 'rb')
	
	# Header information
	Nhalo = np.fromfile(f, 'i4', 1)[0] # Number of groups (larger haloes) in this file
	Nhalotot = np.fromfile(f, 'i4', 1)[0] # Total number of groups for the snapshot
	N = np.fromfile(f, 'i4', 1)[0] # Number of particles IDs in this file
	Ntot = np.fromfile(f, 'i8', 1)[0] # Total number of particles IDs for all files for this redshift
	Nfiles = np.fromfile(f, 'i4', 1)[0] # Number of subfind files for this snapshot (same for all _ids files)
	Nprevsum = np.fromfile(f, 'i4', 1)[0] # Sum of the values of N for all the group_ids_1366. files where the last number is smaller than this file
	
	# Data
	IDs = np.fromfile(f, 'u8', N) # What I presume are particle IDs for this FOF group
	f.close()
	
	return Nprevsum, N, IDs


def e5haloidfull(fpre, dir=0):
	# Read in full list IDs related to a given snapshot and hopefully understand what this achieves
	
	if type(dir) != str:
		dir = '/Users/astevens/Documents/6-MonthProject/e5/' # Default directory for files
	
	Nhalo, Nhalotot, Ntot, Nfiles = e5halohead(fpre, dir) # Get header info for the haloes for this snapshot
	
	IDs = np.zeros(Ntot,dtype='u8') # Initialize the full ID list
	
	for i in xrange(Nfiles):
		fname = fpre + '.' + str(i) # Set the filename for this loop
		Nprevsum, N, IDpart = e5haloidfile(fname, dir) # Read in IDs for each file
		IDs[Nprevsum:Nprevsum+N] = IDpart # Add to the array
	
	return IDs






def e5halotabhead(fname=None, dir=0, t=1366):
	
	if fname==None:
		fname = 'subhalo_tab_'+str(int(t))+'.0'
	
	if type(dir) != str:
		dir = '/Users/astevens/Documents/6-MonthProject/e5/'
	
	if fname[-2:-1] != '.': # Don't need to specify a file to read a header in for the function, but one needs to be internally specified if one hasn't been already.
		fname += '.0'
	
	f = open(dir+fname,'rb')
	Ngroups = np.fromfile(f, 'i4', 1)[0] # Same as Nhalo for subhalo ID files
	Ntotgroups = np.fromfile(f, 'i4', 1)[0] # Same as Nhalotot for subhalo ID files
	Nids = np.fromfile(f, 'i4', 1)[0] # Same as N for subhalo ID files
	TotNids = np.fromfile(f, 'u8', 1)[0] # Same as Ntot for subhalo ID files
	Nfiles = np.fromfile(f, 'i4', 1)[0] # Number of tab files for this snapshot
	Nsubgroups = np.fromfile(f, 'i4', 1)[0] # Number of subhaloes in this file
	NtotSubgroups = np.fromfile(f, 'i4', 1)[0] # Total number of subhaloes across all files
	f.close()
	
	return Ntotgroups, TotNids, Nfiles, NtotSubgroups


def e5halotabfile(fname=None, dir=0, t=1366,n=0):
	# Read in information from a single subhalo tab file
	
	if fname==None: # Have a default file.  Also means don't have to bother with the leading part of the filenames
		fname = 'subhalo_tab_'+str(int(t))+'.'+str(int(n))
	
	if type(dir) != str:
		dir = '/Users/astevens/Documents/6-MonthProject/e5/'
	
	f = open(dir+fname,'rb')
	
	## Header information
	Ngroups = np.fromfile(f, 'i4', 1)[0] # Same as Nhalo for subhalo ID files
	Ntotgroups = np.fromfile(f, 'i4', 1)[0] # Same as Nhalotot for subhalo ID files
	Nids = np.fromfile(f, 'i4', 1)[0] # Same as N for subhalo ID files
	TotNids = np.fromfile(f, 'u8', 1)[0] # Same as Ntot for subhalo ID files
	Nfiles = np.fromfile(f, 'i4', 1)[0] # Number of tab files for this snapshot
	Nsubgroups = np.fromfile(f, 'i4', 1)[0] # Number of subhaloes in this file
	NtotSubgroups = np.fromfile(f, 'i4', 1)[0] # Total number of subhaloes across all files
	
	floattype = 'f4'
	idtype = 'u8'
	
	## Data for groups
	length = np.fromfile(f, 'i4', Ngroups) # Number of particles in each group
	offset = np.fromfile(f, 'i4', Ngroups)
	mass = np.fromfile(f, floattype, Ngroups) # Mass of each group
	pos = np.fromfile(f, np.dtype((floattype,3)), Ngroups) # COM of each group
	mmean200 = np.fromfile(f, floattype, Ngroups) # Mass within radius enclosing an average density 200x that of the mean density of the Universe
	rmean200 = np.fromfile(f, floattype, Ngroups) # The radius for the above
	mcrit200 = np.fromfile(f, floattype, Ngroups) # As for mmean200 but critical density of the Universe
	rcrit200 = np.fromfile(f, floattype, Ngroups) # Radius for the above
	mtoph200 = np.fromfile(f, floattype, Ngroups) # As above masses but for a tophat.  Not sure about this...
	rtoph200 = np.fromfile(f, floattype, Ngroups)
	veldispmean200 = np.fromfile(f, floattype, Ngroups)  # This and the next are the velocity dispersions for the various radii
	veldispcrit200 = np.fromfile(f, floattype, Ngroups)
	veldisptoph200 = np.fromfile(f, floattype, Ngroups)
	lencontam = np.fromfile(f, 'i4', Ngroups) # Number of particles in the group that are not associated with a subhalo
	masscontam = np.fromfile(f, floattype, Ngroups) # Summed mass of those above particles
	nhalo = np.fromfile(f, 'i4', Ngroups) # Number of subhaloes within each group
	firsthalo = np.fromfile(f, 'i4', Ngroups)  # Related to concatenation and "offset".  Can probably ignore.
	
	# Data for subhaloes
	halolen = np.fromfile(f, 'i4', Nsubgroups) # Number of particles in each subhalo
	halooffset = np.fromfile(f, 'i4', Nsubgroups)
	haloparent = np.fromfile(f, 'i4', Nsubgroups) # Index of parent (neither Yu nor I are sure what this is)
	halomass = np.fromfile(f, floattype, Nsubgroups) # Mass of each subhalo (sum of all particle types)
	halopos = np.fromfile(f, np.dtype((floattype,3)), Nsubgroups) # Position of each subhalo in intrinsic units (kpc/h, I believe)
	halovel = np.fromfile(f, np.dtype((floattype,3)), Nsubgroups) # Halo velocity in km/s
	halocom = np.fromfile(f, np.dtype((floattype,3)), Nsubgroups) # COM location of each halo (almost identical to halopos)
	halospin = np.fromfile(f, np.dtype((floattype,3)), Nsubgroups) # Spin of each halo (presumably the dimensionless spin parameter?)
	haloveldisp = np.fromfile(f, floattype, Nsubgroups) # Velocity dispersion of each subhalo
	halovmax = np.fromfile(f, floattype, Nsubgroups) # Maximum circular velocity of each subhalo
	halovmaxrad = np.fromfile(f, floattype, Nsubgroups) # Radius for the maximum circular velocity
	halohalfmassradius = np.fromfile(f, floattype, Nsubgroups) # Radius encompasses half the subhalo's mass
	haloid = np.fromfile(f, idtype, Nsubgroups) # ID of the most bound particle in the subhalo
	halogroup = np.fromfile(f, 'u4', Nsubgroups) # Index of group that this subhalo is in.
	halopartmass = np.fromfile(f, np.dtype(('f4',6)), Nsubgroups) # Integrated mass of the halo for each individual particle type
	
	f.close()
	
	return Ngroups, Nsubgroups, length, nhalo, halolen, halomass, halogroup, halopartmass, haloid, halovmax, halovmaxrad


def e5halotabfull(fpre=None, dir=0, t=1366):
	
	if fpre==None:
		fpre = 'subhalo_tab_'+str(int(t))
	
	if type(dir) != str:
		dir = '/Users/astevens/Documents/6-MonthProject/e5/'
	
	# Read in header information
	Ntotgroups, TotNids, Nfiles, NtotSubgroups = e5halotabhead(fpre, dir)
	
	# Initialize variables: Group number of particles, group number of subhaloes, subhalo number of particles, subhalo most-bound-particle ID, subhalo group ID
	gp_Nparts, gp_Nsh, sh_Nparts, sh_pid, sh_gpid, sh_vmax, sh_vmaxr = np.zeros(Ntotgroups), np.zeros(Ntotgroups), np.zeros(NtotSubgroups), np.zeros(NtotSubgroups), np.zeros(NtotSubgroups), np.zeros(NtotSubgroups), np.zeros(NtotSubgroups)
	sh_ptmass = np.zeros((NtotSubgroups,6)) # Mass of subhalo for each particle type
	
	NgroupPrev, NsubgroupPrev = 0, 0 # Initialize the variables that will remember where to slot in the next file's info to the appropriate arrays
	
	for i in xrange(Nfiles):
		fname = fpre + '.' + str(i) # Set the filename for this loop
		Ngroups, Nsubgroups, length, nhalo, halolen, halomass, halogroup, halopartmass, haloid, halovmax, halovmaxrad = e5halotabfile(fname, dir)
		
		gp_Nparts[NgroupPrev:NgroupPrev+Ngroups] = length
		gp_Nsh[NgroupPrev:NgroupPrev+Ngroups] = nhalo
		sh_Nparts[NsubgroupPrev:NsubgroupPrev+Nsubgroups] = halolen
		sh_gpid[NsubgroupPrev:NsubgroupPrev+Nsubgroups] = halogroup
		sh_ptmass[NsubgroupPrev:NsubgroupPrev+Nsubgroups,:] = halopartmass
		sh_pid[NsubgroupPrev:NsubgroupPrev+Nsubgroups] = haloid
		sh_vmax[NsubgroupPrev:NsubgroupPrev+Nsubgroups] = halovmax
		sh_vmaxr[NsubgroupPrev:NsubgroupPrev+Nsubgroups] = halovmaxrad
		
		NgroupPrev += Ngroups
		NsubgroupPrev += Nsubgroups
	
	return gp_Nparts, gp_Nsh, sh_Nparts, sh_gpid, sh_ptmass, sh_pid, sh_vmax, sh_vmaxr





def e5loadhalo(id1, t, idarr=None, dir=0):
	"""
		Load in the particles for a single halo (either a group or subhalo)
		
		===INPUTS===
		id1 = The ID of the subhalo/group that should be read in.
		t = Time of the snapshot, which is really the numbers in the snapshot file, eg t=1366.  Will work as a string or number.
		idarr = Array of subhalo/group IDs as read in from the files I produce from idmatch.py. Will actually read in this file and assume subhalo if not specified
		dir = Directory where the files are
		"""
	
	t = str(int(t)) # Make sure it's a string of an integer
	
	if type(dir) != str:
		dir = '/Users/astevens/Documents/6-MonthProject/e5/' # Default directory for files
	
	z, Ntot, Nfiles, boxsize, Omega_M, Omega_L, h = e5snaphead('snapshot_'+t, dir) # Read in header info
	
	if idarr==None:
		f = open(dir+'snapshotids_'+t,'rb')
		f.seek(4*sum(Ntot))
		idarr = np.fromfile(f, 'i4', sum(Ntot)) # Subhalo IDs
		f.close()
	
	print t, id1, idarr, len(idarr)
	idargs = np.argwhere(idarr==id1) # Arguments where idarr is equal to id1
	
	con_g = (idargs < Ntot[0]) # Condition to find the total number of gas particles in the halo
	con_dm = (idargs >= Ntot[0]) * (idargs < sum(Ntot[:2])) # Condition ... dark matter
	con_s = (idargs >= sum(Ntot[:2])) * (idargs < sum(Ntot[:5])) # Condition ... stars
	con_bh = (idargs >= sum(Ntot[:5])) # Condition ... black holes
	#print con_bh
	
	Nptot = np.array([len(idargs[con_g]), len(idargs[con_dm]), 0, 0, len(idargs[con_s]), len(idargs[con_bh])]) # Total number of particles in the halo in particle type
	print 'Nptot is', Nptot
	Naccum, Npaccum = np.zeros(6), np.zeros(6) # Initialize the accumulation of N
	
	# Initialize arrays that will contain the particle data
	pos_g, pos_dm, pos_s, pos_bh = np.zeros((Nptot[0],3)), np.zeros((Nptot[1],3)), np.zeros((Nptot[4],3)), np.zeros((Nptot[5],3))
	vel_g, vel_dm, vel_s, vel_bh = np.zeros((Nptot[0],3)), np.zeros((Nptot[1],3)), np.zeros((Nptot[4],3)), np.zeros((Nptot[5],3))
	id_g, id_dm, id_s, id_bh = np.zeros(Nptot[0]), np.zeros(Nptot[1]), np.zeros(Nptot[4]), np.zeros(Nptot[5])
	mass_g, mass_dm, mass_s, mass_bh = np.zeros(Nptot[0]), np.zeros(Nptot[1]), np.zeros(Nptot[4]), np.zeros(Nptot[5])
	u, rho, sfr = np.zeros(Nptot[0]), np.zeros(Nptot[0]), np.zeros(Nptot[0])
	
	for fno in xrange(Nfiles): # fno is file number
		Nbytes = [] # Initialize list that will contain information on the block sizes in each file
		f = open(dir+'snapshot_'+t+'.'+str(fno), 'rb')
		Nbytes += list(np.fromfile(f, 'i4', 1))
		
		# Read header information
		N = np.fromfile(f, 'u4', 6) # Number of particles for each particle type in this file
		Nsum = sum(N) # Total number of particles in the file
		mass_pt = np.fromfile(f, 'f8', 6) # Mass of each particle type.  If 0 then it varies for each particle of that type
		Nmass = sum(N[np.argwhere(mass_pt==0.0)]) # Number of particles in the file with individual masses to be read in
		a = np.fromfile(f, 'f8', 1)[0] # Expansion factor (normalised to 1 at z=0)
		z = np.fromfile(f, 'f8', 1)[0] # Redshift of snapshot
		flag_sfr = np.fromfile(f, 'i4', 1)[0] # Flag for star formation rate
		flag_feedback = np.fromfile(f, 'i4', 1)[0] # Flag for feedback
		Ntot = np.fromfile(f, 'u4', 6) # Total number of particles for each particle type in the entire simulation
		flag_cool = np.fromfile(f, 'i4', 1)[0] # Flag for cooling
		Nfiles = np.fromfile(f, 'i4', 1)[0] # Number of files for each snapshot
		boxsize = np.fromfile(f, 'f8', 1)[0] # Size of box if periodic boundary conditions are used
		Omega_M = np.fromfile(f, 'f8', 1)[0] # Omega Matter
		Omega_L = np.fromfile(f, 'f8', 1)[0] # Omega (Lambda) Dark Energy
		h = np.fromfile(f, 'f8', 1)[0] # Little Hubble h
		flag_StarAge = np.fromfile(f, 'i4', 1)[0] # Flag for the creation times of stars
		flag_metals = np.fromfile(f, 'i4', 1)[0] # Flag for metallicity values
		
		g_con = (idargs < N[0]+Naccum[0]) * (idargs >= Naccum[0]) # Gas condition to figure out how many particles for the halo are in this file
		dm_con = (idargs-Ntot[0] < N[1]+Naccum[1]) * (idargs-Ntot[0] >= Naccum[1]) # Dark matter condition...
		s_con = (idargs-sum(Ntot[:4]) < N[4]+Naccum[4]) * (idargs-sum(Ntot[:4]) >= Naccum[4]) # Star condition...
		bh_con = (idargs-sum(Ntot[:5]) < N[5]+Naccum[5]) * (idargs-sum(Ntot[:5]) >= Naccum[5]) # Black hole condition...
		
		g_args = idargs[g_con] - Naccum[0] # Location of gas particles to extract in the data file
		dm_args = idargs[dm_con] - Ntot[0] - Naccum[1] # Ditto dark matter
		s_args = idargs[s_con] - sum(Ntot[:4]) - Naccum[4] # Ditto stars
		bh_args = idargs[bh_con] - sum(Ntot[:5]) - Naccum[5] # Ditto black holes
		
		Nparts = np.array([ len(g_args), len(dm_args), 0, 0, len(s_args), len(bh_args) ])
		
		Nmass_pt = np.zeros(6,dtype='u4')
		for i in xrange(6):
			if mass_pt[i]==0: Nmass_pt[i] = N[i]
		
		## Extract positions
		for i, arg in enumerate(g_args):
			f.seek(sum(Nbytes) + 4*3 + arg*24)
			pos_g[Npaccum[0]+i,:] = np.fromfile(f, 'f8', 3)
		for i, arg in enumerate(dm_args):
			f.seek(sum(Nbytes) + 4*3 + (arg+N[0])*24)
			pos_dm[Npaccum[1]+i,:] = np.fromfile(f, 'f8', 3)
		for i, arg in enumerate(s_args):
			f.seek(sum(Nbytes) + 4*3 + (arg+sum(N[:4]))*24)
			pos_s[Npaccum[4]+i,:] = np.fromfile(f, 'f8', 3)
		for i, arg in enumerate(bh_args):
			f.seek(sum(Nbytes) + 4*3 + (arg+sum(N[:5]))*24)
			pos_bh[Npaccum[5]+i,:] = np.fromfile(f, 'f8', 3)
		##
		
		f.seek(sum(Nbytes)+4*2)
		Nbytes += list(np.fromfile(f, 'i4', 1))
		
		## Extract velocities
		for i, arg in enumerate(g_args):
			f.seek(sum(Nbytes) + 4*5 + arg*24)
			vel_g[Npaccum[0]+i,:] = np.fromfile(f, 'f8', 3)
		for i, arg in enumerate(dm_args):
			f.seek(sum(Nbytes) + 4*5 + (arg+N[0])*24)
			vel_dm[Npaccum[1]+i,:] = np.fromfile(f, 'f8', 3)
		for i, arg in enumerate(s_args):
			f.seek(sum(Nbytes) + 4*5 + (arg+sum(N[:4]))*24)
			vel_s[Npaccum[4]+i,:] = np.fromfile(f, 'f8', 3)
		for i, arg in enumerate(bh_args):
			f.seek(sum(Nbytes) + 4*5 + (arg+sum(N[:5]))*24)
			vel_bh[Npaccum[5]+i,:] = np.fromfile(f, 'f8', 3)
		##
		
		f.seek(sum(Nbytes)+4*4)
		Nbytes += list(np.fromfile(f, 'i4', 1))
		
		## Extract IDs
		for i, arg in enumerate(g_args):
			f.seek(sum(Nbytes) + 4*7 + arg*8)
			id_g[Npaccum[0]+i] = np.fromfile(f, 'u8', 1)
		for i, arg in enumerate(dm_args):
			f.seek(sum(Nbytes) + 4*7 + (arg+N[0])*8)
			id_dm[Npaccum[1]+i] = np.fromfile(f, 'u8', 1)
		for i, arg in enumerate(s_args):
			f.seek(sum(Nbytes) + 4*7 + (arg+sum(N[:4]))*8)
			id_s[Npaccum[4]+i] = np.fromfile(f, 'u8', 1)
		for i, arg in enumerate(bh_args):
			f.seek(sum(Nbytes) + 4*7 + (arg+sum(N[:5]))*8)
			id_bh[Npaccum[5]+i] = np.fromfile(f, 'u8', 1)
		##
		
		f.seek(sum(Nbytes)+4*6)
		Nbytes += list(np.fromfile(f, 'i4', 1))
		
		## Extract Masses
		if mass_pt[0]==0:
			for i, arg in enumerate(g_args):
				f.seek(sum(Nbytes) + 4*9 + arg*8)
				mass_g[Npaccum[0]+i] = np.fromfile(f, 'f8', 1)
		else:
			mass_g[Npaccum[0]:Npaccum[0]+Nparts[0]] = mass_pt[0]*np.ones(Nparts[0])
		
		if mass_pt[1]==0:
			for i, arg in enumerate(dm_args):
				f.seek(sum(Nbytes) + 4*9 + (arg+Nmass_pt[0])*8)
				mass_dm[Npaccum[1]+i] = np.fromfile(f, 'f8', 1)
		else:
			mass_dm[Npaccum[1]:Npaccum[1]+Nparts[1]] = mass_pt[1]*np.ones(Nparts[1])
		
		if mass_pt[4]==0:
			for i, arg in enumerate(s_args):
				f.seek(sum(Nbytes) + 4*9 + (arg+sum(Nmass_pt[:4]))*8)
				mass_s[Npaccum[4]+i] = np.fromfile(f, 'f8', 1)
		else:
			mass_s[Npaccum[4]:Npaccum[4]+Nparts[4]] = mass_pt[4]*np.ones(Nparts[4])
		
		if mass_pt[5]==0:
			for i, arg in enumerate(bh_args):
				f.seek(sum(Nbytes) + 4*9 + (arg+sum(Nmass_pt[:5]))*8)
				mass_bh[Npaccum[5]+i] = np.fromfile(f, 'f8', 1)
		else:
			mass_bh[Npaccum[5]:Npaccum[5]+Nparts[5]] = mass_pt[5]*np.ones(Nparts[5])
		##
		
		f.seek(sum(Nbytes)+4*8)
		Nbytes += list(np.fromfile(f, 'i4', 1))
		
		## Extract gas properties
		for i, arg in enumerate(g_args):
			f.seek(sum(Nbytes) + 4*11 + arg*8)
			u[Npaccum[0]+i] = np.fromfile(f, 'f8', 1)
		
		f.seek(sum(Nbytes)+4*10)
		Nbytes += list(np.fromfile(f, 'i4', 1))
		
		for i, arg in enumerate(g_args):
			f.seek(sum(Nbytes) + 4*13 + arg*8)
			rho[Npaccum[0]+i] = np.fromfile(f, 'f8', 1)
		
		f.seek(sum(Nbytes)+4*12)
		Nbytes += list(np.fromfile(f, 'i4', 1))
		
		if flag_cool==1:
			f.seek(sum(Nbytes)+4*14)
			Nbytes += list(np.fromfile(f, 'i4', 1))
			f.seek(sum(Nbytes)+4*16)
			Nbytes += list(np.fromfile(f, 'i4', 1))
			extra = 16
		else:
			extra = 0

		f.seek(sum(Nbytes) + 4*14 + extra)
		Nbytes += list(np.fromfile(f, 'i4', 1))

		for i, arg in enumerate(g_args):
			f.seek(sum(Nbytes) + 4*17 + extra + arg*8)
			sfr[Npaccum[0]+i] = np.fromfile(f, 'f8', 1)
		##
		
		Naccum += N
		Npaccum += Nparts
		f.close()
	
	
	## Extract premeasured properties of halo from tab file
	Ntotgroups, TotNids, Nfiles, NtotSubgroups = e5halotabhead(t=t)
	Naccum = 0
	floattype = 'f4'
	idtype = 'u8'
	
	if np.max(idarr) == NtotSubgroups-1: # Determine if the code is getting a group or subhalo
		for i in xrange(Nfiles):
			f = open(dir+'subhalo_tab_'+t+'.'+str(i))
			Ngroups = np.fromfile(f, 'i4', 1)[0] # Number of groups in this file
			f.seek(24)
			Nsubgroups = np.fromfile(f, 'i4', 1)[0] # Number of subhaloes in this file
			
			if id1<=Naccum+Nsubgroups:
				id2 = id1-Naccum # Number of haloes' data to skip (number listed in this file)
				f.seek(32 + 19*4*Ngroups)
				# Data for subhaloes
				Nparts = np.fromfile(f, 'i4', Nsubgroups)[id2] # Number of particles in the subhalo
				print 'Nparts', Nparts
				offset = np.fromfile(f, 'i4', Nsubgroups)[id2]
				parent = np.fromfile(f, 'i4', Nsubgroups)[id2] # Index of parent (neither Yu nor I are sure what this is)
				mass = np.fromfile(f, floattype, Nsubgroups)[id2] *1e10/h # Mass of the subhalo in solar masses (sum of all particle types)
				pos = np.fromfile(f, np.dtype((floattype,3)), Nsubgroups)[id2] *1e3/h # Position of the subhalo in pc
				vel = np.fromfile(f, np.dtype((floattype,3)), Nsubgroups)[id2] # Halo velocity in km/s
				com = np.fromfile(f, np.dtype((floattype,3)), Nsubgroups)[id2] *1e3/h # COM location of the halo (almost identical to halopos)
				spin = np.fromfile(f, np.dtype((floattype,3)), Nsubgroups)[id2] # Spin of the halo (presumably the dimensionless spin parameter?)
				veldisp = np.fromfile(f, floattype, Nsubgroups)[id2] # Velocity dispersion of the subhalo
				vmax = np.fromfile(f, floattype, Nsubgroups)[id2] # Maximum circular velocity of the subhalo (presumably km/s)
				vmaxrad = np.fromfile(f, floattype, Nsubgroups)[id2]*1e3/h # Radius for the maximum circular velocity in pc
				halfmassr = np.fromfile(f, floattype, Nsubgroups)[id2]*1e3/h # Radius encompasses half the subhalo's mass in pc
				idbp = np.fromfile(f, idtype, Nsubgroups)[id2] # ID of the most bound particle in the subhalo
				idgp = np.fromfile(f, 'u4', Nsubgroups)[id2] # Index of group that this subhalo is in.
				partmass = np.fromfile(f, np.dtype(('f4',6)), Nsubgroups)[id2]*1e10/h # Integrated mass of the halo for each individual particle type in solar masses
				tabdata = [Nparts, offset, parent, mass, pos, vel, com, spin, veldisp, vmax, vmaxrad, halfmassr, idbp, idgp, partmass]
				break
			Naccum += Nsubgroups
	
	elif np.max(idarr) == Ntotgroups-1:
		for i in xrange(Nfiles):
			f = open(dir+'subhalo_tab_'+t+'.'+str(i))
			Ngroups = np.fromfile(f, 'i4', 1)[0] # Number of groups in this file
			
			if id1<=Naccum+Ngroups:
				f.seek(32)
				id2 = id1-Naccum
				Nparts = np.fromfile(f, 'i4', Ngroups)[id2] # Number of particles in each group
				offset = np.fromfile(f, 'i4', Ngroups)[id2]
				mass = np.fromfile(f, floattype, Ngroups)[id2]*1e10/h # Mass of each group
				pos = np.fromfile(f, np.dtype((floattype,3)), Ngroups)[id2]*1e3/h # COM of the group
				mmean200 = np.fromfile(f, floattype, Ngroups)[id2]*1e10/h # Mass within radius enclosing an average density 200x that of the mean density of the Universe
				rmean200 = np.fromfile(f, floattype, Ngroups)[id2]*1e3/h # The radius for the above
				mcrit200 = np.fromfile(f, floattype, Ngroups)[id2]*1e10/h # As for mmean200 but critical density of the Universe
				rcrit200 = np.fromfile(f, floattype, Ngroups)[id2]*1e3/h # Radius for the above
				mtoph200 = np.fromfile(f, floattype, Ngroups)[id2]*1e10/h # As above masses but for a tophat.  Not sure about this...
				rtoph200 = np.fromfile(f, floattype, Ngroups)[id2]*1e3/h
				veldispmean200 = np.fromfile(f, floattype, Ngroups)[id2]  # This and the next are the velocity dispersions for the various radii
				veldispcrit200 = np.fromfile(f, floattype, Ngroups)[id2]
				veldisptoph200 = np.fromfile(f, floattype, Ngroups)[id2]
				lencontam = np.fromfile(f, 'i4', Ngroups)[id2] # Number of particles in the group that are not associated with a subhalo
				masscontam = np.fromfile(f, floattype, Ngroups)[id2]*1e10/h # Summed mass of those above particles
				nhalo = np.fromfile(f, 'i4', Ngroups)[id2] # Number of subhaloes within the group
				firsthalo = np.fromfile(f, 'i4', Ngroups)[id2]  # Related to concatenation and "offset".  Can probably ignore.
				tabdata = [Nparts, offset, mass, pos, mmean200, rmean200, mcrit200, rcrit200, mtoph200, rtoph200, veldispmean200, veldispcrit200, veldisptoph200, lencontam, masscontam, nhalo, firsthalo]
				break
			Naccum += Ngroups
	##
	return [pos_g[:,0]*1e3*a/h, pos_g[:,1]*1e3*a/h, pos_g[:,2]*1e3*a/h, vel_g[:,0], vel_g[:,1], vel_g[:,2], id_g, mass_g*1e10/h, u*1e6, rho*10*h*h/(a**3), sfr], [pos_dm[:,0]*1e3*a/h, pos_dm[:,1]*1e3*a/h, pos_dm[:,2]*1e3*a/h, vel_dm[:,0], vel_dm[:,1], vel_dm[:,2], id_dm, mass_dm*1e10/h], [pos_s[:,0]*1e3*a/h, pos_s[:,1]*1e3*a/h, pos_s[:,2]*1e3*a/h, vel_s[:,0], vel_s[:,1], vel_s[:,2], id_s, mass_s*1e10/h], [pos_bh[:,0]*1e3*a/h, pos_bh[:,1]*1e3*a/h, pos_bh[:,2]*1e3*a/h, vel_bh[:,0], vel_bh[:,1], vel_bh[:,2], id_bh, mass_bh*1e10/h], tabdata


def e5v5out(t=1366):
	f = open('trye5out/'+str(t)+'/v5out','rb')
	Nsh = np.fromfile(f, 'f8', 1)[0]
	Nsh = int(Nsh)
	r_abs = np.fromfile(f, 'f8', Nsh)
	r_rel = np.fromfile(f, 'f8', Nsh)
	r_vir = np.fromfile(f, 'f8', Nsh)
	tsm1 = np.fromfile(f, 'f8', Nsh)
	tsm2 = np.fromfile(f, 'f8', Nsh)
	tsm3 = np.fromfile(f, 'f8', Nsh)
	tgm1 = np.fromfile(f, 'f8', Nsh)
	tgm2 = np.fromfile(f, 'f8', Nsh)
	tgm3 = np.fromfile(f, 'f8', Nsh)
	tdmm1 = np.fromfile(f, 'f8', Nsh)
	tdmm2 = np.fromfile(f, 'f8', Nsh)
	tbhm1 = np.fromfile(f, 'f8', Nsh)
	tbhm2 = np.fromfile(f, 'f8', Nsh)
	sfr1 = np.fromfile(f, 'f8', Nsh)
	sfr2 = np.fromfile(f, 'f8', Nsh)
	sfr3 = np.fromfile(f, 'f8', Nsh)
	f.close()
	return [r_abs,r_rel,r_vir], [tsm1,tsm2,tsm3], [tgm1,tgm2,tgm3], [tdmm1,tdmm2], [tbhm1,tbhm2], [sfr1,sfr2,sfr3]


def e5v10outfile(t,fno,run=2):
	# Read in a single file from the output of trye5v10.py
	f = open('trye5out/'+str(t)+'/v10out'+str(run)+'_'+str(fno),'rb')
	Nsh = np.fromfile(f, 'i8', 1)[0]
	Ntech = np.fromfile(f, 'i8', 1)[0]
	Radii = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
	SM = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
	CGM = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
	HGM = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
	DMM = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
	BHM = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
	nBH = np.fromfile(f, np.dtype(('i8',Ntech)), Nsh)
	SFR = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
	if run>1: subhaloes = np.fromfile(f, 'i8', Nsh)
	#TBL = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
	#TvL = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
	#TuL = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
	#TkL = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
	#TjL = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
	#TicL = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
	f.close()
	#return [z, Nsh, Ntech, SM, CGM, HGM, DMM, BHM, nBH, SFR, TBL, TvL, TuL, TkL, TjL, TicL]
	return [Radii, SM, CGM, HGM, DMM, BHM, nBH, SFR, subhaloes]

def e5v10out(t=1366,run=2):
	# Read in full data output from trye5v10.
	
	# Read in first part
	[Radii, SM, CGM, HGM, DMM, BHM, nBH, SFR, shid] = e5v10outfile(t, 1, run)

	for i in xrange(19):
		fd = e5v10outfile(t, i+2, run) # File data
		Radii = np.append(Radii, fd[0], 0)
		SM = np.append(SM, fd[1], 0)
		CGM = np.append(CGM, fd[2], 0)
		HGM = np.append(HGM, fd[3], 0)
		DMM = np.append(DMM, fd[4], 0)
		BHM = np.append(BHM, fd[5], 0)
		nBH = np.append(BHM, fd[6], 0)
		SFR = np.append(SFR, fd[7], 0)
		shid = np.append(shid, fd[8])

	return [Radii, SM, CGM, HGM, DMM, BHM, nBH, SFR, shid]



def MBIIanalysisoutfile(fno,sno='085',run=8):
	# Read in the output of MBIIanalysisv2.py from a single section
	if fno!=1:
		f = open('MBIIanalysis/'+sno+'/run'+str(run)+'_'+str(fno), 'rb')
		Nsh = np.fromfile(f, 'i8', 1)[0]
		Ntech = np.fromfile(f, 'i8', 1)[0]
		Radii = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
		SM = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
		CGM = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
		HGM = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
		DMM = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
		BHM = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
		nBH = np.fromfile(f, np.dtype(('i8',Ntech)), Nsh)
		SFR = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
		subhaloes = np.fromfile(f, 'i8', Nsh)
		if run>5:
			b = np.fromfile(f, 'f8', Nsh)
			m = np.fromfile(f, 'f8', Nsh)
		f.close()


	else:
		# Do the first subsection
		f = open('MBIIanalysis/'+sno+'/run'+str(run)+'_1.1', 'rb')
		Nsh = np.fromfile(f, 'i8', 1)[0]
		Ntech = np.fromfile(f, 'i8', 1)[0]
		Radii = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
		SM = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
		CGM = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
		HGM = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
		DMM = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
		BHM = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
		nBH = np.fromfile(f, np.dtype(('i8',Ntech)), Nsh)
		SFR = np.fromfile(f, np.dtype(('f8',Ntech)), Nsh)
		subhaloes = np.fromfile(f, 'i8', Nsh)
		if run>5:
			b = np.fromfile(f, 'f8', Nsh)
			m = np.fromfile(f, 'f8', Nsh)
		f.close()

		for i in np.arange(29)+2:
			try:
				f = open('MBIIanalysis/'+sno+'/run'+str(run)+'_1.'+str(i), 'rb')
				Nsh = np.fromfile(f, 'i8', 1)[0]
				Ntech = np.fromfile(f, 'i8', 1)[0]
				Radii = np.append(Radii, np.fromfile(f, np.dtype(('f8',Ntech)), Nsh), 0)
				SM = np.append(SM, np.fromfile(f, np.dtype(('f8',Ntech)), Nsh), 0)
				CGM = np.append(CGM, np.fromfile(f, np.dtype(('f8',Ntech)), Nsh), 0)
				HGM = np.append(HGM, np.fromfile(f, np.dtype(('f8',Ntech)), Nsh), 0)
				DMM = np.append(DMM, np.fromfile(f, np.dtype(('f8',Ntech)), Nsh), 0)
				BHM = np.append(BHM, np.fromfile(f, np.dtype(('f8',Ntech)), Nsh), 0)
				nBH = np.append(nBH, np.fromfile(f, np.dtype(('i8',Ntech)), Nsh), 0)
				SFR = np.append(SFR, np.fromfile(f, np.dtype(('f8',Ntech)), Nsh), 0)
				subhaloes = np.append(subhaloes, np.fromfile(f, 'i8', Nsh))
				if run>5:
					b = np.append(b, np.fromfile(f, 'f8', Nsh))
					m = np.append(m, np.fromfile(f, 'f8', Nsh))
				f.close()
			except:
				continue
	
	# Get rid of the bizarre SUBFIND outputs that might exist
	tno = 6 if run<6 else 3
	GM = CGM + HGM
	ind1 = np.transpose(np.argwhere(Radii[:,tno]>1e-5))[0] # Indices that are trustworthy
	ind2 = np.transpose(np.argwhere(SM[:,tno]>=1e8))[0]
	ind3 = np.transpose(np.argwhere(GM[:,tno]>=1e8))[0] # Also just get the ones that feasibly have galaxies
	ind = np.intersect1d(np.intersect1d(ind1,ind2),ind3)
	Radii = Radii[ind,:]
	SM = SM[ind,:]
	CGM = CGM[ind,:]
	HGM = HGM[ind,:]
	DMM = DMM[ind,:]
	BHM = BHM[ind,:]
	nBH = nBH[ind,:]
	SFR = SFR[ind,:]
	subhaloes = subhaloes[ind]

	if run>5:
		b = b[ind]
		m = m[ind]
		return [Radii, SM, CGM, HGM, DMM, BHM, nBH, SFR, subhaloes, b, m]
	else:
		return [Radii, SM, CGM, HGM, DMM, BHM, nBH, SFR, subhaloes]



def MBIIanalysisout(sno='085',run=8,secmin=1,secmax=30):
	# Read in full data output from MBIIanalysis.
	loop = True
	
	if run>5:
		# Read in first part
		while loop:
			try:
				[Radii, SM, CGM, HGM, DMM, BHM, nBH, SFR, shid, b, m] = MBIIanalysisoutfile(secmin, sno, run)
				loop = False
			except:
				secmin +=1
				
		
		for i in range(secmin+1,secmax+1):
			try:
				fd = MBIIanalysisoutfile(i, sno, run) # File data
				Radii = np.append(Radii, fd[0], 0)
				SM = np.append(SM, fd[1], 0)
				CGM = np.append(CGM, fd[2], 0)
				HGM = np.append(HGM, fd[3], 0)
				DMM = np.append(DMM, fd[4], 0)
				BHM = np.append(BHM, fd[5], 0)
				nBH = np.append(nBH, fd[6], 0)
				SFR = np.append(SFR, fd[7], 0)
				shid = np.append(shid, fd[8])
				b = np.append(b, fd[9])
				m = np.append(m, fd[10])
			except:
				continue

		return [Radii, SM, CGM, HGM, DMM, BHM, nBH, SFR, shid, b, m]

	else:
		while loop:
			try:
				[Radii, SM, CGM, HGM, DMM, BHM, nBH, SFR, shid] = MBIIanalysisoutfile(secmin, sno, run)
				loop = False
			except:
				secmin +=1
			
		for i in range(secmin+1,secmax+1):
			try:
				fd = MBIIanalysisoutfile(i, sno, run) # File data
				Radii = np.append(Radii, fd[0], 0)
				SM = np.append(SM, fd[1], 0)
				CGM = np.append(CGM, fd[2], 0)
				HGM = np.append(HGM, fd[3], 0)
				DMM = np.append(DMM, fd[4], 0)
				BHM = np.append(BHM, fd[5], 0)
				nBH = np.append(nBH, fd[6], 0)
				SFR = np.append(SFR, fd[7], 0)
				shid = np.append(shid, fd[8])
			except:
				continue
		
		return [Radii, SM, CGM, HGM, DMM, BHM, nBH, SFR, shid]


def MBIIsubhalotab(sno='085',dir=None):
	# Read in the subhalo tab data from a MassiveBlack-II snapshot

	if dir==None: dir = '/share/spellbound/www/MB-IIa/subhalos/'
	if type(sno)!=str: sno = str(sno)
	if len(sno)==2: sno = '0'+sno

	fname = dir+sno+'/subhalotab.raw'
	#Ngal = os.stat(fname).st_size/104

	subdtype = np.dtype([
							('mass', 'f4'),
							('len', 'i4'),
							('pos', ('f4', 3)),           # potential min pos
							('vel', ('f4', 3)),
							('vdisp', 'f4'),
							('vcirc', 'f4'),
							('rcirc', 'f4'),
							('parent', 'i4'),             # parent structure
							('massbytype', ('f4', 6)),
							('lenbytype', ('u4',6)),
							('unused', 'f4'),
							('groupid', 'u4'),            # group id
							])
	f = open(fname,'rb')
	shtd = np.fromfile(f, subdtype) # SubHalo Tab Data
	f.close()
	return shtd


def MBIIgrouptab(sno='085',dir=None):
	# Read in group tab data from MB-II snapshot
	if dir==None: dir = '/share/spellbound/www/MB-IIa/subhalos/'
	if type(sno)!=str: sno = str(sno)
	if len(sno)==2: sno = '0'+sno
	
	fname = dir+sno+'/grouphalotab.raw'
	#Ngroup = os.stat(fname).st_size/84

	groupdtype = np.dtype([
							  ('mass', 'f4'),
							  ('len', 'i4'),
							  ('pos', ('f4', 3)),
							  ('vel', ('f4', 3)),
							  ('nhalo', 'i4'),            # number of subhalos + 1(contamination)
							  ('massbytype', ('f4', 6)),
							  ('lenbytype', ('u4',6)),
							  ])
	f = open(fname,'rb')
	gtd = np.fromfile(f, groupdtype) # SubHalo Tab Data
	f.close()
	return gtd


def MBIIgrouplen(sno='085',dir=None):
	# Just get the number of particles (and each type) for MB-II groups
	
	if dir==None: dir = '/share/spellbound/www/MB-IIa/subhalos/'
	if type(sno)!=str: sno = str(sno)
	if len(sno)==2: sno = '0'+sno

	gtd = MBIIgrouptab(sno,dir)
	return gtd['lenbytype']



def MBIIheader(sno='085',dir=None):
	# Read the important header info for an MBII snapshot
	if dir==None: dir = '/share/spellbound/www/MB-IIa/subhalos/'
	if type(sno)!=str: sno = str(sno)
	if len(sno)==2: sno = '0'+sno
	
	fname = dir+sno+'/header.txt'
	f = open(fname,'r')
	fr = f.read()
	fs = fr.split()
	f.close()

	for i in xrange(len(fs)):
		if fs[i]=='OmegaL(header)': Omega_L = float(fs[i+2])
		if fs[i]=='OmegaM(header)':Omega_M = float(fs[i+2])
		if fs[i]=='h(header)':h = float(fs[i+2]) # Hubble little h
		if fs[i]=='redshift(header)':z = float(fs[i+2]) # Redshift

	return Omega_L, Omega_M, h, z


def MBIIgroupparticledata(gno, sno='085', shtd=None, sh_groupid=None, sh_lenbytype=None, gp_Npart=None, dir=None):
	# Read in all the necessary particle data for a group in MassiveBlack-II
	# Need to know which group number and snapshot number

	if dir==None: dir = '/share/spellbound/www/MB-IIa/subhalos/'
	if type(sno)!=str: sno = str(sno)
	if len(sno)==2: sno = '0'+sno
	
	Omega_L, Omega_M, h, z = MBIIheader(sno,dir)
	a = 1./(1+z)
	
	# If subhalo tab data hasn't been fed in, then this will need to collect it first
	if (shtd==None) and (sh_groupid==None) and (sh_lenbytype==None): shtd = MBIIsubhalotab(sno,dir)

	# If the particle numbers (per type) aren't given as an input, read in the info
	if gp_Npart==None: gp_Npart = MBIIgrouplen(sno,dir)
	
	if sh_groupid==None: sh_groupid = shtd['groupid']
	gf = (sh_groupid==gno) # Group filter
	shid = np.array(np.transpose(np.argwhere(gf==True)),dtype='i4') # IDs of subhaloes in this group
	Nsh = len(shid) # Number of subhaloes in the group
	if sh_lenbytype==None: sh_lenbytype = shtd['lenbytype'][gf]
	
	
	Npart = gp_Npart[gno] # Get the number of particles (per type) for the group of interest


	# Read gas data
	Ng = Npart[0]

	f = open(dir+sno+'/0/pos.raw','rb')
	f.seek(4*3*sum(gp_Npart[:gno,0]))
	pos_g = np.fromfile(f, np.dtype(('f4',3)), Ng)*a*1e3/h
	f.close()

	f = open(dir+sno+'/0/vel.raw','rb')
	f.seek(4*3*sum(gp_Npart[:gno,0]))
	vel_g = np.fromfile(f, np.dtype(('f4',3)), Ng)
	f.close()

	f = open(dir+sno+'/0/mass.raw','rb')
	f.seek(4*sum(gp_Npart[:gno,0]))
	mass_g = np.fromfile(f, 'f4', Ng)*1e10/h
	f.close()

	f = open(dir+sno+'/0/sfr.raw','rb')
	f.seek(8*sum(gp_Npart[:gno,0]))
	sfr_g = np.fromfile(f, 'f8', Ng)
	f.close()

	"""
		Shouldn't need this as there doesn't appear to be any useful data here
	f = open(dir+sno+'/0/type.raw','rb')
	f.seek(sum(gp_Npart[:gno,0]))
	type_g = np.fromfile(f, 'i1', Ng)
	f.close()
	"""
	
	shid_g = []
	for i in xrange(Nsh): shid_g += ([shid[i]]*sh_lenbytype[i,0])
	shid_g = np.array(shid_g,dtype='i4')

	
	# Read DM data
	Ndm = Npart[1]

	f = open(dir+sno+'/1/pos.raw','rb')
	f.seek(4*3*sum(gp_Npart[:gno,1]))
	pos_dm = np.fromfile(f, np.dtype(('f4',3)), Ndm)*a*1e3/h
	f.close()

	f = open(dir+sno+'/1/vel.raw','rb')
	f.seek(4*3*sum(gp_Npart[:gno,1]))
	vel_dm = np.fromfile(f, np.dtype(('f4',3)), Ndm)
	f.close()

	f = open(dir+sno+'/1/mass.raw','rb')
	f.seek(4*sum(gp_Npart[:gno,1]))
	mass_dm = np.fromfile(f, 'f4', Ndm)*1e10/h
	f.close()

	shid_dm = []
	for i in xrange(Nsh): shid_dm += ([shid[i]]*sh_lenbytype[i,1])
	shid_dm = np.array(shid_dm,dtype='i4')

	
	# Read star data
	Ns = Npart[4]

	f = open(dir+sno+'/4/pos.raw','rb')
	f.seek(4*3*sum(gp_Npart[:gno,4]))
	pos_s = np.fromfile(f, np.dtype(('f4',3)), Ns)*a*1e3/h
	f.close()

	f = open(dir+sno+'/4/vel.raw','rb')
	f.seek(4*3*sum(gp_Npart[:gno,4]))
	vel_s = np.fromfile(f, np.dtype(('f4',3)), Ns)
	f.close()

	f = open(dir+sno+'/4/mass.raw','rb')
	f.seek(4*sum(gp_Npart[:gno,4]))
	mass_s = np.fromfile(f, 'f4', Ns)*1e10/h
	f.close()
	
	shid_s = []
	for i in xrange(Nsh): shid_s += ([shid[i]]*sh_lenbytype[i,4])
	shid_s = np.array(shid_s,dtype='i4')
	
	
	# Read black hole data
	Nbh = Npart[5]
	
	f = open(dir+sno+'/5/pos.raw','rb')
	f.seek(4*3*sum(gp_Npart[:gno,5]))
	pos_bh = np.fromfile(f, np.dtype(('f4',3)), Nbh)*a*1e3/h
	f.close()
	
	f = open(dir+sno+'/5/vel.raw','rb')
	f.seek(4*3*sum(gp_Npart[:gno,5]))
	vel_bh = np.fromfile(f, np.dtype(('f4',3)), Nbh)
	f.close()
	
	f = open(dir+sno+'/5/mass.raw','rb')
	f.seek(4*sum(gp_Npart[:gno,5]))
	mass_bh = np.fromfile(f, 'f4', Nbh)*1e10/h
	f.close()
	
	shid_bh = []
	for i in xrange(Nsh): shid_bh += ([shid[i]]*sh_lenbytype[i,5])
	shid_bh = np.array(shid_bh,dtype='i4')

	return [pos_g[0],pos_g[1],pos_g[2],vel_g[0],vel_g[1],vel_g[2],mass_g,sfr_g,shid_g], [pos_dm[0],pos_dm[1],pos_dm[2],vel_dm[0],vel_dm[1],vel_dm[2],mass_dm,shid_dm], [pos_s[0],pos_s[1],pos_s[2],vel_s[0],vel_s[1],vel_s[2],mass_s,shid_s], [pos_bh[0],pos_bh[1],pos_bh[2],vel_bh[0],vel_bh[1],vel_bh[2],mass_bh,shid_bh], shid



def MBIIsubhaloparticledata(shno, sno='085', shtd=None, dir=None, head=None, lookup=None):
	# Read in all the necessary particle data for a group in MassiveBlack-II
	# Need to know which group number and snapshot number
	
	if dir==None: dir = '/share/spellbound/www/MB-IIa/subhalos/'
	if type(sno)!=str: sno = str(sno)
	if len(sno)==2: sno = '0'+sno
	
	if head==None:
		Omega_L, Omega_M, h, z = MBIIheader(sno,dir)
	else:
		Omega_L, Omega_M, h, z = head[0], head[1], head[2], head[3]
	a = 1./(1+z)
	
	# If subhalo tab data hasn't been fed in, then this will need to collect it first
	if shtd==None: shtd = MBIIsubhalotab(sno,dir)
		
	sh_Npart = shtd['lenbytype']
	
	
	# Read gas data
	Ng = sh_Npart[shno,0]
	print 'No. gas parts.', Ng
	
	if Ng>0:
		f = open(dir+sno+'/0/pos.raw','rb')
		f.seek(4*3*sum(sh_Npart[:shno,0]))
		pos_g = np.fromfile(f, np.dtype(('f4',3)), Ng)*a*1e3/h
		x_g, y_g, z_g = pos_g[:,0], pos_g[:,1], pos_g[:,2]
		f.close()
		
		f = open(dir+sno+'/0/vel.raw','rb')
		f.seek(4*3*sum(sh_Npart[:shno,0]))
		vel_g = np.fromfile(f, np.dtype(('f4',3)), Ng)
		vx_g, vy_g, vz_g = vel_g[:,0], vel_g[:,1], vel_g[:,2]
		f.close()
		
		f = open(dir+sno+'/0/mass.raw','rb')
		f.seek(4*sum(sh_Npart[:shno,0]))
		mass_g = np.fromfile(f, 'f4', Ng)*1e10/h
		f.close()
		
		# For whatever reason, some snapshots have 32-bit SFRs and some 64-bit
		if os.path.getsize(dir+sno+'/0/mass.raw')==os.path.getsize(dir+sno+'/0/sfr.raw'):
			Nbyte = 4
			dtype = 'f4'
		else:
			Nbyte = 8
			dtype = 'f8'
		f = open(dir+sno+'/0/sfr.raw','rb')
		f.seek(Nbyte*sum(sh_Npart[:shno,0]))
		sfr_g = np.fromfile(f, dtype, Ng)
		f.close()
	else:
		x_g, y_g, z_g, vx_g, vy_g, vz_g, mass_g, sfr_g = np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4')


	# Read DM data
	Ndm = sh_Npart[shno,1]
	print 'No. DM parts.', Ndm

	if Ndm>0:
		f = open(dir+sno+'/1/pos.raw','rb')
		f.seek(4*3*sum(sh_Npart[:shno,1]))
		pos_dm = np.fromfile(f, np.dtype(('f4',3)), Ndm)*a*1e3/h
		x_dm, y_dm, z_dm = pos_dm[:,0], pos_dm[:,1], pos_dm[:,2]
		f.close()
		
		f = open(dir+sno+'/1/vel.raw','rb')
		f.seek(4*3*sum(sh_Npart[:shno,1]))
		vel_dm = np.fromfile(f, np.dtype(('f4',3)), Ndm)
		vx_dm, vy_dm, vz_dm = vel_dm[:,0], vel_dm[:,1], vel_dm[:,2]
		f.close()
		
		f = open(dir+sno+'/1/mass.raw','rb')
		f.seek(4*sum(sh_Npart[:shno,1]))
		mass_dm = np.fromfile(f, 'f4', Ndm)*1e10/h
		f.close()
	else:
		x_dm, y_dm, z_dm, vx_dm, vy_dm, vz_dm, mass_dm = np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4')


	# Read star data
	Ns = sh_Npart[shno,4]
	print 'No. star parts.', Ns

	if Ns>0:
		f = open(dir+sno+'/4/pos.raw','rb')
		f.seek(4*3*sum(sh_Npart[:shno,4]))
		pos_s = np.fromfile(f, np.dtype(('f4',3)), Ns)*a*1e3/h
		x_s, y_s, z_s = pos_s[:,0], pos_s[:,1], pos_s[:,2]
		print len(x_s)
		f.close()
		
		f = open(dir+sno+'/4/vel.raw','rb')
		f.seek(4*3*sum(sh_Npart[:shno,4]))
		vel_s = np.fromfile(f, np.dtype(('f4',3)), Ns)
		vx_s, vy_s, vz_s = vel_s[:,0], vel_s[:,1], vel_s[:,2]
		f.close()
		
		f = open(dir+sno+'/4/mass.raw','rb')
		f.seek(4*sum(sh_Npart[:shno,4]))
		mass_s = np.fromfile(f, 'f4', Ns)*1e10/h
		f.close()
		
		f = open(dir+sno+'/4/sft.raw','rb')
		f.seek(8*sum(sh_Npart[:shno,4]))
		birth_a = np.fromfile(f, 'f8', Ns)
		f.close()
		
		if lookup==None:
			tarr, zarr = gc.ztlookup(H_0=100*h,Omega_M=Omega_M,Omega_Lambda=Omega_L)
		else:
			tarr, zarr = lookup[0], lookup[1]
		birth_z = (1./birth_a)-1
		birth = np.interp(birth_z,zarr,tarr)
		
		# Some metallicities stored as 32-bit and some 64-bit
		if os.path.getsize(dir+sno+'/4/mass.raw')==os.path.getsize(dir+sno+'/4/met.raw'):
			Nbyte = 4
			dtype = 'f4'
		else:
			Nbyte = 8
			dtype = 'f8'
		f = open(dir+sno+'/4/met.raw','rb')
		f.seek(Nbyte*sum(sh_Npart[:shno,4]))
		met_s = np.fromfile(f, dtype, Ns)
		f.close()
		
		
	else:
		x_s, y_s, z_s, vx_s, vy_s, vz_s, mass_s, birth, met_s = np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4')


	# Read black hole data
	Nbh = sh_Npart[shno,5]
	print 'No. black holes', Nbh

	if Nbh>0:
		f = open(dir+sno+'/5/pos.raw','rb')
		f.seek(4*3*sum(sh_Npart[:shno,5]))
		pos_bh = np.fromfile(f, np.dtype(('f4',3)), Nbh)*a*1e3/h
		x_bh, y_bh, z_bh = pos_bh[:,0], pos_bh[:,1], pos_bh[:,2]
		f.close()
		
		f = open(dir+sno+'/5/vel.raw','rb')
		f.seek(4*3*sum(sh_Npart[:shno,5]))
		vel_bh = np.fromfile(f, np.dtype(('f4',3)), Nbh)
		vx_bh, vy_bh, vz_bh = vel_bh[:,0], vel_bh[:,1], vel_bh[:,2]
		f.close()
		
		f = open(dir+sno+'/5/mass.raw','rb')
		f.seek(4*sum(sh_Npart[:shno,5]))
		mass_bh = np.fromfile(f, 'f4', Nbh)*1e10/h
		f.close()
	else:
		x_bh, y_bh, z_bh, vx_bh, vy_bh, vz_bh, mass_bh = np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4'), np.array([],dtype='f4')

	[x_s,x_g,x_dm,x_bh], [y_s,y_g,y_dm,y_bh], [z_s,z_g,z_dm,z_bh], [vx_s,vx_g,vx_dm,vx_bh], [vy_s,vy_g,vy_dm,vy_bh], [vz_s,vz_g,vz_dm,vz_bh] = gc.recenrotgalall([x_s,x_g,x_dm,x_bh], [y_s,y_g,y_dm,y_bh], [z_s,z_g,z_dm,z_bh], [vx_s,vx_g,vx_dm,vx_bh], [vy_s,vy_g,vy_dm,vy_bh], [vz_s,vz_g,vz_dm,vz_bh], [mass_s,mass_g,mass_dm], r=min(5e4,np.max(np.sqrt(x_s**2 + y_s**2 + z_s**2))))

	return [x_g,y_g,z_g,vx_g,vy_g,vz_g,mass_g,sfr_g], [x_dm,y_dm,z_dm,vx_dm,vy_dm,vz_dm,mass_dm], [x_s,y_s,z_s,vx_s,vy_s,vz_s,mass_s,birth,met_s], [x_bh,y_bh,z_bh,vx_bh,vy_bh,vz_bh,mass_bh]


def MBIIsubhalofile(hno, sno):
	# Read in the subhalo particle data for one that I've put into a single file already.
	hno = str(hno)
	f = open('MBII/'+sno+'/'+hno+'/particledata', 'rb')
	N = np.fromfile(f, 'i8', 6)
	print 'N is', N

	if N[0]>0:
		x_g = np.fromfile(f, 'f4', N[0])
		y_g = np.fromfile(f, 'f4', N[0])
		z_g = np.fromfile(f, 'f4', N[0])
		vx_g = np.fromfile(f, 'f4', N[0])
		vy_g = np.fromfile(f, 'f4', N[0])
		vz_g = np.fromfile(f, 'f4', N[0])
		mass_g = np.fromfile(f, 'f4', N[0])
		sfr = np.fromfile(f, 'f8', N[0])
		rho = np.fromfile(f, 'f8', N[0])
	else:
		x_g, y_g, z_g, vx_g, vy_g, vz_g, mass_g, sfr, rho = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

	if N[1]>0:
		x_dm = np.fromfile(f, 'f4', N[1])
		y_dm = np.fromfile(f, 'f4', N[1])
		z_dm = np.fromfile(f, 'f4', N[1])
		vx_dm = np.fromfile(f, 'f4', N[1])
		vy_dm = np.fromfile(f, 'f4', N[1])
		vz_dm = np.fromfile(f, 'f4', N[1])
		mass_dm = np.fromfile(f, 'f4', N[1])
	else:
		x_dm, y_dm, z_dm, vx_dm, vy_dm, vz_dm, mass_dm = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

	if N[4]>0:
		x_s = np.fromfile(f, 'f4', N[4])
		y_s = np.fromfile(f, 'f4', N[4])
		z_s = np.fromfile(f, 'f4', N[4])
		vx_s = np.fromfile(f, 'f4', N[4])
		vy_s = np.fromfile(f, 'f4', N[4])
		vz_s = np.fromfile(f, 'f4', N[4])
		mass_s = np.fromfile(f, 'f4', N[4])
	else:
		x_s, y_s, z_s, vx_s, vy_s, vz_s, mass_s = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

	if N[5]>0:
		x_bh = np.fromfile(f, 'f4', N[5])
		y_bh = np.fromfile(f, 'f4', N[5])
		z_bh = np.fromfile(f, 'f4', N[5])
		vx_bh = np.fromfile(f, 'f4', N[5])
		vy_bh = np.fromfile(f, 'f4', N[5])
		vz_bh = np.fromfile(f, 'f4', N[5])
		mass_bh = np.fromfile(f, 'f4', N[5])
	else:
		x_bh, y_bh, z_bh, vx_bh, vy_bh, vz_bh, mass_bh = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

	return [x_g,y_g,z_g,vx_g,vy_g,vz_g,mass_g,sfr,rho], [x_dm,y_dm,z_dm,vx_dm,vy_dm,vz_dm,mass_dm], [x_s,y_s,z_s,vx_s,vy_s,vz_s,mass_s], [x_bh,y_bh,z_bh,vx_bh,vy_bh,vz_bh,mass_bh]


def MBIIsubhaloSED(hno,sno='085'):
	# Read in the spectrum from a subhalo
	dir = '/share/spellbound/www/MB-IIa/subhalos/'+sno+'/SED/'

	f = open(dir+'index.raw', 'rb')
	f.seek(hno*8)
	offset = np.fromfile(f, 'i8', 1)[0]
	f.close()

	if offset != -1:
		f = open(dir+'full.raw', 'rb')
		f.seek(offset*1220*4)
		spec = np.fromfile(f, 'f4', 1220)
		f.close()
		
	else:
		spec = None

	return spec

def MBIIwavelengthSED():
	wl = np.array([
				   91, 94, 96, 98, 100, 102, 104, 106, 108, 110,
				   114, 118, 121, 125, 127, 128, 131, 132, 134, 137,
				   140, 143, 147, 151, 155, 159, 162, 166, 170, 173,
				   177, 180, 182, 186, 191, 194, 198, 202, 205, 210,
				   216, 220, 223, 227, 230, 234, 240, 246, 252, 257,
				   260, 264, 269, 274, 279, 284, 290, 296, 301, 308,
				   318, 328, 338, 348, 357, 366, 375, 385, 395, 405,
				   414, 422, 430, 441, 451, 460, 470, 480, 490, 500,
				   506, 512, 520, 530, 540, 550, 560, 570, 580, 590,
				   600, 610, 620, 630, 640, 650, 658, 665, 675, 685,
				   695, 705, 716, 726, 735, 745, 755, 765, 775, 785,
				   795, 805, 815, 825, 835, 845, 855, 865, 875, 885,
				   895, 905, 915, 925, 935, 945, 955, 965, 975, 985,
				   995, 1005, 1015, 1025, 1035, 1045, 1055, 1065, 1075, 1085,
				   1095, 1105, 1115, 1125, 1135, 1145, 1155, 1165, 1175, 1185,
				   1195, 1205, 1215, 1225, 1235, 1245, 1255, 1265, 1275, 1285,
				   1295, 1305, 1315, 1325, 1335, 1345, 1355, 1365, 1375, 1385,
				   1395, 1405, 1415, 1425, 1435, 1442, 1447, 1455, 1465, 1475,
				   1485, 1495, 1505, 1512, 1517, 1525, 1535, 1545, 1555, 1565,
				   1575, 1585, 1595, 1605, 1615, 1625, 1635, 1645, 1655, 1665,
				   1672, 1677, 1685, 1695, 1705, 1715, 1725, 1735, 1745, 1755,
				   1765, 1775, 1785, 1795, 1805, 1815, 1825, 1835, 1845, 1855,
				   1865, 1875, 1885, 1895, 1905, 1915, 1925, 1935, 1945, 1955,
				   1967, 1976, 1984, 1995, 2005, 2015, 2025, 2035, 2045, 2055,
				   2065, 2074, 2078, 2085, 2095, 2105, 2115, 2125, 2135, 2145,
				   2155, 2165, 2175, 2185, 2195, 2205, 2215, 2225, 2235, 2245,
				   2255, 2265, 2275, 2285, 2295, 2305, 2315, 2325, 2335, 2345,
				   2355, 2365, 2375, 2385, 2395, 2405, 2415, 2425, 2435, 2445,
				   2455, 2465, 2475, 2485, 2495, 2505, 2513, 2518, 2525, 2535,
				   2545, 2555, 2565, 2575, 2585, 2595, 2605, 2615, 2625, 2635,
				   2645, 2655, 2665, 2675, 2685, 2695, 2705, 2715, 2725, 2735,
				   2745, 2755, 2765, 2775, 2785, 2795, 2805, 2815, 2825, 2835,
				   2845, 2855, 2865, 2875, 2885, 2895, 2910, 2930, 2950, 2970,
				   2990, 3010, 3030, 3050, 3070, 3090, 3110, 3130, 3150, 3170,
				   3190, 3210, 3230, 3250, 3270, 3290, 3310, 3330, 3350, 3370,
				   3390, 3410, 3430, 3450, 3470, 3490, 3510, 3530, 3550, 3570,
				   3590, 3610, 3630, 3640, 3650, 3670, 3690, 3710, 3730, 3750,
				   3770, 3790, 3810, 3830, 3850, 3870, 3890, 3910, 3930, 3950,
				   3970, 3990, 4010, 4030, 4050, 4070, 4090, 4110, 4130, 4150,
				   4170, 4190, 4210, 4230, 4250, 4270, 4290, 4310, 4330, 4350,
				   4370, 4390, 4410, 4430, 4450, 4470, 4490, 4510, 4530, 4550,
				   4570, 4590, 4610, 4630, 4650, 4670, 4690, 4710, 4730, 4750,
				   4770, 4790, 4810, 4830, 4850, 4870, 4890, 4910, 4930, 4950,
				   4970, 4990, 5010, 5030, 5050, 5070, 5090, 5110, 5130, 5150,
				   5170, 5190, 5210, 5230, 5250, 5270, 5290, 5310, 5330, 5350,
				   5370, 5390, 5410, 5430, 5450, 5470, 5490, 5510, 5530, 5550,
				   5570, 5590, 5610, 5630, 5650, 5670, 5690, 5710, 5730, 5750,
				   5770, 5790, 5810, 5830, 5850, 5870, 5890, 5910, 5930, 5950,
				   5970, 5990, 6010, 6030, 6050, 6070, 6090, 6110, 6130, 6150,
				   6170, 6190, 6210, 6230, 6250, 6270, 6290, 6310, 6330, 6350,
				   6370, 6390, 6410, 6430, 6450, 6470, 6490, 6510, 6530, 6550,
				   6570, 6590, 6610, 6630, 6650, 6670, 6690, 6710, 6730, 6750,
				   6770, 6790, 6810, 6830, 6850, 6870, 6890, 6910, 6930, 6950,
				   6970, 6990, 7010, 7030, 7050, 7070, 7090, 7110, 7130, 7150,
				   7170, 7190, 7210, 7230, 7250, 7270, 7290, 7310, 7330, 7350,
				   7370, 7390, 7410, 7430, 7450, 7470, 7490, 7510, 7530, 7550,
				   7570, 7590, 7610, 7630, 7650, 7670, 7690, 7710, 7730, 7750,
				   7770, 7790, 7810, 7830, 7850, 7870, 7890, 7910, 7930, 7950,
				   7970, 7990, 8010, 8030, 8050, 8070, 8090, 8110, 8130, 8150,
				   8170, 8190, 8210, 8230, 8250, 8270, 8290, 8310, 8330, 8350,
				   8370, 8390, 8410, 8430, 8450, 8470, 8490, 8510, 8530, 8550,
				   8570, 8590, 8610, 8630, 8650, 8670, 8690, 8710, 8730, 8750,
				   8770, 8790, 8810, 8830, 8850, 8870, 8890, 8910, 8930, 8950,
				   8970, 8990, 9010, 9030, 9050, 9070, 9090, 9110, 9130, 9150,
				   9170, 9190, 9210, 9230, 9250, 9270, 9290, 9310, 9330, 9350,
				   9370, 9390, 9410, 9430, 9450, 9470, 9490, 9510, 9530, 9550,
				   9570, 9590, 9610, 9630, 9650, 9670, 9690, 9710, 9730, 9750,
				   9770, 9790, 9810, 9830, 9850, 9870, 9890, 9910, 9930, 9950,
				   9970, 9990, 10025, 10075, 10125, 10175, 10225, 10275, 10325, 10375,
				   10425, 10475, 10525, 10575, 10625, 10675, 10725, 10775, 10825, 10875,
				   10925, 10975, 11025, 11075, 11125, 11175, 11225, 11275, 11325, 11375,
				   11425, 11475, 11525, 11575, 11625, 11675, 11725, 11775, 11825, 11875,
				   11925, 11975, 12025, 12075, 12125, 12175, 12225, 12275, 12325, 12375,
				   12425, 12475, 12525, 12575, 12625, 12675, 12725, 12775, 12825, 12875,
				   12925, 12975, 13025, 13075, 13125, 13175, 13225, 13275, 13325, 13375,
				   13425, 13475, 13525, 13575, 13625, 13675, 13725, 13775, 13825, 13875,
				   13925, 13975, 14025, 14075, 14125, 14175, 14225, 14275, 14325, 14375,
				   14425, 14475, 14525, 14570, 14620, 14675, 14725, 14775, 14825, 14875,
				   14925, 14975, 15025, 15075, 15125, 15175, 15225, 15275, 15325, 15375,
				   15425, 15475, 15525, 15575, 15625, 15675, 15725, 15775, 15825, 15875,
				   15925, 15975, 16050, 16150, 16250, 16350, 16450, 16550, 16650, 16750,
				   16850, 16950, 17050, 17150, 17250, 17350, 17450, 17550, 17650, 17750,
				   17850, 17950, 18050, 18150, 18250, 18350, 18450, 18550, 18650, 18750,
				   18850, 18950, 19050, 19150, 19250, 19350, 19450, 19550, 19650, 19750,
				   19850, 19950, 20050, 20150, 20250, 20350, 20450, 20550, 20650, 20750,
				   20850, 20950, 21050, 21150, 21250, 21350, 21450, 21550, 21650, 21750,
				   21850, 21950, 22050, 22150, 22250, 22350, 22450, 22550, 22650, 22750,
				   22850, 22950, 23050, 23150, 23250, 23350, 23450, 23550, 23650, 23750,
				   23850, 23950, 24050, 24150, 24250, 24350, 24450, 24550, 24650, 24750,
				   24850, 24950, 25050, 25150, 25250, 25350, 25450, 25550, 25650, 25750,
				   25850, 25950, 26050, 26150, 26250, 26350, 26450, 26550, 26650, 26750,
				   26850, 26950, 27050, 27150, 27250, 27350, 27450, 27550, 27650, 27750,
				   27850, 27950, 28050, 28150, 28250, 28350, 28450, 28550, 28650, 28750,
				   28850, 28950, 29050, 29150, 29250, 29350, 29450, 29550, 29650, 29750,
				   29850, 29950, 30050, 30150, 30250, 30350, 30450, 30550, 30650, 30750,
				   30850, 30950, 31050, 31150, 31250, 31350, 31450, 31550, 31650, 31750,
				   31850, 31950, 32100, 32300, 32500, 32700, 32900, 33100, 33300, 33500,
				   33700, 33900, 34100, 34300, 34500, 34700, 34900, 35100, 35300, 35500,
				   35700, 35900, 36100, 36300, 36500, 36700, 36900, 37100, 37300, 37500,
				   37700, 37900, 38100, 38300, 38500, 38700, 38900, 39100, 39300, 39500,
				   39700, 39900, 40100, 40300, 40500, 40700, 40900, 41100, 41300, 41500,
				   41700, 41900, 42100, 42300, 42500, 42700, 42900, 43100, 43300, 43500,
				   43700, 43900, 44100, 44300, 44500, 44700, 44900, 45100, 45300, 45500,
				   45700, 45900, 46100, 46300, 46500, 46700, 46900, 47100, 47300, 47500,
				   47700, 47900, 48100, 48300, 48500, 48700, 48900, 49100, 49300, 49500,
				   49700, 49900, 50100, 50300, 50500, 50700, 50900, 51100, 51300, 51500,
				   51700, 51900, 52100, 52300, 52500, 52700, 52900, 53100, 53300, 53500,
				   53700, 53900, 54100, 54300, 54500, 54700, 54900, 55100, 55300, 55500,
				   55700, 55900, 56100, 56300, 56500, 56700, 56900, 57100, 57300, 57500,
				   57700, 57900, 58100, 58300, 58500, 58700, 58900, 59100, 59300, 59500,
				   59700, 59900, 60100, 60300, 60500, 60700, 60900, 61100, 61300, 61500,
				   61700, 61900, 62100, 62300, 62500, 62700, 62900, 63100, 63300, 63500,
				   63700, 63900, 64200, 64600, 65000, 65400, 65800, 66200, 66600, 67000,
				   67400, 67800, 68200, 68600, 69000, 69400, 69800, 70200, 70600, 71000,
				   71400, 71800, 72200, 72600, 73000, 73400, 73800, 74200, 74600, 75000,
				   75400, 75800, 76200, 76600, 77000, 77400, 77800, 78200, 78600, 79000,
				   79400, 79800, 80200, 80600, 81000, 81400, 81800, 82200, 82600, 83000,
				   83400, 83800, 84200, 84600, 85000, 85400, 85800, 86200, 86600, 87000,
				   87400, 87800, 88200, 88600, 89000, 89400, 89800, 90200, 90600, 91000,
				   91400, 91800, 92200, 92600, 93000, 93400, 93800, 94200, 94600, 95000,
				   95400, 95800, 96200, 96600, 97000, 97400, 97800, 98200, 98600, 99000,
				   99400, 99800, 100200, 200000, 400000, 600000, 800000, 1e+06, 1.2e+06, 1.4e+06])
	return wl

#==============OTHER GADGET EXAMPLES (ANITA 2014 WORKSHOP)=========

def gadgethead(f):
	# Just read a GADGET header for a file already loaded.  Could really use this for all my other codes above too...
	Nbytes1 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the header uses
	N = np.fromfile(f, 'u4', 6) # Number of particles for each particle type in this file
	Nsum = sum(N) # Total number of particles in the file
	mass_pt = np.fromfile(f, 'f8', 6) # Mass of each particle type.  If 0 then it varies for each particle of that type
	Nmass = sum(N[np.argwhere(mass_pt==0.0)]) # Number of particles in the file with individual masses to be read in
	a = np.fromfile(f, 'f8', 1)[0] # Expansion factor (normalised to 1 at z=0)
	z = np.fromfile(f, 'f8', 1)[0] # Redshift of snapshot
	flag_sfr = np.fromfile(f, 'i4', 1)[0] # Flag for star formation rate
	flag_feedback = np.fromfile(f, 'i4', 1)[0] # Flag for feedback
	Ntot = np.fromfile(f, 'u4', 6) # Total number of particles for each particle type in the entire simulation
	flag_cool = np.fromfile(f, 'i4', 1)[0] # Flag for cooling
	Nfiles = np.fromfile(f, 'i4', 1)[0] # Number of files for each snapshot
	boxsize = np.fromfile(f, 'f8', 1)[0] # Size of box if periodic boundary conditions are used
	Omega_M = np.fromfile(f, 'f8', 1)[0] # Omega Matter
	Omega_L = np.fromfile(f, 'f8', 1)[0] # Omega (Lambda) Dark Energy
	h = np.fromfile(f, 'f8', 1)[0] # Little Hubble h
	flag_StarAge = np.fromfile(f, 'i4', 1)[0] # Flag for the creation times of stars
	flag_metals = np.fromfile(f, 'i4', 1)[0] # Flag for metallicity values
	NallHW = np.fromfile(f, 'u4', 6) # Don't fully understand this.  To do with simulations that use >2^32 particles
	flag_entropy = np.fromfile(f, 'i4', 1)[0] # Flag that the initial conditions contain entropy instead of thermal energy in the thermal energy block
	flag_double = np.fromfile(f, 'i4', 1)[0] # Don't know what this flag is for
	flag_ic_info = np.fromfile(f, 'i4', 1)[0] # Don't know what this flag is for
	flag_scale = np.fromfile(f, 'i4', 1)[0] # Don't know what this flag is for
	unused = np.fromfile(f, 'u4', 12) # Apparently unused stuff
	Nbytes1 = np.fromfile(f, 'i4', 1)[0] # Number of bytes the header uses (again)
	return N, mass_pt, a, z, boxsize


def galmerg(sno, fpre=None):
	# Read in the galaxy merger example
	if fpre==None:
		f = open('/home/adam/Gadget-2.0.7/Examples/GalMerger/snapshot_00'+str(sno), 'rb')
	else:
		f = open(fpre+str(sno), 'rb')

	Nbytes1 = np.fromfile(f, 'i4', 1)[0]	
	N = np.fromfile(f, 'u4', 6) # Number of particles for each particle type in this file
	print N
	Nsum = sum(N) # Total number of particles in the file
	mass_pt = np.fromfile(f, 'f8', 6) # Mass of each particle type.  If 0 then it varies for each particle of that type
	Nmass = sum(N[np.argwhere(mass_pt==0.0)]) # Number of particles in the file with individual masses to be read in

	f.seek(Nbytes1+8)
	Nbytes2 = np.fromfile(f, 'i4', 1)[0]
	pos = np.fromfile(f, np.dtype(('f4',3)), Nsum)*1e3
	Nbytes2 = np.fromfile(f, 'i4', 1)[0]

	Nbytes3 = np.fromfile(f, 'i4', 1)[0]
	vel = np.fromfile(f, np.dtype(('f4',3)), Nsum)

	x_dm, y_dm, z_dm = pos[:N[1],0], pos[:N[1],1], pos[:N[1],2]
	x_s, y_s, z_s = pos[N[1]:,0], pos[N[1]:,1], pos[N[1]:,2]

	vx_dm, vy_dm, vz_dm = vel[:N[1],0], vel[:N[1],1], vel[:N[1],2]
	vx_s, vy_s, vz_s = vel[N[1]:,0], vel[N[1]:,1], vel[N[1]:,2]

	mass_dm = np.ones(N[1])*mass_pt[1]*1e10
	mass_s = np.ones(N[2])*mass_pt[2]*1e10

	return [x_s,y_s,z_s,vx_s,vy_s,vz_s,mass_s], [x_dm,y_dm,z_dm,vx_dm,vy_dm,vz_dm,mass_dm]
	

def galmerg2(sno, fpre=None):
	# Read in the galaxy merger example
	if fpre==None:
		f = open('/home/adam/Gadget-2.0.7/Examples/GalwHaloMerger/snapshot_00'+str(sno), 'rb')
	else:
		f = open(fpre+str(sno), 'rb')

	Nbytes1 = np.fromfile(f, 'i4', 1)[0]	
	N = np.fromfile(f, 'u4', 6) # Number of particles for each particle type in this file
	print N
	Nsum = sum(N) # Total number of particles in the file
	mass_pt = np.fromfile(f, 'f8', 6) # Mass of each particle type.  If 0 then it varies for each particle of that type
	Nmass = sum(N[np.argwhere(mass_pt==0.0)]) # Number of particles in the file with individual masses to be read in

	f.seek(Nbytes1+8)
	Nbytes2 = np.fromfile(f, 'i4', 1)[0]
	pos = np.fromfile(f, np.dtype(('f4',3)), Nsum)*1e3
	Nbytes2 = np.fromfile(f, 'i4', 1)[0]

	Nbytes3 = np.fromfile(f, 'i4', 1)[0]
	vel = np.fromfile(f, np.dtype(('f4',3)), Nsum)

	Nstars = N[3]+N[4]
	Ndm = N[5]

	x_s, y_s, z_s = pos[:Nstars,0], pos[:Nstars,1], pos[:Nstars,2]
	x_dm, y_dm, z_dm = pos[Nstars:,0], pos[Nstars:,1], pos[Nstars:,2]

	vx_s, vy_s, vz_s = vel[:Nstars,0], vel[:Nstars,1], vel[:Nstars,2]
	vx_dm, vy_dm, vz_dm = vel[Nstars:,0], vel[Nstars:,1], vel[Nstars:,2]

	mass_dm = np.ones(Ndm)*mass_pt[3]*1e10
	mass_s = np.ones(Nstars)*mass_pt[5]*1e10

	return [x_dm,y_dm,z_dm,vx_dm,vy_dm,vz_dm,mass_dm]


def lcdm_example(sno, fpre=None):
	# Read in the galaxy merger example
	if fpre==None:
		f = open('/home/adam/Gadget-2.0.7/Examples/LCDM/snapshot_00'+str(sno), 'rb')
	else:
		f = open(fpre+str(sno), 'rb')

	Nbytes1 = np.fromfile(f, 'i4', 1)[0]	
	N = np.fromfile(f, 'u4', 6) # Number of particles for each particle type in this file
	print N
	Nsum = sum(N) # Total number of particles in the file
	mass_pt = np.fromfile(f, 'f8', 6) # Mass of each particle type.  If 0 then it varies for each particle of that type
	Nmass = sum(N[np.argwhere(mass_pt==0.0)]) # Number of particles in the file with individual masses to be read in
	a = np.fromfile(f, 'f8', 1)[0] # Expansion factor (normalised to 1 at z=0)
	z = np.fromfile(f, 'f8', 1)[0] # Redshift of snapshot
	flag_sfr = np.fromfile(f, 'i4', 1)[0] # Flag for star formation rate
	flag_feedback = np.fromfile(f, 'i4', 1)[0] # Flag for feedback
	Ntot = np.fromfile(f, 'u4', 6) # Total number of particles for each particle type in the entire simulation
	flag_cool = np.fromfile(f, 'i4', 1)[0] # Flag for cooling
	Nfiles = np.fromfile(f, 'i4', 1)[0] # Number of files for each snapshot
	boxsize = np.fromfile(f, 'f8', 1)[0] # Size of box if periodic boundary conditions are used
	Omega_M = np.fromfile(f, 'f8', 1)[0] # Omega Matter
	Omega_L = np.fromfile(f, 'f8', 1)[0] # Omega (Lambda) Dark Energy
	h = np.fromfile(f, 'f8', 1)[0] # Little Hubble h

	f.seek(Nbytes1+8)
	Nbytes2 = np.fromfile(f, 'i4', 1)[0]
	pos = np.fromfile(f, np.dtype(('f4',3)), Nsum)*1e3*a/h
	Nbytes2 = np.fromfile(f, 'i4', 1)[0]

	Nbytes3 = np.fromfile(f, 'i4', 1)[0]
	vel = np.fromfile(f, np.dtype(('f4',3)), Nsum)

	Ndm = N[1]

	x_dm, y_dm, z_dm = pos[:Ndm,0], pos[:Ndm,1], pos[:Ndm:,2]
	x_dm = x_dm + (np.max(x_dm)-np.min(x_dm))/2
	y_dm = y_dm + (np.max(y_dm)-np.min(y_dm))/2
	z_dm = z_dm + (np.max(z_dm)-np.min(z_dm))/2

	vx_dm, vy_dm, vz_dm = vel[:Ndm,0], vel[:Ndm,1], vel[:Ndm,2]

	mass_dm = np.ones(Ndm)*mass_pt[1]*1e10/h

	return [x_dm,y_dm,z_dm,vx_dm,vy_dm,vz_dm,mass_dm]



#===============READING IN OBSERVATIONAL DATA=======================

def obsgasids():
	# Find all the shared GASS ids between the relevant files for obsgasmass
	f1 = open('Saintonge2011aTab1Data.txt','r')
	d1 = f1.read()
	ds1 = d1.split()
	GASSid1 = np.array(ds1[0::9],dtype=int)
	f1.close()
	
	f2 = open('Saintonge2011aTab2Data.txt','r')
	d2 = f2.read()
	ds2 = d2.split()
	GASSid2 = np.array(ds2[0::9],dtype=int)
	f2.close()
	
	f3 = open('Catinella2012bTab2Data.txt','r')
	d3 = f3.read()
	ds3 = d3.split()
	GASSid3 = np.array(ds3[0::18],dtype=int)
	f3.close()
	
	GASSids = GASSid1[np.in1d(GASSid1,GASSid2)]
	GASSids = GASSids[np.in1d(GASSids,GASSid3)]
	return GASSids

def obsgasmass():
	comids = obsgasids() # Common GASS ids between all the files
	
	# Load in file that contains stellar masses and redshifts of galaxies
	with open('Saintonge2011aTab1Data.txt','r') as f:
		data = f.read()
		ds = data.split()
	
	SM = np.array(ds[3::9],dtype=float) # Stellar masses of galaxies (log solar masses)
	f1 = (SM>=10.6)*(SM<=10.65) # filter1, corresponds to the right mass range
	f2 = (SM>10.5)*(SM<10.6) + (SM>10.65)*(SM<10.76) # filter2, uncertainty overlaps the right mass range
	
	GASSid = np.array(ds[0::9],dtype=int) # GASS identifiers
	GASSid1 = GASSid[f1] # GASS ids that correspond to filter1
	GASSid2 = GASSid[f2] # Ditto filter 2
	
	#sort1 = np.argsort(GASSid1) # Indices to sort the arrays corresponding to the first GASS id set
	#sort2 = np.argsort(GASSid2)
	
	#GASSid1 = GASSid1[sort1]
	#GASSid2 = GASSid2[sort2] # Do the sorting
	
	cf1 = np.in1d(GASSid1,comids) # Common filter 1
	cf2 = np.in1d(GASSid2,comids) # Common filter 2
	
	GASSid1 = GASSid1[cf1]
	GASSid2 = GASSid2[cf2]
	
	z = np.array(ds[2::9],dtype=float) # Redshifts of the galaxies
	z1 = z[f1][cf1]
	z2 = z[f2][cf2]
	
	
	
	# Load in file that contains molecular hydrogen masses of galaxies
	with open('Saintonge2011aTab2Data.txt','r') as f:
		data = f.read()
		ds = data.split()
	
	GASSid = np.array(ds[0::9],dtype=int) # GASS identifiers in the new file
	sort = np.argsort(GASSid) # Find indices for sorting these GASS ids
	
	idf1 = np.searchsorted(GASSid,GASSid1) # ID filter 1
	idf2 = np.searchsorted(GASSid,GASSid2) # ID filter 2
	
	# If there are GASSids in one file with a higher value than ones that exist in the other, then searchsorted will list that as an index beyond the array's size.
	# Accounting for this issue
	#idf1 = idf1[idf1<len(GASSid)]
	#idf2 = idf2[idf2<len(GASSid)]
	
	MH2_1 = np.array(ds[6::9],dtype=float)[sort][idf1] # H2 masses for the first filter in log solar masses
	MH2_2 = np.array(ds[6::9],dtype=float)[sort][idf2] # and for the second
	
	
	
	# Load in file that contains neutral hydrogen masses of galaxies
	with open('Catinella2012bTab2Data.txt','r') as f:
		data = f.read()
		ds = data.split()
	GASSid = np.array(ds[0::18],dtype=int) # GASS identifiers in the new file
	sort = np.argsort(GASSid) # Find indices for sorting these GASS ids
	idf1 = np.searchsorted(GASSid,GASSid1) # ID filter 1
	idf2 = np.searchsorted(GASSid,GASSid2) # ID filter 2
	#idf1 = idf1[idf1<len(GASSid)]
	#idf2 = idf2[idf2<len(GASSid)]
	
	MHI = np.array(ds[15::18],dtype=float)[sort]
	MHI_1 = MHI[idf1] # HI masses for the first filter in log solar masses
	MHI_2 = MHI[idf2] # and the second
	
	# Calculate total hydrogen mass by summing the HI and H2 values (units of solar masses)
	THM1 = 10**MHI_1 + 10**MH2_1
	THM2 = 10**MHI_2 + 10**MH2_2
	
	return THM1, THM2, z1, z2



def GASSdata(dir='/Users/adam/Dropbox/Swinburne Shared/6-MonthProject/', h=0.6774):
	# Read in all the data on the GASS galaxies from the works of Catinella (2010,2012b) and Saintonge (2011a)

	### READ IN THE DATA I'M INTERESTED IN ###
	with open(dir+'Saintonge2011aTab1Data.txt','r') as f:
		data = f.read()
		S11_1 = data.split() # Saintonge 2011 data 1

	id_11_1 = np.array(S11_1[0::9],dtype=int)
	z_11_1 = np.array(S11_1[2::9],dtype=float)
	Mstar_11_1 = 10**np.array(S11_1[3::9],dtype=float) # Mass in solar masses
	mu_11_1 = 10**(np.array(S11_1[4::9],dtype=float) - 6) # Average stellar surface density within the half-light radius in the z-band (solar masses per square parsec)


	with open(dir+'Saintonge2011aTab2Data.txt','r') as f:
		data = f.read()
		S11_2 = data.split() # Saintonge 2011 data 2

	id_11_2 = np.array(S11_2[0::9],dtype=int)
	MH2_11_2 = 10**np.array(S11_2[6::9],dtype=float)


	with open(dir+'Catinella2010Tab1Data.txt','r') as f:
		data = f.read()
		C10_1 = data.split()

	id_10_1 = np.array(C10_1[0::13],dtype=int)
	z_10_1 = np.array(C10_1[2::13],dtype=float)
	Mstar_10_1 = 10**np.array(C10_1[3::13],dtype=float)
	mu_10_1 = 10**(np.array(C10_1[7::13],dtype=float) - 6)


	with open(dir+'Catinella2010Tab2Data.txt','r') as f:
		data = f.read()
		C10_2 = data.split()

	id_10_2 = np.array(C10_2[0::18],dtype=int)
	MHI_10_2 = 10**np.array(C10_2[15::18],dtype=float)


	with open(dir+'Catinella2012bTab1Data.txt','r') as f:
		data = f.read()
		C12_1 = data.split() # Catinella 2012 data

	id_12_1 = np.array(C12_1[0::16],dtype=int)
	z_12_1 = np.array(C12_1[3::16],dtype=float)
	Mstar_12_1 = 10**np.array(C12_1[4::16],dtype=float)
	mu_12_1 = 10**(np.array(C12_1[8::16],dtype=float) - 6)


	with open(dir+'Catinella2012bTab2Data.txt','r') as f:
		data = f.read()
		C12_2 = data.split() # Catinella 2012 data

	id_12_2 = np.array(C12_2[0::18],dtype=int)
	MHI_12_2 = 10**np.array(C12_2[15::18],dtype=float)
	### ================================= ###

	# Obtain an array with every unique GASS ID. Ordered to have lowest IDs first
	id = np.sort(np.array(list(set(np.concatenate((id_10_1,id_10_2,id_11_1,id_11_2,id_12_1,id_12_2))))))

	Ngal = len(id)

	# Initialise corresponding arrays for the other quantities
	z, Mstar, MH2, MHI, mu = np.zeros(Ngal), np.zeros(Ngal), np.zeros(Ngal), np.zeros(Ngal), np.zeros(Ngal)

	# Obtain the indices in the main arrays that will correspond to indices in each of the original datasets
	f101 = np.searchsorted(id,id_10_1)
	f102 = np.searchsorted(id,id_10_2)
	f111 = np.searchsorted(id,id_11_1)
	f112 = np.searchsorted(id,id_11_2)
	f121 = np.searchsorted(id,id_12_1)
	f122 = np.searchsorted(id,id_12_2)

	# Add values from original data to the new arrays.  Newer data will overwrite old data.
	z[f101], Mstar[f101], mu[f101] = z_10_1, Mstar_10_1, mu_10_1
	MHI[f102] = MHI_10_2
	z[f111], Mstar[f111], mu[f111] = z_11_1, Mstar_11_1, mu_11_1
	MH2[f112] = MH2_11_2
	z[f121], Mstar[f121], mu[f121] = z_12_1, Mstar_12_1, mu_12_1
	MHI[f122] = MHI[f122]

	return id, z, Mstar*0.7/h, MH2*0.7/h, MHI*0.7/h, mu


def fr13data(h=0.678):
    # Data from Fall & Romanowsky (2013)
	FR13x = np.array([  9.233,   8.949,   8.906,   9.743,   9.573,   9.899,  10.   ,
                      10.038,  10.197,  10.266,  10.34 ,  10.326,  10.25 ,  10.387,
                      10.484,  10.59 ,  10.569,  10.577,  10.565,  10.453,  10.32 ,
                      10.333,  10.425,  10.405,  10.398,  10.349,  10.518,  10.606,
                      10.694,  10.772,  10.75 ,  10.764,  10.853,  10.798,  10.845,
                      10.87 ,  10.948,  10.986,  10.997,  11.004,  11.034,  11.044,
                      11.037,  11.151,  11.121,  11.11 ,  10.984,  10.939,  10.964,
                      10.934,  10.884,  10.807,  10.771,  10.15 ,  10.105,   9.842,
                      9.467]) + 2*np.log10(0.72/h)
	FR13y = np.array([ 2.15 ,  2.422,  2.782,  2.283,  2.699,  2.847,  2.798,  2.768,
                  2.735,  2.639,  2.737,  2.787,  2.912,  2.907,  2.999,  2.933,
                  3.101,  3.188,  3.235,  3.209,  3.089,  3.174,  3.251,  3.27 ,
                  3.343,  3.369,  3.339,  3.452,  3.392,  3.385,  3.259,  3.259,
                  3.378,  3.45 ,  3.492,  3.462,  3.555,  3.525,  3.476,  3.56 ,
                  3.553,  3.584,  3.702,  3.73 ,  3.478,  3.405,  3.325,  3.334,
                  3.304,  3.272,  3.224,  3.104,  3.099,  3.016,  3.016,  2.611,
                  2.897]) + np.log10(0.72/h)
        #FR13y =  np.log10(1.01) + 1.3*(FR13y - 3) + 3
	return FR13x, FR13y

def md14data(h=0.678, Omega_R=0, Omega_M=0.3089, Omega_L=0.6911):
    # Data from Madau & Dickinson (2014)
    data = np.array([[0.01, 0.1, -1.82, 0.09, 0.02],
                     [0.2, 0.4, -1.50, 0.05, 0.05],
                     [0.4, 0.6, -1.39, 0.15, 0.08],
                     [0.6, 0.8, -1.20, 0.31, 0.13],
                     [0.8, 1.2, -1.25, 0.31, 0.13],
                     [0.05, 0.05, -1.77, 0.08, 0.09],
                     [0.05, 0.2, -1.75, 0.18, 0.18],
                     [0.2, 0.4, -1.55, 0.12, 0.12],
                     [0.4, 0.6, -1.44, 0.1, 0.1],
                     [0.6, 0.8, -1.24, 0.1, 0.1],
                     [0.8, 1.0, -0.99, 0.09, 0.08],
                     [1, 1.2, -0.94, 0.09, 0.09],
                     [1.2, 1.7, -0.95, 0.15, 0.08],
                     [1.7, 2.5, -0.75, 0.49, 0.09],
                     [2.5, 3.5, -1.04, 0.26, 0.15],
                     [3.5, 4.5, -1.69, 0.22, 0.32],
                     [0.92, 1.33, -1.02, 0.08, 0.08],
                     [1.62, 1.88, -0.75, 0.12, 0.12],
                     [2.08, 2.37, -0.87, 0.09, 0.09],
                     [1.9, 2.7, -0.75, 0.09, 0.11],
                     [2.7, 3.4, -0.97, 0.11, 0.15],
                     [3.8, 3.8, -1.29, 0.05, 0.05],
                     [4.9, 4.9, -1.42, 0.06, 0.06],
                     [5.9, 5.9, -1.65, 0.08, 0.08],
                     [7, 7, -1.79, 0.1, 0.1],
                     [7.9, 7.9, -2.09, 0.11, 0.11],
                     [7, 7, -2, 0.1, 0.11],
                     [0.03, 0.03, -1.72, 0.02, 0.03],
                     [0.03, 0.03, -1.95, 0.2, 0.2],
                     [0.4, 0.7, -1.34, 0.22, 0.11],
                     [0.7, 1, -0.96, 0.15, 0.19],
                     [1, 1.3, -0.89, 0.27, 0.21],
                     [1.3, 1.8, -0.91, 0.17, 0.21],
                     [1.8, 2.3, -0.89, 0.21, 0.25],
                     [0.4, 0.7, -1.22, 0.08, 0.11],
                     [0.7, 1, -1.1, 0.1, 0.13],
                     [1, 1.3, -0.96, 0.13, 0.2],
                     [1.3, 1.8, -0.94, 0.13, 0.18],
                     [1.8, 2.3, -0.8, 0.18, 0.15],
                     [0, 0.3, -1.64, 0.09, 0.11],
                     [0.3, 0.45, -1.42, 0.03, 0.04],
                     [0.45, 0.6, -1.32, 0.05, 0.05],
                     [0.6, 0.8, -1.14, 0.06, 0.06],
                     [0.8, 1, -0.94, 0.05, 0.06],
                     [1, 1.2, -0.81, 0.04, 0.05],
                     [1.2, 1.7, -0.84, 0.04, 0.04],
                     [1.7, 2, -0.86, 0.02, 0.03],
                     [2, 2.5, -0.91, 0.09, 0.12],
                     [2.5, 3, -0.86, 0.15, 0.23],
                     [3, 4.2, -1.36, 0.23, 0.5]])
                     
    my_cosmo = [100*h, Omega_R, Omega_M, Omega_L]
    MD_cosmo = [70.0, 0., 0.3, 0.7]

    z = (data[:,0]+data[:,1])/2
    z_err = (data[:,1]-data[:,0])/2
    SFRD = data[:,2] - 0.2 # converts from Salpeter to Chabrier
    for i in xrange(len(z)):
        SFRD[i] += np.log10( gc.z2dA(z[i], *my_cosmo) / gc.z2dA(z[i], *MD_cosmo) )**2 # adjust for assumed-cosmology influence on SFR calculations
    w = np.where(z_err>0)[0]
    for i in w:
        SFRD[i] += np.log10( (gc.comoving_distance(data[i,1], *MD_cosmo)**3 - gc.comoving_distance(data[i,0], *MD_cosmo)**3) / (gc.comoving_distance(data[i,1], *my_cosmo)**3 - gc.comoving_distance(data[i,0], *my_cosmo)**3) )# adjust for assumed-cosmology influence on comoving volume calculation
    SFRD_err_high = data[:,3]
    SFRD_err_low = data[:,4]
    return z, z_err, SFRD, SFRD_err_high, SFRD_err_low


def Driver18_data(h=0.678, Omega_R=0, Omega_M=0.3089, Omega_L=0.6911):
    data = np.array([[0.85, 0.02, 0.08, -1.95, 0.03, 0.00, 0.07, 0.00],
                  [1.52, 0.06, 0.14, -1.82, 0.03, 0.01, 0.05, 0.01],
                  [2.16, 0.14, 0.20, -1.90, 0.02, 0.00, 0.04, 0.00],
                  [2.90, 0.20, 0.28, -1.77, 0.01, 0.00, 0.05, 0.00],
                  [3.65, 0.28, 0.36, -1.75, 0.01, 0.00, 0.06, 0.01],
                  [4.35, 0.36, 0.45, -1.79, 0.01, 0.01, 0.06, 0.01],
                  [5.11, 0.45, 0.56, -1.73, 0.04, 0.01, 0.09, 0.03],
                  [5.86, 0.56, 0.68, -1.56, 0.05, 0.00, 0.07, 0.02],
                  [6.59, 0.68, 0.82, -1.42, 0.06, 0.01, 0.06, 0.04],
                  [7.36, 0.82, 1.00, -1.29, 0.05, 0.00, 0.07, 0.01],
                  [8.11, 1.00, 1.20, -1.31, 0.04, 0.00, 0.05, 0.01],
                  [8.82, 1.20, 1.45, -1.27, 0.03, 0.00, 0.06, 0.02],
                  [9.50, 1.45, 1.75, -1.17, 0.02, 0.00, 0.06, 0.03],
                  [10.21, 1.75, 2.20, -1.30, 0.04, 0.01, 0.07, 0.06],
                  [10.78, 2.20, 2.60, -1.29, 0.04, 0.01, 0.04, 0.09],
                  [11.29, 2.60, 3.25, -1.28, 0.04, 0.01, 0.04, 0.11],
                  [11.69, 3.25, 3.75, -1.33, 0.03, 0.01, 0.03, 0.08],
                  [11.95, 3.75, 4.25, -1.42, 0.04, 0.04, 0.05, 0.02],
                  [12.19, 4.25, 5.00, -1.45, 0.03, 0.04, 0.04, 0.04]])
                  
    my_cosmo = [100*h, Omega_R, Omega_M, Omega_L]
    D18_cosmo = [70.0, 0., 0.3, 0.7]
    
    z = 0.5*(data[:,1] + data[:,2])
    dz = data[:,2] - data[:,1]
    SFRD = data[:,3]
    for i in xrange(len(z)):
        SFRD[i] += np.log10( gc.z2dA(z[i], *my_cosmo) / gc.z2dA(z[i], *D18_cosmo) )**2 # adjust for assumed-cosmology influence on SFR calculations
        SFRD[i] += np.log10( (gc.comoving_distance(data[i,2], *D18_cosmo)**3 - gc.comoving_distance(data[i,1], *D18_cosmo)**3) / (gc.comoving_distance(data[i,2], *my_cosmo)**3 - gc.comoving_distance(data[i,1], *my_cosmo)**3) )# adjust for assumed-cosmology influence on comoving volume
    SFRD_err = np.sum(data[:,5:],axis=1)

    return [z, dz, SFRD, SFRD_err]

def cooltables(dir=None):
    if dir==None: dir = '/Users/astevens/Dropbox/Swinburne Not Shared/SAGE/extra/CoolFunctions/'
    logLambda = np.zeros((91,8))
    fnames = ['mzero.cie', 'm-30.cie', 'm-20.cie', 'm-15.cie', 'm-10.cie', 'm-05.cie', 'm-00.cie', 'm+05extra.cie']
    met = np.array([-5,-3,-2,-1.5,-1,-0.5,0,0.5])+np.log10(0.02)
    logTemp = np.arange(4,8.51,0.05)
    for i, f in enumerate(fnames): logLambda[:,i] = np.loadtxt(dir+f, skiprows=4)[:,5]
    return logLambda, logTemp, met




def HaardtMadau12():
    return np.array([[  0.00000000e+00,   5.00000000e-02,   1.00000000e-01,
            1.60000000e-01,   2.10000000e-01,   2.70000000e-01,
            3.30000000e-01,   4.00000000e-01,   4.70000000e-01,
            5.40000000e-01,   6.20000000e-01,   6.90000000e-01,
            7.80000000e-01,   8.70000000e-01,   9.60000000e-01,
            1.05000000e+00,   1.15000000e+00,   1.26000000e+00,
            1.37000000e+00,   1.49000000e+00,   1.61000000e+00,
            1.74000000e+00,   1.87000000e+00,   2.01000000e+00,
            2.16000000e+00,   2.32000000e+00,   2.48000000e+00,
            2.65000000e+00,   2.83000000e+00,   3.02000000e+00,
            3.21000000e+00,   3.42000000e+00,   3.64000000e+00,
            3.87000000e+00,   4.11000000e+00,   4.36000000e+00,
            4.62000000e+00,   4.89000000e+00,   5.18000000e+00,
            5.49000000e+00],
           [  8.89000000e-14,   1.11000000e-13,   1.39000000e-13,
            1.73000000e-13,   2.15000000e-13,   2.66000000e-13,
            3.29000000e-13,   4.05000000e-13,   4.96000000e-13,
            6.05000000e-13,   7.34000000e-13,   8.85000000e-13,
            1.06000000e-12,   1.26000000e-12,   1.49000000e-12,
            1.75000000e-12,   2.03000000e-12,   2.32000000e-12,
            2.62000000e-12,   2.90000000e-12,   3.17000000e-12,
            3.41000000e-12,   3.60000000e-12,   3.74000000e-12,
            3.81000000e-12,   3.82000000e-12,   3.75000000e-12,
            3.63000000e-12,   3.46000000e-12,   3.25000000e-12,
            3.02000000e-12,   2.79000000e-12,   2.57000000e-12,
            2.36000000e-12,   2.18000000e-12,   2.02000000e-12,
            1.89000000e-12,   1.78000000e-12,   1.67000000e-12,
            1.48000000e-12]])


def FaucherGiguere09():
    qdot_HI = np.array([0.158, 0.311, 0.569, 0.929, 1.371, 1.841, 2.260, 2.574, 2.768, 2.852, 2.839, 2.762, 2.642, 2.511, 2.384, 2.272, 2.171, 2.083, 2.002, 1.921, 1.833, 1.745, 1.661, 1.573, 1.487, 1.399, 1.305, 1.216, 1.127])*1e-12
    return np.array([np.arange(0,7.25,0.25),
                     qdot_HI])


def csv_dict(fname, skiprows, dtypes, keylist=None, delimiter=',', head_delimiter=','):
    with open(fname, 'r') as f:
        for i in xrange(skiprows): line = f.readline()
    keys = line.split(head_delimiter)
    keys[-1] = keys[-1][:-1] # gets rid of \n at the end
    if keylist==None: keylist = keys
    while keys[0][0]=='#' or keys[0][0]==' ': keys[0] = keys[0][1:] # get rid of comment mark for header if present
    print 'Number of properties =', len(keys)
    dict = {}
    for i, key in enumerate(keys):
        if key in keylist:
            j = np.where(np.array(keylist)==key)[0][0]
            print 'Reading', j, i, key
            dict[key] = np.loadtxt(fname, skiprows=skiprows, usecols=(i,), dtype=dtypes[j], delimiter=delimiter)
    return dict


def csv_dict_multifile(fpre, fsuf, fnumbers, skiprows, dtypes, keylist=None, delimiter=','):
    with open(fpre+str(fnumbers[0])+fsuf, 'r') as f:
        for i in xrange(skiprows[0]): line = f.readline()
    keys = line.split(',')
    keys[-1] = keys[-1][:-1] # gets rid of \n at the end
    if keylist==None: keylist = keys
    if type(skiprows)==int: skiprows = np.array([skiprows]*len(fnumbers))
    print 'Number of properties =', len(keys)
    dict = {}
    fnum = np.array([],dtype=np.int32)
    for i, key in enumerate(keys):
        if key in keylist:
            j = np.where(np.array(keylist)==key)[0][0]
            print 'Reading', j, i, key
            prop = np.array([], dtype=dtypes[j])
            for fi, fno in enumerate(fnumbers):
                prop = np.append(prop, np.loadtxt(fpre+str(fno)+fsuf, skiprows=skiprows[fi], usecols=(i,), dtype=dtypes[j], delimiter=delimiter))
                if key==keylist[0]: fnum = np.append(fnum, fno*np.ones(len(prop)-len(fnum),dtype=np.int32))
            dict[key] = prop
    dict['FileNumber'] = fnum
    print 'finished reading csv file'
    return dict


def zlist_eagle():
    return ['20.000', '15.132', '9.993', '8.988', '8.075', '7.050', '5.971', '5.487', '5.037', '4.485', '3.984', '3.528', '3.017', '2.478', '2.237', '2.012', '1.737', '1.487', '1.259', '1.004', '0.865', '0.736', '0.615', '0.503', '0.366', '0.271', '0.183', '0.101', '0.000']


def Brown_HI_fractions_satcen(h):
    ### Observational data for HI frac with m* for sat/cen
    logM = np.array([[9.2209911, 9.6852989, 10.180009, 10.665453, 11.098589],
                     [9.2085762, 9.6402225, 10.141238, 10.599669, 11.026575],
                     [9.2121296, 9.6528578, 10.139588, 10.615245, 11.054683]]) + 2*np.log10(0.7/h) + np.log10(0.61/0.66)

    logHIfrac = np.array([[ 0.37694988,  0.0076254,  -0.45345795, -0.90604609, -1.39503932],
                          [ 0.15731917, -0.16941574, -0.6199488,  -0.99943721, -1.30476058],
                          [ 0.19498822, -0.27559358, -0.74410361, -1.12869251, -1.49363434]]) - np.log10(0.61/0.66)

    Ngal = np.array([[120, 318, 675, 1132, 727],
                     [3500, 4359, 3843, 2158, 268],
                     [2203, 3325, 2899, 1784, 356]])

    logHIfrac_err = np.array([[0.044044334229208275, 0.036943084240966269, 0.015790838627011444, 0.017046581090793569, 0.033233075844263882],
                              [0.0097479231133465009, 0.0080264859983698953, 0.0026645765732298959, 0.013221412853921718, 0.038249326158796226],
                              [0.019933333457406766, 0.008115267925045043, 0.016097124168277393, 0.017141607902103756, 0.042458119603051923]])

    logM_cen = np.log10((10**logM[0,:] * Ngal[0,:] + 10**logM[1,:] * Ngal[1,:]) / (Ngal[0,:]+Ngal[1,:]))
    logHIfrac_cen = np.log10((10**logHIfrac[0,:] * Ngal[0,:] + 10**logHIfrac[1,:] * Ngal[1,:]) / (Ngal[0,:]+Ngal[1,:]))
    logHIfrac_err_cen = np.sqrt((logHIfrac_err[0,:]*Ngal[0,:])**2 + (logHIfrac_err[1,:]*Ngal[1,:])**2) / (Ngal[0,:]+Ngal[1,:])

    logM_sat = logM[2,:]
    logHIfrac_sat = logHIfrac[2,:]
    logHIfrac_err_sat = logHIfrac_err[2,:]
    ### ===================


    ### Observational data for sSFR plot
    logsSFR = 9+np.array([[-11.984118, -10.985614, -10.039345, -9.3237801],
                          [-11.822327, -10.961091, -9.9707727, -9.3765173],
                          [-11.829823, -11.01907, -10.010489, -9.3605328]]) + np.log10(0.63/0.67) - np.log10(0.61/0.66)

    logHIfrac_sSFR = np.array([[-1.19850969, -0.68417609, -0.1457182,   0.3424041 ],
                               [-1.22263718, -0.6337834,  -0.08348971,  0.2980288 ],
                               [-1.24741423, -0.69113463,  0.01750424,  0.50645626]]) - np.log10(0.61/0.66)

    logHIfrac_sSFR_err = np.array([[0.0101160053080814, 0.038387703324331612, 0.015336414293913735, 0.035059814816840985],
                                   [0.070986564156433327, 0.0089297650771804002, 0.0025412392934365089, 0.011826374505400481],
                                   [0.038234615293696067, 0.029941097250883182, 0.014058875616031209, 0.031036043642364026]])

    Ngal_sSFR = np.array([[1299, 645, 966, 62],
                          [1834, 2930, 8737, 627],
                          [2550, 3616, 4100, 300]])

    logHIfrac_sSFR_cen = np.log10((10**logHIfrac_sSFR[0,:] * Ngal_sSFR[0,:] + 10**logHIfrac_sSFR[1,:] * Ngal_sSFR[1,:]) / (Ngal_sSFR[0,:]+Ngal_sSFR[1,:]))
    logsSFR_cen = np.log10((10**logsSFR[0,:] * Ngal_sSFR[0,:] + 10**logsSFR[1,:] * Ngal_sSFR[1,:]) / (Ngal_sSFR[0,:]+Ngal_sSFR[1,:]))
    logHIfrac_sSFR_err_cen = np.sqrt((logHIfrac_sSFR_err[0,:]*Ngal_sSFR[0,:])**2 + (logHIfrac_sSFR_err[1,:]*Ngal_sSFR[1,:])**2) / (Ngal_sSFR[0,:]+Ngal_sSFR[1,:])

    logHIfrac_sSFR_sat = logHIfrac_sSFR[2,:]
    logsSFR_sat = logsSFR[2,:]
    logHIfrac_sSFR_err_sat = logHIfrac_sSFR_err[2,:]
    ### ===================

    # Added calculation for all galaxies
    logM_all = np.log10((10**logM[0,:] * Ngal[0,:] + 10**logM[1,:] * Ngal[1,:] + 10**logM[2,:] * Ngal[2,:]) / (Ngal[0,:]+Ngal[1,:]+Ngal[2,:]))
    logHIfrac_all = np.log10((10**logHIfrac[0,:] * Ngal[0,:] + 10**logHIfrac[1,:] * Ngal[1,:] + 10**logHIfrac[2,:] * Ngal[2,:]) / (Ngal[0,:]+Ngal[1,:]+Ngal[2,:]))
    logHIfrac_sSFR_all = np.log10((10**logHIfrac_sSFR[0,:] * Ngal_sSFR[0,:] + 10**logHIfrac_sSFR[1,:] * Ngal_sSFR[1,:] + 10**logHIfrac_sSFR[2,:] * Ngal_sSFR[2,:]) / (Ngal_sSFR[0,:]+Ngal_sSFR[1,:]+Ngal_sSFR[2,:]))
    logsSFR_all = np.log10((10**logsSFR[0,:] * Ngal_sSFR[0,:] + 10**logsSFR[1,:] * Ngal_sSFR[1,:] + 10**logsSFR[2,:] * Ngal_sSFR[2,:]) / (Ngal_sSFR[0,:]+Ngal_sSFR[1,:]+Ngal_sSFR[2,:]))

    logHIfrac_all_err = np.array([np.log10(1.511+0.011)-np.log10(1.511),
                                  np.log10(0.643+0.011)-np.log10(0.643),
                                  np.log10(0.232+0.005)-np.log10(0.232),
                                  np.log10(0.096+0.002)-np.log10(0.096),
                                  np.log10(0.039+0.001)-np.log10(0.039)]) # taken straight from Table 1 of Brown+15

    return [logM_cen, logHIfrac_cen, logHIfrac_err_cen], [logM_sat, logHIfrac_sat, logHIfrac_err_sat], [logHIfrac_sSFR_cen, logsSFR_cen, logHIfrac_sSFR_err_cen], [logHIfrac_sSFR_sat, logsSFR_sat, logHIfrac_sSFR_err_sat], [logM_all, logHIfrac_all, logHIfrac_all_err], [logsSFR_all, logHIfrac_sSFR_all]


def Brown_HIfrac_satenv_4bins(h=0.6774, indir='/Users/adam/ALFALFA/'):
    data = np.loadtxt(indir+'HIfrac_env_4bin.txt', delimiter=', ')
    data[:4,:] += (2*np.log10(0.7/h) + np.log10(0.61/0.66)) # adjust for cosmology and IMF
    data[4:8,:] /= (0.61/0.66)
    data[12:16,:] += (np.log10(0.63/0.67) - np.log10(0.61/0.66))
    data[16:20,:] /= (0.61/0.66)
    
#    data[8:12,:] = np.log10((data[4:8,:]+data[8:12,:])/data[4:8,:])
#    data[20:24,:] = np.log10((data[16:20,:]+data[20:24,:])/data[16:20,:])
    data[4:8,:] = np.log10(data[4:8,:])
    data[16:20,:] = np.log10(data[16:20,:])

    B17_12_m = data[[0,4,8],:3] # number after _ is the highest halo mass of the bin, m means mass
    B17_12_s = data[[12,16,20],:3] # s means ssfr
    B17_13_m = data[[1,5,9],:4]
    B17_13_s = data[[13,17,21],:4]
    B17_14_m = data[[2,6,10],:5]
    B17_14_s = data[[14,18,22],:4]
    B17_15_m = data[[3,7,11],:5]
    B17_15_s = data[[15,19,23],:4]

    return [B17_12_m, B17_13_m, B17_14_m, B17_15_m], [B17_12_s, B17_13_s, B17_14_s, B17_15_s]


def Brown_deltaHIfrac_satenv_4bins(h=0.6774, indir='/Users/adam/ALFALFA/'):
    B17_m, B17_s = Brown_HIfrac_satenv_4bins(h, indir)
    [logM_cen, logHIfrac_cen, logHIfrac_err_cen], [logM_sat, logHIfrac_sat, logHIfrac_err_sat], [logHIfrac_sSFR_cen, logsSFR_cen, logHIfrac_sSFR_err_cen], [logHIfrac_sSFR_sat, logsSFR_sat, logHIfrac_sSFR_err_sat], [logM_all, logHIfrac_all, logHIfrac_all_err], [logsSFR_all, logHIfrac_sSFR_all] = Brown_HI_fractions_satcen(h)

    for i in xrange(len(B17_m)):
        B17_m[i][1,:] -= np.interp(B17_m[i][0,:], logM_sat, logHIfrac_sat)
        B17_s[i][1,:] -= np.interp(B17_s[i][0,:], logsSFR_sat-9, logHIfrac_sSFR_sat)

    return B17_m, B17_s

def carnage(fname, dir='', addH=False, fields_of_interest=None):
    # For full details on fields, see http://cosmiccarnage2015.pbworks.com/w/page/95901731/File%20Formats
    all_fields = [['haloid', np.int64],
                  ['galaxyhostid', np.int64],
                  ['galaxy_is_orphan', bool],
                  ['X', np.float32],
                  ['Y', np.float32],
                  ['Z', np.float32],
                  ['Vx', np.float32],
                  ['Vy', np.float32],
                  ['Vz', np.float32],
                  ['Mcold', np.float32],
                  ['Mhot', np.float32],
                  ['Mstar', np.float32],
                  ['Mbh', np.float32],
                  ['Z_gas', np.float32],
                  ['Z_stars', np.float32],
                  ['T_stars', np.float32],
                  ['SFR', np.float32],
                  ['SFRbulge', np.float32],
                  ['M_hot,halo', np.float32],
                  ['M_cold,halo', np.float32],
                  ['Meject', np.float32],
                  ['M_outflowed', np.float32],
                  ['M_gas,disk', np.float32],
                  ['M_gas,spheroid', np.float32],
                  ['M_stars,disk', np.float32],
                  ['M_stars,spheroid', np.float32],
                  ['M_bh', np.float32],
                  ['M_ICstars', np.float32],
                  ['M_total', np.float32],
                  ['MZ_hot,halo', np.float32],
                  ['MZ_cold,halo', np.float32],
                  ['MZeject', np.float32],
                  ['MZ_outflowed', np.float32],
                  ['MZ_gas,disk', np.float32],
                  ['MZ_gas,spheroid', np.float32],
                  ['MZ_stars,disk', np.float32],
                  ['MZ_stars,spheroid', np.float32],
                  ['MZ_bh', np.float32],
                  ['MZ_ICstars', np.float32],
                  ['BoverT', np.float32],
                  ['r1/2', np.float32],
                  ['r1/2_bulge', np.float32],
                  ['r1/2_disk', np.float32],
                  ['nuv_ext', np.float32],
                  ['B_ext', np.float32],
                  ['V_ext', np.float32],
                  ['g_ext', np.float32],
                  ['r_ext', np.float32],
                  ['K_ext', np.float32],
                  ['nuv', np.float32],
                  ['B', np.float32],
                  ['V', np.float32],
                  ['g', np.float32],
                  ['r', np.float32],
                  ['K', np.float32]]

    if fields_of_interest==None: # none means all in this case
        fields = all_fields
        cols = range(len(fields))
    else:
        fields = []
        cols = []
        for i, field in enumerate(all_fields):
            if field[0] in fields_of_interest:
                fields += [field]
                cols += [i]
    data = np.loadtxt(dir+fname, skiprows=1)
    dict = {}
    for i, field in enumerate(fields):
        print 'Reading', field[0]
        dict[field[0]] = np.array(data[:,cols[i]], dtype=field[1])

    if addH:
        dict['M_HI'] = np.loadtxt(dir+'Mass_HI.txt')
        dict['M_H2'] = np.loadtxt(dir+'Mass_H2.txt')
        dict['Mhalo'] = np.loadtxt(dir+'CentralMvir.txt')
        dict['Mvir'] = np.loadtxt(dir+'Mvir.txt')
    return dict


def read_binary(fname, dtype, return_GroupRange=False):
    # Read binary file of particle data I built from EAGLE
    f = open(fname, 'rb')
    ID_range = np.fromfile(f, 'i4', 2)
    Npart = np.fromfile(f, 'i4', 1)[0]
    property = np.fromfile(f, dtype, Npart)
    f.close()
    if return_GroupRange:
        return property, ID_range
    else:
        return property


def U_MW_FG09_Dec11():
    # Kindly supplied by Benedikt Diemer
    # Columns 1-3 are computed from the flux at 1000A, the integrated flux in the Lyman-Werner band, and the integrated flux in the photoheating band 8-13.6 eV, respectively.
    return np.array([[0.00, 1.37e-03, 1.38e-03, 1.75e-03],
                    [0.20, 2.35e-03, 2.38e-03, 3.05e-03],
                    [0.40, 3.77e-03, 3.81e-03, 4.89e-03],
                    [0.60, 5.73e-03, 5.81e-03, 7.46e-03],
                    [0.80, 8.31e-03, 8.41e-03, 1.08e-02],
                    [1.00, 1.13e-02, 1.15e-02, 1.47e-02],
                    [1.20, 1.50e-02, 1.52e-02, 1.95e-02],
                    [1.40, 1.92e-02, 1.95e-02, 2.49e-02],
                    [1.60, 2.39e-02, 2.42e-02, 3.11e-02],
                    [1.80, 2.91e-02, 2.95e-02, 3.80e-02],
                    [2.00, 3.47e-02, 3.51e-02, 4.54e-02],
                    [2.20, 4.12e-02, 4.17e-02, 5.40e-02],
                    [2.40, 4.82e-02, 4.88e-02, 6.32e-02],
                    [2.60, 5.58e-02, 5.63e-02, 7.26e-02],
                    [2.80, 6.27e-02, 6.34e-02, 8.16e-02],
                    [3.00, 7.12e-02, 7.19e-02, 9.20e-02],
                    [3.20, 7.98e-02, 8.04e-02, 1.03e-01],
                    [3.40, 8.82e-02, 8.89e-02, 1.13e-01],
                    [3.60, 9.91e-02, 9.96e-02, 1.24e-01],
                    [3.80, 1.08e-01, 1.08e-01, 1.33e-01],
                    [4.00, 1.17e-01, 1.17e-01, 1.42e-01],
                    [4.20, 1.26e-01, 1.26e-01, 1.51e-01],
                    [4.40, 1.35e-01, 1.35e-01, 1.60e-01],
                    [4.60, 1.43e-01, 1.43e-01, 1.67e-01],
                    [4.80, 1.53e-01, 1.51e-01, 1.74e-01],
                    [5.00, 1.60e-01, 1.58e-01, 1.79e-01],
                    [5.20, 1.65e-01, 1.63e-01, 1.83e-01],
                    [5.40, 1.71e-01, 1.69e-01, 1.86e-01],
                    [5.60, 1.77e-01, 1.75e-01, 1.89e-01],
                    [5.80, 1.83e-01, 1.79e-01, 1.90e-01],
                    [6.00, 1.89e-01, 1.84e-01, 1.91e-01]])


def GASS_COLDGASS_data(dir='/Users/adam/Dropbox/Swinburne Shared/6-MonthProject/', h=0.6774):
    # Read in all relevant data from files
    S11_id = np.loadtxt(dir+'Saintonge2011aTab1Data.txt', usecols=0, dtype=np.int32)
    S11_Mstar = 10**np.loadtxt(dir+'Saintonge2011aTab1Data.txt', usecols=3, dtype=np.float32)
    S11_MH2 = 10**np.loadtxt(dir+'Saintonge2011aTab2Data.txt', usecols=6, dtype=np.float32)
    S11_detflag = 10**np.loadtxt(dir+'Saintonge2011aTab2Data.txt', usecols=8, dtype=np.int32)
    #
    C10_id = np.loadtxt(dir+'Catinella2010Tab1Data.txt', usecols=0, dtype=np.int32)
    C10_Mstar = 10**np.loadtxt(dir+'Catinella2010Tab1Data.txt', usecols=3, dtype=np.float32)
    C10_det_id = np.loadtxt(dir+'Catinella2010Tab2Data.txt', usecols=0, dtype=np.int32)
    C10_det_MHI = 10**np.loadtxt(dir+'Catinella2010Tab2Data.txt', usecols=15, dtype=np.float32)
    C10_nondet_id = np.loadtxt(dir+'Catinella2010Tab3Data.txt', usecols=1, dtype=np.int32)
    C10_nondet_MHI = 10**np.loadtxt(dir+'Catinella2010Tab3Data.txt', usecols=5, dtype=np.float32)
    #
    C12_id = np.loadtxt(dir+'Catinella2012bTab1Data.txt', usecols=0, dtype=np.int32)
    C12_Mstar = 10**np.loadtxt(dir+'Catinella2012bTab1Data.txt', usecols=4, dtype=np.float32)
    C12_det_id = np.loadtxt(dir+'Catinella2012bTab2Data.txt', usecols=0, dtype=np.int32)
    C12_det_MHI = 10**np.loadtxt(dir+'Catinella2012bTab2Data.txt', usecols=15, dtype=np.float32)
    C12_nondet_id = np.loadtxt(dir+'Catinella2012bTab3Data.txt', usecols=1, dtype=np.int32)
    C12_nondet_MHI = 10**np.loadtxt(dir+'Catinella2012bTab3Data.txt', usecols=5, dtype=np.float32)

    # Check the data are as I expect
    assert len(C10_id)==len(C10_det_id)+len(C10_nondet_id)
    assert np.all(~np.in1d(C10_det_id,C10_nondet_id))
    assert len(C12_id)==len(C12_det_id)+len(C12_nondet_id)
    assert np.all(~np.in1d(C12_det_id,C12_nondet_id))

    # Sort and reorganise data
    asort = np.argsort(S11_id)
    S11_id = S11_id[asort]
    S11_Mstar = S11_Mstar[asort]
    S11_MH2 = S11_MH2[asort]
    S11_detflag = S11_detflag[asort]
    #
    asort = np.argsort(C10_id)
    C10_id = C10_id[asort]
    C10_Mstar = C10_Mstar[asort]
    #
    asort = np.argsort(C10_det_id)
    C10_det_id = C10_det_id[asort]
    C10_det_MHI = C10_det_MHI[asort]
    #
    asort = np.argsort(C10_nondet_id)
    C10_nondet_id = C10_nondet_id[asort]
    C10_nondet_MHI = C10_nondet_MHI[asort]
    #
    asort = np.argsort(C12_id)
    C12_id = C12_id[asort]
    C12_Mstar = C12_Mstar[asort]
    #
    asort = np.argsort(C12_det_id)
    C12_det_id = C12_det_id[asort]
    C12_det_MHI = C12_det_MHI[asort]
    #
    asort = np.argsort(C12_nondet_id)
    C12_nondet_id = C12_nondet_id[asort]
    C12_nondet_MHI = C12_nondet_MHI[asort]

    # Get information for galaxies that are common for both GASS and COLD GASS
    ids = np.intersect1d(S11_id, np.append(C10_id,C12_id))
    Ngal = len(ids)
    MHI = np.zeros(Ngal, dtype=np.float32)
    Mstar_C, Mstar_S = np.zeros(Ngal, dtype=np.float32), np.zeros(Ngal, dtype=np.float32)
    HI_det = np.zeros(Ngal, dtype=bool)
    #
    MHI[np.in1d(ids,C10_det_id)] = 1.0*C10_det_MHI[np.in1d(C10_det_id,ids)]
    HI_det[np.in1d(ids,C10_det_id)] = True
    MHI[np.in1d(ids,C10_nondet_id)] = 1.0*C10_nondet_MHI[np.in1d(C10_nondet_id,ids)]
    #
    MHI[np.in1d(ids,C12_det_id)] = 1.0*C12_det_MHI[np.in1d(C12_det_id,ids)]
    HI_det[np.in1d(ids,C12_det_id)] = True
    MHI[np.in1d(ids,C12_nondet_id)] = 1.0*C12_nondet_MHI[np.in1d(C12_nondet_id,ids)]
    #
    MH2 = 1.0*S11_MH2[np.in1d(S11_id,ids)]
    H2_det = np.array(2-S11_detflag[np.in1d(S11_id,ids)], dtype=bool)
    #
    Mstar_C[np.in1d(ids,C10_id)] = 1.0*C10_Mstar[np.in1d(C10_id,ids)]
    Mstar_C[np.in1d(ids,C12_id)] = 1.0*C12_Mstar[np.in1d(C12_id,ids)]
    Mstar_S = 1.0*S11_Mstar[np.in1d(S11_id,ids)]
#    print Mstar_C
#    print Mstar_S
    assert np.allclose(Mstar_C, Mstar_S)

    return id, Mstar_S, MHI, MH2, HI_det, H2_det

    
def Tremonti04(h):
    x_obs = np.array([8.52, 8.57, 8.67, 8.76, 8.86, 8.96, 9.06, 9.16, 9.26, 9.36, 9.46, 9.57, 9.66, 9.76, 9.86, 9.96, 10.06, 10.16, 10.26, 10.36, 10.46, 10.56, 10.66, 10.76, 10.86, 10.95, 11.05, 11.15, 11.25, 11.30])
    y_low = np.array([8.25, 8.25, 8.28, 8.32, 8.37, 8.46, 8.56, 8.59, 8.60, 8.63, 8.66, 8.69, 8.72, 8.76, 8.80, 8.83, 8.85, 8.88, 8.92, 8.94, 8.96, 8.98, 9.00, 9.01, 9.02, 9.03, 9.03, 9.04, 9.03, 9.03])
    y_high= np.array([8.64, 8.64, 8.65, 8.70, 8.73, 8.75, 8.82, 8.82, 8.86, 8.88, 8.92, 8.94, 8.96, 8.99, 9.01, 9.05, 9.06, 9.09, 9.10, 9.11, 9.12, 9.14, 9.15, 9.15, 9.16, 9.17, 9.17, 9.18, 9.18, 9.18])
    x_obs += np.log10(1.5/1.8) + 2*np.log10(0.7/h) # Accounts for difference in Kroupa & Chabrier IMFs and the difference in h
    return x_obs, y_low, y_high

def Baldry08(h=None):
    B = np.array([   [7.05, 1.3531e-01, 6.0741e-02],
                     [7.15, 1.3474e-01, 6.0109e-02],
                     [7.25, 2.0971e-01, 7.7965e-02],
                     [7.35, 1.7161e-01, 3.1841e-02],
                     [7.45, 2.1648e-01, 5.7832e-02],
                     [7.55, 2.1645e-01, 3.9988e-02],
                     [7.65, 2.0837e-01, 4.8713e-02],
                     [7.75, 2.0402e-01, 7.0061e-02],
                     [7.85, 1.5536e-01, 3.9182e-02],
                     [7.95, 1.5232e-01, 2.6824e-02],
                     [8.05, 1.5067e-01, 4.8824e-02],
                     [8.15, 1.3032e-01, 2.1892e-02],
                     [8.25, 1.2545e-01, 3.5526e-02],
                     [8.35, 9.8472e-02, 2.7181e-02],
                     [8.45, 8.7194e-02, 2.8345e-02],
                     [8.55, 7.0758e-02, 2.0808e-02],
                     [8.65, 5.8190e-02, 1.3359e-02],
                     [8.75, 5.6057e-02, 1.3512e-02],
                     [8.85, 5.1380e-02, 1.2815e-02],
                     [8.95, 4.4206e-02, 9.6866e-03],
                     [9.05, 4.1149e-02, 1.0169e-02],
                     [9.15, 3.4959e-02, 6.7898e-03],
                     [9.25, 3.3111e-02, 8.3704e-03],
                     [9.35, 3.0138e-02, 4.7741e-03],
                     [9.45, 2.6692e-02, 5.5029e-03],
                     [9.55, 2.4656e-02, 4.4359e-03],
                     [9.65, 2.2885e-02, 3.7915e-03],
                     [9.75, 2.1849e-02, 3.9812e-03],
                     [9.85, 2.0383e-02, 3.2930e-03],
                     [9.95, 1.9929e-02, 2.9370e-03],
                     [10.05, 1.8865e-02, 2.4624e-03],
                     [10.15, 1.8136e-02, 2.5208e-03],
                     [10.25, 1.7657e-02, 2.4217e-03],
                     [10.35, 1.6616e-02, 2.2784e-03],
                     [10.45, 1.6114e-02, 2.1783e-03],
                     [10.55, 1.4366e-02, 1.8819e-03],
                     [10.65, 1.2588e-02, 1.8249e-03],
                     [10.75, 1.1372e-02, 1.4436e-03],
                     [10.85, 9.1213e-03, 1.5816e-03],
                     [10.95, 6.1125e-03, 9.6735e-04],
                     [11.05, 4.3923e-03, 9.6254e-04],
                     [11.15, 2.5463e-03, 5.0038e-04],
                     [11.25, 1.4298e-03, 4.2816e-04],
                     [11.35, 6.4867e-04, 1.6439e-04],
                     [11.45, 2.8294e-04, 9.9799e-05],
                     [11.55, 1.0617e-04, 4.9085e-05],
                     [11.65, 3.2702e-05, 2.4546e-05],
                     [11.75, 1.2571e-05, 1.2571e-05],
                     [11.85, 8.4589e-06, 8.4589e-06],
                     [11.95, 7.4764e-06, 7.4764e-06]
                     ], dtype=np.float32)
    if h is not None:
        B[:,0] -= (np.log10(h**2)+0.26) # adjust to Charbier too
        B[:,1] *= h**3
        B[:,2] *= h**3

    return B

def Keres03(h):
    Keres_high = np.array([-1.051, -1.821, -1.028, -1.341, -1.343, -1.614, -1.854, -2.791,  -3.54 , -5.021]) + 3*np.log10(h)
    Keres_mid = np.array([-1.271, -1.999, -1.244, -1.477, -1.464, -1.713, -1.929, -2.878,   -3.721, -5.22 ]) + 3*np.log10(h)
    Keres_low = np.array([-1.706, -2.302, -1.71 , -1.676, -1.638, -1.82 , -2.033, -2.977,   -4.097, -5.584]) + 3*np.log10(h)
    Keres_M = np.array([  6.953,   7.353,   7.759,   8.154,   8.553,   8.96 ,   9.365,  9.753,  10.155,  10.558]) - 2*np.log10(h)
    return Keres_M, Keres_low, Keres_mid, Keres_high


def OG14(h, disconly=False):
    BTT = np.array([0.04, 0.05, 0.02, 0.1, 0.0, 0.02, 0.03, 0.14, 0.16, 0.22, 0.32, 0.17 ,0.09, 0.1, 0.16, 0.01])
    logM_star = np.array([10.1, 9.9, 9.7, 10.8, 9.0, 10.3, 10.1, 10.4, 10.7, 10.6, 10.3, 10.8, 10.6, 10.5, 10.9, 9.5]) + 2*np.log10(0.7/h) + np.log10(0.61/0.66) # adjusts from Kroupa to Chabrier IMF
    logM_HIH2 = np.array([9.8, 9.8, 9.5, 10.1, 8.4, 9.7, 10.1, 9.4, 10.1, 9.4, 9.0, 10.2, 9.8, 10.0, 10.2, 9.2]) + 2*np.log10(0.7/h)
    logj_star = np.array([2.98, 2.94, 2.62, 3.4, 2.03, 3.03, 2.97, 2.91, 3.06, 2.84, 2.34, 3.18, 3.18, 3.02, 3.35, 2.43]) + np.log10(0.7/h)
    logj_HIH2 = np.array([3.23, 3.09, 3.04, 3.91, 2.11, 3.25, 3.45, 3.1, 3.38, 2.82, 2.63, 3.59, 3.25, 3.17, 3.52, 2.61]) + np.log10(0.7/h)

    if disconly:
        logM_star += np.log10(1-BTT)
        logj_star -= np.log10(1-BTT)

    return BTT, logM_star, logM_HIH2, logj_star, logj_HIH2


def xGASS_xCOLDGASS(indir=None, h=0.6774, extra=False, extra2=False, Omega_L=0.6911, Omega_M=0.3089):
    if indir is None: indir = '/Users/adam/xGASS/'
    keys = ['GASS', 'lgMstar', 'SFR_best', 'lgMHI', 'HIsrc', 'env_code_B', 'logMh_Mst_B', 'lvir_ratB', 'zSDSS', 'lgmust', 'petrR50_r', 'Dlum']
    key_dtypes = [np.int32, np.float32, np.float32, np.float32, np.uint16, np.int16, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]
    G = csv_dict(indir+'xGASS_data.txt', 1, key_dtypes, keylist=keys, delimiter=None, head_delimiter=', ')

    import pyfits
    C = pyfits.open(indir+'xCOLDGASS_PubCat.fits')[1].data

    xGASS_ID = np.unique(np.append(C['ID'], G['GASS']))
    Ngal = len(xGASS_ID)
    
    arg1 = np.searchsorted(xGASS_ID, G['GASS'])
    arg2 = np.searchsorted(xGASS_ID, C['ID'])
    
    xGASS_redshift = np.zeros(Ngal, dtype=np.float32)
    xGASS_redshift[arg2] = C['Z_SDSS']
    xGASS_redshift[arg1] = G['zSDSS']
    
    xGASS_LumDist = np.zeros(Ngal, dtype=np.float32)
    xGASS_LumDist[arg2] = C['LUMDIST']
    xGASS_LumDist[arg1] = G['Dlum']
#    assert np.allclose(xGASS_LumDist, np.array([gc.z2dA(z, H_0=70.0, Omega_Lambda=0.7, Omega_M=0.3) for z in xGASS_redshift]) * (1+xGASS_redshift)**2 * 1e-6, atol=0.01, rtol=1e-3) # checking consistency of luminosity distance in the file with what I would calculate from their assumed cosmology
    Dlum = np.array([gc.z2dA(z, H_0=100*h, Omega_Lambda=Omega_L, Omega_M=Omega_M) for z in xGASS_redshift]) * (1+xGASS_redshift)**2 * 1e-6 # luminosity distance based on desired cosmology
    logDfac = np.log10(Dlum / xGASS_LumDist)
    
    xGASS_logMstar = np.zeros(Ngal, dtype=np.float32)
    xGASS_logMstar[arg2] = C['LOGMSTAR']
    xGASS_logMstar[arg1] = G['lgMstar']
    xGASS_logMstar[xGASS_logMstar!=0] += 2*logDfac[xGASS_logMstar!=0]

    xGASS_logSFR = np.zeros(Ngal, dtype=np.float32)
    xGASS_logSFR[arg2] = C['LOGSFR_BEST']
    xGASS_logSFR[arg1] = np.log10(G['SFR_best'])
    xGASS_logSFR[xGASS_logSFR!=0] += 2*logDfac[xGASS_logSFR!=0]

    xGASS_logMHI = np.zeros(Ngal, dtype=np.float32) # If 0, no measurement has been made at all
    xGASS_logMHI[arg1] = G['lgMHI']
    xGASS_logMHI[xGASS_logMHI!=0] += 2*logDfac[xGASS_logMHI!=0]

    xGASS_HIdet = np.zeros(Ngal, dtype=bool) # Non-detections have their upper limit provided as their mass entry
    xGASS_HIdet[arg1[G['HIsrc']<=3]] = True

    xGASS_Type = -np.ones(Ngal, dtype=np.int16) # -1 means it isn't in the group finder
    xGASS_Type[arg1[G['env_code_B']==0]] = 1
    xGASS_Type[arg1[G['env_code_B']>=1]] = 0

    xGASS_logMhalo = np.zeros(Ngal, dtype=np.float32)
    xGASS_logMhalo[arg1] = G['logMh_Mst_B']
    xGASS_logMhalo[xGASS_logMhalo!=0] += 2*logDfac[xGASS_logMhalo!=0]

    xGASS_logRonRvir = np.zeros(Ngal, dtype=np.float32)
    xGASS_logRonRvir[arg1] = G['lvir_ratB']

    xGASS_logMH2 = np.zeros(Ngal, dtype=np.float32) # If 0, no measurement has been made at all
    xGASS_logMH2[arg2] = np.log10(C['XCO_A17']*C['LCO'])
    xGASS_logMH2[xGASS_logMH2!=0] += (2*logDfac[xGASS_logMH2!=0] - np.log10(1.36)) # 1.36 is the "helium contribution" to the conversion factor (wut)

    xGASS_logMH2_corr = np.zeros(Ngal, dtype=np.float32) # _corr means the aperture correction has been applied
    xGASS_logMH2_corr[arg2[C['FLAG_CO']==1]] = C['LOGMH2'][C['FLAG_CO']==1]
    xGASS_logMH2_corr[arg2[C['FLAG_CO']==2]] = C['LIM_LOGMH2'][C['FLAG_CO']==2]
    xGASS_logMH2_corr[xGASS_logMH2_corr!=0] += (2*logDfac[xGASS_logMH2_corr!=0] - np.log10(1.36))

    xGASS_H2det = np.zeros(Ngal, dtype=bool)
    xGASS_H2det[arg2[C['FLAG_CO']==1]] = True
    
    if extra:
        xGASS_LCO_corr = np.zeros(Ngal, dtype=np.float32)
        xGASS_LCO_corr[arg2] = C['LCO_COR']
        xGASS_LCO_corr_err = np.zeros(Ngal, dtype=np.float32)
        xGASS_LCO_corr_err[arg2] = C['LCO_COR_ERR']
        xGASS_logMH2_corr_err = np.zeros(Ngal, dtype=np.float32)
        xGASS_logMH2_corr_err[arg2] = C['LOGMH2_ERR']
        
        
        return xGASS_ID, xGASS_logMstar, xGASS_logSFR, xGASS_logMHI, xGASS_HIdet, xGASS_Type, xGASS_logMhalo, xGASS_logRonRvir, xGASS_logMH2, xGASS_H2det, xGASS_logMH2_corr, xGASS_redshift, xGASS_LCO_corr, xGASS_LCO_corr_err, xGASS_logMH2_corr_err

    elif extra2:
        xGASS_logMuStar = np.zeros(Ngal, dtype=np.float32)
        xGASS_logMuStar[arg2] = C['LOGMUST']
        xGASS_logMuStar[arg1] = G['lgmust']

        xGASS_r50 = np.zeros(Ngal, dtype=np.float32)
        xGASS_r50[arg2] = C['R50KPC']
        xGASS_r50[arg1] = np.tan(G['petrR50_r']/60.**2 * np.pi/180.) * G['Dlum']*1e3
        xGASS_r50[xGASS_r50>0] += logDfac[xGASS_r50>0]

        return xGASS_ID, xGASS_logMstar, xGASS_logSFR, xGASS_logMHI, xGASS_HIdet, xGASS_Type, xGASS_logMhalo, xGASS_logRonRvir, xGASS_logMH2, xGASS_H2det, xGASS_logMH2_corr, xGASS_redshift, xGASS_logMuStar, xGASS_r50

    else:
        return xGASS_ID, xGASS_logMstar, xGASS_logSFR, xGASS_logMHI, xGASS_HIdet, xGASS_Type, xGASS_logMhalo, xGASS_logRonRvir, xGASS_logMH2, xGASS_H2det, xGASS_logMH2_corr, xGASS_redshift
    

def HI_surface_density_profiles_obs(dir='/Users/adam/HI profiles/'):
    
    # Read THINGS and LITTLE THINGS data
#    import pandas as PDF
    from rpy2.robjects import r
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    r.load(dir+'DataAdamStevens.img')
    T = pandas2ri.ri2py(r.galaxy)

    # Read Bluedisk and LVHIS data
    import pyfits
    B = pyfits.open(dir+'Bluedisk.fits')[1].data
    L = pyfits.open(dir+'LVHIS.fits')[1].data
    Lc = pyfits.open(dir+'LVHIS_cat.fits')[1].data

    nT = len(T)
    nB = len(B)
    nL = len(L)
    ntot = nT+nB+nL

    Sigma_cen = np.zeros(ntot)
    rHI = np.zeros(ntot)
    rHI50 = np.zeros(ntot)
    mHI = np.zeros(ntot)
    r_cen = np.zeros(ntot)
    survey = []
    r_profile = []
    SigmaHI_profile = []
    gal_name = []

    for i in xrange(ntot):
        if i<nT:
            rad = np.array(T[i][-2])
            SigHI = np.array(T[i][-1])
            mHI[i] = np.float32(T[i][5])[0]
        elif i<nT+nB:
            j = i-nT
            rad = B['RADIUS'][j,:] * np.pi / (180*60*60) * B['DIS'][j] * 1e6
            SigHI = B['MU'][j,:]
            mHI[i] = 10**B['MHI'][j]
        else:
            j = i-nT-nB
            if L['R1'][j]/L['BMAJ'][j] < 1.5: continue
            rad = L['RADI'][j,:] * np.pi / (180*60*60) * Lc['DIS'][j] * 1e6
            SigHI = L['DENS'][j,:]
            mHI[i] = 10**L['MHI_TOT'][j]

        f = np.isfinite(rad) * np.isfinite(SigHI)
        rad, SigHI = rad[f], SigHI[f]
        if len(rad)<10: continue
        annuli = np.append(np.append(0., rad[:-1]+0.5*np.diff(rad)), rad[-1]+0.5*np.diff(rad)[-1])
        area = np.pi * (annuli[1:]**2 - annuli[:-1]**2)
        r_cen[i] = rad[SigHI>=0.9*np.max(SigHI)][-1]
        try:
            w = np.where((SigHI>1.)*(rad>r_cen[i]))[0][-1]+1
        except IndexError:
            continue
#        rHI[i] = np.interp(0., np.log10(SigHI[w-1:w+1])[::-1], rad[w-1:w+1][::-1])
        rHI[i] = np.interp(1., SigHI[w-1:w+1][::-1], rad[w-1:w+1][::-1])
#        if i>=nT and i<nT+nB: print 'rHI calc v read:', rHI[i], B['R1'][j] * np.pi / (180*60*60) * B['DIS'][j] * 1e6
#        if i>=nT+nB: print 'rHI calc v read:', rHI[i], L['R1'][j] * np.pi / (180*60*60) * Lc['DIS'][j] * 1e6
        Sigma_cen[i] = min(np.sum((SigHI*area)[rad<=r_cen[i]]) / (np.pi * r_cen[i]**2), np.max(SigHI))
        r_profile += [rad/rHI[i]]
        SigmaHI_profile += [SigHI]
#        print 'mHI read/calc, read/calc(r<rHI):', mHI[i] / np.sum(SigHI*area), mHI[i] / np.sum((SigHI*area)[rad<=rHI[i]])
        mHI[i] = max( mHI[i], np.sum(SigHI*area) )
        rHI50[i] = np.interp(0.5, np.append(0,np.cumsum(SigHI*area)/mHI[i]), annuli)
        
        if i<nT:
            survey += [T[i][0][0]]
            gal_name += [T[i][1][0]]
        elif i<nT+nB:
            survey += ['BlueDisk']
            gal_name += [B['NAME'][j]]
        else:
            survey += ['LVHIS']
            gname = Lc['NAME'][j]
            if gname[0]==' ': 
                gname = gname[1:]
            elif gname[-1]==' ': 
                gname = gname[:gname.find(' ')]
            gal_name += [gname]

    rho = r_cen / rHI
    f = (mHI>0) * (rHI > 0)
    rho, Sigma_cen, mHI, rHI, rHI50 = rho[f], Sigma_cen[f], mHI[f], rHI[f], rHI50[f]
    survey = np.array(survey)
    return r_profile, SigmaHI_profile, rho, Sigma_cen, rHI, mHI, survey, rHI50, gal_name

        
def DS_HI_profilefits(file_suffices, h, dir='./', alt=False, btt_ssfr=False):
    # file_suffices is a list of length-2 lists giving the file number boundaries used to generate each file
    Ngal_f_DS_total = 0
    
    for i, fno in enumerate(file_suffices):
        fname = dir + 'HI_ProfileFits_DarkSage_'
        fname = fname + str(fno[0]) + '_' + str(fno[1]) if not alt else fname + 'alt'
        fname = fname + '.bin'
        f = open(fname, 'rb')
        Ngal_f_DS = np.fromfile(f, 'i4', 1)[0]
        Ngal_f_DS_total += Ngal_f_DS
        Nann_f_DS = np.fromfile(f, 'i4', 1)[0]
        if not alt: Nfit_f_DS = np.fromfile(f, 'i4', 1)[0]
        GalaxyIDs_f_DS = np.fromfile(f, 'i8', Ngal_f_DS)
        
        if btt_ssfr:
            fname2 = dir + 'DarkSage_BTT_sSFR' + str(fno[0]) + '_' + str(fno[1])+ '.bin'
            f2 = open(fname2, 'rb')
            Ngal_2 = np.fromfile(f2, 'i4', 1)[0]
            assert Ngal_2==Ngal_f_DS
            GalaxyIDs_2 = np.fromfile(f2, 'i8', Ngal_2)
            assert np.all(GalaxyIDs_f_DS==GalaxyIDs_2)

        if i==0:
            fit1_Sig_f_DS = np.fromfile(f, 'f4', Ngal_f_DS)
            fit1_rb_f_DS = np.fromfile(f, 'f4', Ngal_f_DS)
            fit1_chi_f_DS = np.fromfile(f, 'f4', Ngal_f_DS)
            fit1_residual_f_DS = np.fromfile(f,np.dtype(('f4',Nann_f_DS)), Ngal_f_DS)
            if not alt: fit1_residual_fit_f_DS = np.fromfile(f, np.dtype(('f4',Nfit_f_DS)), Ngal_f_DS)
            fit2_Sig_f_DS = np.fromfile(f, 'f4', Ngal_f_DS)
            fit2_rb_f_DS = np.fromfile(f, 'f4', Ngal_f_DS)
            fit2_chi_f_DS = np.fromfile(f, 'f4', Ngal_f_DS)
            fit2_residual_f_DS = np.fromfile(f, np.dtype(('f4',Nann_f_DS)), Ngal_f_DS)
            if not alt: fit2_residual_fit_f_DS = np.fromfile(f, np.dtype(('f4',Nfit_f_DS)), Ngal_f_DS)
            fit3_dlSig_f_DS = np.fromfile(f, 'f4', Ngal_f_DS)
            fit3_rd_f_DS = np.fromfile(f, 'f4', Ngal_f_DS)
            fit3_chi_f_DS = np.fromfile(f, 'f4', Ngal_f_DS)
            fit3_residual_f_DS = np.fromfile(f, np.dtype(('f4',Nann_f_DS)), Ngal_f_DS)
            if not alt: 
                fit3_residual_fit_f_DS = np.fromfile(f, np.dtype(('f4',Nfit_f_DS)), Ngal_f_DS)
                StellarMass_f_DS = np.fromfile(f, 'f4', Ngal_f_DS) * 1e10/h
                Type_f_DS = np.fromfile(f, 'i2', Ngal_f_DS)
                MvirCen = np.fromfile(f, 'f4', Ngal_f_DS) * 1e10/h
                mHI = np.fromfile(f, 'f4', Ngal_f_DS)
                rHI = np.fromfile(f, 'f4', Ngal_f_DS)
                fit3_mHI_f_DS = np.fromfile(f, 'f4', Ngal_f_DS)
                DiscRadiiNorm = np.fromfile(f, np.dtype(('f4',Nann_f_DS)), Ngal_f_DS)
                    
            if btt_ssfr:
                BTT = np.fromfile(f2, 'f4', Ngal_2)
                sSFR = np.fromfile(f2, 'f4', Ngal_2)

        else:
            fit1_Sig_f_DS = np.append(fit1_Sig_f_DS, np.fromfile(f, 'f4', Ngal_f_DS))
            fit1_rb_f_DS = np.append(fit1_rb_f_DS, np.fromfile(f, 'f4', Ngal_f_DS))
            fit1_chi_f_DS = np.append(fit1_chi_f_DS, np.fromfile(f, 'f4', Ngal_f_DS))
            fit1_residual_f_DS =  np.vstack((fit1_residual_f_DS, np.fromfile(f, np.dtype(('f4',Nann_f_DS)), Ngal_f_DS)))
            if not alt: fit1_residual_fit_f_DS = np.vstack((fit1_residual_fit_f_DS, np.fromfile(f, np.dtype(('f4',Nfit_f_DS)), Ngal_f_DS)))
            fit2_Sig_f_DS = np.append(fit2_Sig_f_DS, np.fromfile(f, 'f4', Ngal_f_DS))
            fit2_rb_f_DS = np.append(fit2_rb_f_DS, np.fromfile(f, 'f4', Ngal_f_DS))
            fit2_chi_f_DS = np.append(fit2_chi_f_DS, np.fromfile(f, 'f4', Ngal_f_DS))
            fit2_residual_f_DS = np.vstack((fit2_residual_f_DS, np.fromfile(f, np.dtype(('f4',Nann_f_DS)), Ngal_f_DS)))
            if not alt: fit2_residual_fit_f_DS = np.vstack((fit2_residual_fit_f_DS, np.fromfile(f, np.dtype(('f4',Nfit_f_DS)), Ngal_f_DS)))
            fit3_dlSig_f_DS = np.append(fit3_dlSig_f_DS, np.fromfile(f, 'f4', Ngal_f_DS))
            fit3_rd_f_DS = np.append(fit3_rd_f_DS, np.fromfile(f, 'f4', Ngal_f_DS))
            fit3_chi_f_DS = np.append(fit3_chi_f_DS, np.fromfile(f, 'f4', Ngal_f_DS))
            fit3_residual_f_DS = np.vstack((fit3_residual_f_DS, np.fromfile(f, np.dtype(('f4',Nann_f_DS)), Ngal_f_DS)))
            if not alt: 
                fit3_residual_fit_f_DS = np.vstack((fit3_residual_fit_f_DS, np.fromfile(f, np.dtype(('f4',Nfit_f_DS)), Ngal_f_DS)))
                StellarMass_f_DS = np.append(StellarMass_f_DS, np.fromfile(f, 'f4', Ngal_f_DS) * 1e10/h)
                Type_f_DS = np.append(Type_f_DS, np.fromfile(f, 'i2', Ngal_f_DS))
                MvirCen = np.append(MvirCen, np.fromfile(f, 'f4', Ngal_f_DS) * 1e10/h)
                mHI = np.append(mHI, np.fromfile(f, 'f4', Ngal_f_DS))
                rHI = np.append(rHI, np.fromfile(f, 'f4', Ngal_f_DS))
                fit3_mHI_f_DS = np.append(fit3_mHI_f_DS, np.fromfile(f, 'f4', Ngal_f_DS))
                DiscRadiiNorm = np.vstack((DiscRadiiNorm, np.fromfile(f, np.dtype(('f4',Nann_f_DS)), Ngal_f_DS)))
            if btt_ssfr:
                BTT = np.append(BTT, np.fromfile(f2, 'f4', Ngal_2))
                sSFR = np.append(sSFR, np.fromfile(f2, 'f4', Ngal_2))

        f.close()

    if not alt:
        if not btt_ssfr:
            return Ngal_f_DS_total, fit1_Sig_f_DS, fit1_rb_f_DS, fit1_chi_f_DS, fit1_residual_f_DS, fit1_residual_fit_f_DS, fit2_Sig_f_DS, fit2_rb_f_DS, fit2_chi_f_DS, fit2_residual_f_DS, fit2_residual_fit_f_DS, fit3_dlSig_f_DS, fit3_rd_f_DS, fit3_chi_f_DS, fit3_residual_f_DS, fit3_residual_fit_f_DS, StellarMass_f_DS, Type_f_DS, MvirCen, mHI, rHI, fit3_mHI_f_DS, DiscRadiiNorm
        else:
            return Ngal_f_DS_total, fit1_Sig_f_DS, fit1_rb_f_DS, fit1_chi_f_DS, fit1_residual_f_DS, fit1_residual_fit_f_DS, fit2_Sig_f_DS, fit2_rb_f_DS, fit2_chi_f_DS, fit2_residual_f_DS, fit2_residual_fit_f_DS, fit3_dlSig_f_DS, fit3_rd_f_DS, fit3_chi_f_DS, fit3_residual_f_DS, fit3_residual_fit_f_DS, StellarMass_f_DS, Type_f_DS, MvirCen, mHI, rHI, fit3_mHI_f_DS, DiscRadiiNorm, BTT, sSFR
    else:
        return Ngal_f_DS_total, fit1_Sig_f_DS, fit1_rb_f_DS, fit1_chi_f_DS, fit1_residual_f_DS, fit2_Sig_f_DS, fit2_rb_f_DS, fit2_chi_f_DS, fit2_residual_f_DS

def Pearson_MS(h=0.6774):
    z_av = np.array([0.37, 0.66, 0.95, 1.24, 1.59, 2.02, 2.59, 3.23, 4.34, 5.18])
    alpha = np.array([0.43, 0.50, 0.46, 0.48, 0.51, 0.74, 0.83, 0.7, 0.93, 1.0])
    beta = np.array([0.58, 0.92, 1.10, 1.22, 1.31, 1.39, 1.59, 1.77, 1.87, 1.92]) # log(Msun/yr)
    hfrac = 2*np.log10(0.704/h)
    beta -= (hfrac*(1-alpha)) # normalization changes due to different little h assumption
    return z_av, alpha, beta


def TNG_gasprops(fname, highz=True, fname2=False, fmock=False, fmock_sfr=False):
    # Read the data produced by z0_gasprops_fromFOF_v2.py OR zhigh_gasprops_fromFOF.py if highz=True, and put into a dictionary
    if fname[-4:]=='.pkl': # reading a pickle file
        import cPickle as pickle
        with open(fname, 'rb') as f: dict = pickle.load(f)
        Ngal = dict['Ngal']
        Nradii = dict['Nradii']
    else: # reading a raw binary file with specific structure
        dict = {}
        
        mainbox = 'L75n1820TNG' in fname
        
        f = open(fname, 'rb')
        Ngal = dict['Ngal'] = np.fromfile(f, 'i8', 1)[0]
        Nradii_3D = dict['Nradii_3D'] = np.fromfile(f, 'i4', 1)[0]
        Nradii_2D = dict['Nradii_2D'] = np.fromfile(f, 'i4', 1)[0]
        Nradii_2D_proj = dict['Nradii_2D_proj'] = np.fromfile(f, 'i4', 1)[0]
        Nradii_cylinder = dict['Nradii_cylinder'] = np.fromfile(f, 'i4', 1)[0]
        Nradii = dict['Nradii'] = Nradii_3D + Nradii_2D + Nradii_2D_proj + 2*Nradii_cylinder
        Nradii_mock = dict['Nradii_mock'] = np.fromfile(f, 'i4', 1)[0]
        Nbin_profiles = dict['Nbin_profiles'] = np.fromfile(f, 'i4', 1)[0]

        dict['subhaloes'] = np.fromfile(f, 'i8', Ngal)
        dict['groupnr'] = np.fromfile(f, 'i8', Ngal)
        dict['Type'] = np.fromfile(f, 'i2', Ngal)
        dict['mass_halo'] = np.fromfile(f, 'f8', Ngal)
        
        dict['radii_3D'] = np.fromfile(f, np.dtype(('f4',Nradii_3D)), Ngal)
        dict['radii_2D'] = np.fromfile(f, np.dtype(('f4',Nradii_2D)), Ngal)
        dict['radii_2D_proj'] = np.fromfile(f, np.dtype(('f4',Nradii_2D_proj)), Ngal)
        dict['radii_cylinder'] = np.fromfile(f, np.dtype(('f4',Nradii_cylinder)), Ngal)
        dict['height_cylinder'] = np.fromfile(f, np.dtype(('f4',Nradii_cylinder)), Ngal)
        if not highz and Nradii_mock>0: dict['mock_aperture'] = np.fromfile(f, np.dtype(('f4',Nradii_mock)), Ngal)

        dict['mass_stars'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        dict['mass_gas'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        dict['mass_dm'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        dict['SFR'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        
        dict['j_stars'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        dict['j_gas'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        
        dict['mass_neutral'] = np.fromfile(f, np.dtype(('f4',Nradii+Nradii_mock)), Ngal)
        dict['mass_HI_GK11'] = np.fromfile(f, np.dtype(('f4',Nradii+Nradii_mock)), Ngal)
        dict['mass_H2_GK11'] = np.fromfile(f, np.dtype(('f4',Nradii+Nradii_mock)), Ngal)
        dict['mass_HI_K13'] = np.fromfile(f, np.dtype(('f4',Nradii+Nradii_mock)), Ngal)
        dict['mass_H2_K13'] = np.fromfile(f, np.dtype(('f4',Nradii+Nradii_mock)), Ngal)
        dict['mass_HI_GD14'] = np.fromfile(f, np.dtype(('f4',Nradii+Nradii_mock)), Ngal)
        dict['mass_H2_GD14'] = np.fromfile(f, np.dtype(('f4',Nradii+Nradii_mock)), Ngal)

        if highz:
            dict['metallicity_gas'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
            dict['metallicity_neutral'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
            
        dict['j_neutral'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        dict['j_HI_GK11'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        dict['j_H2_GK11'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        dict['j_HI_K13'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        dict['j_H2_K13'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        dict['j_HI_GD14'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        dict['j_H2_GD14'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        
        dict['veldisp_neutral'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        dict['veldisp_baryon'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        dict['velstd_gas'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        dict['velstd_baryon'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        
        if not highz and mainbox: dict['Time_Since_Infall'] = np.fromfile(f, 'f4', Ngal)
        dict['inclination'] = np.fromfile(f, 'f4', Ngal)
        if not highz and 'v1' not in fname and 'v2' not in fname and 'v3' not in fname: 
            dict['inclination_axis'] = np.fromfile(f, np.dtype(('f4',3)), Ngal)
        
        dict['veldisp_HIz_grided'] = np.fromfile(f, np.dtype(('f4',Nradii_cylinder)), Ngal)
        dict['veldisp_HIr_grided'] = np.fromfile(f, np.dtype(('f4',Nradii_cylinder)), Ngal)
        dict['veldisp_HIz_thermal_grided'] = np.fromfile(f, np.dtype(('f4',Nradii_cylinder)), Ngal)
        dict['veldisp_HIr_thermal_grided'] = np.fromfile(f, np.dtype(('f4',Nradii_cylinder)), Ngal)
        
        dict['positions_galaxies'] = np.fromfile(f, np.dtype(('f4',3)), Ngal)
        dict['velocities_galaxies'] = np.fromfile(f, np.dtype(('f4',3)), Ngal)
        
        if not highz and mainbox:
            dict['redshifts_dA'] = np.fromfile(f, np.dtype(('f4',4)), Ngal)
            dict['redshifts_vel'] = np.fromfile(f, np.dtype(('f4',4)), Ngal)

        dict['SigmaHI_profiles'] = np.fromfile(f, np.dtype(('f4',Nbin_profiles)), Ngal)

        if highz or 'v1' not in fname:
            dict['SigmaH2_profiles'] = np.fromfile(f, np.dtype(('f4',Nbin_profiles)), Ngal)

        if highz:
            if 'v1' not in fname and 'v2' not in fname: # applies for v3 and above
                dict['Hfrac_gas'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                dict['Hfrac_neutral'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
        
                dict['Cfrac_gas'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                dict['Cfrac_neutral'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                
                dict['Ofrac_gas'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                dict['Ofrac_neutral'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                
                dict['SFR_short'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                dict['SFR_long'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
            
                if 'v3' not in fname: # applies for v4 and above
                    dict['metallicity_molecular_GK11'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                    dict['metallicity_molecular_K13'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                    dict['metallicity_molecular_GD14'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                    
                    dict['Hfrac_molecular_GK11'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                    dict['Hfrac_molecular_K13'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                    dict['Hfrac_molecular_GD14'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                    
                    dict['Cfrac_molecular_GK11'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                    dict['Cfrac_molecular_K13'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                    dict['Cfrac_molecular_GD14'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                    
                    dict['Ofrac_molecular_GK11'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                    dict['Ofrac_molecular_K13'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)
                    dict['Ofrac_molecular_GD14'] = np.fromfile(f, np.dtype(('f4',Nradii)), Ngal)


    if not highz and fname2:
        f2 = open(fname2, 'rb')
        Ngal2 = np.fromfile(f2, 'i4', 1)[0]
        assert Ngal2==Ngal
        Nedge = 1+np.fromfile(f2, 'i4', 1)[0]
        dict['Nbin_prof'] = Nedge-1
        dict['RadArray'] = np.fromfile(f2, np.dtype(('f4',Nedge)), Ngal)
        dict['HIProf_GK11'] = np.fromfile(f2, np.dtype(('f4',Nedge)), Ngal)
        dict['HIProf_K13'] = np.fromfile(f2, np.dtype(('f4',Nedge)), Ngal)
        dict['HIProf_GD14'] = np.fromfile(f2, np.dtype(('f4',Nedge)), Ngal)
        dict['H2Prof_GK11'] = np.fromfile(f2, np.dtype(('f4',Nedge)), Ngal)
        dict['H2Prof_K13'] = np.fromfile(f2, np.dtype(('f4',Nedge)), Ngal)
        dict['H2Prof_GD14'] = np.fromfile(f2, np.dtype(('f4',Nedge)), Ngal)
        f2.close()

    if not highz and fmock:
        f3 = open(fmock, 'rb')
        Ngal3 = np.fromfile(f3, 'i8', 1)[0]
        if not Ngal3==Ngal: print Ngal, Ngal3
        assert Ngal3==Ngal
        dict['mass_halo_mock'] = np.fromfile(f3, 'f4', Ngal)
        dict['mass_stars_mock'] = np.fromfile(f3, 'f4', Ngal)
        dict['mass_stars'] = np.append(dict['mass_stars'], dict['mass_stars_mock'][np.newaxis].T, axis=1)
        dict['Type_mock'] = np.fromfile(f3, 'i2', Ngal)
        f3.close()
        
    if not highz and fmock_sfr:
        # Wants fmock_ssfr to be a list of 2 files
        f4 = open(fmock_sfr[0], 'rb')
        Ngal4 = np.fromfile(f4, 'i8', 1)[0]
        if not Ngal4==Ngal: print Ngal, Ngal4
        assert Ngal4==Ngal
        dict['SFR_recent'] = np.fromfile(f4, 'f4', Ngal)
        f4.close()
        
        f4 = open(fmock_sfr[1], 'rb')
        Ngal4 = np.fromfile(f4, 'i8', 1)[0]
        if not Ngal4==Ngal: print Ngal, Ngal4
        assert Ngal4==Ngal
        dict['SFR_old'] = np.fromfile(f4, 'f4', Ngal)
        f4.close()
            


    f.close()
    return dict

def HIdatacube(file):
    f = open(file, 'rb')
    Npix = np.fromfile(f, 'i4', 1)[0] # number of pixels in each dimension
    Nchan = np.fromfile(f, 'i4', 1)[0] # number of velocity channels
    xedge = np.fromfile(f, 'f4', Npix+1) # array of pixels edges [kpc]
    vedge = np.fromfile(f, 'f4', Nchan+1) # velocity bins [km/s]
    data_cube = np.fromfile(f, 'f4', Npix*Npix*Nchan).reshape(Npix,Npix,Nchan) # "direct" datacube from simulation.  Units are solar masses/kpc^2/(km/s)
    beamSmeared_cube = np.fromfile(f, 'f4', Npix*Npix*Nchan).reshape(Npix,Npix,Nchan) # data cube after beam smearing
    noiseAdded_cube = np.fromfile(f, 'f4', Npix*Npix*Nchan).reshape(Npix,Npix,Nchan) # data cub after adding noise
    f.close()

    return xedge, vedge, data_cube, beamSmeared_cube, noiseAdded_cube


def TNG_HIlines(file, mainbox=True, extended=False):
    dict = {}
    f = open(file, 'rb')
    Ngal = dict['Ngal'] = np.fromfile(f, 'i8', 1)[0]
    Nchan = dict['Nchan'] = np.fromfile(f, 'i8', 1)[0]
    dict['subhaloes'] = np.fromfile(f, 'i8', Ngal)
    dict['groupnr'] = np.fromfile(f, 'i8', Ngal)
    dict['mass_stars'] = np.fromfile(f, 'f4', Ngal)
    if mainbox: 
        dict['mass_stars_mock'] = np.fromfile(f, 'f4', Ngal)
        dict['mass_HI'] = np.fromfile(f, 'f8', Ngal)
        dict['mass_HI_mock'] = np.fromfile(f, 'f8', Ngal)
    else:
        dict['mass_HI'] = np.fromfile(f, 'f4', Ngal)
    dict['mass_halo'] = np.fromfile(f, 'f4', Ngal)
    if mainbox: dict['mass_halo_mock'] = np.fromfile(f, 'f4', Ngal)
    dict['SFR'] = np.fromfile(f, 'f4', Ngal)
    if mainbox: dict['SFR_mock'] = np.fromfile(f, 'f4', Ngal)
    dict['vbins'] = np.fromfile(f, 'f4', Nchan+1)
    dict['HIline_true'] = np.fromfile(f, np.dtype(('f4',Nchan)), Ngal)
    if mainbox: dict['HIline_mock'] = np.fromfile(f, np.dtype(('f4',Nchan)), Ngal)
    dict['Type'] = np.fromfile(f, 'i2', Ngal) # -1 is isolated central, 0 is a central with satellite(s), 1 is a satellite
    if mainbox: dict['Type_mock'] = np.fromfile(f, 'i2', Ngal)
    if extended:
        dict['pos_rel'] = np.fromfile(f, np.dtype(('f4',3)), Ngal) # kpc
        dict['vel_rel'] = np.fromfile(f, np.dtype(('f4',3)), Ngal) # km/s
    f.close()
    return dict
