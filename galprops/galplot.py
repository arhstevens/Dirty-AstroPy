# Adam Stevens, 2013-2017
# Plotting tools for galaxies

from pylab import *
from scipy import signal as ss
from scipy import stats
import galcalc as gc
from random import sample



def figure():
	# Generate an empty figure that is forced to not have excess white space (a problem that arises when legends are added otherwise)
	fig = plt.figure()
	plt.clf()
	fig.subplots_adjust(left=0, bottom=0)
	plt.subplot(111)


def contour(x,y,Nbins=None,weights=None,range=None,Nlevels=25,c='k',ls='-',lw=2,pcs=None,smooth=True):
	# Plot a 2D contour by first doing a 2D histogram of data with axis positions x and y
	
	if range==None: range = [[np.min(x),np.max(x)],[np.min(y),np.max(y)]]
	if Nbins==None: Nbins = len(x)/10
	
	finite = np.isfinite(x) * np.isfinite(y)
	x, y = x[finite], y[finite]
	im, xedges, yedges = np.histogram2d(x, y, bins=Nbins, weights=weights, range=range)
	xd, yd = xedges[1]-xedges[0], yedges[1]-yedges[0]
	xp, yp = xedges[1:]-xd, yedges[1:]-yd
	
    #k = gc.sphere2dk(3, 1, 7)
    #im = ss.convolve2d(im,k,mode='same') # Smooth the image for cleaner contours
	
	if pcs is not None: # Passing pcs trumps Nlevels, where Nlevels now essentially becomes len(pcs)
	    print 'pcs is not None and len(x)=', len(x)
	    if not smooth:
	        xx, yy = np.array([(Nbins+1)*[xedges]]), np.array([(Nbins+1)*[yedges]])
	        xx, yy = xx[0,:,:], yy[0,:,:].T
	        positions = np.vstack([xx.ravel(), yy.ravel()])
	        values = np.vstack([x, y])
	        kernel = stats.gaussian_kde(values)
	        zz = np.reshape(kernel(positions).T, xx.shape) # A better version of im essentially
	        xp, yp = xedges, yedges
	    else:
	        if len(x)<1e7: print 'Warning: smoothing with lots of particles will take time!'
	        k = gc.sphere2dk(3, 1, 7)
	        im = ss.convolve2d(im,k,mode='same') # Smooth the image for cleaner contours
	        zz = im.transpose()
		#arr = np.array(im, dtype=int)
		#vals = np.unique(arr)
		#siglevel = 0
	    zzlevel = np.zeros(len(pcs))
	    zzsum = np.sum(zz)
	    tol = 1e-3
		
	    for level in xrange(len(pcs)):
	        high_bound = np.max(zz)
	        low_bound = np.min(zz)
	        for i in xrange(1000): # Arbitrary maximum number of iterations of 1000
	            zval = (high_bound + low_bound)/2.
	            pc_within = np.sum(zz[zz>=zval])/zzsum
	            if pc_within < pcs[level]-tol:
	                high_bound = float(zval)
	            elif pc_within > pcs[level]+tol:
	                low_bound = float(zval)
	            else:
	                break
	        zzlevel[level] = zval
#        print 'zzlevel', zzlevel
	    if type(ls)==str:
	        plt.contour(xp, yp, zz, zzlevel, colors=c, linestyles=ls, linewidths=lw, zorder=1)
	    else:
	        CS = plt.contour(xp, yp, zz, zzlevel, colors=c, linestyles='-', linewidths=lw, zorder=1)
	        for c in CS.collections: c.set_dashes([(0, tuple(ls))])
	else:
	    if type(ls)==str:
	        plt.contour(xp, yp, im.transpose(), levels=Nlevels, colors=c, linestyles=ls, linewidths=lw, zorder=1)
	    else:
	        CS = plt.contour(xp, yp, im.transpose(), levels=Nlevels, colors=c, linestyles='-', linewidths=lw, zorder=1)
	        for c in CS.collections: c.set_dashes([(0, tuple(ls))])


def surfdens(r,Sigma,colour='red',bright=False,fsize=28,rmax=None):
	# Plot the surface density of a galaxy with some standard plotting parameters
	
	###  INPUTS ###
	# r = radius vector in parsecs
	# Sigma = corresponding surface density vector in solar masses per square parsec
	# colour = colour of line plotted
	# bright = set to True if doing surface brightness rather than surface density
	### ------- ###
	
	r = r/1000.
	plot(r, Sigma, color=colour, linewidth=2, label='Data')
	yscale('log')
	xlabel(r'$R$ [kpc]', fontsize=fsize)
	if bright is False:
		ylabel(r'Surface Density [$M_{\odot}$ pc$^{-2}$]', fontsize=fsize)
	else:
		ylabel(r'Surface Brightness', fontsize=fsize)
	grid(True)
	rmax = np.max(r) if rmax==None else rmax*1e-3  # Allow for a manual end to the plotting range
	axis([np.min(r),rmax,np.min(Sigma[r<=rmax]),np.max(Sigma[r<=rmax])])
	


def surfdensfit(r_vec,Sigma_fit,dsr,n):
	# Overplot the fitted surface density profile on the surface density profile plot (or can be by itself)
	# Strings for plotting legends
	n_str = str(n)
	try:
		if n_str[4]=='.':
			n_str = n_str[:4]
		else:
			n_str = n_str[:5]

		if dsr>1000:
			dsr_str = str(dsr*1e-3)
			if dsr_str[3]=='.':
				dsr_str = dsr_str[:3] + r' kpc'
			else:
				dsr_str = dsr_str[:4] + r' kpc'
		else:
			dsr_str = str(dsr)
			dsr_str = dsr_str[:3] + r' pc'

		plt.plot(r_vec/1000., Sigma_fit, 'g-', linewidth=2, label=r'DSR = '+dsr_str+r', $n$ = '+n_str)

	except IndexError:
		n = n # Do nothing if the inputs aren't working.  Will be a result of galfit's outputs when there's no convergence



def circle(r,colour='white',centre=[0,0],lw=3,ls='-',half='f',inclination=0,fill=False,alpha=1,part=1):
	# Plot a circle with radius r centred on coordinates "centre"
	# "Half" indicates whether to do a full circle (f), or just the top (t) or bottom (b) halves
	r = abs(r)
	x = np.linspace(-r, r, 500)
	y1 = np.sqrt(r**2 - x**2)
	if inclination>0: y1 *= np.cos(inclination*np.pi/180.)
	y1[0] = 0
	y1[-1] = 0
	y2 = -y1
	x, y1, y2 = x + centre[0], y1 + centre[1], y2 + centre[1]
	if half=='t': plot(x, y1, linewidth=lw, color=colour, linestyle=ls)
	if half=='b': plot(x, y2, linewidth=lw, color=colour, linestyle=ls)
	if half=='f':
		arg = np.where(y1==np.max(y1))[0][0]
		xp = np.concatenate((x[arg:], x[::-1], x[:arg]))
		yp = np.concatenate((y1[arg:], y2[::-1], y1[:arg]))
		if part<1 and part>0:
			filt = (yp < np.max(yp) - (1-part)*(np.max(yp)-np.min(yp)))
			xp, yp = xp[filt], yp[filt]
			top = np.max(yp)
		plot(xp, yp, linewidth=lw, color=colour, linestyle=ls)
	if fill:
		if part<1 and part>0:
			filt = (y2<top)
			x, y1, y2 = x[filt], y1[filt], y2[filt]
			y1[y1>top] = top
			print top
		plt.fill_between(x, y1, y2, color=colour, alpha=alpha)



def savepng(filename,xpix=2560,ypix=1440,ss=27,xpixplot=1024,ypixplot=None,fsize=28,addpdf=False,transparent=False, fig=None):
	# Save a plotted image a png file in the working directory with optimized resolution
	
	###  INPUTS ###
	# filename = filename of image to be saved
	
	# xpix = horizontal pixels of monitor
	# ypix = vertical pixels of monitor
	# ss = diagonal screen size in inches.  DEFAULTS SET TO MY SWINBURNE MAC
	# xpixplot = saved image size along the x-axis in pixels
	# fsize = fontsize of numbers in the plot
	### ------- ###
	
	#rcParams.update({'font.size': int(xpixplot/80.)}) # Set size of numbers on graph to be dependent on image size.
	#rcParams.update({'font.size': fsize})
	
	if ypixplot==None:
		ypixplot = int(xpixplot*9./16)
	
	mydpi = np.sqrt(xpix**2 + ypix**2)/ss  # The dpi of your screen
	xinplot = xpixplot*(9./7.)/mydpi
	yinplot = ypixplot*(9./7.)/mydpi
	if fig is None: fig = plt.gcf()
	fig.set_size_inches(xinplot,yinplot)
	fig.set_dpi(mydpi)
	
	
	filename = str(filename)
	if filename[-4:] != '.png':
		filename = filename+'.png'
	fig.savefig(filename, dpi=mydpi, bbox_inches='tight', transparent=transparent)
	if addpdf==True: fig.savefig(filename[:-4]+'.pdf', bbox_inches='tight')


def axlogic(x,y,yscale='linear'):
	# Give logical axis bounds for plots, i.e. based on the max/min values to be plotted
	
	if yscale=='linear': # The reasonable bounds to plot are simple for linear yscale...
		yex = 0.05*(np.max(y)-np.min(y)) # "y ex" = extra room on y-axis
		axis([np.min(x),np.max(x),np.min(y)-yex,np.max(y)+yex])
	
	elif yscale=='log': # ... but isn't so simple for log plots
		x, y = x[y>0], y[y>0]
		axis([np.min(x),np.max(x),(np.min(y)**1.05)/(np.max(y)**0.05),(np.max(y)**1.05)/(np.min(y)**0.05)])
		


def plotgal(x,y,z,mass,thetadeg=0,phideg=0,xmax=40000.,Npixels=2048,pt=1,bright=False,fsize=28,vran=None,rsmooth=None,gridon=True,xlab=r'$x$ [kpc]',ylab=None,vel=False):
	# Plot galaxy as a heat map based on surface density (or brightness) in any orientation.  This is what example.py does.
	
	###  INPUTS ###
	# x,y,z = particle coordinates in pc
	# mass = mass of particles
	# thetadeg = angle in degrees to rotate about the z-axis (0-360)
	# phideg = angle in degrees to rotate about the x'-axis (0-180)
	
	# xmax = plotting range on each side of box in pc (one side will be collapsed).
	# Npixels = size of 2D bins for the plot.  Should be equivalent to xpixplot once saved using savepng.
	# pt = particle type (0=gas, 1=stars, 2=dm, 3=bh)
	# bright = True if instead of mass, luminosity of particles are input to plot
	# vran = vmin and vmax values for imshow.  If not specified, has default values
	# rsmooth = radius by which to smooth particles in a spherically symmetric way (so mass is distributed accordingly in neighbouring pixels)
	# gridon = switch to False to prevent the grid from being part of the plot
	# xlab = xlabel
	# ylab = ylabel
	# vel = True if instead of mass, velocities are input
	### ------- ###
	
	xmin = -xmax
	
	theta, phi = np.deg2rad(thetadeg), np.deg2rad(phideg) # Change to radians
	xprime, ytemp = x*np.cos(theta) + y*np.sin(theta), y*np.cos(theta) - x*np.sin(theta) # Rotate about z-axis by angle theta
	yprime, zprime = ytemp*np.cos(phi) + z*np.sin(phi), z*np.cos(phi) - ytemp*np.sin(phi) # Rotate about x'-axis by angle phi
	
	
	filter_rad=(xprime>xmin)*(xprime<xmax)*(yprime>xmin)*(yprime<xmax)*(zprime>xmin)*(zprime<xmax)
	xf, yf = xprime[filter_rad], yprime[filter_rad] # x and y filtered
	
	# Build a 2D array for plotting
	im, xedges, yedges = np.histogram2d(xf, yf, bins=Npixels, weights=mass[filter_rad], range=[[xmin,xmax],[xmin,xmax]])
	extent = [xedges[0]*1e-3, xedges[-1]*1e-3, yedges[0]*1e-3, yedges[-1]*1e-3] # Make the scale on the plot in kpc
	Lpix = (xmax-xmin)/float(Npixels) # Length of each pixel
	
	if vel==False:
		if pt!=3:
			im = im / Lpix**2 # Gives surface density in Msun/pc2.  Remains as just Msun if using BHs
		
		if rsmooth!=None:
			if Lpix<rsmooth:
				k = gc.sphere2dk(rsmooth, Lpix, 2*rsmooth/Lpix)
				im = ss.convolve2d(im,k,mode='same')

		im2plot = np.log10(im.transpose() + 1e-10)
	
	else:
		im = im / (np.histogram2d(xf, yf, bins=Npixels, range=[[xmin,xmax],[xmin,xmax]])[0] + 1e-10) # stop division by zero
		im2plot = im.transpose()
		

	
	if vran==None or (len(vran)!=2 and len(vran)!=0):
		if pt==0: vmin, vmax = -1, 2 # Shows a log scale, from 10^-1 to 10^2 Msun/pc^2
		if pt==1 or pt==2: vmin, vmax = -1, 3.5 # Shows a log scale, from 10^-1 to 10^3.5 Msun/pc^2 (for stars)
		if pt==3: vmin, vmax = 4, 8
		if bright==True: vmin, vmax = 32, 36
	elif len(vran)==0:
		if vel==False:
			vmin, vmax = max(np.min(im2plot[im2plot>-10])-0.2,-10.02), np.max(im2plot) # If vran is an empty list, then plot the minimum to maximum range (brightening the dim end slightly)
		else: # not plotting logarithmically, so can be simpler bounds
			vmin, vmax = np.min(im2plot), np.max(im2plot)
		print 'Plotting range', vmin, 'to', vmax
	else:
		vmin, vmax = vran[0], vran[1]
	
	if pt==0:
		imshow(im2plot,extent=extent, interpolation='nearest',cmap=plt.cm.bone,vmin=vmin,vmax=vmax, origin="lower") 
		grid(True,color='red',linewidth=1)
	elif pt==3:
		imshow(im2plot,extent=extent, interpolation='bilinear',cmap=plt.cm.bone, vmin=vmin,vmax=vmax, origin="lower")
		grid(True,color='red',linewidth=1)
	elif vel==True:
		imshow(im2plot,extent=extent, interpolation='nearest',cmap=plt.cm.RdYlBu, vmin=vmin,vmax=vmax, origin="lower")
		grid(True,color='white',linewidth=1)
	else:
		imshow(im2plot,extent=extent, interpolation='nearest',cmap=plt.cm.hot,vmin=vmin,vmax=vmax, origin="lower") 
		grid(True,color='blue',linewidth=1)
	xlabel(xlab, fontsize=fsize)
	if gridon is not True: grid(False)
	if ylab is not None: ylabel(ylab, fontsize=fsize)
	

	
def tzplot(t,y,z,mstyle='b-',msize=10,lstyle='-',lwidth=3,fsize=28,label=r'',rate=0,text=r''):
	# Make a plot against time with redshift also shown at the top.  Probably usable for any plot with two x-scales.
	
	### INPUTS ###
	# t = time data to be plotted on x-axis (x-scale at bottom)
	# y = corresponding data to be plotted on y-axis
	# z = corresponding redshift data (x-scale at top)
	
	# msize = markersize
	# lstyle = linestyle
	# lwidth = linewidth
	# fsize = fontsize
	# label = label for plot on a legend
	# rate = whether a rate (1) or values (0) at times are being plotted.  For a rate, the first time entry is assumed to be ignored.
	# text = words to put in upper left corner of graph
	### ------ ###
	t, y = t[y!=np.inf], y[y!=np.inf] # Just incase there's an infinity...
	
	grid(True)
	plot(t,y,mstyle,markersize=msize,linestyle=lstyle,linewidth=lwidth,label=label)
	xlabel(r'Time (Gyr)',fontsize=fsize)
	
	yran = np.max(y)-np.min(y)
	axis([np.min(t[rate:]),np.max(t),np.min(y)-0.05*yran,np.max(y)+0.05*yran])
	addtext(t,y,text)
	
	addz(t[rate:],fsize=fsize,H_0=67.3,Omega_M=0.324,Omega_Lambda=0.676)
	gcf()

	
def addz(t,bounds=None,fsize=28,H_0=67.3,Omega_R=0,Omega_M=0.315,Omega_Lambda=0.685):
	# This just executes the command to add the redshift points to a plot.
	# Should do this generally as the last command in the plotting.
	
	### INPUTS ###
	# t = the time vector that has already been plotted
	# bounds = the left and right bounds of the plot in terms of the t-coordinates.  Will default to min and max while set to "None".  Otherwise should be a two-entry list
	# fsize = fontsize for the word "redshift"
	# H_0, Omegas = Cosmological parameters for working out where to place the ticks
	
	if bounds==None:
		bounds = [np.min(t),np.max(t)]

	ztix = np.array([0,0.1,0.2,0.4,0.6,0.8,1,1.5,2,3,5,8]) # Values of z to be shown on the plot
	ax1 = gca()
	ax2 = ax1.twiny()
	ax2.set_xlabel(r'Redshift',fontsize=fsize)
	ax2.set_xlim(bounds[0],bounds[1]) # Tells the program where to place the ticks by telling it the bounds of the plot.  Therefore, should have the restricted the plot with plt.axis accordingly before.
	tickpos = [gc.z2t(ztix[i],Omega_R=Omega_R,Omega_M=Omega_M,Omega_Lambda=Omega_Lambda,H_0=H_0) for i in range(len(ztix)) if gc.z2t(ztix[i],Omega_R=Omega_R,Omega_M=Omega_M,Omega_Lambda=Omega_Lambda,H_0=H_0) >= min(bounds)]
	ax2.set_xticks(tickpos)
	labs = [r'$0$',r'$0.1$',r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1$',r'$1.5$',r'$2$',r'$3$',r'$5$',r'$8$']
	ax2.set_xticklabels(labs[:len(tickpos)])



def addtl(tarr=None,zarr=None,zmin=0,zmax=2.5,fsize=28,H_0=67.3,Omega_R=0,Omega_M=0.315,Omega_Lambda=0.685):
	# Same purpose as addz, except adds look-time time ticks, assuming redshift is plotted on the x-axis.

	# The zmin and zmax values are equivalent to "bounds" in addz.
	# tarr is from ztlookup.  If this hasn't been precalculated, it will be done here.
	
	if tarr==None:
		tarr = gc.ztlookup(zmax=zmax, H_0=H_0, Omega_R=Omega_R, Omega_M=Omega_M, Omega_Lambda=Omega_Lambda)
	
	t0 = gc.z2t(0, H_0=H_0, Omega_R=Omega_R, Omega_M=Omega_M, Omega_Lambda=Omega_Lambda)
	
	ttix = np.array([0,3,5,6,7,8,9,10,10.5,11,11.5,12,13])
	labs = [r'$0$',r'$3$',r'$5$',r'$6$',r'$7$',r'$8$',r'$9$',r'$10$',r'$10.5$',r'$11$',r'$11.5$',r'$12$',r'$13$']
	#labs = [r'0',r'3',r'5',r'6',r'7',r'8',r'9',r'10',r'10.5',r'11',r'11.5',r'12',r'13']
	ax1 = gca()
	ax2 = ax1.twiny()
	ax2.set_xlabel(r'Look-back time [Gyr]',fontsize=fsize)
	ax2.set_xlim(zmin,zmax)
	tickpos = [gc.t2z(t0-ttix[i],tarr,zarr) for i in range(len(ttix)) if gc.t2z(t0-ttix[i],tarr,zarr)<=zmax]
	ax2.set_xticks(tickpos)
	ax2.set_xticklabels(labs[:len(tickpos)])



def plot4(t,y,extra=1,ylabel=r'',title=r'Main Galaxy',yscale='linear',fsize=28,yfilt=True):
	# Plot for the 4 radii.  t is time in Myr.  y-data whatever.
	# Extra==1 means there'll be extra plotting stuff added (addz etc.), otherwise set to 0.
	# yflit is any filter that should be applied to each set of ydata, eg. (t<10)
	t = t*1e-3
	y0, y1, y2, y3 = y[:,0], y[:,1], y[:,2], y[:,3]
	con0, con1, con2, con3 = (abs(y0)!=np.inf)*yfilt*(True-isnan(y0)), (abs(y1)!=np.inf)*yfilt*(True-isnan(y1)), (abs(y2)!=np.inf)*yfilt*(True-isnan(y2)), (abs(y3)!=np.inf)*yfilt*(True-isnan(y3)) # Remove points from plot that are infinities and nans.
	
	plt.plot(t[con0],y0[con0],'r-',linewidth=3,label=r'Star Rel') 
	plt.plot(t[con1],y1[con1],'y-',linewidth=2,label=r'Star Abs')
	plt.plot(t[con2],y2[con2],'c--',linewidth=3,label=r'Gas Rel')
	plt.plot(t[con3],y3[con3],'b--',linewidth=3,label=r'Gas Abs')
	
	
	if extra==1:
		plt.legend(fontsize=22,title=title,loc='best')
		plt.xlabel(r'Time (Gyr)',fontsize=fsize)
		plt.ylabel(ylabel,fontsize=fsize)
		plt.yscale(yscale)
		plt.grid(True)
		
		axlogic(t,list(y0[con0])+list(y1[con1])+list(y2[con2])+list(y3[con3]),yscale=yscale)
		addz(t,H_0=67.3,Omega_M=0.324,Omega_Lambda=0.676) # Cosmo parameters chosen for Marie's sims to give last snapshot at z=0


def addtext(x,y,words,fsize=28,ax='logic'):
	# Put words in the upper left corner.  x and y are the data plotted on the axes.
	xran, yran = np.max(x)-np.min(x), 1.1*(np.max(y)-np.min(y)) # Given how axlogic works, these are the ranges of the plot
	plt.text(np.min(x)+0.02*xran, np.max(y)-0.02*yran, words, fontsize=fsize)



def allthecircles(r_rel,r_abs,r_rel_g,r_abs_g,r_vir,lw=3):
	# Place circles on a plot for all the baryonic radii and the virial radius.  Inputs in pc.
	circle(r_rel/1000.,lw=lw)
	circle(r_abs/1000.,colour='yellow',lw=lw)
	circle(r_rel_g/1000.,colour='cyan',lw=lw)
	circle(r_abs_g/1000.,colour='blue',lw=lw)
	circle(r_vir/1000.,colour='magenta',lw=lw)


def analysiscircles(r_vir, r_abs, r_rel_g, r_od, h=0.7,lw=3):
	circle(30/h, 'white',lw=lw)
	circle(r_rel_g*1e-3, 'cyan',lw=lw)
	circle(r_abs*1e-3, 'y',lw=lw)
	circle(r_vir*1.5e-4, 'red',lw=lw)
	circle(r_vir*1e-4, 'magenta',lw=lw)
	circle(r_od*1e-3, 'blue',lw=lw)
	

def smoothplot(x,y,dx,plottype='b-',lw=1,label=r''):
	# Do a plot but smooth it by averaging all the values within each range dx.  First averaged block starts at the lowest value of x.
	
	if x[1]<x[0]: # Check to see if x is increasing.  If not, assume it's always decreasing and reverse.
		x = x[::-1]
		y = y[::-1]
	
	n = int((np.max(x)-np.min(x))/dx) + 1 # Number of bins
	xplot = np.arange(n)*dx + min(x) # x-data to plot
	yplot = np.zeros(n) # Initialize y-data
	#xint = np.arange(100*n)*dx/100 + min(x) # Same as xplot but with more points in between
	#yint = np.interp(xint, x, y) # y-data but interpolated at lots of points for easiness
	
	for i in xrange(n):
		x1 = min(x)+dx*i # First x-point in the bin
		x2 = min(x)+dx*(i+1) # Last x-point in the bin
		ydata = y[(x>=x1)*(x<x2)]
		if len(ydata)>0:
			yplot[i] = np.mean(ydata)
			xplot[i] = np.mean(x[(x>=x1)*(x<x2)])
		else:
			yplot[i] = np.interp(xplot[i],x,y)
	
	xplot = np.append(np.array([x[0]]),xplot) # Still keep the first point so the overall positions of data points don't move drastically.
	yplot = np.append(np.array([y[0]]),yplot)
	
	plt.plot(xplot,yplot,plottype,linewidth=lw,label=label)


def point_with_spread(x, data, low=16, high=84, mean=False, stddev=False, ls='.', c='k', ms=4, alpha=1.0, lw=1):
	# Plots a single point for an array of data with an errorbar covering the low to high percentile of that data or standard deviation dependent on choice
    if len(data)==0:
        return

    if mean:
        y = np.mean(data)
    else:
        y = np.median(data)
		
    if stddev:
        ylow = np.std(data)
        yhigh = np.std(data)
    else:
        data = np.sort(data)
        i_low = int(len(data)*low/100.)
        i_high = int(len(data)*high/100.)
        ylow = y - data[i_low]
        yhigh = data[i_high] - y

    #print 'point_with_spread info', len(data), np.min(data), np.max(data), y, ylow, yhigh
    plt.errorbar(x, y, yerr=[np.array([ylow]), np.array([yhigh])], ecolor=c, fmt=ls, ms=ms, alpha=alpha, color=c, lw=lw)
	
	
def massfunction(mass, Lbox, colour='k', label=r'Input', extra=0, fsize=28, pt=1, Nbins=50, h=0.7, step=True, ls='-', lw=2, binwidth=None, loc=None, leg=True, hfac1=False, range=None, ax=None, Print=None):
	# Make a mass function plot.  Input vector of halo masses in solar masses and length of box in Mpc.
	# Setting h=1 means masses are being put in with a factor of h^2 already considered.  Else use physical units.
    #masslog = np.log10(gc.cleansample(mass))
    masslog = np.log10(mass[(mass>0)*(np.isfinite(mass))])
    if range==None:
        lbound, ubound = max(8,np.min(masslog)), min(12.5,np.max(masslog))
    else:
        lbound, ubound = range[0], range[1]
    if binwidth is not None: Nbins = int((ubound - lbound) / binwidth)
    if pt==0 or pt==1 or pt==4:
        N, edges = np.histogram(masslog, bins=Nbins, range=[lbound,ubound]) # number of galaxies in each bin.  Doing a specific range for stars and gas.
        xhist, edges = np.histogram(masslog, bins=Nbins, range=[lbound,ubound], weights=10**masslog)
    elif pt==3:
        N, edges = np.histogram(masslog, bins=Nbins, range=[6+2*np.log10(0.7/h),9+np.log10(5)+2*np.log10(0.7/h)])
        xhist, edges = np.histogram(masslog, bins=Nbins, range=[6+2*np.log10(0.7/h),9+np.log10(5)+2*np.log10(0.7/h)], weights=10**masslog)
    binwidth = edges[1]-edges[0]
    x = edges[:-1] + binwidth/2
    x[N>0] = np.log10(xhist[N>0] / N[N>0])
    y = N/(binwidth*Lbox**3)
    if ax is None:
        if step==True:
            plt.step(x, y, where='mid', color=colour, linewidth=lw, label=label)
        else:
            if type(ls)==str:
                plt.plot(x, y, colour+ls, linewidth=lw, label=label)
            elif type(ls)==list:
                plt.plot(x, y, colour+'-', linewidth=lw, label=label, dashes=ls)
        plt.yscale('log', nonposy='clip')
    else:
        if step==True:
            ax.step(x, y, where='mid', color=colour, linewidth=lw, label=label)
        else:
            if type(ls)==str:
                ax.plot(x, y, colour+ls, linewidth=lw, label=label)
            elif type(ls)==list:
                ax.plot(x, y, colour+'-', linewidth=lw, label=label, dashes=ls)
        ax.set_yscale('log', nonposy='clip')

    if Print is not None: print Print, '\n', x, '\n', y, '\n'

    if extra>0:
        axlogic(x,y,yscale='log') # Set axis
        if pt==0:
            xlab = r'$\log_{10}(M_{\mathrm{gas}}\ [\mathrm{M}_{\bigodot}])$' if h!=1 else r'$\log_{10}(M_{\mathrm{gas}} h^2\ [\mathrm{M}_{\bigodot}])$'
        elif pt==1:
            xlab = r'$\log_{10}(M_{\mathrm{stars}}\ [\mathrm{M}_{\odot}])$' if h!=1 else r'$\log_{10}(M_{\mathrm{stars}} h^2\ [\mathrm{M}_{\odot}])$'
        elif pt==2:
            xlab = r'$\log_{10}(M_{\mathrm{DM}}\ [\mathrm{M}_{\odot}])$' if h!=1 else r'$\log_{10}(M_{\mathrm{DM}} h^2\ [\mathrm{M}_{\odot}])$'
        elif pt==3:
            xlab = r'$\log_{10}(M_{\mathrm{BH}}\ [\mathrm{M}_{\odot}])$' if h!=1 else r'$\log_{10}(M_{\mathrm{BH}} h^2\ [\mathrm{M}_{\odot}])$'
        elif pt==4:
            xlab = r'$\log_{10}(M_{\mathrm{baryons}}\ [\mathrm{M}_{\odot}])$' if h!=1 else r'$\log_{10}(M_{\mathrm{baryons}} h^2\ [\mathrm{M}_{\odot}])$'
        plt.xlabel(xlab, fontsize=fsize)
        ylab = r'$\Phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$' if h!=1 else r'$\Phi h^{-3}\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$'
        plt.ylabel(ylab, fontsize=fsize)
        #plt.ylabel(r'$\mathrm{Number\ Density}\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$', fontsize=fsize)
        
        if extra>1:
            
            if pt==1:
                
                # Data for Baldry et al 2008 (copied from SAGE's allresults.py script)
                
                B = np.array([
			        [7.05, 1.3531e-01, 6.0741e-02],
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
                
                #  Looking at the actual Baldry paper, the x-axis is physical units assuming h=0.7, while the y-data is phi*h**-3 (i.e. need to multiply by h**3 to agree with their Fig. 6)
                if ax is None:
                    plt.fill_between(B[:,0]-np.log10(h**2)-0.26, (B[:,1]+B[:,2])*h**3, (B[:,1]-B[:,2])*h**3, facecolor='purple', alpha=0.2) if not hfac1 else plt.fill_between(B[:,0]-np.log10(h)-0.26, (B[:,1]+B[:,2]), (B[:,1]-B[:,2]), facecolor='purple', alpha=0.2)
                    plt.plot([1,1], [1,2], color='purple', linewidth=8, alpha=0.3, label=r'Baldry et al.~(2008)') # Just for the legend
                else:
                    ax.fill_between(B[:,0]-np.log10(h**2)-0.26, (B[:,1]+B[:,2])*h**3, (B[:,1]-B[:,2])*h**3, facecolor='purple', alpha=0.2) if not hfac1 else plt.fill_between(B[:,0]-np.log10(h)-0.26, (B[:,1]+B[:,2]), (B[:,1]-B[:,2]), facecolor='purple', alpha=0.2)
                    ax.plot([1,1], [1,2], color='purple', linewidth=8, alpha=0.3, label=r'Baldry et al.~(2008)') # Just for the legend

                """
                Phi_low, Phi_high, logM = gc.doubleschechter2(10.648, 4.26, -0.46, 0.58, -1.58, 0.013, 0.09, 0.05, 0.07, 0.02, h=h)
                #print Phi_low
                plt.fill_between(logM+0.26, Phi_high, Phi_low, color='r', alpha=0.3)
                plt.plot([1,1], [1,2], color='r', linewidth=4, alpha=0.3, label=r'Baldry et al. (2008)') # Just for the legend
                """
            elif pt==4:
                Phi_low, Phi_high, logM = gc.schechter2(0.0108*h**3, 5.3e10 *h**(-2), -1.21, 0.0006*h**3, 3e9 *h**(-2), 0.05)
                plt.fill_between(logM, Phi_high, Phi_low, color='b', alpha=0.3)
                Phi, logM = gc.doubleschechter(10.675,4.9,-0.42,0.61,-1.87,h=h) # From Baldry et al (2008)
                plt.plot(logM+0.26, Phi, 'r-', lw=2, label=r'Baldry et al.~(2008)')
                plt.plot([0,1],[1,1], 'b-', lw=4, alpha=0.5, label=r'Bell et al.~(2003)')
            elif pt==3:
                # Add observations from Shankar 2004				
                Phi_low, Phi_high, logM = gc.bhmf2(7.7e-3, 6.4e7, -1.11, 0.49, 3e-4, 1.1e7, 0.02, 0.02)
                plt.fill_between(logM+2*np.log10(0.7/h), Phi_high*(h/0.7)**3, Phi_low*(h/0.7)**3, color='b', alpha=0.3)
                plt.plot([0,1],[1,1], 'b-', linewidth=4, alpha=0.5, label=r'Shankar et al.~(2004)')
                plt.xlim(6+2*np.log10(0.7/h),9+np.log10(5)+2*np.log10(0.7/h))
        if leg:
            if loc is not None:
                plt.legend(fontsize=fsize-6, loc=loc, frameon=False)
            elif pt==1:
                plt.legend(fontsize=fsize-6, loc='lower left', frameon=False)#, title=r'$h='+str(h)+'$, $\kappa_R = 0.10$, $\kappa_Q = 0.002$')
            else:
                plt.legend(fontsize=fsize-6, loc='best', frameon=False)


def massfunction_HI_H2_obs(h=0.678, HI=True, H2=True, K=True, OR=False, ax=None, Z=True, M=False):
    Zwaan = np.array([[6.933,   -0.333],
                    [7.057,   -0.490],
                    [7.209,   -0.698],
                    [7.365,   -0.667],
                    [7.528,   -0.823],
                    [7.647,   -0.958],
                    [7.809,   -0.917],
                    [7.971,   -0.948],
                    [8.112,   -0.927],
                    [8.263,   -0.917],
                    [8.404,   -1.062],
                    [8.566,   -1.177],
                    [8.707,   -1.177],
                    [8.853,   -1.312],
                    [9.010,   -1.344],
                    [9.161,   -1.448],
                    [9.302,   -1.604],
                    [9.448,   -1.792],
                    [9.599,   -2.021],
                    [9.740,   -2.406],
                    [9.897,   -2.615],
                    [10.053,  -3.031],
                    [10.178,  -3.677],
                    [10.335,  -4.448],
                    [10.492,  -5.083]])



    Martin_data = np.array([[6.302,	-0.504],
                        [6.500,	-0.666],
                        [6.703,	-0.726],
                        [6.904,	-0.871],
                        [7.106,	-1.135],
                        [7.306,	-1.047],
                        [7.504,	-1.237],
                        [7.703,	-1.245],
                        [7.902,	-1.254],
                        [8.106,	-1.414],
                        [8.306,	-1.399],
                        [8.504,	-1.476],
                        [8.705,	-1.591],
                        [8.906,	-1.630],
                        [9.104,	-1.695],
                        [9.309,	-1.790],
                        [9.506,	-1.981],
                        [9.707,	-2.141],
                        [9.905,	-2.317],
                        [10.108,	-2.578],
                        [10.306,	-3.042],
                        [10.509,	-3.780],
                        [10.703,	-4.534],
                        [10.907,	-5.437]])
    Martin_mid = Martin_data[:,1] + 3*np.log10(h/0.7)
    Martin_x = Martin_data[:,0] + 2*np.log10(0.7/h)
    Martin_high = np.array([-0.206, -0.418, -0.571, -0.725, -1.003, -0.944, -1.144, -1.189, -1.189, -1.358, -1.344, -1.417, -1.528, -1.586, -1.651, -1.753, -1.925, -2.095, -2.281, -2.537, -3.003, -3.729, -4.451, -5.222]) + 3*np.log10(h/0.7)
    Martin_low = np.array([-0.806, -0.910, -0.885, -1.019, -1.268, -1.173, -1.313, -1.314, -1.320, -1.459, -1.443, -1.530, -1.647, -1.669, -1.736, -1.838, -2.021, -2.191, -2.359, -2.621, -3.098, -3.824, -4.618, -5.663]) + 3*np.log10(h/0.7)

    Keres_high = np.array([-1.051, -1.821, -1.028, -1.341, -1.343, -1.614, -1.854, -2.791,  -3.54 , -5.021]) + 3*np.log10(h)
    Keres_mid = np.array([-1.271, -1.999, -1.244, -1.477, -1.464, -1.713, -1.929, -2.878,   -3.721, -5.22 ]) + 3*np.log10(h)
    Keres_low = np.array([-1.706, -2.302, -1.71 , -1.676, -1.638, -1.82 , -2.033, -2.977,   -4.097, -5.584]) + 3*np.log10(h)
    Keres_M = np.array([  6.953,   7.353,   7.759,   8.154,   8.553,   8.96 ,   9.365,  9.753,  10.155,  10.558]) - 2*np.log10(h)
    
    ObrRaw_high = np.array([-0.905, -1.122, -1.033, -1.1  , -1.242, -1.418, -1.707, -2.175, -2.984, -4.868]) + 3*np.log10(h)
    ObrRaw_mid = np.array([-1.116, -1.308, -1.252, -1.253, -1.373, -1.509, -1.806, -2.261,  -3.198, -5.067]) + 3*np.log10(h)
    ObrRaw_low = np.array([-1.563, -1.602, -1.73 , -1.448, -1.537, -1.621, -1.918, -2.369,  -3.556, -5.413]) + 3*np.log10(h)
    ObrRaw_M = np.array([ 7.301,  7.586,  7.862,  8.133,  8.41 ,  8.686,  8.966,  9.242,    9.514,  9.788]) - 2*np.log10(h)

    HI_x = Zwaan[:,0] - 2*np.log10(h)
    HI_y = 10**Zwaan[:,1] * h**3

    if ax is None:
        #if HI: plt.plot(HI_x, HI_y, 'g-', lw=8, alpha=0.4, label=r'H\,\textsc{i} (Zwaan et al.~2005)')
        if HI and Z: plt.plot(HI_x, HI_y, '-', color='purple', lw=8, alpha=0.4, label=r'Zwaan et al.~(2005)')
        if HI and M: plt.fill_between(Martin_x, 10**Martin_high, 10**Martin_low, color='c', alpha=0.4)
        if HI and M: plt.plot([0,1], [1,1], 'c-', lw=8, alpha=0.4, label=r'Martin et al.~(2010)')

        #plt.fill_between(ObrRaw_M, 10**ObrRaw_high, 10**ObrRaw_low, color='purple', alpha=0.4)
        if H2 and K: plt.fill_between(Keres_M, 10**Keres_high, 10**Keres_low, color='purple', alpha=0.4)
        #if H2 and K: plt.plot([0,1], [1,1], 'b-', lw=8, alpha=0.4, label=r'H$_2$ (Keres et al.~2003)')
        if H2 and K: plt.plot([0,1], [1,1], '-', color='purple', lw=8, alpha=0.4, label=r'Keres et al.~(2003)')
        
        if H2 and OR: plt.fill_between(ObrRaw_M, 10**ObrRaw_high, 10**ObrRaw_low, color='c', alpha=0.4)
        if H2 and OR: plt.plot([0,1], [1,1], '-', color='c', lw=8, alpha=0.4, label=r'Obreschkow \& Rawlings (2009)')
        
        #plt.plot([0,1], [1,1], '-', color='purple', lw=8, alpha=0.4, label=r'H$_2$ (Obreschkow+09)')
        
        plt.xlabel(r'$\log_{10}(M_{\mathrm{H}\,\huge\textsc{i}}\ \mathrm{or}\ M_{\mathrm{H}_2}\ [\mathrm{M}_{\bigodot}])$')
        plt.ylabel(r'$\Phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
        plt.axis([8,11.5,1e-6,1e-1])
    else:
        if HI and Z: ax.plot(HI_x, HI_y, 'g-', lw=8, alpha=0.4, label=r'Zwaan et al.~(2005)')
        
        if H2 and K: ax.fill_between(Keres_M, 10**Keres_high, 10**Keres_low, color='c', alpha=0.4)
        if H2 and K: ax.plot([0,1], [1,1], 'c-', lw=8, alpha=0.4, label=r'Keres et al.~(2003)')
        
        if H2 and OR: ax.fill_between(ObrRaw_M, 10**ObrRaw_high, 10**ObrRaw_low, color='purple', alpha=0.4)
        if H2 and OR: ax.plot([0,1], [1,1], '-', color='purple', lw=8, alpha=0.4, label=r'H$_2$ (Obreschkow \& Rawlings 2009)')
        
        ax.xlabel(r'$\log_{10}(M_{\mathrm{H}\,\huge\textsc{i}}\ \mathrm{or}\ M_{\mathrm{H}_2}\ [\mathrm{M}_{\bigodot}])$')
        ax.ylabel(r'$\Phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
        ax.axis([8,11.5,1e-6,1e-1])

def btf(M_bary, V_rot, extra=False, c='k', ls='-', lw=2, h=0.7, label=r'Input', fsize=28, pcs=None):
	# Create baryonic Tully-Fisher relation plot with contours
	filt = (V_rot>0) * (M_bary>0)
	x, y = np.log10(V_rot[filt]), np.log10(M_bary[filt])
	contour(x,y,Nbins=50,weights=None,range=None,Nlevels=5,c=c,ls=ls,lw=lw,pcs=pcs)
	xlab = r'$\log_{10} (V_{\rm max}\ [\mathrm{km}\ \mathrm{s}^{-1}])$'
	ylab = r'$\log_{10} (M_{\rm baryons}\ [\mathrm{M}_{\bigodot}])$' if h!=1 else r'$\log_{10} (M_{\rm baryons} h^2\ [\mathrm{M}_{\bigodot}])$'
	plt.xlabel(xlab, fontsize=fsize) 
	plt.ylabel(ylab, fontsize=fsize)
	plt.plot([-1,-1],[-1,-2],c+ls,linewidth=lw, label=label) if type(ls)==str else plt.plot([-1,-1],[-1,-2],c,linewidth=lw, label=label, dashes=ls)
	if extra:
		x_obs = np.linspace(1,3,100)
		y_obs_arr = np.array([[4.01*x_obs + 2.05], [4.01*x_obs + 1.53], [3.87*x_obs + 2.05], [3.87*x_obs + 1.53]]) # Random + systematic
		#y_obs_arr = np.array([[4.02*x_obs + 2.04], [4.02*x_obs + 1.54], [3.86*x_obs + 2.04], [3.86*x_obs + 1.54]])
		y_obs_min = np.min(y_obs_arr, axis=0)[0] + 2*np.log10(0.75/h)
		y_obs_max = np.max(y_obs_arr, axis=0)[0] + 2*np.log10(0.75/h) # h=0.75 used in the Stark+ paper
		plt.fill_between(x_obs, y_obs_max, y_obs_min, color='purple', alpha=0.2)
		plt.plot([-1,-1],[-1,-2], color='purple', ls='-', lw=8, alpha=0.3, label=r'Stark et al.~(2009)')
		plt.legend(fontsize=fsize-6, loc='lower right', frameon=True)
	plt.axis([np.min(x)-0.1, np.max(x)+0.1, np.min(y)-0.1, np.max(y)+0.1])


def smf_nifty_obs(h=0.7, z=0, c='purple', haxes=True, fulldata=True, bc03=False):
    # Plot observed SMF that is used for the constraint for the nIFTy workshop.  Note how the h's are handled
    if z==0:
        if fulldata:
            data = np.array([[7.06520,      7.31520,     0.216271,    0.0755382],
                         [7.31520,      7.56520,     0.204020,    0.0712595],
                         [7.56520,      7.81520,     0.151049,    0.0527579],
                         [7.81520,      8.06520,     0.112048,    0.0391356],
                         [8.06520,      8.31520,    0.0762307,    0.0266256],
                         [8.31520,      8.56520,    0.0560568,    0.0135117],
                         [8.56520,      8.81520,    0.0423269,    0.0102622],
                         [8.81520,      9.06520,    0.0331105,   0.00837041],
                         [9.06520,      9.31520,    0.0256968,   0.00525948],
                         [9.31520,      9.56520,    0.0214038,   0.00473875],
                         [9.56520,      9.81520,    0.0195365,   0.00418913],
                         [9.81520,      10.0652,    0.0186109,   0.00360372],
                         [10.0652,      10.3152,    0.0152146,   0.00202494],
                         [10.3152,      10.5652,    0.0107248,   0.00209060],
                         [10.5652,      10.8152,    0.00517901,  0.000978219],
                         [10.8152,      11.0652,    0.00142977,  0.000428163],
                         [11.0652,      11.3152,    0.000139376,  0.000104390]])
        else:
            data = np.array([[9.31520,      9.56520,    0.0214038,   0.00473875],
                             [9.56520,      9.81520,    0.0195365,   0.00418913],
                             [9.81520,      10.0652,    0.0186109,   0.00360372],
                             [10.0652,      10.3152,    0.0152146,   0.00202494],
                             [10.3152,      10.5652,    0.0107248,   0.00209060],
                             [10.5652,      10.8152,    0.00517901,  0.000978219]])
    elif z==2:
        if fulldata:
            data = np.array([[8.56520,      8.81520,    0.0164884,   0.00565431],
                              [8.81520,      9.06520,    0.0107285,   0.00367909],
                              [9.06520,      9.31520,    0.00868969,   0.00297992],
                              [9.31520,      9.56520,    0.00651444,   0.00223397],
                              [9.56520,      9.81520,    0.00490829,   0.00168318],
                              [9.81520,      10.0652,    0.00371451,   0.00127380],
                              [10.0652,      10.3152,    0.00272899,  0.000935842],
                              [10.3152,      10.5652,    0.00170499,  0.000864231],
                              [10.5652,      10.8152,    0.000731956,  0.000387435],
                              [10.8152,      11.0652,    0.000167129,  0.000112849],
                              [11.0652,      11.3152,    2.60057e-05,  1.96621e-05]])
        else:
            data = np.array([[9.31520,      9.56520,    0.00651444,   0.00223397],
                             [9.56520,      9.81520,    0.00490829,   0.00168318],
                             [9.81520,      10.0652,    0.00371451,   0.00127380],
                             [10.0652,      10.3152,    0.00272899,  0.000935842],
                             [10.3152,      10.5652,    0.00170499,  0.000864231],
                             [10.5652,      10.8152,    0.000731956,  0.000387435]])
    """(m,n) = data.shape
    for i in xrange(m):
        plt.fill_between(data[i,0:2]/h, np.array([data[i,2],data[i,2]])+data[i,3], np.array([data[i,2],data[i,2]])-data[i,3], color='purple', alpha=0.3)"""
    #xdata = np.append(data[0,0], data[:,1]) - np.log10(h)
    #yhigh = np.append(data[0,2]+data[0,3], data[:,2]+data[:,3])
    #ylow = np.append(data[0,2]-data[0,3], data[:,2]-data[:,3])
    if haxes:
        xdata = (data[:,0]+data[:,1])/2 - np.log10(h)
        if bc03: xdata += 0.14
        yhigh = data[:,2]+data[:,3]
        ylow = data[:,2]-data[:,3]
        plt.fill_between(xdata, yhigh, ylow, color=c, alpha=0.3)
        plt.xlabel(r'$\log_{10}(M_{\mathrm{stars}}\ [h^{-1}\mathrm{M}_{\bigodot}])$')
        plt.ylabel(r'$\Phi\ [h^3\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
        plt.plot([0,1],[0,1], color=c, lw=8, alpha=0.3, label=r'Observations ($z\sim2$)')
        plt.axis([8.0, 11.999, 4e-4, 2e-1])
    else:
        xdata = (data[:,0]+data[:,1])/2 - 2*np.log10(h)
        if bc03: xdata += 0.14
        yhigh = (data[:,2]+data[:,3]) * h**3
        ylow = (data[:,2]-data[:,3]) * h**3
        plt.fill_between(xdata, yhigh, ylow, color=c, alpha=0.3)
        plt.xlabel(r'$\log_{10}(M_{\mathrm{stars}}\ [\mathrm{M}_{\bigodot}])$')
        plt.ylabel(r'$\Phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
        plt.plot([0,1],[0,1], color=c, lw=8, alpha=0.3, label=r'Observations')
        plt.plot([8,11,5e-6,0.2])

def passive_nifty_obs(h=0.7, z=0):
    if z==0:
        data = np.array([[7.63040,      7.88040,     0.210240,    0.0842091],
                          [7.88040,      8.13040,     0.140601,    0.0430960],
                          [8.13040,      8.38040,     0.179360,    0.0461142],
                          [8.38040,      8.63040,     0.216695,    0.0531785],
                          [8.63040,      8.88040,     0.306115,    0.0650996],
                          [8.88040,      9.13040,     0.384735,    0.0697630],
                          [9.13040,      9.38040,     0.466144,    0.0705114],
                          [9.38040,      9.63040,     0.546098,    0.0718740],
                          [9.63040,      9.88040,     0.609358,    0.0763614],
                          [9.88040,      10.1304,     0.687935,    0.0908160],
                          [10.1304,      10.3804,     0.755346,    0.0789445],
                          [10.3804,      10.6304,     0.865670,    0.0492323]])
    elif z==2:
        data = np.array([[8.94020,      9.19020,    0.0304564,    0.0190173],
                          [9.19020,     9.44020,    0.0555014,    0.0337603],
                          [9.44020,      9.69020,     0.115423,    0.0657547],
                          [9.69020,      9.94020,     0.224003,     0.111169],
                          [9.94020,      10.1902,     0.272901,     0.128215],
                          [10.1902,      10.4402,     0.267389,     0.151138],
                          [10.4402,      10.6902,     0.314906,     0.201316],
                          [10.6902,      10.9402,     0.406142,     0.234631],
                          [10.9402,      11.1902,     0.481301,     0.333481]])
    xdata = np.append(data[0,0], data[:,1]) - np.log10(h)
    yhigh = np.append(data[0,2]+data[0,3], data[:,2]+data[:,3])
    ylow = np.append(data[0,2]-data[0,3], data[:,2]-data[:,3])
    plt.fill_between(xdata, yhigh, ylow, color='purple', alpha=0.3)
    plt.xlabel(r'$\log_{10}(M_{\mathrm{stars}}\ [h^{-1}\mathrm{M}_{\bigodot}])$')
    plt.ylabel(r'Passive fraction')
    plt.plot([0,1],[-2,-1], color='purple', lw=8, alpha=0.3, label=r'Observations')
    plt.axis([np.min(xdata), np.max(xdata), 0, 1])


def red_fraction_magntiude(u, r, h=0.678, label=None, extra=False, c='k', ls='-', lw=2):
    # Expects u and r to be magntiudes-5*log(h)
    edge = np.arange(-23.5,-15,0.5) + 5*np.log10(h)
    rf = (u-r < 2.06-0.244*np.tanh((r+20.07/1.09)))
    Ntot, edge = np.histogram(r, bins=edge)
    Nred, edge = np.histogram(r[rf], bins=edge)
    red_frac = (1.0*Nred)/Ntot
    mag = (edge[1:]+edge[:-1])/2
    plt.plot(mag, red_frac, c+ls, lw=lw, label=label) if type(ls)==str else plt.plot(mag, red_frac, c, lw=lw, label=label, dashes=ls)
    plt.xlabel(r'$r$-band magnitude')
    plt.ylabel(r'Red fraction')
    if extra:
        rmax = np.array([ 1.000,  0.852,  0.642,  0.495,  0.44 ,  0.414,  0.391,  0.365,    0.321,  0.266,  0.204,  0.198,  0.178,  0.179,  0.123,  0.162])
        rmean = np.array([ 0.985,  0.801,  0.619,  0.482,  0.429,  0.405,  0.384,  0.356,   0.31 ,  0.254,  0.192,  0.182,  0.157,  0.147,  0.092,  0.123])
        rmin = np.array([ 0.821,  0.75 ,  0.596,  0.468,  0.417,  0.395,  0.374,  0.345,    0.298,  0.241,  0.179,  0.164,  0.135,  0.119,  0.061,  0.083])
        plt.fill_between(mag, rmax, rmin, color='purple', alpha=0.3)
        plt.plot([0,1], [0,1], '-', color='purple', lw=8, alpha=0.3, label=r'Baldry et al.~(2004)') # for legend
    plt.axis([edge[-1], edge[0], 0, 1])



def bhbulge(M_BH, M_bulge, extra=0, c='k', ls='-', lw=2, h=0.7, label=r'Input', fsize=28, Nbins=50, pcs=None, alpha=0.3):
	# Plot a black hole--bulge relation
	y, x = np.log10(M_BH), np.log10(M_bulge)
	contour(x,y,Nbins=Nbins,weights=None,range=None,Nlevels=5,c=c,ls=ls,lw=lw,pcs=pcs)
	xlab = r'$\log_{10} (M_{\rm bulge}\ [\mathrm{M}_{\bigodot}])$' if h!=1 else r'$\log_{10} (M_{\rm bulge} h^2\ [\mathrm{M}_{\bigodot}])$'
	ylab = r'$\log_{10} (M_{\rm BH}\ [\mathrm{M}_{\bigodot}])$' if h!=1 else r'$\log_{10} (M_{\rm BH} h^2\ [\mathrm{M}_{\bigodot}])$'
	plt.xlabel(xlab, fontsize=fsize) 
	plt.ylabel(ylab, fontsize=fsize)
	plt.plot([-1,-1],[-1,-2],c+ls,linewidth=lw, label=label) if type(ls)==str else plt.plot([-1,-1],[-1,-2],c,linewidth=lw, label=label, dashes=ls)
	if extra==1:
		x_obs = np.linspace(8,13,1e4)
		y_large = np.array([[0.83*(x_obs-np.log10(3e11))+9.18], [1.11*(x_obs-np.log10(3e11))+9.18], [0.83*(x_obs-np.log10(3e11))+9.36], [1.11*(x_obs-np.log10(3e11))+9.36]])
		y_large_min, y_large_max = np.min(y_large, axis=0)[0], np.max(y_large, axis=0)[0]
		y_small = np.array([[1.64*(x_obs-np.log10(2e10))+7.71], [2.8*(x_obs-np.log10(2e10))+7.71], [1.64*(x_obs-np.log10(2e10))+8.07], [2.8*(x_obs-np.log10(2e10))+8.07]])
		y_small_min, y_small_max = np.min(y_small, axis=0)[0], np.max(y_small, axis=0)[0]
		min_index = np.argwhere(y_small_min-y_large_min>=0)[0][0]
		max_index = np.argwhere(y_small_max-y_large_max>=0)[0][0]
		y_obs_min = np.append(y_small_min[:min_index], y_large_min[min_index:]) + 2*np.log10(0.7/h)
		y_obs_max = np.append(y_small_max[:max_index], y_large_max[max_index:]) + 2*np.log10(0.7/h)
		plt.fill_between(x_obs+2*np.log10(0.7/h), y_obs_max, y_obs_min, color='purple', alpha=0.2)
		plt.plot([-1,-1],[-1,-2], color='purple', ls='-', lw=8, alpha=0.3, label=r'Scott et al.~(2013)')
		plt.legend(fontsize=fsize-6, loc='lower right', frameon=False)
	elif extra==2:
		BH_bulge_obs(h)
		plt.legend(fontsize=fsize-6, loc='lower right', frameon=True, numpoints=1)
	elif extra==3:
		data = BH_bulge_carnage(h)
		plt.errorbar(data[:,0], data[:,2], xerr=data[:,1], yerr=[data[:,3],data[:,4]], color='purple', alpha=0.5, lw=2, ms=0, ls='none')
	plt.axis([np.min(x), np.max(x)+0.1, np.min(y)-0.1, np.max(y)+0.1])
	
def BH_bulge_obs(h=0.678):
	M_BH_obs = (0.7/h)**2*1e8*np.array([39, 11, 0.45, 25, 24, 0.044, 1.4, 0.73, 9.0, 58, 0.10, 8.3, 0.39, 0.42, 0.084, 0.66, 0.73, 15, 4.7, 0.083, 0.14, 0.15, 0.4, 0.12, 1.7, 0.024, 8.8, 0.14, 2.0, 0.073, 0.77, 4.0, 0.17, 0.34, 2.4, 0.058, 3.1, 1.3, 2.0, 97, 8.1, 1.8, 0.65, 0.39, 5.0, 3.3, 4.5, 0.075, 0.68, 1.2, 0.13, 4.7, 0.59, 6.4, 0.79, 3.9, 47, 1.8, 0.06, 0.016, 210, 0.014, 7.4, 1.6, 6.8, 2.6, 11, 37, 5.9, 0.31, 0.10, 3.7, 0.55, 13, 0.11])
	M_BH_hi = (0.7/h)**2*1e8*np.array([4, 2, 0.17, 7, 10, 0.044, 0.9, 0.0, 0.9, 3.5, 0.10, 2.7, 0.26, 0.04, 0.003, 0.03, 0.69, 2, 0.6, 0.004, 0.02, 0.09, 0.04, 0.005, 0.2, 0.024, 10, 0.1, 0.5, 0.015, 0.04, 1.0, 0.01, 0.02, 0.3, 0.008, 1.4, 0.5, 1.1, 30, 2.0, 0.6, 0.07, 0.01, 1.0, 0.9, 2.3, 0.002, 0.13, 0.4, 0.08, 0.5, 0.03, 0.4, 0.38, 0.4, 10, 0.2, 0.014, 0.004, 160, 0.014, 4.7, 0.3, 0.7, 0.4, 1, 18, 2.0, 0.004, 0.001, 2.6, 0.26, 5, 0.005])
	M_BH_lo = (0.7/h)**2*1e8*np.array([5, 2, 0.10, 7, 10, 0.022, 0.3, 0.0, 0.8, 3.5, 0.05, 1.3, 0.09, 0.04, 0.003, 0.03, 0.35, 2, 0.6, 0.004, 0.13, 0.1, 0.05, 0.005, 0.2, 0.012, 2.7, 0.06, 0.5, 0.015, 0.06, 1.0, 0.02, 0.02, 0.3, 0.008, 0.6, 0.5, 0.6, 26, 1.9, 0.3, 0.07, 0.01, 1.0, 2.5, 1.5, 0.002, 0.13, 0.9, 0.08, 0.5, 0.09, 0.4, 0.33, 0.4, 10, 0.1, 0.014, 0.004, 160, 0.007, 3.0, 0.4, 0.7, 1.5, 1, 11, 2.0, 0.004, 0.001, 1.5, 0.19, 4, 0.005])
	M_sph_obs = (0.7/h)**2*1e10*np.array([69, 37, 1.4, 55, 27, 2.4, 0.46, 1.0, 19, 23, 0.61, 4.6, 11, 1.9, 4.5, 1.4, 0.66, 4.7, 26, 2.0, 0.39, 0.35, 0.30, 3.5, 6.7, 0.88, 1.9, 0.93, 1.24, 0.86, 2.0, 5.4, 1.2, 4.9, 2.0, 0.66, 5.1, 2.6, 3.2, 100, 1.4, 0.88, 1.3, 0.56, 29, 6.1, 0.65, 3.3, 2.0, 6.9, 1.4, 7.7, 0.9, 3.9, 1.8, 8.4, 27, 6.0, 0.43, 1.0, 122, 0.30, 29, 11, 20, 2.8, 24, 78, 96, 3.6, 2.6, 55, 1.4, 64, 1.2])
	M_sph_hi = (0.7/h)**2*1e10*np.array([59, 32, 2.0, 80, 23, 3.5, 0.68, 1.5, 16, 19, 0.89, 6.6, 9, 2.7, 6.6, 2.1, 0.91, 6.9, 22, 2.9, 0.57, 0.52, 0.45, 5.1, 5.7, 1.28, 2.7, 1.37, 1.8, 1.26, 1.7, 4.7, 1.7, 7.1, 2.9, 0.97, 7.4, 3.8, 2.7, 86, 2.1, 1.30, 1.9, 0.82, 25, 5.2, 0.96, 4.9, 3.0, 5.9, 1.2, 6.6, 1.3, 5.7, 2.7, 7.2, 23, 5.2, 0.64, 1.5, 105, 0.45, 25, 10, 17, 2.4, 20, 67, 83, 5.2, 3.8, 48, 2.0, 55, 1.8])
	M_sph_lo = (0.7/h)**2*1e10*np.array([32, 17, 0.8, 33, 12, 1.4, 0.28, 0.6, 9, 10, 0.39, 2.7, 5, 1.1, 2.7, 0.8, 0.40, 2.8, 12, 1.2, 0.23, 0.21, 0.18, 2.1, 3.1, 0.52, 1.1, 0.56, 0.7, 0.51, 0.9, 2.5, 0.7, 2.9, 1.2, 0.40, 3.0, 1.5, 1.5, 46, 0.9, 0.53, 0.8, 0.34, 13, 2.8, 0.39, 2.0, 1.2, 3.2, 0.6, 3.6, 0.5, 2.3, 1.1, 3.9, 12, 2.8, 0.26, 0.6, 57, 0.18, 13, 5, 9, 1.3, 11, 36, 44, 2.1, 1.5, 26, 0.8, 30, 0.7])
	core = np.array([1,1,0,1,1,0,0,0,1,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,1,0,1,0,1,1,1,0,0,1,0,1,0])
	yerr2, yerr1 = gc.logten(gc.divide(M_BH_obs+M_BH_hi, M_BH_obs)), -gc.logten(gc.divide(M_BH_obs-M_BH_lo, M_BH_obs))
	xerr2, xerr1 = gc.logten(gc.divide(M_sph_obs+M_sph_hi, M_sph_obs)), -gc.logten(gc.divide(M_sph_obs-M_sph_lo, M_sph_obs))
	plt.errorbar(np.log10(M_sph_obs[core==0]), np.log10(M_BH_obs[core==0]), yerr=[yerr1[core==0],yerr2[core==0]], xerr=[xerr1[core==0],xerr2[core==0]], color='purple', alpha=0.3, label=r'S13 core', ls='none', lw=2, ms=0)
	plt.errorbar(np.log10(M_sph_obs[core==1]), np.log10(M_BH_obs[core==1]), yerr=[yerr1[core==1],yerr2[core==1]], xerr=[xerr1[core==1],xerr2[core==1]], color='c', alpha=0.5, label=r'S13 S\`{e}rsic', ls='none', lw=2, ms=0)


def BH_bulge_carnage(h=0.678):
    k13 = np.array([[9.05000, 0.10000, 6.38917, 0.38242, 0.38057],
                 [11.84000, 0.09000, 8.22789, 0.33559, 0.75077],
                 [11.27000, 0.09000, 9.16732, 0.33099, 0.86629],
                 [10.65000, 0.09000, 8.77085, 0.32292, 1.06328],
                 [11.50000, 0.09000, 8.94498, 0.48007, 0.29564],
                 [11.74000, 0.09000, 9.66745, 0.33385, 1.05467],
                 [11.33000, 0.09000, 9.58771, 0.33398, 0.73645],
                 [10.26000, 0.09000, 7.16137, 0.47812, 0.00000],
                 [11.06000, 0.09000, 7.03342, 0.30900, 1.33445],
                 [11.61000, 0.09000, 9.57054, 0.30740, 0.86297],
                 [10.50000, 0.09000, 8.25042, 0.40279, 0.28194],
                 [10.91000, 0.09000, 8.61909, 0.35218, 0.60206],
                 [11.26000, 0.09000, 8.13672, 0.36707, 0.46462],
                 [11.01000, 0.09000, 8.66745, 0.34496, 0.67182],
                 [11.77000, 0.09000, 9.95856, 0.35298, 0.50986],
                 [11.65000, 0.09000, 8.72346, 0.34287, 0.69003],
                 [10.85000, 0.09000, 8.99034, 0.36339, 0.50179],
                 [11.62000, 0.09000, 8.96614, 0.32278, 1.02662],
                 [11.51000, 0.09000, 7.11394, 1.26080, 0.00000],
                 [10.88000, 0.09000, 7.84261, 0.34066, 0.71550],
                 [11.84000, 0.09000, 9.40483, 0.34798, 1.40483],
                 [10.85000, 0.09000, 7.95424, 0.39794, 0.30103],
                 [11.72000, 0.09000, 9.78888, 0.31424, 1.22067],
                 [10.04000, 0.09000, 7.15836, 0.37439, 0.44236],
                 [9.64000, 0.10000, 8.77815, 0.39794, 0.47712],
                 [11.64000, 0.09000, 9.67394, 0.34642, 0.65275],
                 [10.97000, 0.09000, 8.30535, 0.35266, 0.60638],
                 [11.16000, 0.09000, 9.23300, 0.31355, 0.95424],
                 [12.09000, 0.09000, 10.30103, 0.45179, 0.12205],
                 [11.28000, 0.09000, 8.93197, 0.39946, 0.28069],
                 [11.05000, 0.09000, 7.75511, 0.33901, 0.73808],
                 [11.65000, 0.09000, 9.67669, 0.33950, 0.38889],
                 [11.60000, 0.09000, 9.56703, 0.30688, 0.54999],
                 [11.00000, 0.09000, 8.43616, 0.35201, 0.53854],
                 [10.57000, 0.09000, 8.68753, 0.36439, 0.50284],
                 [11.69000, 0.09000, 9.57287, 0.39280, 0.51217],
                 [11.88000, 0.09000, 8.78817, 0.36777, 0.47641],
                 [11.25000, 0.09000, 9.32222, 0.36173, 1.32222],
                 [11.61000, 0.10000, 8.59770, 0.43088, 0.40457],
                 [11.65000, 0.09000, 9.36173, 0.39794, 1.32034],
                 [11.75000, 0.09000, 9.12710, 0.37669, 0.51432],
                 [11.60000, 0.09000, 9.39445, 0.34115, 1.11570],
                 [11.81000, 0.10000, 9.57287, 0.32476, 0.85687],
                 [11.78000, 0.09000, 9.11394, 0.33321, 0.83519],
                 [10.35000, 0.09000, 8.15534, 0.42101, 0.66397],
                 [10.42000, 0.09000, 7.81291, 0.37742, 0.63682],
                 [11.26000, 0.09000, 8.93802, 0.32396, 1.27526],
                 [10.98000, 0.09000, 8.21748, 0.38890, 0.35416],
                 [10.53000, 0.09000, 7.61595, 0.32307, 0.99270],
                 [10.64000, 0.09000, 7.85003, 0.31104, 1.34488],
                 [11.00000, 0.09000, 10.23045, 0.33775, 0.75333],
                 [9.71000, 0.09000, 7.16137, 0.32999, 0.10446],
                 [10.92000, 0.09000, 8.95279, 0.31461, 0.51031],
                 [10.69000, 0.09000, 8.37840, 0.32489, 0.49758],
                 [11.26000, 0.09000, 8.51720, 0.38752, 0.75377],
                 [10.67000, 0.09000, 8.92686, 0.31866, 1.10731],
                 [10.67000, 0.09000, 8.35603, 0.38827, 0.43175],
                 [10.33000, 0.09000, 8.25527, 0.36798, 0.71120],
                 [9.86000, 0.09000, 7.57749, 0.30332, 1.97543],
                 [10.31000, 0.09000, 8.65610, 0.41246, 0.48584],
                 [11.02000, 0.09000, 8.65418, 0.36369, 0.64134],
                 [10.38000, 0.09000, 7.94498, 0.35844, 0.55937],
                 [11.47000, 0.09000, 8.82282, 0.31390, 1.21004],
                 [10.20000, 0.09000, 7.88480, 0.39556, 0.37425],
                 [9.56000, 0.09000, 6.95424, 0.41311, 0.22185],
                 [10.09000, 0.10000, 6.63347, 0.31884, 1.07717],
                 [9.63000, 0.14000, 6.05690, 0.33755, 0.75587],
                 [10.92000, 0.10000, 6.92376, 0.31227, 1.28031],
                 [9.84000, 0.10000, 7.87795, 0.47133, 0.31447],
                 [10.08000, 0.09000, 6.93500, 0.31248, 1.27225],
                 [9.41000, 0.10000, 7.64738, 0.37956, 0.38731],
                 [9.78000, 0.09000, 7.60959, 0.32186, 0.89359],
                 [9.99000, 0.09000, 7.32222, 0.36709, 0.27300],
                 [10.26000, 0.09000, 6.88423, 0.34237, 0.69954],
                 [10.34000, 0.09000, 7.03342, 0.38982, 0.34323],
                 [10.48000, 0.09000, 7.19590, 0.41900, 0.20026],
                 [10.11000, 0.09000, 6.77379, 0.33070, 0.85471],
                 [10.50000, 0.09000, 6.94448, 0.59081, 0.00000],
                 [9.41000, 0.10000, 6.86392, 0.30605, 1.60864],
                 [10.13000, 0.10000, 6.83059, 0.34839, 0.63746],
                 [10.14000, 0.09000, 6.19312, 0.35218, 0.60206],
                 [9.35000, 0.12000, 6.13033, 0.39858, 0.44909],
                 [10.36000, 0.09000, 7.48855, 0.30384, 1.88649],
                 [9.94000, 0.09000, 7.00432, 0.30531, 2.00432],
                 [10.02000, 0.10000, 7.74115, 0.34946, 0.76343],
                 [10.12000, 0.09000, 6.69984, 0.33076, 0.00000],
                 [10.39000, 0.09000, 6.98453, 0.33458, 0.79420]])
    k13[:,0] += 2*np.log10(0.705/h)
    k13[:,2] += 2*np.log10(0.705/h)


    m13 = np.array([[11.487139, 0.2500,  9.447158,  0.143907,  0.243038],
                [8.881955,  0.2500,  6.414973,  0.076388,  0.092754],
                [11.283301, 0.2500,  8.230449,  0.149762,  0.230449],
                [10.812244, 0.2500,  7.602060,  0.041393,  0.045757],
                [10.762678, 0.2500,  8.770852,  0.042061,  0.038458],
                [11.599883, 0.2500,  8.707570,  0.048305,  0.064117],
                [11.599883, 0.2500,  9.113943,  0.141330,  0.335792],
                [12.000000, 0.2500,  9.672098,  0.060295,  0.048849],
                [10.298853, 0.2500,  7.146128,  0.029963,  0.146128],
                [11.195900, 0.2500,  8.949390,  0.196737,  0.156999],
                [10.845098, 0.2500,  8.322219,  0.092754,  0.146128],
                [10.371068, 0.2500,  8.255273,  0.176091,  0.301030],
                [10.836324, 0.2500,  8.623249,  0.092754,  0.131887],
                [10.278753, 0.2500,  7.041393,  0.162727,  0.263242],
                [11.204120, 0.2500,  8.518514,  0.162727,  0.087150],
                [10.884229, 0.2500,  8.672098,  0.083776,  0.103896],
                [12.190331, 0.2500,  9.986772,  0.117032,  0.129439],
                [10.448707, 0.2500,  8.255273,  0.124938,  0.079182],
                [11.916980, 0.2500,  8.724276,  0.081904,  0.101027],
                [10.998260, 0.2500,  8.991226,  0.119364,  0.165152],
                [10.255273, 0.2500,  8.662758,  0.194574,  0.171396],
                [11.558708, 0.2500,  8.963788,  0.044812,  0.039509],
                [11.953277, 0.2500,  9.397940,  0.093422,  0.017729],
                [11.206826, 0.2500,  7.949390,  0.177715,  0.296177],
                [12.117271, 0.2500,  9.792392,  0.020521,  0.028964],
                [10.668386, 0.2500,  7.944483,  0.104735,  0.138303],
                [11.887617, 0.2500,  9.672098,  0.001015,  0.000925],
                [11.110590, 0.2500,  8.301030,  0.041392,  0.045757],
                [12.243038, 0.2500,  10.32221,  0.245983,  0.544067],
                [11.563481, 0.2500,  8.903090,  0.210854,  0.230991],
                [10.981365, 0.2500,  8.230449,  0.070581,  0.116506],
                [10.526340, 0.2500,  8.690196,  0.115984,  0.171682],
                [12.155336, 0.2500,  9.579783,  0.160580,  0.164810],
                [11.748188, 0.2500,  8.778152,  0.124938,  0.176091],
                [11.544068, 0.2500,  8.602060,  0.230449,  0.221849],
                [12.064458, 0.2500,  9.113943,  0.141330,  0.159700]])
    m13[:,0] += 2*np.log10(0.7/h)
    m13[:,2] += 2*np.log10(0.7/h)

    data = np.append(k13,m13,axis=0)
    return data


def massmet(M_star, lOH, extra=False, c='k', ls='-', lw=2, h=0.7, label=r'Input', fsize=28, Nbins=50, pcs=None):
	# stellar mass -- metallicity relationship.  lOH = 12+log(O/H)
	logM = np.log10(M_star)
	contour(logM,lOH,Nbins=Nbins,weights=None,Nlevels=5,c=c,ls=ls,lw=lw,pcs=pcs)
	#contour(logM,lOH,Nbins=Nbins,weights=None,range=[[8,11.5],[7.5,10.5]],Nlevels=5,c=c,ls=ls,lw=lw,pcs=pcs)
	plt.plot([-1,-1],[-1,-2],c+ls,linewidth=lw, label=label) if type(ls)==str else plt.plot([-1,-1],[-1,-2],c,linewidth=lw, label=label, dashes=ls)
	xlab = r'$\log_{10}(M_{\mathrm{stars}}\ [\mathrm{M}_{\bigodot}])$' if h!=1 else r'$\log_{10}(M_{\mathrm{stars}} h^2\ [\mathrm{M}_{\bigodot}])$'
	ylab = r'$12 + \log_{10}(\mathrm{O} / \mathrm{H})$'
	plt.xlabel(xlab, fontsize=fsize)
	plt.ylabel(ylab, fontsize=fsize)
	if extra:
		"""x_obs = np.linspace(8.5,11,1e4)
		y_obs = -1.492 + 1.847*x_obs - 0.08026*x_obs**2
		x_obs += 2*np.log10(0.7/h)
		plt.fill_between(x_obs, y_obs+0.1, y_obs-0.1, color='b', alpha=0.3)"""
		x_obs, y_low, y_high = gr.Tremonti04(h)
		plt.fill_between(x_obs, y_high, y_low, color='purple', alpha=0.2)
		plt.plot([-1,-1],[-1,-2], color='purple', ls='-', lw=8, alpha=0.3, label=r'Tremonti et al.~(2004)')
		plt.legend(fontsize=fsize-6, loc='lower right', frameon=True)
		plt.xlim(np.min(x_obs), np.max(x_obs))
	plt.ylim(7.999,10)


def quiescent(M_star, SFR, sSFRcut=1e-11, c='k', ls='-', lw=2, label=r'Input', fsize=28, Nbins=30, h=0.7, extra=False, range=None, addh=False):
	M_star, SFR = M_star[M_star>0], SFR[M_star>0]
	sSFR = SFR/M_star
	logM = np.log10(M_star)
	if range==None: range=[8.5,np.max(logM)]
	Ntot, edge = np.histogram(logM, bins=Nbins, range=range) if not addh else np.histogram(logM+np.log10(h), bins=Nbins, range=range)
	Nred, edge = np.histogram(logM[sSFR<sSFRcut], bins=Nbins, range=range) if not addh else np.histogram(logM[sSFR<sSFRcut]+np.log10(h), bins=Nbins, range=range)
	logMplot = (edge[1:]+edge[:-1])/2.
	plt.plot(logMplot, gc.divide(1.0*Nred,1.0*Ntot), c+ls, linewidth=lw, label=label) if type(ls)==str else plt.plot(logMplot, gc.divide(1.0*Nred,Ntot), c+'-', linewidth=lw, label=label, dashes=ls)
	xlab = r'$\log_{10}(M_{\mathrm{stars}}\ [\mathrm{M}_{\bigodot}])$' if h!=1 else r'$\log_{10}(M_{\mathrm{stars}} h^2\ [\mathrm{M}_{\bigodot}])$'
	ylab = r'Quiescent fraction'
	plt.xlabel(xlab, fontsize=fsize)
	plt.ylabel(ylab, fontsize=fsize)
	plt.axis([range[0], range[1], 0, 1])
	if extra:
		"""
		obs = np.array([[  7.6304   ,   7.8804   ,   0.21024  ,   0.0842091],
               [  7.8804   ,   8.1304   ,   0.140601 ,   0.043096 ],
               [  8.1304   ,   8.3804   ,   0.17936  ,   0.0461142],
               [  8.3804   ,   8.6304   ,   0.216695 ,   0.0531785],
               [  8.6304   ,   8.8804   ,   0.306115 ,   0.0650996],
               [  8.8804   ,   9.1304   ,   0.384735 ,   0.069763 ],
               [  9.1304   ,   9.3804   ,   0.466144 ,   0.0705114],
               [  9.3804   ,   9.6304   ,   0.546098 ,   0.071874 ],
               [  9.6304   ,   9.8804   ,   0.609358 ,   0.0763614],
               [  9.8804   ,  10.1304   ,   0.687935 ,   0.090816 ],
               [ 10.1304   ,  10.3804   ,   0.755346 ,   0.0789445],
               [ 10.3804   ,  10.6304   ,   0.86567  ,   0.0492323]])
		plt.fill_between((obs[:,0]+obs[:,1])/2 - 2*np.log10(h), obs[:,2]+obs[:,3], obs[:,2]-obs[:,3], color='purple', alpha=0.5)
		"""
        #data = np.loadtxt('/Users/astevens/Dropbox/Swinburne Shared/SAGE plots/obs_data_Toby_full.dat', skiprows=1)
        #red = data[:,18]
        #logM = data[:,11] + 2*np.log10(0.7/h)
        #sSFR = 10**data[:,16]
		
		data = np.loadtxt('/Users/astevens/Dropbox/Swinburne Shared/SAGE plots/mpa_jhu_ms_sfr.dat', dtype=str)
		logM = np.array(data[:,25], dtype='f8') + 2*np.log10(0.7/h)
		SFR = 10**np.array(data[:,35], dtype='f8')
		sSFR = SFR / 10**logM
      
		Nbins = 25
		bins = np.linspace(9, 11.5, Nbins+1) + 2*np.log10(0.7/h)
		N, v, Ntot = np.zeros(Nbins, dtype=int), np.zeros(Nbins), np.zeros(Nbins, dtype=int)
		Ntot_obs, bins = np.histogram(logM, bins=bins)
		Nqui_obs, bins = np.histogram(logM[sSFR<sSFRcut], bins=bins)
		      #Nred_obs, bins = np.histogram(logM[red==1], bins=bins)
		      #redfrac_obs = (1.0*Nred_obs) / Ntot_obs
		quifrac_obs = (1.0*Nqui_obs) / Ntot_obs
		yerr = np.sqrt(Nqui_obs)/Ntot_obs
		xplot = (bins[1:]+bins[:-1])/2
		plt.fill_between(xplot, quifrac_obs+yerr, quifrac_obs-yerr, color='purple', alpha=0.5)



def quiescenthalo(M_star, SFR, M_halo, sSFRcut=1e-11, c='k', ls='-', lw=2, label=r'Input', fsize=28, Nbins=30, h=0.7, extra=False):
	M_star, SFR, M_halo = M_star[M_star>0], SFR[M_star>0], M_halo[M_star>0]
	sSFR = SFR/M_star
	M_halo /= (0.745 - 0.0006*(np.log10(M_halo)-7.0)**2.45)
	logM = np.log10(M_halo)
	Ntot, edge = np.histogram(logM, bins=Nbins, range=[max(12,np.min(logM)),min(15,np.max(logM))])
	Nred, edge = np.histogram(logM[sSFR<sSFRcut], bins=Nbins, range=[max(12,np.min(logM)),min(15,np.max(logM))])
	logMplot = (edge[1:]+edge[:-1])/2.
	plt.plot(logMplot, (1.0*Nred)/Ntot, c+ls, linewidth=lw, label=label)
	xlab = r'$\log_{10}(M_{\mathrm{halo}}\ [\mathrm{M}_{\bigodot}])$' if h!=1 else r'$\log_{10}(M_{\mathrm{vir}} h^2\ [\mathrm{M}_{\bigodot}])$'
	ylab = r'Quiescent fraction'
	plt.xlabel(xlab, fontsize=fsize)
	plt.ylabel(ylab, fontsize=fsize)
	if extra:
		x_obs = np.array([12, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15]) + np.log10(0.7) - 2*np.log10(h)
		y_high = 1-np.array([0.48, 0.48, 0.43, 0.38, 0.26, 0.22, 0.17, 0.17])
		y_low = 1-np.array([0.74, 0.74, 0.61, 0.56, 0.38, 0.27, 0.35, 0.35])
        
		bin1 = np.array([[0.74, 0.68, 0.67, 0.60, 0.53, 0.54, 0.48],
                         [69, 108, 186, 308, 449, 437, 66]])
		bin2 = np.array([[0.61, 0.56, 0.54, 0.45, 0.47, 0.43, 0.43],
                         [114, 189, 343, 489, 1101, 1235, 389]])
		bin3 = np.array([[0.56, 0.48, 0.44, 0.44, 0.39, 0.39, 0.38, 0.38],
                         [80, 133, 265, 449, 987, 1425, 2005, 810]])
		bin4 = np.array([[0.38, 0.28, 0.33, 0.30, 0.31, 0.32, 0.29, 0.26],
                         [84, 213, 392, 878, 1172, 1624, 1568, 408]])
		bin5 = np.array([[0.23, 0.22, 0.27, 0.23, 0.23, 0.24, 0.26, 0.23],
                         [74, 96, 246, 654, 771, 1057, 1000, 621]])
		bin6 = np.array([[0.24, 0.35, 0.20, 0.17, 0.23, 0.22, 0.18],
                         [67, 98, 208, 296, 380, 368, 283]])
            
		y_obs = 1.0-np.array([ np.sum(bin1[0,:]*bin1[1,:])/np.sum(bin1[1,:]),  np.sum(bin2[0,:]*bin2[1,:])/np.sum(bin2[1,:]), np.sum(bin3[0,:]*bin3[1,:])/np.sum(bin3[1,:]), np.sum(bin4[0,:]*bin4[1,:])/np.sum(bin4[1,:]), np.sum(bin5[0,:]*bin5[1,:])/np.sum(bin5[1,:]), np.sum(bin6[0,:]*bin6[1,:])/np.sum(bin6[1,:]) ])
		x_obs = x_obs[1:-1]
        
		#plt.fill_between(x_obs, y_high, y_low, color='purple', alpha=0.2)
		#plt.plot([-1,-1],[-1,-2], color='purple', ls='-', lw=8, alpha=0.3, label=r'Weinmann et al.~(2006)')
		plt.plot(x_obs, y_obs, color='purple', ls='-', lw=8, alpha=0.3, label=r'Weinmann et al.~(2006)')
		plt.legend(fontsize=fsize-6, loc='lower right', frameon=False)
		#plt.axis([12+np.log10(0.7/h**2), np.max(logMplot), 0, 1])
		plt.axis([12.25+np.log10(0.7/h**2), 14.75+np.log10(0.7/h**2), 0, 1])


def bulgedom(M_star, M_bulge, c='k', ls='-', lw=2, label=r'Input', Nbins=30, h=0.7, extra=False, range=None):
	M_star, M_bulge = M_star[M_star>0], M_bulge[M_star>0]
	logM = np.log10(M_star)
	filt = (M_bulge > 0.7*M_star)
	if range==None: range=[7.76-np.log10(h), 11.7-np.log10(h)]
	Ntot, edge = np.histogram(logM, bins=Nbins, range=range)
	Ndom, edge = np.histogram(logM[filt], bins=Nbins, range=range)
	logMplot = (edge[1:]+edge[:-1])/2.
	plt.plot(logMplot, gc.divide(1.0*Ndom,Ntot), c+ls, linewidth=lw, label=label)
	if extra:
		data = np.array([[7.76220,      8.29020,  0.130,	0.160],
                         [8.29020,      8.78970,  0.189,    0.095],
                         [8.78970,      9.27370,  0.137,	0.077],
                         [9.27370,      9.76970,  0.218,	0.053],
                         [9.76970,      10.2592,  0.288,	0.039],
                         [10.2592,      10.7187,  0.421,	0.030],
                         [10.7187,      11.1882,  0.631,	0.040],
                         [11.1882,      11.6922,  0.926,	0.082]])
		xdata = np.append(data[0,0], data[:,1]) - np.log10(h)
		yhigh = np.append(data[0,2]+data[0,3], data[:,2]+data[:,3])
		ylow = np.append(data[0,2]-data[0,3], data[:,2]-data[:,3])
		plt.fill_between(xdata, yhigh, ylow, color='purple', alpha=0.3)
		plt.xlabel(r'$\log_{10}(M_{\mathrm{stars}}\ [h^{-1}\mathrm{M}_{\bigodot}])$')
		plt.ylabel(r'Bulge-dominated fraction')
		plt.plot([0,1],[-2,-1], color='purple', lw=8, alpha=0.3, label=r'Observations')
		plt.axis([7.76-np.log10(h), 11.69-np.log10(h), 0, 1])



def coolinglum(Vvir, Cooling, extra=0, Nbins=50, ls='x', lw=2, c='k', label=r'Input', fsize=28, points=0, ms=6, alpha=1.0, mew=1):
	# Plot cooling luminosity for SAGE galaxies.  Vvir and Cooling are direct values from the SAGE outputs.
	HaloTemp = 35.9*(Vvir*Vvir) / 11604.5 / 1.0e3
	CoolingLum = 10**(Cooling - 40.0)
	HaloTemp, CoolingLum = gc.logten(HaloTemp), gc.logten(CoolingLum)
	print 'average log cooling lum = ', np.mean(CoolingLum[HaloTemp>-0.3])
	filt = (HaloTemp>-0.3)*(CoolingLum>-1) * (HaloTemp<1.1)*(CoolingLum<6)
	HaloTemp, CoolingLum = HaloTemp[filt], CoolingLum[filt]
    #print 'N plotted for coolinglum =', len(HaloTemp)
	#print len(HaloTemp), len(HaloTemp[HaloTemp>0.3]), len(HaloTemp[filt])
	if len(HaloTemp)==0: return

	
	if points>0:
		inds = np.arange(len(HaloTemp))
		if len(inds) > points:
			inds = sample(inds, points)
		print 'points in axes/printed/requested for coolinglum is', len(HaloTemp), len(HaloTemp[inds]), points
		plt.plot(HaloTemp[inds], CoolingLum[inds], ls, c=c, ms=ms, alpha=alpha, mew=mew, label=label) if label is not None else plt.plot(HaloTemp[inds], CoolingLum[inds], ls, c=c, ms=ms, alpha=alpha, mew=mew)
	else:
		contour(HaloTemp,CoolingLum,Nbins=Nbins,weights=None,range=None,Nlevels=5,c=c,ls=ls,lw=lw)
		plt.plot([-5,-5],[-5,-4],c+ls,linewidth=lw, label=label)
	plt.axis([-0.3,1.1,-1,6])
	plt.ylabel(r'log$_{10}$(Net cooling [$10^{40}$ erg s$^{-1}$])', fontsize=fsize)
	plt.xlabel(r'$\log_{10}(T_{\rm vir}\ [\mathrm{keV}])$', fontsize=fsize)
	
	if extra>0:
		Obs = np.array([
		        [6.2, 2.7, 3.7, 0.2, 6.2, 4.8, 1.1, 1.2, 46.0, 58.0            ], 
		        [0.0, 0.0, 0.0, 0.0, 5.1, 0.0, 0.0, 0.0, -1.0, 0.0             ], 
		        [0.0, 0.0, 0.0, 0.0, 2.4, 0.3, 0.02, 0.02, -1.0, 131.0         ],
		        [3.6, 0.3, 0.7, 0.1, 3.6, 0.5, 0.03, 0.04, -1.0, -1.0          ],
		        [5.8, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0             ],
		        [0.0, 0.0, 0.0, 0.0, 7.8, 0.9, 1.6, 0.9, -1.0, 0.0             ],
		        [4.1, 7.8, 0.3, 0.7, 4.1, 7.0, 1.2, 1.2, 690.0, -1.0           ], 
		        [5.5, 6.8, 999.9, 0.0, 5.5, 13.7, 0.4, 0.4, 42000.0, 21200.0   ], 
		        [3.0, 3.8, 1.3, 0.3, 3.0, 5.0, 0.3, 0.3, -1.0, 24.1            ],
		        [0.0, 0.0, 0.0, 0.0, 5.5, 0.4, 1.0, 0.4, -1.0, -1.0            ],
		        [6.8, 14.5, 0.5, 0.3, 6.8, 17.3, 1.7, 2.7, -1.0, 35.5          ],
		        [6.2, 0.1, 0.03, 0.03, 6.2, 0.0, 0.7, 0.0, -1.0, -1.0          ],
		        [4.7, 1.9, 0.2, 0.1, 4.7, 2.0, 0.1, 0.2, 44.0, -1.0            ], 
		        [0.0, 0.0, 0.0, 0.0, 5.2, 0.0, 0.1, 0.0, 1922.0, -1.0          ], 
		        [4.3, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0             ],
		        [8.5, 34.9, 1.8, 1.8, 8.5, 43.6, 4.6, 2.8, 480.0, 2370.0       ],
		        [6.6, 4.5, 0.6, 1.0, 6.6, 4.1, 2.3, 0.6, -1.0, -1.0            ],
		        [8.7, 0.0, 0.0, 0.0, 8.7, 0.0, 0.7, 0.0, -1.0, 9.0             ],
		        [3.8, 4.7, 999.9, 0.0, 3.8, 4.7, 1.0, 1.2, 14000.0, 40800.0    ], 
		        [3.3, 0.1, 0.01, 0.01, 3.3, 0.15, 0.04, 0.1, -1.0, 0.0         ], 
		        [0.0, 0.0, 0.0, 0.0, 3.5, 0.0, 0.0, 0.0, -1.0, 0.0             ],
		        [2.4, 0.1, 999.9, 0.0, 2.4, 0.3, 0.01, 0.01, 61000.0, 22400.0  ],
		        [3.6, 0.4, 999.9, 0.0, 3.6, 0.5, 0.1, 0.1, 1530.0, -1.0        ],
		        [8.0, 0.01, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 84.0, 207.0          ],
		        [4.7, 0.2, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 112.0, 99.0           ], 
		        [0.0, 0.0, 0.0, 0.0, 4.4, 0.0, 0.3, 0.0, 440.0, 1160.0         ], 
		        [5.5, 5.4, 4.1, 0.9, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0            ],
		        [0.0, 0.0, 0.0, 0.0, 7.0, 3.3, 1.0, 1.0, -1.0, -1.0            ],
		        [10.1, 18.0, 1.0, 1.0, 10.1, 26.1, 9.5, 1.5, -1.0, 0.0         ],
		        [4.6, 0.02, 0.03, 999.9, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0         ],
		        [6.5, 0.8, 0.4, 0.6, 6.5, 1.0, 0.9, 0.2, -1.0, 4.5             ], 
		        [0.0, 0.0, 0.0, 0.0, 3.8, 0.5, 0.3, 0.3, -1.0, 0.0             ], 
		        [7.6, 1.8, 0.6, 0.7, 7.6, 1.6, 1.2, 0.5, -1.0, 8.4             ],
		        [5.1, 10.4, 0.4, 0.4, 5.1, 8.5, 0.6, 0.2, 260.0, 930.0         ],
		        [7.8, 16.8, 3.5, 1.1, 7.8, 17.8, 1.2, 2.9, 89.0, 550.0         ],
		        [3.4, 1.7, 0.2, 0.1, 3.4, 1.9, 0.4, 0.02, 1030.0, 5400.0       ],
		        [3.0, 1.4, 0.4, 0.5, 3.0, 2.3, 0.1, 0.6, -1.0, 126.0           ], 
		        [8.4, 0.5, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, -1.0, 14.0            ], 
		        [0.0, 0.0, 0.0, 0.0, 4.1, 0.6, 0.1, 0.3, -1.0, 15.0            ],
		        [11.0, 11.2, 2.1, 2.8, 11.0, 13.0, 1.8, 4.6, -1.0, 0.0         ],
		        [4.7, 3.0, 0.2, 0.5, 4.7, 2.7, 0.3, 0.1, 480.0, 3700.0         ],
		        [9.0, 40.3, 8.8, 4.2, 9.0, 40.1, 4.6, 4.6, -1.0, 70.1          ],
		        [0.0, 0.0, 0.0, 0.0, 7.9, 0.7, 1.7, 999.9, -1.0, -1.0          ], 
		        [0.0, 0.0, 0.0, 0.0, 7.1, 6.6, 0.8, 4.0, -1.0, 2.41            ], 
		        [7.5, 0.0, 0.26, 0.0, 7.5, 0.0, 0.2, 0.0, -1.0, 0.0            ],
		        [9.0, 1.3, 0.1, 0.01, 9.0, 3.8, 0.9, 2.6, -1.0, 29.1           ],
		        [0.0, 0.0, 0.0, 0.0, 7.3, 0.0, 0.1, 0.0, -1.0, 0.0             ],
		        [9.9, 0.1, 2.6, 999.9, 9.9, 0.5, 1.5, 999.9, -1.0, 0.0         ],
		        [7.3, 8.1, 0.3, 0.4, 7.3, 9.3, 0.8, 1.1, 210000.0, -1.0        ], 
		        [0.0, 0.0, 0.0, 0.0, 6.5, 0.0, 0.2, 0.0, -1.0, -1.0            ], 
		        [6.0, 8.2, 0.7, 0.9, 6.0, 8.2, 1.3, 1.9, 410.0, 1880.0         ],
		        [0.0, 0.0, 0.0, 0.0, 3.3, 1.1, 0.3, 0.2, -1.0, 28.4            ],
		        [3.5, 1.5, 0.4, 0.1, 3.5, 1.6, 0.2, 0.2, 117.0, 1290.0         ], 
		      ], dtype=np.float32)

		cut = 10000.0

		OtempH = Obs[:, 0]
		OlumH = Obs[:, 1] * 1.0e4
		OlumHerrU = Obs[:, 2] * 1.0e4
		OlumHerrD = Obs[:, 3] * 1.0e4

		OtempP = Obs[:, 4]
		OlumP = Obs[:, 5] * 1.0e4
		OlumPerrU = Obs[:, 6] * 1.0e4
		OlumPerrD = Obs[:, 7] * 1.0e4

		radioL = Obs[:, 8]
		radioS = Obs[:, 9]

		w = np.where((OlumHerrU < cut*OtempH) & ((OlumHerrD < cut*OtempH)))[0]
		xplot, yplot = gc.logten(OtempH[w]), gc.logten(OlumH[w])
		yerr2, yerr1 = gc.logten(gc.divide(OlumH[w]+OlumHerrU[w], OlumH[w])), -gc.logten(gc.divide(OlumH[w]-OlumHerrD[w], OlumH[w]))
		plt.errorbar(xplot, yplot, yerr=[yerr1,yerr2], color='c', lw=2.0, alpha=0.6, marker='s', markersize=8, ls='none', label='P98 HRI', mew=1)
		
		w = np.where((OlumPerrU < cut*OtempP) & ((OlumPerrD < cut*OtempP)))[0]
		xplot, yplot = gc.logten(OtempP[w]), gc.logten(OlumP[w])
		yerr2, yerr1 = gc.logten(gc.divide(OlumP[w]+OlumPerrU[w], OlumP[w])), -gc.logten(gc.divide(OlumP[w]-OlumPerrD[w], OlumP[w]))
		plt.errorbar(xplot, yplot, yerr=[yerr1,yerr2], color='c', lw=2.0, alpha=0.6, marker='*', markersize=12, ls='none', label='P98 PSPC', mew=1)

	if extra>1:
		# Data from Ponman et al 1996
		P_temp = np.array([0.89, 0.44, 0.3, 0.61, 0.91, 0.67, 0.82, 1.09, 0.82, 0.64, 0.96, 0.82, 0.54, 0.59, 0.68, 0.75, 0.87])
		P_temp_err = np.array([0.12, 0.08, 0.05, 0.3, 0.18, 0.11, 0.03, 0.21, 0.27, 0.19, 0.04, 0.19, 0.15, 0.0, 0.12, 0.08, 0.05])
		P_loglum = np.array([42.31, 41.8, 41.68, 41.77, 42.35, 42.12, 42.16, 41.58, 41.98, 41.89, 43.04, 41.69, 41.27, 42.43, 41.48, 42.16, 42.78])
		P_loglum_err = np.array([0.08, 0.12, 0.06, 0.11, 0.11, 0.06, 0.02, 0.14, 0.21, 0.11, 0.03, 0.1, 0.26, 0.24, 0.09, 0.04, 0.02])
        
		P_xplot = np.log10(P_temp)
		P_xerr = np.log10(P_temp+P_temp_err) - P_xplot
		plt.errorbar(P_xplot, P_loglum-40, xerr=P_xerr, yerr=P_loglum_err, color='brown', alpha=0.6, marker='>', markersize=8, ls='none', label=r'P96', mew=1, lw=2)
        
        # Data from Bharadwaj et al (2015)
		B_temp = np.array([1.9, 1.79, 2.1, 1.43, 0.98, 1.98, 3.58, 2.01, 3.25, 2.06, 1.45, 14.3, 0.81, 1.25, 1.40, 1.06, 0.97, 1.05, 1.96, 2.05, 2.13, 0.64, 2.05, 1.43, 1.78, 0.91])
		B_temp_errH = np.array([0.09, 0.07, 0.05, 0.04, 0.02, 0.04, 0.14, 0.03, 0.08, 0.08, 0.02, 0.04, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.12, 0.1, 0.1, 0.01, 0.1, 0.06, 0.29, 0.01])
		B_temp_errL = np.array([0.09, 0.07, 0.07, 0.05, 0.02, 0.05, 0.14, 0.03, 0.08, 0.09, 0.02, 0.04, 0.02, 0.01, 0.01, 0.01, 0.02, 0.01, 0.15, 0.12, 0.05, 0.01, 0.12, 0.1, 0.22, 0.01])
		B_lum = np.array([2.69, 1.22, 2.02, 0.399, 0.358, 3.42, 3.1, 2.86, 6.92, 3.22, 1.67, 0.413, 0.232, 0.929, 2., 1.25, 0.293, 0.445, 0.444, 2.66, 3.63, 0.111, 2.62, 0.602, 1.98, 0.666])
		B_lum_err = np.array([0.38, 0.17, 0.24, 0.053, 0.139, 0.17, 0.25, 0.05, 0.58, 0.42, 0.02, 0.066, 0.032, 0.15, 0.11, 0.1, 0.04, 0.076, 0.064, 0.22, 0.56, 0.012, 0.74, 1.09, 0.33, 0.045])

		B_xplot = np.log10(B_temp)
		B_xerr1 = np.log10(B_temp+B_temp_errH) - B_xplot
		B_xerr2 = B_xplot - np.log10(B_temp-B_temp_errL)
		B_yplot = np.log10(B_lum)+3.0
		B_yerr = np.log10(B_lum+B_lum_err)+3.0 - B_yplot

		plt.errorbar(B_xplot, B_yplot, xerr=[B_xerr1, B_xerr2], yerr=B_yerr, color='orange', alpha=0.6, marker='^', markersize=8, ls='none', label=r'B15', mew=1, lw=2.0)

	if extra>2:
		A_Ltot = np.array([43.82, 43.46, 43.39, 42.98, 42.64, 42.34, 41.80, 41.52, 41.29, 40.97, 40.58, 40.40, 39.96, 40.10, 39.60, 38.96, 39.94, 40.00, 39.60])
		A_Ltot_uncert_m = np.array([0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.04, 0.07, 0.09, 0.27, 0.19, 0.97, 0.86, 0.21, 0.19, 0.97])
		A_Ltot_uncert_b = np.array([0.21, 0.11, 0.09, 0.06, 0.05, 0.06, 0.06, 0.05, 0.07, 0.11, 0.10, 0.19, 0.46, 0.63, 0.78, 0.83, 0.28, 0.47, 0.86])
		A_Ltot_uncert = np.max(np.array([A_Ltot_uncert_m, A_Ltot_uncert_b]), axis=0)
		A_temp = np.array([5.0, 4.0, 3.4, 2.5, 1.9, 1.5, 1.1, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
		plt.errorbar(np.log10(A_temp), A_Ltot-40, yerr=A_Ltot_uncert, marker='_', alpha=0.8, color='purple', markersize=15, ls='none', label=r'A15', mew=2, lw=2)


def SFRD_obs(h, alpha=0.3, ax=None):
	### Observational data as first compiled in Somerville et al. (2001) ###
	ObsSFRdensity = np.array([
							  [0, 0.0158489, 0, 0, 0.0251189, 0.01000000],
							  [0.150000, 0.0173780, 0, 0.300000, 0.0181970, 0.0165959],
							  [0.0425000, 0.0239883, 0.0425000, 0.0425000, 0.0269153, 0.0213796],
							  [0.200000, 0.0295121, 0.100000, 0.300000, 0.0323594, 0.0269154],
							  [0.350000, 0.0147911, 0.200000, 0.500000, 0.0173780, 0.0125893],
							  [0.625000, 0.0275423, 0.500000, 0.750000, 0.0331131, 0.0229087],
							  [0.825000, 0.0549541, 0.750000, 1.00000, 0.0776247, 0.0389045],
							  [0.625000, 0.0794328, 0.500000, 0.750000, 0.0954993, 0.0660693],
							  [0.700000, 0.0323594, 0.575000, 0.825000, 0.0371535, 0.0281838],
							  [1.25000, 0.0467735, 1.50000, 1.00000, 0.0660693, 0.0331131],
							  [0.750000, 0.0549541, 0.500000, 1.00000, 0.0389045, 0.0776247],
							  [1.25000, 0.0741310, 1.00000, 1.50000, 0.0524807, 0.104713],
							  [1.75000, 0.0562341, 1.50000, 2.00000, 0.0398107, 0.0794328],
							  [2.75000, 0.0794328, 2.00000, 3.50000, 0.0562341, 0.112202],
							  [4.00000, 0.0309030, 3.50000, 4.50000, 0.0489779, 0.0194984],
							  [0.250000, 0.0398107, 0.00000, 0.500000, 0.0239883, 0.0812831],
							  [0.750000, 0.0446684, 0.500000, 1.00000, 0.0323594, 0.0776247],
							  [1.25000, 0.0630957, 1.00000, 1.50000, 0.0478630, 0.109648],
							  [1.75000, 0.0645654, 1.50000, 2.00000, 0.0489779, 0.112202],
							  [2.50000, 0.0831764, 2.00000, 3.00000, 0.0512861, 0.158489],
							  [3.50000, 0.0776247, 3.00000, 4.00000, 0.0416869, 0.169824],
							  [4.50000, 0.0977237, 4.00000, 5.00000, 0.0416869, 0.269153],
							  [5.50000, 0.0426580, 5.00000, 6.00000, 0.0177828, 0.165959],
							  [3.00000, 0.120226, 2.00000, 4.00000, 0.173780, 0.0831764],
							  [3.04000, 0.128825, 2.69000, 3.39000, 0.151356, 0.109648],
							  [4.13000, 0.114815, 3.78000, 4.48000, 0.144544, 0.0912011],
							  [0.350000, 0.0346737, 0.200000, 0.500000, 0.0537032, 0.0165959],
							  [0.750000, 0.0512861, 0.500000, 1.00000, 0.0575440, 0.0436516],
							  [1.50000, 0.0691831, 1.00000, 2.00000, 0.0758578, 0.0630957],
							  [2.50000, 0.147911, 2.00000, 3.00000, 0.169824, 0.128825],
							  [3.50000, 0.0645654, 3.00000, 4.00000, 0.0776247, 0.0512861],
							  ], dtype=np.float32)
	
	ObsRedshift = ObsSFRdensity[:, 0]
	xErrLo = ObsSFRdensity[:, 0]-ObsSFRdensity[:, 2]
	xErrHi = ObsSFRdensity[:, 3]-ObsSFRdensity[:, 0]
	
	ObsSFR = np.log10(ObsSFRdensity[:, 1]*h/0.7)
	yErrLo = np.log10(ObsSFRdensity[:, 1]*h/0.7)-np.log10(ObsSFRdensity[:, 4]*h/0.7)
	yErrHi = np.log10(ObsSFRdensity[:, 5]*h/0.7)-np.log10(ObsSFRdensity[:, 1]*h/0.7)
	### ================= ###
	
	if ax is None:
		plt.errorbar(ObsRedshift, ObsSFR, yerr=[yErrLo, yErrHi], xerr=[xErrLo, xErrHi], color='purple', lw=2.0, alpha=alpha, ls='none', label=r'Somerville et al.~(2001)', ms=0)
	else:
		ax.errorbar(ObsRedshift, ObsSFR, yerr=[yErrLo, yErrHi], xerr=[xErrLo, xErrHi], color='purple', lw=2.0, alpha=alpha, ls='none', label=r'Somerville et al.~(2001)', ms=0)



def SFR_function_obs(h=0.678):
	data = np.array([[-0.5, 0.0, -1.78, 0.23],
                     [0.0, 0.5, -2.26, 0.05],
                     [0.5, 1.0, -2.53, 0.04],
                     [1.0, 1.5, -3.29, 0.05],
                     [1.5, 2.0, -4.92, 0.31],
                     [2.0, 2.5, -5.22, 0.43]])
	xplot = (data[:,0]+data[:,1])/2
	ymax = 10**(data[:,2]+data[:,3])*(h/0.71)**3
	ymin = 10**(data[:,2]-data[:,3])*(h/0.71)**3
	plt.fill_between(xplot, ymax, ymin, color='purple', alpha=0.5)
	plt.plot([-10,-11],[1,2], color='purple', lw=6, alpha=0.5, label=r'Observations')
	plt.axis([-0.5, 2.5, 1e-6, 1e-1])
	plt.ylabel(r'$\Phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
	plt.xlabel(r'$\log_{10}(\mathrm{SFR}\ [\mathrm{M}_{\bigodot}\ \mathrm{yr}^{-1}])$')



def gas_fraction_obs(h=0.678):
	data = np.array([[6.715, 0.87, 1.0808, 0.6592],
                      [7.08273, 0.57, 1.29, 0.324],
                      [7.54601, 0.745, 1.19484, 0.3392],
                      [8.16125, 0.765, 0.924, 0.677854],
                      [8.58396, 0.55, 0.6352, 0.411652],
                      [9.07265, 0.1139, 0.342, -0.00281077],
                      [9.56721, -0.23395, 0.2196, -0.494536],
                      [10.1131, -0.30334, 0.0693901, -1.08455],
                      [10.5787, -0.650294, -0.292155, -1.26095],
                      [10.9901, -0.765097, -0.472351, -1.08804],
                      [11.4426, -1.17901, -0.914211, -1.24687]])
	plt.fill_between(data[:,0]+2*np.log10(0.7/h), data[:,2]-np.log10(0.75)+0.1, data[:,3]-np.log10(0.75)-0.1, color='purple', alpha=0.5) # additional 0.1 to account for uncertainty in conversion from CO measurements
	plt.xlabel(r'$\log_{10}(M_{\rm stars}\ [\mathrm{M}_{\bigodot}])$')
	plt.ylabel(r'$\log_{10}(M_{\rm gas} / M_{\rm stars})$')
	plt.axis([6.5, 11.5, -1.25, 1.3])




def Leroygals(HI=False, H2=False, HighVvir=True, LowVvir=False, ax=None, SFR=False, h=0.678, c='k', alpha=0.5, lw=2):
	# Plot galaxy surface density profiles for select galaxies.  Stars done by default
	N628 = np.array([[0.2, 0.5, 0.9, 1.2, 1.6, 1.9, 2.3, 2.7, 3.0, 3.4, 3.7, 4.1, 4.4, 4.8, 5.1, 5.5, 5.8, 6.2, 6.5, 6.9, 7.3, 7.6, 8.0, 8.3, 8.7, 9.0, 9.4, 9.7, 10.1, 10.4, 10.8, 11.1, 11.5, 11.9, 12.2],
					 [1.6, 2.1, 2.6, 3.1, 3.7, 4.6, 5.3, 5.8, 6.1, 6.5, 7.3, 7.9, 8.1, 7.9, 8.2, 8.5, 8.6, 8.6, 8.8, 8.8, 8.6, 8.2, 7.6, 7.1, 6.7, 6.5, 6.0, 5.2, 4.5, 4.1, 3.9, 3.9, 4.0, 4.3, 4.6],
					 [0.3, 0.3, 0.4, 0.4, 0.3, 0.3, 0.4, 0.5, 0.5, 0.5, 0.7, 0.8, 0.8, 0.9, 1.0, 1.0, 0.8, 0.7, 0.6, 0.5, 0.5, 0.6, 0.6, 0.6, 0.5, 0.4, 0.5, 0.4, 0.4, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5],
					 [22.7, 20.2, 16.1, 12.7, 11.4, 11.1, 11.1, 10.6, 8.9, 7.2, 6.2, 5.9, 5.4, 4.3, 3.1, 2.1, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					 [1.2, 1.3, 1.2, 0.8, 1.1, 1.2, 1.7, 1.9, 1.5, 1.2, 1.5, 1.7, 1.5, 1.1, 0.8, 0.7, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					 [1209.4, 557.8, 313.6, 231.9, 194.3, 163.5, 143.9, 123.5, 107.5, 151.0, 81.6, 68.0, 61.6, 48.3, 41.8, 37.0, 33.2, 37.0, 52.9, 19.5, 18.9, 18.7, 12.9, 17.6, 17.0, 10.8, 8.0, 7.5, 5.0, 4.1, 3.6, 3.9, 4.4, 9.5, 5.8],
					 [18.3, 4.8, 1.0, 0.5, 0.5, 0.7, 0.8, 0.5, 0.4, 10.5, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.4, 2.3, 6.1, 0.1, 0.1, 0.7, 0.1, 1.3, 1.6, 0.4, 0.1, 0.2, 0.1, 0.0, 0.0, 0.1, 0.2, 0.9, 0.2],
                     [105.1, 92.3, 76.7, 65.5, 62.2, 72.4, 90.2, 90.7, 71.9, 57.9, 55.8, 59.6, 59.9, 48.8, 37.4, 33.5, 30.2, 23.5, 17.4, 13.6, 11.6, 9.8, 7.5, 5.4, 4.1, 3.2, 2.5, 1.8, 1.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                     [14, 9.9, 5.1, 4.2, 3.7, 12.2, 23.5, 21.3, 11.9, 8.5, 11.3, 14.1, 15.2, 11, 6.6, 8.7, 10, 6.5, 3.1, 1.9, 2.3, 2.2, 1.5, 0.9, 0.6, 0.4, 0.3, 0.3, 0.2, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0]])
	
	#[0.02, 0.07, 0.11, 0.16, 0.20, 0.25, 0.29, 0.34, 0.38, 0.43, 0.47, 0.52, 0.56, 0.61, 0.65, 0.70, 0.74, 0.79, 0.83, 0.88, 0.92, 0.97, 1.01, 1.06, 1.10, 1.15, 1.19],
	
	N3184 = np.array([[0.3, 0.8, 1.3, 1.9, 2.4, 3.0, 3.5, 4.0, 4.6, 5.1, 5.7, 6.2, 6.7, 7.3, 7.8, 8.3, 8.9, 9.4, 10.0, 10.5, 11.0, 11.6, 12.1, 12.6, 13.2, 13.7, 14.3],
					  [3.7, 3.2, 3.3, 3.8, 4.7, 5.5, 5.7, 5.9, 6.5, 7.3, 7.5, 7.8, 8.1, 8.0, 7.3, 7.0, 7.0, 6.7, 6.1, 5.4, 5.0, 4.6, 4.0, 3.3, 2.9, 2.8, 2.7],
					  [0.5, 0.3, 0.3, 0.3, 0.5, 0.4, 0.4, 0.5, 0.4, 0.3, 0.5, 0.6, 0.6, 0.5, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2],
					  [44.2, 20.8, 14.5, 11.9, 12.6, 12.6, 11.0, 9.6, 7.4, 6.2, 5.5, 4.3, 2.7, 1.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					  [9.6, 3.3, 2.0, 1.6, 2.0, 2.1, 2.1, 0.9, 1.0, 0.6, 0.9, 0.8, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					  [701.8, 270.5, 200.3, 146.5, 121.1, 113.0, 100.2, 94.2, 83.6, 74.6, 96.2, 61.2, 46.3, 34.6, 27.5, 22.4, 19.3, 14.9, 12.9, 9.5, 7.9, 6.6, 5.1, 4.7, 3.7, 3.0, 3.8],
					  [22.8, 1.1, 0.6, 1.0, 0.5, 0.4, 0.4, 0.5, 0.3, 0.3, 6.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1],
                      [282.1, 85.4, 47.1, 47.1, 48.6, 50.8, 51.3, 50.0, 42.5, 38.4, 36.5, 37.6, 29.9, 20.4, 13.9, 9.9, 8.4, 6.9, 4.3, 3.0, 2.4, 1.7, 1.1, 0.0, 0.0, 0.0, 0.0],
                      [78.8, 23.5, 6.5, 4.8, 6.9, 7.7, 8.8, 10.4, 6.5, 4.7, 4.5, 7.2, 5.0, 2.3, 1.4, 1.1, 1.1, 0.8, 0.4, 0.6, 0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0]])
	
	N3521 = np.array([[0.3, 0.8, 1.3, 1.8, 2.3, 2.9, 3.4, 3.9, 4.4, 4.9, 5.4, 6.0, 6.5, 7.0, 7.5, 8.0, 8.6, 9.1, 9.6, 10.1, 10.6, 11.2, 11.7, 12.2, 12.7, 13.2, 13.7, 14.3, 14.8, 15.3],
					  [4.5, 4.8, 5.7, 7.1, 8.3, 8.7, 8.7, 8.9, 9.9, 10.6, 10.5, 10.2, 10.2, 9.4, 8.6, 8.4, 8.5, 8.6, 8.5, 8.2, 8.1, 8.1, 8.2, 8.3, 8.2, 8.0, 7.9, 7.7, 7.1, 6.5],
					  [0.2, 0.3, 0.6, 0.7, 0.8, 0.8, 0.6, 0.8, 0.9, 0.6, 0.5, 0.4, 0.5, 0.3, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 1.0, 0.9, 0.8, 0.6],
					  [25.7, 35.4, 43.4, 44.6, 41.5, 36.8, 30.6, 24.6, 22.2, 21.0, 17.5, 12.5, 8.4, 5.2, 3.2, 2.1, 1.7, 1.6, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					  [4.7, 5.5, 3.0, 1.5, 2.1, 2.7, 2.6, 2.3, 2.0, 2.1, 2.4, 2.5, 2.0, 1.4, 0.9, 0.6, 0.5, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					  [4545.9, 1442.2, 929.8, 589.1, 462.9, 381.3, 322.9, 250.8, 212.3, 192.2, 169.7, 134.7, 106.7, 82.4, 66.1, 55.4, 47.7, 41.7, 35.8, 30.5, 27.1, 25.4, 22.9, 20.0, 17.4, 15.7, 14.4, 13.3, 12.0, 10.9],
					  [287.3, 23.0, 8.9, 5.4, 2.9, 2.0, 1.9, 1.8, 1.4, 1.3, 1.1, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [106, 152.7, 183.4, 186.8, 176.9, 160.3, 138.2, 109, 89.2, 80.1, 70.3, 53.5, 36.1, 23.5, 16.2, 12.9, 11.6, 10.7, 9.4, 7.5, 6.1, 5.3, 4.6, 3.9, 3.1, 2.5, 2.1, 1.8, 1.4, 1.1],
                      [27.9, 22.8, 11.5, 8.6, 8.7, 10.9, 16.3, 15.1, 11.5, 9.3, 9.2, 9.2, 7.6, 5.2, 3.6, 3.0, 3.0, 2.5, 2.0, 1.3, 1.3, 1.3, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.2]])
	
	N5194 = np.array([[0.2, 0.6, 1.0, 1.4, 1.7, 2.1, 2.5, 2.9, 3.3, 3.7, 4.1, 4.5, 4.8, 5.2, 5.6, 6.0, 6.4, 6.8, 7.2, 7.6, 8.0, 8.3, 8.7, 9.1, 9.5, 9.9, 10.3, 10.7],
					  [4.5, 5.5, 6.1, 6.1, 6.7, 7.9, 8.5, 7.5, 6.2, 6.1, 7.2, 9.1, 11.2, 12.8, 12.7, 11.1, 9.4, 8.4, 7.8, 7.8, 7.8, 7.8, 7.8, 7.3, 6.4, 5.8, 5.1, 4.5],
					  [0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.9, 0.9 ,0.7, 0.5, 0.7, 0.9, 0.9, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 1.0, 1.1, 1.2, 1.2, 1.1, 1.0, 1.0, 0.9, 0.9],
					  [197.4, 207.7, 181.6, 134.5, 106.8, 94.8, 72.6, 40.9, 19.1, 14.6, 22.6, 33.6, 35.9, 28.0, 14.9, 4.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					  [35.0, 33.5, 30.5, 41.2, 33.8, 19.2, 15.7 ,12.2, 5.6, 3.3, 7.5, 10.3, 9.6, 7.0, 4.7, 8.2, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					  [4912.2, 2352.5, 1251.1, 703.1, 471.8, 417.5, 394.9, 334.1, 286.9, 253.2, 236.7, 224.2, 227.8, 206.0, 176.5, 148.9, 106.3, 77.5, 64.3, 50.2, 45.0, 46.4, 53.2, 69.2, 71.2, 86.5, 210.3, 102.7],
					  [111.6, 15.3, 6.5, 3.8, 1.3, 1.6, 1.9, 1.5, 1.3, 2.0, 1.7, 1.2, 1.4, 1.1, 0.9, 0.8, 0.4, 0.3, 0.6, 0.3, 0.3, 0.3, 0.6, 1.8, 1.3, 1.7, 11.3, 3.1],
                      [1164, 1026.9, 772.8, 463.7, 295.4, 296.7, 304.1, 234.9, 154.7, 113, 113.5, 148.7, 188, 199.4, 180.2, 141.1, 95.9, 58.7, 36.7, 24.9, 19.5, 19.5, 20, 19, 14.8, 10.8, 8.3, 6.6],
                      [96.6, 138.1, 134.6, 84.7, 45.3, 49.5, 49.9, 34.2, 19.4, 8.7, 19.8, 39.9, 48.7, 41.2, 31.7, 27.6, 19.2, 9.9, 5.7, 4.3, 3.9, 4.3, 5.5, 6.8, 5.1, 2.8, 1.8, 1.3]])
	if HighVvir and not SFR:
		if HI and H2:
			if ax is None:
				plt.errorbar(N628[0,:], N628[1,:]+N628[3,:], N628[2,:]+N628[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N3184[0,:], N3184[1,:]+N3184[3,:], N3184[2,:]+N3184[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N3521[0,:], N3521[1,:]+N3521[3,:], N3521[2,:]+N3521[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N5194[0,:], N5194[1,:]+N5194[3,:], N5194[2,:]+N5194[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			else:
				ax.errorbar(N628[0,:], N628[1,:]+N628[3,:], N628[2,:]+N628[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N3184[0,:], N3184[1,:]+N3184[3,:], N3184[2,:]+N3184[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N3521[0,:], N3521[1,:]+N3521[3,:], N3521[2,:]+N3521[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N5194[0,:], N5194[1,:]+N5194[3,:], N5194[2,:]+N5194[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
		elif HI and not H2:
			if ax is None:
				plt.errorbar(N628[0,:], N628[1,:], N628[2,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N3184[0,:], N3184[1,:], N3184[2,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N3521[0,:], N3521[1,:], N3521[2,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N5194[0,:], N5194[1,:], N5194[2,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			else:
				ax.errorbar(N628[0,:], N628[1,:], N628[2,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N3184[0,:], N3184[1,:], N3184[2,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N3521[0,:], N3521[1,:], N3521[2,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N5194[0,:], N5194[1,:], N5194[2,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
		elif H2 and not HI:
			if ax is None:
				plt.errorbar(N628[0,:], N628[3,:], N628[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N3184[0,:], N3184[3,:], N3184[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N3521[0,:], N3521[3,:], N3521[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N5194[0,:], N5194[3,:], N5194[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			else:
				ax.errorbar(N628[0,:], N628[3,:], N628[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N3184[0,:], N3184[3,:], N3184[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N3521[0,:], N3521[3,:], N3521[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N5194[0,:], N5194[3,:], N5194[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
		else:
			if ax is None:
				plt.errorbar(N628[0,:], N628[5,:]*0.61/0.66, N628[6,:]*0.61/0.66, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N3184[0,:], N3184[5,:]*0.61/0.66, N3184[6,:]*0.61/0.66, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N3521[0,:], N3521[5,:]*0.61/0.66, N3521[6,:]*0.61/0.66, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N5194[0,:], N5194[5,:]*0.61/0.66, N5194[6,:]*0.61/0.66, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			else:
				ax.errorbar(N628[0,:], N628[5,:]*0.61/0.66, N628[6,:]*0.61/0.66, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N3184[0,:], N3184[5,:]*0.61/0.66, N3184[6,:]*0.61/0.66, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N3521[0,:], N3521[5,:]*0.61/0.66, N3521[6,:]*0.61/0.66, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N5194[0,:], N5194[5,:]*0.61/0.66, N5194[6,:]*0.61/0.66, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
	elif HighVvir and SFR:
		if ax is None:
			plt.errorbar(N628[0,:], N628[7,:]*1e-4*0.63/0.67, N628[8,:]*1e-4*0.63/0.67, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			plt.errorbar(N3184[0,:], N3184[7,:]*1e-4*0.63/0.67, N3184[8,:]*1e-4*0.63/0.67, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			plt.errorbar(N3521[0,:], N3521[7,:]*1e-4*0.63/0.67, N3521[8,:]*1e-4*0.63/0.67, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			plt.errorbar(N5194[0,:], N5194[7,:]*1e-4*0.63/0.67, N5194[8,:]*1e-4*0.63/0.67, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
		else:
			ax.errorbar(N628[0,:], N628[7,:]*1e-4*0.63/0.67, N628[8,:]*1e-4*0.63/0.67, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			ax.errorbar(N3184[0,:], N3184[7,:]*1e-4*0.63/0.67, N3184[8,:]*1e-4*0.63/0.67, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			ax.errorbar(N3521[0,:], N3521[7,:]*1e-4*0.63/0.67, N3521[8,:]*1e-4*0.63/0.67, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			ax.errorbar(N5194[0,:], N5194[7,:]*1e-4*0.63/0.67, N5194[8,:]*1e-4*0.63/0.67, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)

	N3351 = np.array([[0.2, 0.7, 1.2, 1.7, 2.2, 2.7, 3.2, 3.7, 4.2, 4.7, 5.1, 5.6, 6.1, 6.6, 7.1, 7.6, 8.1, 8.6, 9.1, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5],
					  [1.5, 1.0, 0.0, 0.0, 1.2, 2.0, 2.6, 2.7, 2.9, 3.1, 3.0, 2.9, 2.7, 2.7, 2.7, 2.8, 3.0, 3.3, 3.5, 3.6, 3.5, 3.1, 2.5, 2.1, 1.8, 1.4],
					  [0.3, 0.2, 1.0, 1.0, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1],
					  [161.4, 72.1, 15.7, 3.0, 2.1, 3.6, 4.8, 4.3, 3.1, 2.3, 1.6, 1.3, 1.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					  [29.9, 22.6, 6.3, 1.5, 0.6, 0.7, 0.8, 1.0, 0.7, 0.4, 0.4, 0.4, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					  [4525.1, 973.0, 418.7, 245.3, 189.8, 713.4, 162.3, 124.0, 93.3, 69.6, 55.5, 49.1, 44.5, 38.8, 32.9, 26.3, 20.4, 16.1, 12.6, 10.7, 9.2, 6.7, 5.9, 5.3, 3.7, 3.4],
					  [98.2, 10.4, 2.7, 1.7, 1.2, 0.8, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.0, 0.1, 0.2, 0.1, 0.2],
                      [2559.3, 1052, 194.1, 44.5, 30.6, 33.1, 37.2, 32.3, 23.1, 17.1, 13.6, 11.2, 9.7, 9.0, 8.6, 6.9, 5.2, 4.3, 3.5, 3, 2.6, 1.8, 1.1, 0.0, 0.0, 0.0],
                      [473.1, 337, 68.7, 7.9, 1.8, 2, 2.7, 2.8, 1.7, 1.3, 1.4, 1.2, 1.1, 0.8, 1.1, 1.1, 0.5, 0.4, 0.3, 0.3, 0.3, 0.3, 0.1, 0.0, 0.0, 0.0]])
					
	N3627 = np.array([[0.2, 0.7, 1.1, 1.6, 2.0, 2.5, 2.9, 3.4, 3.8, 4.3, 4.7, 5.2, 5.6, 6.1, 6.5, 7.0, 7.4, 7.9, 8.3, 8.8, 9.2, 9.7, 10.1, 10.6, 11.0, 11.5, 11.9, 12.4, 12.8, 13.3, 13.8, 14.2, 14.7, 15.1, 15.6, 16.0, 16.5],
					  [3.0, 3.5, 4.1, 4.9, 5.7, 6.4, 7.3, 7.7, 7.2, 6.5, 6.0, 5.3, 4.5, 3.7, 3.2, 2.9, 2.3, 1.7, 1.3, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					  [1.0, 0.6, 0.4, 0.6, 0.7, 1.0, 1.5, 1.6, 1.2, 1.1, 1.3, 1.2, 1.0, 0.9, 0.9, 0.9, 0.6, 0.4, 0.3, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
					  [173.3, 109.6, 59.1, 37.3, 38.7, 47.9, 49.6, 39.2, 22.2, 13.4, 10.5, 8.4, 7.2, 6.1, 5.4, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					  [30.1, 23.0, 11.8, 8.9, 16.3, 23.9, 24.3, 19.2, 12.2, 7.0, 4.4, 3.8, 3.7, 3.6, 3.3, 2.8, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					  [5428.1, 1553.9, 846.5, 592.0, 478.9, 289.1, 254.9, 275.1, 216.8, 172.9, 158.3, 132.8, 113.8, 94.8, 78.5, 69.2, 53.4, 41.3, 35.0, 28.7, 24.3, 21.9, 22.2, 57.6, 59.3, 16.5, 10.8, 8.7, 7.5, 8.6, 9.0, 5.4, 4.6, 4.3, 4.3, 3.7, 3.0],
					  [273.0, 14.4, 4.6, 3.5, 3.3, 4.6, 4.7, 2.5, 1.6, 1.0, 1.2, 0.9, 0.8, 0.6, 0.5, 0.6, 0.4, 0.2, 0.3, 0.1, 0.1, 0.1, 0.3, 6.3, 6.7, 0.5, 0.1, 0.1, 0.1, 0.5, 0.6, 0.1, 0.0, 0.1, 0.1, 0.1, 0.0],
                      [251.1, 205.9, 167.2, 197.1, 317.6, 430.4, 431.9, 327.5, 192.2, 127.9, 101, 62.5, 37.6, 27.7, 25.6, 25.3, 19.2, 11.5, 7, 4.6, 3.3, 2.4, 1.7, 1.3, 1.1, 0.0, 0.0, 0.0, 0.0, 1.1, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [27.4, 24, 15.7, 56.5, 152.2, 209.4, 181.4, 128.4, 72.2, 43.2, 40.8, 19.6, 8.9, 8.1, 9.9, 12.8, 9.7, 4.6, 1.8, 0.9, 0.6, 0.4, 0.3, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
					
	N5055 = np.array([[0.2, 0.7, 1.2, 1.7, 2.2, 2.7, 3.2, 3.7, 4.2, 4.7, 5.1, 5.6, 6.1, 6.6, 7.1, 7.6, 8.1, 8.6, 9.1, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.4, 14.9, 15.4, 15.9, 16.4, 16.9, 17.4, 17.9, 18.4, 18.9, 19.3, 19.8, 20.3, 20.8],
					  [5.6, 5.8, 5.9, 5.9, 6.2, 6.6, 6.4, 6.2, 6.5, 7.2, 8.2, 8.7, 8.7, 8.5, 8.5, 8.6, 8.5, 7.9, 7.5, 7.4, 7.2, 7.3, 7.5, 7.3, 6.7, 6.4, 6.2, 5.7, 5.1, 4.5, 4.2, 3.9, 3.7, 3.2, 2.8, 2.6, 2.5, 2.4, 2.1, 1.7, 1.4, 1.3, 1.2],
					  [0.7, 0.5, 0.4, 0.2, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.7, 0.5, 0.5, 0.6, 0.5, 0.5, 0.5, 0.4, 0.4, 0.5, 0.7, 0.8, 0.9, 0.7, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.5, 0.6, 0.5, 0.4, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2],
					  [142.7, 98.8, 62.2, 43.7, 36.6, 32.1, 25.3, 20.3, 19.1, 18.6, 18.8, 17.3, 13.4, 10.9, 10.3, 10.0, 8.7, 6.2, 4.3, 3.1, 2.1, 1.5, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					  [19.4, 16.1, 8.5, 3.9, 2.3, 2.5, 2.7, 2.3, 2.1, 2.1, 2.1, 1.8, 1.1, 1.0, 1.0, 1.2, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					  [4742.4, 1627.7, 987.4, 758.0, 569.7, 417.9, 325.4, 264.3, 230.7, 194.9, 169.5, 150.6, 133.4, 109.1, 94.0, 84.5, 75.3, 62.6, 52.0, 44.5, 40.8, 36.8, 33.4, 60.4, 59.2, 24.8, 21.2, 18.7, 17.3, 18.0, 13.8, 13.2, 11.8, 10.7, 9.8, 9.2, 8.4, 8.4, 7.7, 7.4, 7.1, 13.2, 12.4],
					  [251.1, 11.4, 4.6, 2.2, 1.7, 1.1, 0.9, 0.8, 0.7, 0.5, 0.4, 0.4, 0.4, 0.3, 0.2, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 5.4, 5.7, 0.1, 0.1, 0.1, 0.1, 0.5, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 1.0, 1.4],
                      [249.8, 267.2, 238.7, 192.7, 162.5, 135.2, 104.4, 81.3, 70.8, 67.2, 70.7, 73.4, 63, 51.4, 46.6, 43.3, 38.5, 29.6, 22.7, 20.1, 16.8, 12.6, 10.2, 8.6, 6.7, 5.3, 4.4, 3.7, 3.2, 2.8, 2.5, 2.2, 2, 1.6, 1.2, 1, 1.1, 1.4, 1.2, 0.0, 0.0, 0.0, 0.0],
                      [28.4, 20.9, 13.2, 8.7, 5.9, 5.8, 5.9, 5.4, 5.2, 5.8, 7.3, 9.3, 6.4, 4.6, 5.3, 6.4, 6.4, 3.6, 2.1, 3.6, 3.7, 2.6, 2.0, 1.4, 0.8, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.1, 0.2, 0.4, 0.9, 0.8, 0.0, 0.0, 0.0, 0.0]])
					
	N6946 = np.array([[0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.1, 4.4, 4.7, 5.0, 5.3, 5.6, 5.9, 6.1, 6.4, 6.7, 7.0, 7.3, 7.6, 7.9, 8.2, 8.4, 8.7, 9.0, 9.3, 9.6, 9.9, 10.2, 10.4, 10.7, 11.0, 11.3, 11.6],
					  [6.1, 6.4, 6.4, 5.9, 5.5, 5.5, 5.8, 6.4, 6.9, 7.4, 7.8, 8.2, 8.7, 9.3, 9.5, 9.5, 9.6, 9.6, 9.5, 9.3, 9.3, 9.3, 9.1, 8.8, 8.4, 8.1, 8.0, 8.0, 7.9, 7.4, 6.9, 6.3, 5.8, 5.3, 4.8, 4.5, 4.1, 3.9, 3.7, 3.6, 3.6],
					  [1.1, 1.1, 1.0, 0.7 ,0.6, 0.5, 0.4, 0.4 ,0.4, 0.4, 0.4, 0.5, 0.6, 0.8, 1.0, 1.1, 1.1, 1.1, 1.1, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4],
					  [548.6, 390.7, 214.2, 110.4, 64.2, 46.4, 39.9, 37.9, 36.9, 35.2, 32.4, 29.7, 28.1, 27.5, 25.9, 22.8, 19.2, 15.6, 12.2, 9.4, 7.5, 6.3, 5.1, 4.1, 3.2, 2.3, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					  [68.4, 81.2, 54.9, 31.1, 17.4, 10.6, 7.0, 5.4, 4.9, 4.3, 3.5, 3.3, 3.9, 4.9, 5.7, 5.6, 5.1, 4.4, 3.7, 2.7, 1.9, 1.5, 1.4, 1.4, 1.4, 1.5, 1.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					  [5937.7, 1125.9, 708.0, 496.6, 405.6, 390.4, 356.3, 313.6, 287.1, 258.4, 236.2, 212.1, 200.8, 276.6, 187.3, 159.7, 146.3, 127.2, 110.4, 97.4, 184.5, 105.3, 99.9, 85.8, 71.1, 59.1, 49.0, 40.7, 43.0, 35.2, 35.4, 30.4, 22.6, 26.8, 36.6, 86.9, 20.4, 17.3, 17.2, 17.8, 12.9],
					  [348.7, 7.7, 4.5, 2.6, 1.8, 4.7, 1.9, 0.9, 1.0, 1.0, 1.5, 0.9, 1.5, 12.8, 1.8, 1.3, 1.0, 0.7, 0.6, 0.4, 10.5, 2.0, 4.4, 3.8, 1.1, 0.5, 0.4, 0.3, 1.0, 0.9, 0.9, 1.1, 0.5, 0.8, 1.9, 6.3, 0.5, 0.4, 0.5, 0.7, 0.3],
                      [2286, 1549.6, 771.9, 373.6, 227.6, 186.1, 184.5, 197.2, 205.9, 202.3, 198.2, 201.7, 210.5, 221.8, 230.3, 230.6, 210.8, 175, 138.6, 110, 92, 79.9, 73.2, 74, 80.4, 77.7, 64.9, 45.4, 31.3, 23.1, 18.8, 15.7, 12.9, 10.4, 7.9, 5.6, 3.8, 2.7, 2.3, 2, 1.5],
                      [553.4, 487.4, 265, 113.4, 48.6, 27, 21.2, 21.3, 25.9, 31.2, 24.7, 33.8, 31.9, 42.6, 60.7, 65.8, 57.1, 42.8, 31.8, 25.6, 23.9, 22.7, 22.9, 27.8, 35.7, 36.6, 28.7, 17.9, 9.9, 6.3, 5.9, 5.5, 4.8, 3.7, 2.4, 1.5, 1.3, 1.3, 1.3, 1.4, 1.4]])

	if LowVvir and not SFR:
		if HI and H2:
			if ax is None:
				plt.errorbar(N3351[0,:], N3351[1,:]+N3351[3,:], N3351[2,:]+N3351[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N3627[0,:], N3627[1,:]+N3627[3,:], N3627[2,:]+N3627[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N5055[0,:], N5055[1,:]+N5055[3,:], N5055[2,:]+N5055[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N6946[0,:], N6946[1,:]+N6946[3,:], N6946[2,:]+N6946[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			else:
				ax.errorbar(N3351[0,:], N3351[1,:]+N3351[3,:], N3351[2,:]+N3351[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N3627[0,:], N3627[1,:]+N3627[3,:], N3627[2,:]+N3627[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N5055[0,:], N5055[1,:]+N5055[3,:], N5055[2,:]+N5055[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N6946[0,:], N6946[1,:]+N6946[3,:], N6946[2,:]+N6946[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
		elif HI and not H2:
			if ax is None:
				plt.errorbar(N3351[0,:], N3351[1,:], N3351[2,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N3627[0,:], N3627[1,:], N3627[2,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N5055[0,:], N5055[1,:], N5055[2,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N6946[0,:], N6946[1,:], N6946[2,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			else:
				ax.errorbar(N3351[0,:], N3351[1,:], N3351[2,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N3627[0,:], N3627[1,:], N3627[2,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N5055[0,:], N5055[1,:], N5055[2,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N6946[0,:], N6946[1,:], N6946[2,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
		elif H2 and not HI:
			if ax is None:
				plt.errorbar(N3351[0,:], N3351[3,:], N3351[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N3627[0,:], N3627[3,:], N3627[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N5055[0,:], N5055[3,:], N5055[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N6946[0,:], N6946[3,:], N6946[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			else:
				ax.errorbar(N3351[0,:], N3351[3,:], N3351[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N3627[0,:], N3627[3,:], N3627[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N5055[0,:], N5055[3,:], N5055[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N6946[0,:], N6946[3,:], N6946[4,:], elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
		else:
			if ax is None:
				plt.errorbar(N3351[0,:], N3351[5,:]*0.61/0.66, N3351[6,:]*0.61/0.66, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N3627[0,:], N3627[5,:]*0.61/0.66, N3627[6,:]*0.61/0.66, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N5055[0,:], N5055[5,:]*0.61/0.66, N5055[6,:]*0.61/0.66, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				plt.errorbar(N6946[0,:], N6946[5,:]*0.61/0.66, N6946[6,:]*0.61/0.66, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			else:
				ax.errorbar(N3351[0,:], N3351[5,:]*0.61/0.66, N3351[6,:]*0.61/0.66, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N3627[0,:], N3627[5,:]*0.61/0.66, N3627[6,:]*0.61/0.66, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N5055[0,:], N5055[5,:]*0.61/0.66, N5055[6,:]*0.61/0.66, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
				ax.errorbar(N6946[0,:], N6946[5,:]*0.61/0.66, N6946[6,:]*0.61/0.66, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
	elif LowVvir and SFR:
		if ax is None:
			plt.errorbar(N3351[0,:], N3351[7,:]*1e-4*0.63/0.67, N3351[8,:]*1e-4*0.63/0.67, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			plt.errorbar(N3627[0,:], N3627[7,:]*1e-4*0.63/0.67, N3627[8,:]*1e-4*0.63/0.67, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			plt.errorbar(N5055[0,:], N5055[7,:]*1e-4*0.63/0.67, N5055[8,:]*1e-4*0.63/0.67, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			plt.errorbar(N6946[0,:], N6946[7,:]*1e-4*0.63/0.67, N6946[8,:]*1e-4*0.63/0.67, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
		else:
			ax.errorbar(N3351[0,:], N3351[7,:]*1e-4*0.63/0.67, N3351[8,:]*1e-4*0.63/0.67, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			ax.errorbar(N3627[0,:], N3627[7,:]*1e-4*0.63/0.67, N3627[8,:]*1e-4*0.63/0.67, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			ax.errorbar(N5055[0,:], N5055[7,:]*1e-4*0.63/0.67, N5055[8,:]*1e-4*0.63/0.67, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)
			ax.errorbar(N6946[0,:], N6946[7,:]*1e-4*0.63/0.67, N6946[8,:]*1e-4*0.63/0.67, elinewidth=1, ecolor=c, alpha=alpha, color=c, lw=lw)



def kennicuttschmidt(redshift, GM, SFR, Radii, s3=False):
	"""
	Make a Kennicutt-Schmidt plot for a single galaxy property integration technique.
	Designed for Marie's galaxies
	redshift = array of redshift values
	GM = array or list of arrays of integrated (cold) gas mass values at each redshift
	SFR = ditto but Star Formation Rate
	Radii = ditto but for the Radii of the apertures
	s3 = True if I'm doing 3 galaxies and want different symbols for each on the plot
	"""

	# Do redshifts bins of width 0.5, with everything above 2.5 binned in one
	z1 = (redshift<0.5)
	z2 = (redshift>=0.5)*(redshift<1)
	z3 = (redshift>=1)*(redshift<1.5)
	z4 = (redshift>=1.5)*(redshift<2)
	#z5 = (redshift>=2)*(redshift<2.5) # Even this range is untrustworthy, and certainly z<2.5 is
	zz = z1+z2+z3+z4#+z5

	if type(GM)!=list:
		GM = [GM]
		SFR = [SFR]

	xdata, ydata = np.array([]), np.array([])
	for i in xrange(len(GM)):
		S = np.pi*Radii[i]**2 # Surface area
		f = (S>0)*(GM[i]>0)*(SFR[i]>0) # Filter to remove any zero values

		if s3==True:
			if i==0:
				m, mew = 's', 0
			elif i==1:
				m, mew = 'x', 2
			elif i==2:
				m, mew = '.', 5
		else:
			m, mew = '.', 3

		plt.plot(np.log10(GM[i][z1*f]/S[z1*f]), np.log10(SFR[i][z1*f]/S[z1*f]), marker=m, color='b', ms=10, mew=mew, lw=0)
		plt.plot(np.log10(GM[i][z2*f]/S[z2*f]), np.log10(SFR[i][z2*f]/S[z2*f]), marker=m, color='c', ms=10, mew=mew, lw=0)
		plt.plot(np.log10(GM[i][z3*f]/S[z3*f]), np.log10(SFR[i][z3*f]/S[z3*f]), marker=m, color='g', ms=10, mew=mew, lw=0)
		plt.plot(np.log10(GM[i][z4*f]/S[z4*f]), np.log10(SFR[i][z4*f]/S[z4*f]), marker=m, color='y', ms=10, mew=mew, lw=0)
		#plt.plot(np.log10(GM[i][z5*f]/S[z5*f]), np.log10(SFR[i][z5*f]/S[z5*f]), marker=m, color='m', ms=10, mew=mew, lw=0)

		xdata = np.append(xdata, np.log10(GM[i][zz*f]/S[zz*f]))
		ydata = np.append(ydata, np.log10(SFR[i][zz*f]/S[zz*f])) # Puts all the plotted data into one array
	
			
	# The following plots are just for the legend
	plt.plot([20,21],[12,12],'b.',ms=10,mew=5,label=r'M12, $z < 0.5$')
	plt.plot([20,21],[12,12],'c.',ms=10,mew=5,linewidth=2,label=r'M12, $0.5 \leq z < 1$')
	plt.plot([20,21],[12,12],'g.',ms=10,mew=5,linewidth=2,label=r'M12, $1 \leq z < 1.5$')
	plt.plot([20,21],[12,12],'y.',ms=10,mew=5,linewidth=2,label=r'M12, $1.5 \leq z < 2$')
	#plt.plot([20,21],[12,12],'m.',ms=10,mew=5,linewidth=2,label=r'$2 \leq z < 2.5$')
			
	# Linear fit for all the plotted data
	p = np.polyfit(xdata,ydata,1)
	xp = np.array([-10,20]) # Linear fit so only need 2 points to plot it
	yp = xp*p[0]+p[1]
	chi2, rms = gc.chisqr(ydata, xdata*p[0]+p[1])
	#plt.plot(xp,yp,'k--',linewidth=2, label=r'$y='+str(int(p[0]*100+0.5)/100.)+r'x - '+str(int(abs(p[1])*100+0.5)/100.)+r'$')
	plt.plot(xp,yp,'r--',linewidth=3, label=r'M12 fit, $n = '+str(int(p[0]*100+0.5)/100.)+r'$')
	plt.axis([np.min(xdata)-0.2, np.max(xdata)+0.2, np.min(ydata)-0.2, np.max(ydata)+0.2])
	plt.ylabel(r'$\log_{10} (\Sigma_{\mathrm{SFR}}\ [\mathrm{M}_{\odot}\ \mathrm{yr}^{-1}\ \mathrm{pc}^{-2}])$')
	plt.xlabel(r'$\log_{10} (\Sigma_{\mathrm{gas}}\ [\mathrm{M}_{\odot}\ \mathrm{pc}^{-2}])$')

	# Also do a linear fit with a forced gradient of 1.4
	p0 = np.polyfit(xdata, ydata-1.4*xdata, 0)[0]
	chi2_2, rms14 = gc.chisqr(ydata, xdata*1.4+p0)
	yp2 = xp*1.4 + p0
	#plt.plot(xp,yp2,'k:',linewidth=2, label=r'$y=1.4x - $'+str(int(abs(p0)*100+0.5)/100.))

	# Get deviation from the best-fit line from Kennicutt (1998)
	chi2_3, rms3 = gc.chisqr(ydata, xdata*1.4+np.log10(2.5e-10))
			

	return p, rms, rms14, rms3


def kennicuttschmidt2(GM, SFR, Radii, colour='b', ms=2, alpha=0.5):
	# Designed for MB-II galaxies, but should be generally applicable.
	
	f = (GM>0)*(Radii>0)*(SFR>0) # Filter to get rid of zero-values that are naturally going to be error-some
	
	xdata = np.log10(GM[f] / (4*np.pi*Radii[f]**2))
	ydata = np.log10(SFR[f] / (4*np.pi*Radii[f]**2))
	
	plt.plot(xdata, ydata, colour+'o', ms=ms, mew=0, alpha=alpha)
	
	p = np.polyfit(xdata,ydata,1)
	xp = np.array([-10,20]) # Linear fit so only need 2 points to plot it
	yp = xp*p[0]+p[1]
	chi2, rms = gc.chisqr(ydata, xdata*p[0]+p[1])
	
	#plt.plot(xp,yp,'k--',linewidth=2, label=r'$y='+str(int(p[0]*100+0.5)/100.)+r'x - '+str(int(abs(p[1])*100+0.5)/100.)+r'$')
	plt.plot(xp,yp,'m--',lw=3, label=r'\emph{MB-II} fit, $n = '+str(int(p[0]*100+0.5)/100.)+r'$')
	plt.axis([np.min(xdata)-0.2, np.max(xdata)+0.2, np.min(ydata)-0.2, np.max(ydata)+0.2])
	plt.ylabel(r'$\log_{10} (\Sigma_{\mathrm{SFR}}\ [\mathrm{M}_{\odot}\ \mathrm{yr}^{-1}\ \mathrm{pc}^{-2}])$')
	plt.xlabel(r'$\log_{10} (\Sigma_{\mathrm{gas}}\ [\mathrm{M}_{\odot}\ \mathrm{pc}^{-2}])$')
	
	# Also do a linear fit with a forced gradient of 1.4
	p0 = np.polyfit(xdata, ydata-1.4*xdata, 0)[0]
	chi2_2, rms14 = gc.chisqr(ydata, xdata*1.4+p0)
	yp2 = xp*1.4 + p0

	# Get deviation from the best-fit line from Kennicutt (1998)
	chi2_3, rms3 = gc.chisqr(ydata, xdata*1.4+np.log10(2.5e-10))
	
	return p, rms, rms14, rms3



def smsfr(z, SM, SFR):
	"""
	Create a stellar mass vs SFR plot and relate closeness to observations.
	SM = stellar mass, either an array or a list of arrays
	SFR = star formation rate, same deal
	"""

	z0f = (z>0.015)*(z<0.1) # Filter for the z~0 relation
	z1f = (z>0.8)*(z<1.2) # Ditto z~1 relation
	z2f = (z>1.9)*(z<2.0) # Ditto z~2 relation. Should really be 1.4<z<2.5, but for Marie's sims, don't trust above z=2.

	z0, z1, z2 = z[z0f], z[z1f], z[z2f]
	
	if type(SM)!=list:
		SM = [SM]
		SFR = [SFR]

	SM0, SM1, SM2, SFR0, SFR1, SFR2 = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
	for i in xrange(len(SM)):
		zerof = (SM[i]>0)*(SFR[i]>0) # Don't want zero-values.
		SM0 = np.append(SM0, np.log10(SM[i][z0f*zerof]))
		SM1 = np.append(SM1, np.log10(SM[i][z1f*zerof]))
		SM2 = np.append(SM2, np.log10(SM[i][z2f*zerof]))
		SFR0 = np.append(SFR0, np.log10(SFR[i][z0f*zerof]))
		SFR1 = np.append(SFR1, np.log10(SFR[i][z1f*zerof]))
		SFR2 = np.append(SFR2, np.log10(SFR[i][z2f*zerof]))

	# Plot the input data
	plt.plot(SM0, SFR0, 'b.', mew=5)
	plt.plot(SM1, SFR1, 'g.', mew=5)
	plt.plot(SM2, SFR2, 'r.', mew=5)

	# Plot the observational trends from Elbaz et al. (2007) and Daddi et al. (2007)
	xlin = np.array([1e1,1e20])
	xplot = np.log10(xlin)
	y0max = np.log10((8.7+7.4)*(xlin/1e11)**0.77)
	y0min = np.log10((8.7-3.7)*(xlin/1e11)**0.77)
	y1max = np.log10((7.2+7.2)*(xlin/1e10)**0.9)
	y1min = np.log10((7.2-3.6)*(xlin/1e10)**0.9)
	y2 = np.log10(200*(xlin/1e11)**0.9)
	plt.fill_between(xplot, y0max, y0min, color='b', alpha=0.3)
	plt.fill_between(xplot, y1max, y1min, color='g', alpha=0.3)
	plt.plot(xplot, y2, 'r--', lw=2)

	plt.xlabel(r'$\log_{10}$($M_{\mathrm{stars}}$ [$M_{\odot}$])')
	plt.ylabel(r'$\log_{10}$(SFR [$M_{\odot}$ yr$^{-1}$])')

	plt.axis([np.min(SM2)-0.1, np.max(SM0)+0.1, np.min(SFR0)-0.1, np.max(SFR2)+0.1])

	chisqr0, rms0 = gc.chisqr(SFR0, np.log10(8.7*((10**SM0)/1e11)**0.77))
	chisqr1, rms1 = gc.chisqr(SFR1, np.log10(7.2*((10**SM1)/1e10)**0.9))
	chisqr2, rms2 = gc.chisqr(SFR2, np.log10(200*((10**SM2)/1e11)**0.9))

	return rms0, rms1, rms2


def shm(SM, DMM, ps='g.', label=None, fsize=24):
	# Do a stellar-halo mass relation plot
	# ps = plot style
	mM1, logM = gc.smhm(N=0.0351-0.0058, M1=11.590+0.236, beta=1.376+0.153, gamma=0.608+0.059) # Calculate range for fraction from Moster et al (2013)
	mM2, logM = gc.smhm(N=0.0351+0.0058, M1=11.590-0.236, beta=1.376-0.153, gamma=0.608-0.059)
	mM3, LogM = gc.smhm(N=0.0351+0.0058, M1=11.590+0.236, beta=1.376-0.153, gamma=0.608-0.059)
	mM4, logM = gc.smhm(N=0.0351-0.0058, M1=11.590-0.236, beta=1.376+0.153, gamma=0.608+0.059)
	mM5, logM = gc.smhm(N=0.0351-0.0058, M1=11.590, beta=1.376+0.153, gamma=0.608+0.059)
	mM6, LogM = gc.smhm(N=0.0351+0.0058, M1=11.590, beta=1.376-0.153, gamma=0.608-0.059)
	mM_low = np.array([min(mM1[i],mM2[i],mM3[i],mM4[i],mM5[i],mM6[i]) for i in range(len(logM))])
	mM_high = np.array([max(mM1[i],mM2[i],mM3[i],mM4[i],mM5[i],mM6[i]) for i in range(len(logM))])
	plt.fill_between(logM, mM_high, mM_low, color='b', alpha=0.3)
	plt.plot([1,1],[1,2],'b-',lw=4,alpha=0.5,label=r'Moster et al. (2013)')
	
	f = (DMM>0)*(SM>0)
	DMM, SM = DMM[f], SM[f]
	plt.plot(np.log10(DMM), SM/DMM, ps, label=label, markersize=3)
	plt.yscale('log', nonposy='clip')

	plt.legend(loc='lower right',fontsize=fsize-6,frameon=False)
	plt.ylabel(r'$M_{\mathrm{stars}} / M_{\mathrm{DM}}$', fontsize=fsize)
	plt.xlabel(r'$\log_{10}(M_{\mathrm{DM}}\ [M_{\odot}])$', fontsize=fsize)


def SDSS_LFs(Hubble_h=0.6777):
    # Adapted from Darren's TAO plotting routine
    M = np.arange(-30, -15, 0.1)
    k_correct_zeq0 = -2.5*np.log(1.1) # shift observations from z=0.1 to z=0.0 (Blanton et al. 2003 eq.4)
    
    # SDSS u
    Mstar = -17.93 +5.0*np.log10(Hubble_h) +k_correct_zeq0
    alpha = -0.92
    phistar = 0.0305 *Hubble_h*Hubble_h*Hubble_h
    xval = 10.0 ** (0.4*(Mstar-M))
    yval = 0.4 * np.log(10.0) * phistar * xval ** (alpha+1) * np.exp(-xval)
    plt.plot(M, yval, 'y--', lw=1.0)

    # SDSS g
    Mstar = -19.39 +5.0*np.log10(Hubble_h) +k_correct_zeq0
    alpha = -0.89
    phistar = 0.0218 *Hubble_h*Hubble_h*Hubble_h
    xval = 10.0 ** (0.4*(Mstar-M))
    yval = 0.4 * np.log(10.0) * phistar * xval ** (alpha+1) * np.exp(-xval)
    plt.plot(M, yval, 'b--', lw=1.0)
    
    # SDSS r
    Mstar = -20.44 +5.0*np.log10(Hubble_h) +k_correct_zeq0
    alpha = -1.05
    phistar = 0.0149 *Hubble_h*Hubble_h*Hubble_h
    xval = 10.0 ** (0.4*(Mstar-M))
    yval = 0.4 * np.log(10.0) * phistar * xval ** (alpha+1) * np.exp(-xval)
    plt.plot(M, yval, 'r--', lw=1.0)
            
    # SDSS i
    Mstar = -20.82 +5.0*np.log10(Hubble_h) +k_correct_zeq0
    alpha = -1.00
    phistar = 0.0147 *Hubble_h*Hubble_h*Hubble_h
    xval = 10.0 ** (0.4*(Mstar-M))
    yval = 0.4 * np.log(10.0) * phistar * xval ** (alpha+1) * np.exp(-xval)
    plt.plot(M, yval, 'c--', lw=1.0)

    # SDSS z
    Mstar = -21.18 +5.0*np.log10(Hubble_h) +k_correct_zeq0
    alpha = -1.08
    phistar = 0.0135 *Hubble_h*Hubble_h*Hubble_h
    xval = 10.0 ** (0.4*(Mstar-M))
    yval = 0.4 * np.log(10.0) * phistar * xval ** (alpha+1) * np.exp(-xval)
    plt.plot(M, yval, 'k--', lw=1.0)

def Kband_obs(Hubble_h=0.6777):
    # Cole et al. 2001 K band 2dFGRS LF
    Cole_Phi = np.array([3.1315561E-03, 8.2625253E-03, 0.0000000E+00, 4.6483092E-03, 5.7576019E-03, 9.1649834E-03, 1.1232893E-02,
         1.0536440E-02, 8.5763102E-03, 8.8181989E-03, 6.9448259E-03, 6.0896124E-03, 9.2596142E-03, 6.9631678E-03,
         7.2867479E-03, 6.9923755E-03, 5.9844730E-03, 5.9305103E-03, 5.3865365E-03, 5.8525647E-03, 5.2373926E-03,
         4.9635037E-03, 4.1801766E-03, 2.7171015E-03, 1.8800517E-03, 1.2136410E-03, 6.5419916E-04, 3.4594961E-04,
         1.4771589E-04, 5.5521199E-05, 2.1283222E-05, 9.4211919E-06, 1.0871951E-06, 2.7923562E-07])
    Cole_PhiErr = np.array([3.6377162E-03, 6.6833422E-03, 1.0000000E-10, 4.0996978E-03, 4.3155681E-03, 5.6722397E-03, 6.4211683E-03,
        5.7120644E-03, 4.6346937E-03, 3.8633577E-03, 2.4383855E-03, 1.6279118E-03, 1.6941463E-03, 1.1781409E-03,
        9.7785855E-04, 7.9027453E-04, 6.0649612E-04, 5.1598746E-04, 4.2267537E-04, 3.7395130E-04, 2.8177485E-04,
        2.1805518E-04, 1.6829016E-04, 1.1366483E-04, 8.1871600E-05, 5.7472309E-05, 3.6554517E-05, 2.3141622E-05,
        1.2801432E-05, 6.5092854E-06, 3.3540452E-06, 1.9559407E-06, 5.6035748E-07, 2.8150106E-07])
    Cole_Mag = np.array([-18.00000, -18.25000, -18.50000, -18.75000, -19.00000, -19.25000, -19.50000, -19.75000, -20.00000,
         -20.25000, -20.50000, -20.75000, -21.00000, -21.25000, -21.50000, -21.75000, -22.00000, -22.25000,
         -22.50000, -22.75000, -23.00000, -23.25000, -23.50000, -23.75000, -24.00000, -24.25000, -24.50000,
         -24.75000, -25.00000, -25.25000, -25.50000, -25.75000, -26.00000, -26.25000])

    # Huang et al. 2003 K band Hawaii+AAO LF
    Huang_Phi = np.array([0.0347093, 0.0252148, 0.0437980, 0.0250516, 0.00939655, 0.0193473, 0.0162743, 0.0142267, 0.0174460,
          0.0100971, 0.0136507, 0.00994688, 0.00655286, 0.00528234, 0.00310017, 0.00157789, 0.000721131,
          0.000272634, 8.33409e-05, 2.12150e-05, 3.97432e-06, 5.07697e-06, 5.42939e-07])
    Huang_PhiErr = np.array([ 0.0249755, 0.0181685, 0.0161526, 0.0105895, 0.00479689, 0.00525068, 0.00428192, 0.00308970, 0.00248676,
         0.00166458, 0.00166691, 0.00106289, 0.000704721, 0.000527429, 0.000340814, 0.000170548, 8.25681e-05,
         3.81529e-05, 1.50279e-05, 6.16614e-06, 2.34362e-06, 1.98971e-06, 5.54946e-07])
    Huang_Mag = np.array([-19.8000, -20.1000, -20.4000, -20.7000, -21.0000, -21.3000, -21.6000, -21.9000, -22.2000, -22.5000,
      -22.8000, -23.1000, -23.4000, -23.7000, -24.0000, -24.3000, -24.6000, -24.9000, -25.2000,
      -25.5000, -25.8000, -26.1000, -26.4000])

    plt.errorbar(Cole_Mag+5.0*np.log10(Hubble_h), Cole_Phi*Hubble_h*Hubble_h*Hubble_h, yerr=Cole_PhiErr*Hubble_h*Hubble_h*Hubble_h, color='m', lw=1.0, marker='o', ls='none', label='Cole et al. 2001')
    plt.errorbar(Huang_Mag+5.0*np.log10(Hubble_h), Huang_Phi*Hubble_h*Hubble_h*Hubble_h, yerr=Huang_PhiErr*Hubble_h*Hubble_h*Hubble_h, color='g', lw=1.0, marker='o', ls='none', label='Huang et al. 2003')


