# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt

##############################################################################
# STEP 1
print('STEP 1:')



# Unpacking data
parallax,e_parallax,period,apparentmag,extinction,e_extinction=np.loadtxt('MW_Cepheids.dat',\
                                                                        unpack = True,\
                                                                        usecols=(1,2,3,4,5,6),\
                                                                        dtype=float)

# Finding distance to earth using parallax    
dpc = 1000/parallax
e_dpc = np.sqrt((((-1000/parallax**2)**2)*e_parallax**2))



# Finding absolute magnitude using values calculated of distance from earth
absolutemag = apparentmag - 5*np.log10(dpc)+5-extinction
e_absolutemag = np.sqrt((((-5/(np.log(10)*dpc))**2)*(e_dpc**2))+((e_extinction**2)))

#Allowing for intrinsic dispersion to reduce chi^2
intrinsic = 0.3
e_absolutemag = np.sqrt((e_absolutemag**2)+(intrinsic**2))

# Finding logp and mean of logp to shift data to account for correlation of α and β
logp = np.log10(period)
logpmean = np.mean(logp)
shiftedlogp = logp - 0.864


# Plot of absolute magnitude vs logp
plt.plot(shiftedlogp,absolutemag,marker="o",markersize=5,linestyle = 'None')
plt.errorbar(shiftedlogp,absolutemag,yerr=e_absolutemag,linestyle = 'None',color='tab:blue')
plt.xlabel('Normalized log(P) (days) ')
plt.ylabel('Absolute Magnitude (mag)')


def model(a,x,b):
    return(a*x+b)


# Using curve fit to find values of α and β
guess = [-1.4,-3]
popt,pcov = opt.curve_fit(f = model, xdata = shiftedlogp,ydata = absolutemag,sigma = e_absolutemag,
                    p0=guess,absolute_sigma = True)


# Assigning α and β from optimal parameter matrix
α = popt[0]
β = popt[1]


# Plotting fit of data
plt.plot(shiftedlogp, model(α,shiftedlogp,β))
plt.show()
plt.close()


# Working out error on α and β using covariance matrix
e_α = np.sqrt(pcov[0,0])
e_β = np.sqrt(pcov[1,1])


print('α =',α,'+/-',e_α)
print('β =',β,'+/-',e_β)
print('M = -2.40logp - 3.68')


# Creating correlation matrix to check correlation is minimised
corr = np.zeros((2,2))
corr[0,0]=pcov[0,0]/(e_α*e_α)
corr[1,1]=pcov[1,1]/(e_β*e_β)
corr[0,1]=pcov[0,1]/(e_α*e_β)
corr[1,0]=pcov[1,0]/(e_β*e_α)


expected_y = α*shiftedlogp+β
observed_y = absolutemag + e_absolutemag


def chi2(y,y_m,sig_y):
    """

    Parameters
    ----------
    y : Observed y values.
    y_m : Expected y values.
    sig_y : Errors on y values.

    Returns
    -------
    Chi^2

    """
    chi2 = np.sum(((y-y_m)**2)/(sig_y**2))
    return(chi2)


expected_y = α*shiftedlogp+β
observed_y = absolutemag + e_absolutemag


# Degrees of freedom
degree = len(shiftedlogp)-2


chi2_MW = chi2(observed_y,expected_y,e_absolutemag)
reducedchi2_MW = chi2_MW/degree
e_reducedchi2_MW = np.sqrt(2/10)


print('Chi^2 for fit of M is',chi2_MW)
print('Reduced chi^" = ',reducedchi2_MW,'+/-',e_reducedchi2_MW)



###############################################################################
# STEP 2
print('')
print('STEP 2:')

logp_ngc,apparentmag_ngc = np.loadtxt('ngc4527_cepheids.dat',\
                        unpack = True,\
                        usecols = (1,2),\
                        dtype = float)


# Plot of apparent magnitude vs logp
plt.plot(logp_ngc,apparentmag_ngc,marker="o",markersize=5,linestyle = 'None')
plt.xlabel('log(P)(Days)')
plt.ylabel('Apparent Magnitude (mag)')
plt.show()
plt.close()

# Outlier identified with max value of apparentmag
# Max(apparentmag_ngc) = 26.78
# Value corresponds to 6th location in array logp_ngc and apparentmag_ngc


# Removing outlier
logp_ngc=np.delete(logp_ngc,6,axis=0)
apparentmag_ngc=np.delete(apparentmag_ngc,6,axis=0)


# Plot without outlier
plt.plot(logp_ngc,apparentmag_ngc,marker="o",markersize=5,linestyle = 'None')
plt.xlabel('log(P)(Days)')
plt.ylabel('Apparent Magnitude (mag)')
plt.show()
plt.close()


# Calculating Absolute magnitude for NGC4527 (shifting logp_ngc)
shiftedlogp_ngc = logp_ngc - 0.864
absolutemag_ngc = α*shiftedlogp_ngc + β
e_absolutemag_ngc = np.sqrt((((np.log10(logp_ngc)-0.864)**2)*((e_α)**2))+((e_β)**2))


# Calculating dpc for each cepheid
extinction_ngc = 0.0682
dpc_ngc=10**((apparentmag_ngc+5-extinction_ngc-absolutemag_ngc)/(5))
e_dpc_ngc=np.sqrt((((-np.log(10)*10**((-absolutemag_ngc+apparentmag_ngc-extinction_ngc+5)/(5)))/(5))**2)*((e_absolutemag_ngc)**2))


def weightedmean(x,sig):
    """
    
    Parameters
    ----------
    x : Values of data to calculated weighted mean.
    sig : Errors on values.

    Returns
    -------
    Weighted mean

    """
    w=1/(sig**2)
    weightedmean=(np.sum(w*x))/(np.sum(w))
    return(weightedmean)

def e_weightedmean(sig):
    """
    

    Parameters
    ----------
    sig : Errors on values.

    Returns
    -------
    Error on weighted mean.

    """
    e_weightedmean = np.sqrt(1/(np.sum(1/(sig**2))))
    return(e_weightedmean)
    
#Calculating distance from earth using weighted mean of all distances    
distance_from_earth = weightedmean(dpc_ngc,e_dpc_ngc)*(10**-6)
e_distance_from_earth = e_weightedmean(e_dpc_ngc)*(10**-6)
print('Distance from earth to NGC4527 =' , distance_from_earth,'+/-',e_distance_from_earth,'Mpc')




###############################################################################
#STEP 3

print('')
print('STEP 3:')

vrec,dgal,e_dgal = np.loadtxt('other_galaxies.dat',\
                              unpack = True,\
                              usecols = (1,2,3),\
                              dtype = float)
dgal = np.append(dgal,distance_from_earth)
vrec = np.append(vrec,1152)
e_dgal = np.append(e_dgal,e_distance_from_earth)


y = dgal+e_dgal # Observed data

# Code for brute force gridding (used from lecture notes)
slopes = np.arange(0, 0.04, 0.000001) 
chi = 1.e5 + slopes*0.0  
best_slope = 1.e5 
best_chi2 = 1.e5
i = 0
for m in slopes:
    y_test = m*vrec 
    chi[i] = chi2(y, y_test,e_dgal)
    if (chi[i] < best_chi2):
        # print('arrayval',chi[i])
        best_chi2 = chi[i]
        best_slope = m    
    i=i+1
    
dof = len(vrec)-1.0    
    


chi2_h0 = best_chi2
chi2_h0reduced = best_chi2/dof
e_chi2_h0reduced = np.sqrt(2/8)
hubble_constant = 1/best_slope

#Plot of hubble diagram
plt.plot(dgal,vrec,marker="o",markersize=5,linestyle = 'None')
plt.errorbar(dgal,vrec,yerr= e_dgal, linestyle = 'None',color='tab:blue')
plt.xlabel('Distance to Galaxy (Mpc)')
plt.ylabel('Recession Velocity (km/s)')

 
plt.plot(dgal, model((hubble_constant),dgal,0))
plt.show()
plt.close()

print('Hubble constant = ', hubble_constant, 'km/s/Mpc')
print('Chi^2 for fit of H0 is',chi2_h0 )
print('Reduced chi^2 = ',chi2_h0reduced,'+/-',e_chi2_h0reduced)























