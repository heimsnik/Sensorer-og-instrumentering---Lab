import numpy as np


muabo = np.genfromtxt("C:/Users/cmhei/OneDrive/Dokumenter/Semester 6/TTK4280 Sensorer og instrumentering/Lab/Lab 3/Programmer/muabo.txt", delimiter=",")
muabd = np.genfromtxt("C:/Users/cmhei/OneDrive/Dokumenter/Semester 6/TTK4280 Sensorer og instrumentering/Lab/Lab 3/Programmer/muabd.txt", delimiter=",")

red_wavelength = 700 # Replace with wavelength in nanometres
green_wavelength = 546 # Replace with wavelength in nanometres
blue_wavelength = 436 # Replace with wavelength in nanometres

wavelength = np.array([red_wavelength, green_wavelength, blue_wavelength])

def mua_blood_oxy(x): return np.interp(x, muabo[:, 0], muabo[:, 1])
def mua_blood_deoxy(x): return np.interp(x, muabd[:, 0], muabd[:, 1])

bvf1 = 1 # Blood volume fraction, average blood amount in tissue
bvf0 = 0.01 # Blood volume fraction, average blood amount in tissue
oxy = 0.8 # Blood oxygenation

# Absorption coefficient ($\mu_a$ in lab text)
# Units: 1/m
mua_other = 25 # Background absorption due to collagen, et cetera
mua_blood = (mua_blood_oxy(wavelength)*oxy # Absorption due to
            + mua_blood_deoxy(wavelength)*(1-oxy)) # pure blood
mua1 = mua_blood*bvf1 + mua_other
mua0 = mua_blood*bvf0 + mua_other

# reduced scattering coefficient ($\mu_s^\prime$ in lab text)
# the numerical constants are thanks to N. Bashkatov, E. A. Genina and
# V. V. Tuchin. Optical properties of skin, subcutaneous and muscle
# tissues: A review. In: J. Innov. Opt. Health Sci., 4(1):9-38, 2011.
# Units: 1/m
musr = 100 * (17.6*(wavelength/500)**-4 + 18.78*(wavelength/500)**-0.22)

# mua and musr are now available as shape (3,) arrays
# Red, green and blue correspond to indexes 0, 1 and 2, respectively

C0 = np.sqrt(3*(musr+mua0)*mua0)
C1 = np.sqrt(3*(musr+mua1)*mua1)

# TODO calculate penetration depth

def penetration_depth(mu_s,mu_a):
    return np.sqrt(1/(3*(mu_s+mu_a)*mu_a))

def fluence_rate(d,mu_s,mu_a):
    return 1/(2*penetration_depth(mu_s,mu_a)*mu_a)*np.e**(-C*d)

def T(C,d):
    return np.e**(-C*d)

def reflection_coeff(mu_s, mu_a):
    return np.sqrt(3*(mu_s/mu_a+1))

def exp_z(C,z):
    return np.e**(-C*z)

def K():
    return (np.abs(T(C1,0.0003)-T(C0,0.0003))/T(C0,0.0003))



#print(penetration_depth(musr,mua))
#print(transmission_coeff(fluence_rate(0),fluence_rate(0.017)))
#print(reflection_coeff(musr,mua))

#print(exp_z(2*0.017))

print(T(C0,0.0003))
print(T(C1,0.0003))

#print(abs((transmission_coeff(fluence_rate(0,musr,mua1),fluence_rate(0.0003,musr,mua1))-transmission_coeff(fluence_rate(0,musr,mua0),fluence_rate(0.0003,musr,mua0))))/transmission_coeff(fluence_rate(0,musr,mua0),fluence_rate(0.0003,musr,mua0)))
#print(K())