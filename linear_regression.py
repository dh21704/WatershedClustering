import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('/Users/danielhernandez/Downloads/Watershed_Characteristics.csv')

print(df.info())
#x
a_a_precip = df['Annual Avg. Precipitation']

#y
q = df['Annual Avg. River Flow']

#Linear regression for Precipitation and RiverFlow
q_slope, q_intercept, q_r, q_p, q_std_err = stats.linregress(a_a_precip, q)

regression_line = q_slope * a_a_precip + q_intercept

plt.scatter(a_a_precip, q)
plt.plot(a_a_precip, regression_line)
plt.title('Annual Avg Precipitation vs Annual Avg. River Flow ALL DATA')
plt.xlabel('Annual Avg. Precipitation (mm)', size = 15)
plt.ylabel('Annual Avg. River Flow (mm)', size=15)
plt.grid(True)
plt.show()

print('Annual Avg Precipitation vs Annual Avg. River Flow ALL DATA')
print('q_r is ', q_r)
print('R^2 is ', q_r * q_r)
print('slope: ', q_slope)
print('intercept ', q_intercept)

#######################################################################################


#x
a_a_precip = df['Annual Avg. Precipitation']

#y
aquifier_recharge = df['Annual Avg. Aquifer Recharge']

#Linear regression for Precipitation and RiverFlow
q_slope, q_intercept, q_r, q_p, q_std_err = stats.linregress(a_a_precip, aquifier_recharge)

regression_line = q_slope * a_a_precip + q_intercept

plt.scatter(a_a_precip, aquifier_recharge)
plt.plot(a_a_precip, regression_line)
plt.title('Annual Avg Precipitation vs Annual Avg. Aquifer Recharge ALL DATA')
plt.xlabel('Annual Avg. Precipitation (mm)', size = 15)
plt.ylabel('Annual Avg. Aquifer Recharge (mm)', size=15)
plt.grid(True)
plt.show()

print(), print(), print(), print(), print()
print('Annual Avg Precipitation vs Annual Avg. Aquifer Recharge ALL DATA')
print('q_r is ', q_r)
print('R^2 is ', q_r * q_r)
print('slope: ', q_slope)
print('intercept ', q_intercept)




#######################################################
#x
p_divided_by_potential_evapotranspiration = df['Annual Avg. Precipitation'] / df['Annual Avg. Potential Evapotranspiration']

#y
aquifier_recharge = df['Annual Avg. Aquifer Recharge']

#Linear regression for Precipitation and RiverFlow
q_slope, q_intercept, q_r, q_p, q_std_err = stats.linregress(p_divided_by_potential_evapotranspiration, aquifier_recharge)

regression_line = q_slope * p_divided_by_potential_evapotranspiration + q_intercept

plt.scatter(p_divided_by_potential_evapotranspiration, aquifier_recharge)
plt.plot(p_divided_by_potential_evapotranspiration, regression_line)
plt.title('(Annual Avg. Precipitation / Annual Avg. Potential Evapotranspiration) vs Annual Avg. Aquifer Recharge ALL DATA', size=8)
plt.xlabel('Annual Avg. Precipitation (mm) / Annual Avg. Potential Evapotranspiration')
plt.ylabel('Annual Avg. Aquifer Recharge (mm)', size=15)
plt.grid(True)
plt.show()

print(), print(), print(), print(), print()
print('Annual Avg. Precipitation / Annual Avg. Potential Evapotranspiration) vs Annual Avg. Aquifer Recharge ALL DATA')
print('q_r is ', q_r)
print('R^2 is ', q_r * q_r)
print('slope: ', q_slope)
print('intercept ', q_intercept)

###################################################################################
#x
p_e = df['Annual Avg. Precipitation']

#y
elevation = df['Elevation']

#Linear regression for Precipitation and RiverFlow
a_e_slope, a_e_intercept, a_e_r, a_e_p, a_e_std_err = stats.linregress(p_e, elevation)

regression_line = a_e_slope * p_e + a_e_intercept

plt.scatter(p_e, elevation)
plt.plot(p_e, regression_line)
plt.title('Annual Avg Precipitation vs Elevation DATA')
plt.xlabel('Annual Avg. Precipitation (mm)', size = 15)
plt.ylabel('Elevation (meters)', size=15)
plt.grid(True)
plt.show()

print(), print(), print(), print(), print()
print('Annual Avg Precipitation vs Elevation ALL DATA')
print('q_r is ', a_e_r)
print('R^2 is ', a_e_r * a_e_r)
print('slope: ', a_e_slope)
print('intercept ', a_e_intercept)


###################################################################################
#x
p_e = df['Annual Avg. Precipitation']

#y
elevation = df['Elevation']

#Linear regression for Precipitation and RiverFlow
a_e_slope, a_e_intercept, a_e_r, a_e_p, a_e_std_err = stats.linregress(p_e, elevation)

regression_line = a_e_slope * p_e + a_e_intercept

plt.scatter(p_e, elevation)
plt.plot(p_e, regression_line)
plt.title('Annual Avg Precipitation vs Elevation DATA')
plt.xlabel('Annual Avg. Precipitation (mm)', size = 15)
plt.ylabel('Elevation (meters)', size=15)
plt.grid(True)
plt.show()

print(), print(), print(), print(), print()
print('Annual Avg Precipitation vs Elevation ALL DATA')
print('q_r is ', a_e_r)
print('R^2 is ', a_e_r * a_e_r)
print('slope: ', a_e_slope)
print('intercept ', a_e_intercept)

###################################################################################
#x
p_a = df['Annual Avg. Precipitation']

#y
area = df['Area']

#Linear regression for Precipitation and RiverFlow
p_a_slope, p_a_intercept, p_a_r, p_a_p, p_a_std_err = stats.linregress(p_a, area)

regression_line = p_a_slope * p_a + p_a_intercept

plt.scatter(p_a, area)
plt.plot(p_a, regression_line)
plt.title('Annual Avg Precipitation vs Area DATA')
plt.xlabel('Annual Avg. Precipitation (mm)', size = 15)
plt.ylabel('Area (square kilometers)', size=15)
plt.grid(True)
plt.show()

print(), print(), print(), print(), print()
print('Annual Avg Precipitation vs Area ALL DATA')
print('q_r is ', a_e_r)
print('R^2 is ', p_a_r * p_a_r)
print('slope: ', p_a_slope)
print('intercept ', p_a_intercept)


################################################################################
temp = df['Annual Avg. Temp']
pre = df['Annual Avg. Precipitation']

t_p_slope, t_p_intercept, t_p_r, t_p_p, t_p_std_err = stats.linregress(pre, temp)
regression_line_p_e = t_p_slope * pre + t_p_intercept

plt.scatter(pre, temp)
plt.plot(pre, regression_line_p_e)
plt.title('Precipitation vs Temp')
plt.xlabel('Avg Precipition')
plt.ylabel('Temp')
plt.grid(True)
plt.show()

print(), print(), print(), print(), print()
print('Annual Avg Precipitation vs Temp ALL DATA')
print('q_r is ', t_p_r)
print('R^2 is ', t_p_r * t_p_r)
print('slope: ', t_p_slope)
print('intercept ', t_p_intercept)