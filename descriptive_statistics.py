import pandas as pd

df = pd.read_csv('/Users/danielhernandez/Downloads/Watershed_Characteristics.csv')

#Calculate descriptive statistics such mean, median and standard deviation to describe
# catchment area, precipitation, ET, recharge and streamflow across your 203 catchments.

print(df.info())

print()
print("Area Mean: ", df['Area'].mean())
print("Area Median: ", df['Area'].median())
print("Area Standard Deviation: ", df['Area'].std())
print()

print()
print("Annual Avg. Precipitation Mean: ", df['Annual Avg. Precipitation'].mean())
print("Annual Avg. Precipitation Median: ", df['Annual Avg. Precipitation'].median())
print("Annual Avg. Precipitation Standard Deviation", df['Annual Avg. Precipitation'].std())
print()

print()
print("Annual Avg. Evapotranspiration Mean: ", df['Annual Avg. Evapotranspiration'].mean())
print("Annual Avg. Evapotranspiration Median: ", df['Annual Avg. Evapotranspiration'].median())
print("Annual Avg. Evapotranspiration Standard Deviation: ", df['Annual Avg. Evapotranspiration'].std())

print()
print("Annual Avg. Aquifer Recharge Mean: ", df['Annual Avg. Aquifer Recharge'].mean())
print("Annual Avg. Aquifer Recharge Median: ", df['Annual Avg. Aquifer Recharge'].median())
print("Annual Avg. Aquifer Recharge Standard Deviation: ", df['Annual Avg. Aquifer Recharge'].std())
print()

print()
print("Annual Avg. River Flow Mean: ", df['Annual Avg. River Flow'].mean())
print("Annual Avg. River Flow Median: ", df['Annual Avg. River Flow'].median())
print("Annual Avg. River Flow Standard Deviation: ", df['Annual Avg. River Flow'].std())
print()