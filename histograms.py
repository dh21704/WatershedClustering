import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/danielhernandez/Downloads/Watershed_Characteristics.csv')

print(df.info())
#Plot your streamflow, recharge, precipitation and actual ET using histograms (one histogram per variable). Read about histogram if you are not familiar.

bins = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


#plot streamflow
plt.hist(df['Annual Avg. River Flow'], bins = bins, edgecolor='black', color='purple')
plt.title('Annual Avg. River Flow')
plt.xlabel('River Flow (mm)')
plt.ylabel('Number of River Flow Respondents')
plt.tight_layout()
plt.show()

#########################################################################
#plot recharge
plt.hist(df['Annual Avg. Aquifer Recharge'], bins=bins, edgecolor='black')
plt.title('Annual Avg. Aquifer Recharge')
plt.xlabel('Aquifer Recharge (mm)')
plt.ylabel('Number of Aquifer Recharge Respondents')
plt.tight_layout()
plt.show()

#########################################################################
#plot precipitation
plt.hist(df['Annual Avg. Precipitation'], bins=bins, edgecolor='black')
plt.title('Annual Avg. Precipitation')
plt.xlabel('Annual Avg. Precipitation (mm)')
plt.ylabel('Number of Precipitation Respondents')
plt.tight_layout()
plt.show()

#########################################################################
#plot actual ET
plt.hist(df['Annual Avg. Evapotranspiration'], bins=bins, edgecolor='black')
plt.title('Annual Avg. Evapotranspiration')
plt.xlabel('Evapotranspiration (mm)')
plt.ylabel('Number of Evapotranspiration Respondents')
plt.tight_layout()
plt.show()