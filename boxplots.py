import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/danielhernandez/Downloads/Watershed_Characteristics.csv')
print(df.info())

data = [df['Area'], df['Annual Avg. Precipitation'],
        df['Annual Avg. Evapotranspiration'],
        df['Annual Avg. Aquifer Recharge'],
        df['Annual Avg. River Flow']]

plt.figure(figsize=(15, 10))
plt.boxplot(data, tick_labels=['Area', 'Precipitation', 'Evapotranspiration',
                               'Aquifer Recharge', 'River Flow'], showfliers=False)
plt.ylabel('Area is Square Kilometers; Everything Else is Millimeters')
plt.title('Boxplots of Catchment Area, Precipitation, ET, Recharge and Streamflow Across 203 Catchments')
plt.tight_layout()
plt.show()
