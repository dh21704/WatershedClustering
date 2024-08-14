import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/danielhernandez/Downloads/Watershed_Characteristics.csv')
print(df.info())

data = [df['Annual Avg. Precipitation'],
        df['Annual Avg. River Flow'],
        df['Annual Avg. Snow'],
        df['Aridity Index']]

plt.figure(figsize=(15, 10))
plt.boxplot(data, tick_labels=['Precipitation', 'River Flow', 'Snow', 'Aridity Index'], showfliers=False)
plt.ylabel('Millimeters', size=25)
plt.title('Boxplots of Precipitation, Streamflow, Snow, and Aridity Index Across 203 Catchments', size=22)
plt.tight_layout()
plt.show()