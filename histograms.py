import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/danielhernandez/Downloads/Watershed_Characteristics.csv')

print(df.info())
#Plot your streamflow, recharge, precipitation and actual ET using histograms (one histogram per variable). Read about histogram if you are not familiar.

bins = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def show_historgram(df, title_name, xlabel, ylabel, title):
    data = df[title_name]

    plt.figure(figsize=(10,6))
    plt.hist(df, bins=bins, edgecolor='black')
    plt.title(title, size=15)
    plt.xlabel(xlabel, size=15)
    plt.ylabel(ylabel, size=15)
    plt.tight_layout()
    plt.show()


show_historgram(df, 'Annual Avg. Snow', 'Snow (mm)', 'Frequency', 'Annual Avg. Snow')
#plot streamflow
plt.hist(df['Annual Avg. River Flow'], bins = bins, edgecolor='black')
plt.title('Annual Avg. River Flow', size=15)
plt.xlabel('River Flow (mm)', size=15)
plt.ylabel('Frequncy', size=15)
plt.tight_layout()
plt.show()


#########################################################################
#plot precipitation
plt.hist(df['Annual Avg. Precipitation'], bins=bins, edgecolor='black')
plt.title('Annual Avg. Precipitation', size=15)
plt.xlabel('Precipitation (mm)', size=15)
plt.ylabel('Frequency', size=15)
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

#########################################################################
bins2 = [50, 100, 150, 200, 250, 300, 350, 400, 450]

#plot snow
plt.hist(df['Annual Avg. Snow'], bins=bins2, edgecolor='black')
plt.title('Annual Avg. Snow')
plt.xlabel('Snow (mm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

#########################################################################
bins2 = [50, 100, 150, 200, 250, 300, 350, 400, 450]

#plot snow
plt.hist(df['Annual Avg. Snow'], bins=bins2, edgecolor='black')
plt.title('Annual Avg. Snow', size=15)
plt.xlabel('Snow (mm)', size=15)
plt.ylabel('Frequency', size=15)
plt.tight_layout()
plt.show()
