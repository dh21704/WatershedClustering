import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats


df = pd.read_csv('/Users/danielhernandez/Downloads/Watershed_Characteristics.csv', )
############
#drop rows with missing values
df.dropna(inplace=True)

#initialize scaler
scaler = StandardScaler()

#2. KMeans clusters use all the variables except river flow and recharge
#does this include all the ones that aren't also tempe
wanted_transform = ['Annual Avg. Temp', 'Annual Avg. Snow', 'Annual Avg. Precipitation',
'Area']

#now transform them into a scaled, scales the number from something like -1, to 1
df_scaled = scaler.fit_transform(df[wanted_transform])

# Convert scaled data (which is now a numpy array) back to a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=[col + '_T' for col
in wanted_transform], index=df.index)


#axis=0 refers to concatenating along rows (stacking vertically).
#axis=1 refers to concatenating along columns (joining horizontally).
# Concatenate the scaled columns back to the original DataFrame
# Concatenate scaled data with original DataFrame
df = pd.concat([df, df_scaled], axis = 1)


#function to optimize number of clusters using elbow method
def optimize_k_means(data, max_number):
    means = []

    #inertias helps assess how well the KMeans model has grouped the data
    inertias = []

    for k in range(1, max_number):
        kmeans = KMeans(n_clusters=k)

        kmeans.fit(data)

        means.append(k)

        inertias.append(kmeans.inertia_)

    #generate the elbow plot
    fig = plt.subplots(figsize=(10,8))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Clusters', size = 30)
    plt.ylabel('Inertias', size = 30)
    plt.title('Number of Clusters vs Inertias', size = 30)
    plt.grid(True)
    plt.show()


optimize_k_means(df[['Annual Avg. Precipitation_T', 'Annual Avg. River Flow']], 10)


#Applying KMeans Clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(df[['Annual Avg. Precipitation_T', 'Annual Avg. River Flow']])
df['kmeans_3'] = kmeans.labels_

cluster_1 = df[df['kmeans_3'] == 0]
cluster_2 = df[df['kmeans_3'] == 1]
cluster_3 = df[df['kmeans_3'] == 2]

print(cluster_1.info()), print(), print(), print()
print(cluster_2.info()), print(), print(), print()
print(cluster_3.info()), print(), print(), print()

#extract variables from clusters here and then graph them and make a linear regression model

#plot the results
plt.figure(figsize=(8,6))
plt.scatter(df['Annual Avg. Precipitation_T'], df['Annual Avg. River Flow'], c=df['kmeans_3'])
plt.xlabel('Annual Avg. Precipitation', size = 15)
plt.ylabel('Annual Avg. River Flow', size = 15)
plt.title('Clusters with 3 KMeans', size = 15)
plt.show()

#creeate the subplots
fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (18, 6))

#show the different types of clusters
for k in range(1, 4):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(df[['Annual Avg. Precipitation_T', 'Annual Avg. River Flow']])
    df[f'KMeans_{k}'] = kmeans.labels_

    slope, intercept, r, p, std_error = stats.linregress(df['Annual Avg. Precipitation_T'], df['Annual Avg. River Flow'])
    regression_line = slope * df['Annual Avg. Precipitation_T'] + intercept

    ax = axs[k - 1]
    ax.scatter(df['Annual Avg. Precipitation_T'], df['Annual Avg. River Flow'], c=df[f'KMeans_{k}'])
    ax.plot(df['Annual Avg. Precipitation_T'], regression_line, color = 'red')
    ax.set_xlabel('Annual Avg. Precipitation', size = 25)
    ax.set_ylabel('Annual Avg. River Flow', size = 25)
    ax.set_xlim(df['Annual Avg. Precipitation_T'].min(), df['Annual Avg. Precipitation_T'].max())
    ax.set_ylim(df['Annual Avg. River Flow'].min(), df['Annual Avg. River Flow'].max())
    ax.set_title(f'NClusters_{k}', size = 30)

#adjust plot layout
plt.tight_layout()
plt.show()



############################################################################
#CLUSTER 1 RIVER FLOW AND PRECIPITATION
p_q_c1_slope, p_q_c1_intercept, p_q_c1_r, p_q_c1_p, p_q_c1_std_err\
    = stats.linregress(cluster_1['Annual Avg. Precipitation'], cluster_1['Annual Avg. River Flow'])

p_q_c1_regression_line = p_q_c1_slope * cluster_1['Annual Avg. Precipitation'] + p_q_c1_intercept

plt.scatter(cluster_1['Annual Avg. Precipitation'], cluster_1['Annual Avg. River Flow'], c='green')
plt.plot(cluster_1['Annual Avg. Precipitation'], p_q_c1_regression_line, c = 'black')
plt.title('Cluster 1 Precipitation vs River Flow')
plt.xlabel('Precipitation (mm)')
plt.ylabel('River Flow (mm)')
plt.show()

print("CLUSTER 1 RIVER FLOW AND PRECIPITATION")
print("Slope: ", p_q_c1_slope)
print("Intercept: ", p_q_c1_intercept)
print("R^2: ", p_q_c1_r * p_q_c1_r)


############################################################################
#CLUSTER 2 RIVER FLOW AND PRECIPITATION
p_q_c2_slope, p_q_c2_intercept, p_q_c2_r, p_q_c2_p, p_q_c2_std_err\
    = stats.linregress(cluster_2['Annual Avg. Precipitation'], cluster_2['Annual Avg. River Flow'])

p_q_c2_regression_line = p_q_c2_slope * cluster_2['Annual Avg. Precipitation'] + p_q_c2_intercept

plt.scatter(cluster_2['Annual Avg. Precipitation'], cluster_2['Annual Avg. River Flow'], c = 'blue')
plt.plot(cluster_2['Annual Avg. Precipitation'], p_q_c2_regression_line, c = 'black')
plt.title('Cluster 2 Precipitation vs River Flow')
plt.xlabel('Precipitation (mm)')
plt.ylabel('River Flow (mm)')
plt.show()

print()
print("CLUSTER 2 RIVER FLOW AND PRECIPITATION")
print("Slope: ", p_q_c2_slope)
print("Intercept: ", p_q_c2_intercept)
print("R^2: ", p_q_c2_r * p_q_c2_r)

############################################################################
#CLUSTER 3 RIVER FLOW AND PRECIPITATION
p_q_c3_slope, p_q_c3_intercept, p_q_c3_r, p_q_c3_p, p_q_c3_std_err\
    = stats.linregress(cluster_3['Annual Avg. Precipitation'], cluster_3['Annual Avg. River Flow'])

p_q_c3_regression_line = p_q_c3_slope * cluster_3['Annual Avg. Precipitation'] + p_q_c3_intercept

plt.scatter(cluster_3['Annual Avg. Precipitation'], cluster_3['Annual Avg. River Flow'], c='yellow')
plt.plot(cluster_3['Annual Avg. Precipitation'], p_q_c3_regression_line, c='black')
plt.title('Cluster 3 Precipitation vs River Flow')
plt.xlabel('Precipitation (mm)')
plt.ylabel('River Flow (mm)')
plt.show()

print()
print("CLUSTER 3 RIVER FLOW AND PRECIPITATION")
print("Slope: ", p_q_c3_slope)
print("Intercept: ", p_q_c3_intercept)
print("R^2: ", p_q_c3_r * p_q_c3_r)

############################################################################
#CLUSTER 1 AQUIFIER RECHARGE AND PRECIPITATION
p_r_c1_slope, p_r_c1_intercept, p_r_c1_r, p_r_c1_p, p_r_c1_std_err\
    = stats.linregress(cluster_1['Annual Avg. Precipitation'], cluster_1['Annual Avg. Aquifer Recharge'])

p_r_c1_linear_regression = p_r_c1_slope * cluster_1['Annual Avg. Precipitation'] + p_r_c1_intercept

plt.scatter(cluster_1['Annual Avg. Precipitation'], cluster_1['Annual Avg. River Flow'], c = 'green')
plt.plot(cluster_1['Annual Avg. Precipitation'], p_r_c1_linear_regression, c='black')
plt.title('Cluster 1 Precipitation vs Aquifer Recharge')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Aquifer Recharge (mm)')
plt.show()

print()
print("CLUSTER 1 AQUIFER RECHARGE AND PRECIPITATION")
print("Slope: ", p_r_c1_slope)
print("Intercept: ", p_r_c1_intercept)
print("R^2: ", p_r_c1_r * p_r_c1_r)

############################################################################
#CLUSTER 2 AQUIFIER RECHARGE AND PRECIPITATION
p_r_c2_slope, p_r_c2_intercept, p_r_c2_r, p_r_c2_p, p_r_c2_std_err\
    = stats.linregress(cluster_2['Annual Avg. Precipitation'], cluster_2['Annual Avg. Aquifer Recharge'])

p_r_c2_regression_line = p_r_c2_slope * cluster_2['Annual Avg. Precipitation'] + p_r_c2_intercept

plt.scatter(cluster_2['Annual Avg. Precipitation'], cluster_2['Annual Avg. Aquifer Recharge'], color  ='blue')
plt.plot(cluster_2['Annual Avg. Precipitation'], p_r_c2_regression_line, c='black')
plt.title('Cluster 2 Precipitation vs Aquifer Recharge')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Aquifer Recharge (mm)')
plt.show()

print()
print("CLUSTER 2 AQUIFER RECHARGE AND PRECIPITATION")
print("Slope: ", p_r_c2_slope)
print("Intercept: ", p_r_c2_intercept)
print("R^2: ", p_r_c2_r * p_r_c2_r)


############################################################################
#CLUSTER 3 AQUIFIER RECHARGE AND PRECIPITATION
p_r_c3_slope, p_r_c3_intercept, p_r_c3_r, p_r_c3_p, p_r_c3_std_err\
    = stats.linregress(cluster_3['Annual Avg. Precipitation'], cluster_3['Annual Avg. Aquifer Recharge'])

p_r_c3_regression_line = p_r_c3_slope * cluster_3['Annual Avg. Precipitation'] + p_r_c3_intercept

plt.scatter(cluster_3['Annual Avg. Precipitation'], cluster_3['Annual Avg. Aquifer Recharge'], c='yellow')
plt.plot(cluster_3['Annual Avg. Precipitation'], p_r_c3_regression_line, c='black')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Aquifer Recharge (mm)')
plt.title('Cluster 3 Precipitation vs Aquifer Recharge')
plt.show()

print()
print("CLUSTER 3 AQUIFER RECHARGE AND PRECIPITATION")
print("Slope: ", p_r_c3_slope)
print("Intercept: ", p_r_c3_intercept)
print("R^2: ", p_r_c3_r * p_r_c3_r)

############################################################################
#CLUSTER 1 AQUIFIER RECHARGE AND PRECIPITATION/POTENTIAL EVAPOTRANSPIRATION
p_over_pet_c1 = cluster_1['Annual Avg. Precipitation'] / cluster_1['Annual Avg. Potential Evapotranspiration']


r_p_pet_c1_slope, r_p_pet_c1_intercept, r_p_pet_c1_r, r_p_pet_c1_p, r_p_pet_c1_std_err\
  =  stats.linregress(p_over_pet_c1, cluster_1['Annual Avg. Aquifer Recharge'])

r_p_pet_c1_regression_line = r_p_pet_c1_slope * p_over_pet_c1 + r_p_pet_c1_intercept

plt.scatter(p_over_pet_c1, cluster_1['Annual Avg. Aquifer Recharge'], c='green')
plt.plot(p_over_pet_c1, r_p_pet_c1_regression_line, c='black')
plt.xlabel('Precipitation / Potential Evapotranspiration (mm)')
plt.ylabel('Aquifer Recharge (mm)')
plt.title('Cluster 1 Precipitation / Potential Evapotranspiration vs Aquifer Recharge')
plt.show()

print()
print("CLUSTER 1 AQUIFIER RECHARGE AND PRECIPITATION/POTENTIAL EVAPOTRANSPIRATION")
print("Slope: ", r_p_pet_c1_slope)
print("Intercept: ", r_p_pet_c1_intercept)
print("R^2: ", r_p_pet_c1_r * r_p_pet_c1_r)

############################################################################
#CLUSTER 2 AQUIFIER RECHARGE AND PRECIPITATION/POTENTIAL EVAPOTRANSPIRATION

p_over_pet_c2 = cluster_2['Annual Avg. Precipitation'] / cluster_2['Annual Avg. Potential Evapotranspiration']

r_p_pet_c2_slope, r_p_pet_c2_intercept, r_p_pet_c2_r, r_p_pet_c2_p, r_p_pet_c2_std_err\
    = stats.linregress(p_over_pet_c2, cluster_2['Annual Avg. Aquifer Recharge'])

r_p_pet_c2_linear_regression = r_p_pet_c2_slope * p_over_pet_c2 + r_p_pet_c2_intercept

plt.scatter(p_over_pet_c2, cluster_2['Annual Avg. Aquifer Recharge'], c='blue')
plt.plot(p_over_pet_c2, r_p_pet_c2_linear_regression, c='black')
plt.xlabel('Precipitation / Potential Evapotranspiration (mm)')
plt.ylabel('Aquifer Recharge (mm)')
plt.title('Cluster 2 Precipitation / Potential Evapotranspiration vs Aquifer Recharge')
plt.show()

print()
print("CLUSTER 2 AQUIFIER RECHARGE AND PRECIPITATION/POTENTIAL EVAPOTRANSPIRATION")
print("Slope: ", r_p_pet_c2_slope)
print("Intercept: ", r_p_pet_c2_intercept)
print("R^2: ", r_p_pet_c2_r * r_p_pet_c2_r)

############################################################################
#CLUSTER 3 AQUIFIER RECHARGE AND PRECIPITATION/POTENTIAL EVAPOTRANSPIRATION
p_over_pet_c3 = cluster_3['Annual Avg. Precipitation'] / cluster_3['Annual Avg. Potential Evapotranspiration']

r_p_pet_c3_slope, r_p_pet_c3_intercept, r_p_pet_c3_r, r_p_pet_c3_p, r_p_pet_c3_std_err\
    = stats.linregress(p_over_pet_c3, cluster_3['Annual Avg. Aquifer Recharge'])

r_p_pet_c3_linear_regression = r_p_pet_c3_slope * p_over_pet_c3 + r_p_pet_c3_intercept

plt.scatter(p_over_pet_c3, cluster_3['Annual Avg. Aquifer Recharge'], c='yellow')
plt.plot(p_over_pet_c3, r_p_pet_c3_linear_regression, c='black')
plt.xlabel('Precipitation / Potential Evapotranspiration (mm)')
plt.ylabel('Aquifer Recharge (mm)')
plt.title('Cluster 3 Precipitation / Potential Evapotranspiration vs Aquifer Recharge')
plt.show()

print()
print("CLUSTER 3 AQUIFIER RECHARGE AND PRECIPITATION/POTENTIAL EVAPOTRANSPIRATION")
print("Slope: ", r_p_pet_c3_slope)
print("Intercept: ", r_p_pet_c3_intercept)
print("R^2: ", r_p_pet_c3_r * r_p_pet_c3_r)