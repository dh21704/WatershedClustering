import geopandas as gpd

# Replace this with the actual path to your shapefile
shapefile_path = '/Users/danielhernandez/Downloads/Watershed_Shapefile 3/Watershed_Shapefile.shp'

# Load the shapefile
gdf = gpd.read_file(shapefile_path)

# Print the first few rows
print(gdf.head())
