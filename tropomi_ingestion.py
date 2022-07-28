# Import the necessary dependencies
import glob
import netCDF4 as nc
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame, points_from_xy
from geopandas.tools import sjoin
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


# Write a function that converts a list of NetCDF files to a Pandas DataFrame
def netcdf_files_to_geodataframe(netcdf_files_path, usa_shapefile_path, export_file_path):
    # Read in a list of NetCDF files
    netcdf_files = sorted(glob.glob(netcdf_files_path + '/*.nc'))

    # Specify the NetCDF files' relevant variables
    product_vars = ['methane_mixing_ratio_bias_corrected', 'qa_value', 'longitude', 'latitude']
    input_vars = ['eastward_wind', 'northward_wind']

    # Create an empty list that will store all the converted NetCDF files, i.e., DataFrames.
    processed_dataframes = []
    for i in range(len(netcdf_files)):
        # ingest the NetCDF file
        netcdf_file = netcdf_files[i]
        dataset = nc.Dataset(netcdf_file)
        product_data = dataset.groups['PRODUCT']
        input_data = product_data.groups['SUPPORT_DATA'].groups['INPUT_DATA']

        # extract the 'time_utc' variable
        df_time = pd.DataFrame(product_data['time_utc'][0], columns=['date'])
        df_time['date'] = pd.to_datetime(df_time['date'])

        # extract the 'methane_mixing_ratio_bias_corrected' variable
        df_methane = pd.DataFrame(np.array(product_data[product_vars[0]][0]), index=df_time['date'])
        row_indices, col_indices = np.meshgrid(np.arange(df_methane.shape[0]), np.arange(df_methane.shape[1]))
        table = np.vstack((row_indices.ravel(), df_methane.values.ravel())).T
        df_temp = pd.DataFrame(table, columns=['row_index', product_vars[0]])

        # map each row index to its corresponding time
        row_index_to_time = dict(zip(list(df_temp['row_index'].sort_values().unique()), list(df_methane.index)))
        df_temp['row_index'] = df_temp['row_index'].map(row_index_to_time)
        df_temp.index = df_temp['row_index']
        df_temp.index.rename('date', inplace=True)
        df_temp.drop('row_index', axis=1, inplace=True)

        # extract the remaining relevant variables
        for var in product_vars[1:] + input_vars:
            if var in product_vars[1:]:
                df_temp[var] = np.array(product_data[var][0]).flatten()
            elif var in input_vars:
                df_temp[var] = np.array(input_data[var][0]).flatten()

        # remove missing values and extract the 'high quality' data
        fill_value = product_data[product_vars[0]]._FillValue
        df_temp[product_vars[0]] = df_temp[product_vars[0]].apply(lambda x: np.nan if x == fill_value else x)
        df_temp.dropna(subset=[product_vars[0]], inplace=True)
        df_temp = df_temp[df_temp[product_vars[1]] > 0.5].copy(deep=True)
        df_temp.drop(product_vars[1], axis=1, inplace=True)
        df_temp['count'] = df_temp.shape[0]

        # Add the 'df_temp' DataFrame to the 'processed_dataframes' list
        processed_dataframes.append(df_temp)

    # Vertically stack all the DataFrames in the 'processed_dataframes' list to create a single DataFrame
    df = pd.concat(processed_dataframes, axis=0)
    df[product_vars[0]] = df[product_vars[0]] / 1000
    df.rename({product_vars[0]: 'ppm'}, axis=1, inplace=True)
    df['day'] = df.index
    df['day'] = df['day'].dt.strftime('%b %d, %Y')
    df.index = df.index.time
    df.index.rename('time_utc', inplace=True)
    df = df.iloc[:, [-1, -2, 0, 1, 2, 3, 4]].copy(deep=True)

    # Read in the USA shapefile as a GeoDataFrame
    gdf_usa = gpd.read_file(usa_shapefile_path)
    gdf_usa['geometry'] = gdf_usa['geometry'].to_crs(epsg=4326)
    gdf_usa.columns = gdf_usa.columns.str.lower()

    # filter the 'gdf_usa' GeoDataFrame so that it only contains the 48 mainland states
    non_mainland = ['HI', 'VI', 'MP', 'GU', 'AK', 'AS', 'PR', 'DC']
    non_mainland_indices = []
    for i in range(gdf_usa.shape[0]):
        if gdf_usa.iloc[i]['stusps'] in non_mainland:
            non_mainland_indices.append(gdf_usa.index[i])
    gdf_usa48 = gdf_usa.drop(index=non_mainland_indices).copy(deep=True)

    # extract the centroid coordinates from each state in the 'gdf_usa48' GeoDataFrame
    gdf_usa48['centroid'] = gdf_usa48.centroid
    gdf_usa48['centroid'] = gdf_usa48['centroid'].apply(lambda x: list(x.coords)[0])

    # specify the different regions in the 'gdf_usa48' GeoDataFrame
    regions = ['Northeast', 'Midwest', 'South', 'West']
    region_mapper = dict(zip(list(gdf_usa48['region'].sort_values().unique()), regions))
    gdf_usa48['region'] = gdf_usa48['region'].map(region_mapper)

    # Convert the 'df' DataFrame to a USA-mainland filtered DataFrame whose dates are in ascending order
    days = list(df['day'].sort_values().unique())
    states = list(gdf_usa48['name'].sort_values().unique())
    features = ['day', 'bin_number', 'name', 'stusps', 'division', 'region', 'ppm', 'longitude', 'latitude', 'eastward_wind', 'northward_wind', 'geometry']
    daily_geodataframes = []
    for day in days:
        df_day = df[df['day'] == day].copy(deep=True)
        bins = [i + 1 for i in range(len(df_day['count'].unique()))]
        bin_mapper = dict(zip(list(df_day['count'].unique()), bins))
        df_day['count'] = df_day['count'].map(bin_mapper)
        df_day.rename({'count': 'bin_number'}, axis=1, inplace=True)
        gdf_day = df_day.copy(deep=True)
        gdf_day = GeoDataFrame(gdf_day, geometry=points_from_xy(gdf_day['longitude'], gdf_day['latitude']))
        gdf_day = sjoin(gdf_day, gdf_usa48, how='inner')
        gdf_day = gdf_day[['day', 'bin_number', 'ppm', 'longitude', 'latitude', 'eastward_wind', 'northward_wind', 'geometry']].copy(deep=True)
        state_geodataframes = []
        for state in states:
            state_shapefile = gdf_usa48[gdf_usa48['name'] == state].copy(deep=True)
            gdf_state = sjoin(gdf_day, state_shapefile, how='inner')
            gdf_state = gdf_state[features].copy(deep=True)
            state_geodataframes.append(gdf_state)
        gdf_states = pd.concat(state_geodataframes, axis=0)
        daily_geodataframes.append(gdf_states)

    # Vertically stack all the GeoDataFrames in the 'daily_geodataframes' list to create a final DataFrame
    gdf = pd.concat(daily_geodataframes, axis=0)
    gdf.rename({'name': 'state'}, axis=1, inplace=True)
    gdf.sort_values(['day', 'time_utc'], inplace=True)
    df_final = gdf.drop('geometry', axis=1).copy(deep=True)

    # Export the 'df_final' DataFrame as a csv file and return the 'gdf' GeoDataFrame
    file_name = '/TROPOMI_L2_CH4_USA_' + df_final.iloc[0]['day'].split(' ')[0] + '_' + df_final.iloc[0]['day'].split(' ')[-1] + '.csv'
    df_final.to_csv(export_file_path + file_name)
    return gdf
