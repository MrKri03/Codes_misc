import io
import os
import numpy as np
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon
import fiona
import wget



#Function 1
def access_copernicusHub(user_copernicusHub, password_copernicusHub, urlCopernicusHub):
    """
    Connects to Copernicus Open Hub using personal credentials
    https://scihub.copernicus.eu/dhus
    
    :param user_copernicusHub: name of user of Copernicus Hub
    :type user_copernicusHub: str
    
    :param user_copernicusHub: password of user of Copernicus Hub
    :type user_copernicusHub: str
    
    :return: connection to Copernicus hub via API
    :rtype: SentinelAPI object
    """
   
    return SentinelAPI(user_copernicusHub, password_copernicusHub, urlCopernicusHub)


#Function 2
def search_footprint_data(Sapi_object, 
                          pol, date1, date2, platf_name, 
                          level, cloudCv,tile_name,online):
    """
    Get the SentinelAPI object and search images within the API.
    
    :param Sapi_object: API object storing user connection information
    :type Sapi_object: SentinelAPI
    
    :param pol: polygon used to search for intersecting images
    :type pol: wkt 
    
    :param date*: two dates , 1 and 2, being 1 start date and 2 end date to search images 
    :type date*: 'YYYYMMdd'
    
    :param platf_name: name of sentinel . 
    :type platf_name: str
    
    :param level: Image level processing
    :type level: str
    
    :param cloudCv: percentage of cloud cover of the image
    :type cloudCv: tuple integer
    
    :param online: wether the image is online or it is in Long Term Archive (LTA). If True, it is online
    :type online: bool
    
    :param tile_name: tile of satellite footprint in the study area
    :type tile_name:  str
    
    :return: filtered image
    :rtype: OrdDict

    """
    search =  Sapi_object.query(pol,
                                date=(date1, date2),
                                platformname= platf_name,
                                processinglevel = level,
                                cloudcoverpercentage=cloudCv)
    
    for idx, att in enumerate(search):
        
        # get information from the query
        image_to_download = Sapi_object.get_product_odata(att, full=True)
        
        # Get the tile suitable for study
        tile = image_to_download.get("title").split("_")[-2]
        
        # return FIRST available image, as specific date in November is NOT specified
        if (tile == tile_name ) and (image_to_download.get("Online") == True):
            
            return image_to_download

    
        
#Function 3
def get_bands(name_S2img, resol_folder):
    """
    Retrieve the bands located in the specified resolution folder
    
    :param name_S2img: Name of the image to use
    :type name_S2img: str
    
    :param resol_folder: resolution folder to search for the images
    :type resol_folder: str 
    
    :return: VNIR bands : Blue, Green, Red, Near Infrarred
    :rtype: str
    
    """
    import glob
    
    # Main folder with .SAFE
    s2_name = name_S2img + ".SAFE"
        
    path_to_image = list(glob.iglob(os.path.join(
        
        s2_name,
        "GRANULE",
        "*",
        "IMG_DATA",
        resol_folder,
        "*"
    )))
    
    
    B_blue = [ b for b in path_to_image if b.endswith("_B02_10m.jp2")][0]
    B_green = [ g for g in path_to_image if g.endswith("_B03_10m.jp2")][0]
    B_red = [ r for r in path_to_image if r.endswith("_B04_10m.jp2")][0]
    B_nir = [ nir for nir in path_to_image if nir.endswith("_B08_10m.jp2")][0]
    

    return B_blue, B_green, B_red, B_nir


#Function 4
def stack_vnir_bands(bands):
    
    """
     Use the original VNIR bands 
     and retrieve the stack in tif format
    
    :param bands: list with name of bands
    :type bands: list str
    
    :return: void. writes a raster of image stack in current directory
    
    """
    
    #get metadata of one band
    with rio.open(bands[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count = len(bands))

    # Read each layer and write it to stack
    with rio.open(os.path.join(os.getcwd(),"stack.tif"), 'w', **meta) as dst:
        for id, layer in enumerate(bands, start=1):
            print("raster",id,"is stacked")
            with rio.open(layer) as src1:
                dst.write_band(id, src1.read(1))
    print("Raster stack written")   
    

#Function 5
def crop_image(image,boundingBox,all_touching, crs):
    from rasterio.mask import mask
    
    """
     Crops the stack raster
     keeps the same band order
    
    :param image: path to the stack raster (tif)
    :type image: str
    
    :param boundingBox: geopandas polygon used as mask boundary to crop the image
    :type boundingBox: GeoDataFrame
    
    :param all_touching: Keep the pixels touching the boundaries. If true, all are preserved
    :type all_touching: bool
    
    :param crs: Coordinates Reference System of stack image
    :type crs: rasterio.crs
    
    
    :return: void. writes the raster image cropped
    
    """
    #Change projection
    
    boundingBox['geometry'] = boundingBox['geometry'].to_crs(epsg=32628)

    
    with rio.open(image) as src:
        out_image_clip, out_transform_clip = mask(src, boundingBox["geometry"], 
                                                  all_touched = all_touching, crop=True) 
        
        #get metadata from the clip
        meta_clip = src.meta

    
    #change the metadata
    meta_clip.update({"driver": "GTiff",
                      "height": out_image_clip.shape[1],
                      "width": out_image_clip.shape[2],
                      "transform": out_transform_clip,
                      "nodata": 256})

    with rio.open(os.path.join(os.getcwd(),"clip_stack.tif"), "w", **meta_clip) as dest:
        dest.write(out_image_clip)

        
# Function 6
def create_grid(input_image,output_grid):
    """
     Creates a byte raster grid (0,1)
    
    :param input_image: path to the stack raster (tif) to use to create the grid
    :type input_image: str
    
    :param output_grid: path to create the output grid 
    :type output_grid: str

    :return: void. writes the raster grid
    
    """
    import osgeo
    from osgeo import gdal

    from osgeo import osr

    
    ds = osgeo.gdal.Open(input_image,1) ## select only one image in stack, all have same resolution
    
    #get the size of gdal raster
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    
    #create an array of 0s with size of gdal raster
    ary_x = np.zeros((rows, cols), dtype = int) # get dimensions of raster
    
    # WE GET FIRST (SECOND, STARTING FROM 0) ROW OF ARRAY TO THE END, AND FIRST (ZERO) COLUMN TO THE END, EACH 2: EQUALS 1
    ary_x[1::2, ::2] = 1
    
    ary_x[::2, 1::2] = 1  ## One of this must be 0 !!
    
    band = ds.GetRasterBand(1)
    #nodataNum = band.GetNoDataValue()
    #print(nodataNum)
    # get properties of gdal raster
    geotransform = ds.GetGeoTransform()
    wkt = ds.GetProjection()
    
    driver = gdal.GetDriverByName("GTiff")

    dst_ds = driver.Create(output_grid,
                               band.XSize,
                               band.YSize,
                               1,
                               gdal.GDT_Byte)
    #writting the array into the new created raster dst_ds
    dst_ds.GetRasterBand(1).WriteArray( ary_x )

    #setting nodata value
    #dst_ds.GetRasterBand(1).SetNoDataValue(nodataNum)
    
    dst_ds.SetGeoTransform(geotransform)
    # setting spatial reference of output raster
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dst_ds.SetProjection( srs.ExportToWkt() )

    
# Function 7
def polygonize_grid(input_raster,output_grid):
    """
     Creates a polygon grid matching the input image
    
    :param input_raster: path to the stack raster (tif) to use to create the polygon grid
    :type input_raster: str
    
    :param output_grid: path to create the output with extension of file (creates geoJSON)
    :type output_grid: str

    :return: void. writes the polygon grid

    """
    from rasterio.features import shapes
    mask = None
    
    with rio.Env():
        with rio.open(input_raster) as src:
            image = src.read(1) # first band
            results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) 
            in enumerate(
                
                shapes(image, mask=mask, transform=src.transform)))
            src.close()
    
    geoms = list(results)
    
    gpd_polygonized_raster  = gpd.GeoDataFrame.from_features(geoms)         
    
    gpd_polygonized_raster.crs = 'EPSG:32628'
    
    gpd_polygonized_raster['raster_val'] = gpd_polygonized_raster['raster_val'].astype(int)
    gpd_polygonized_raster.drop(gpd_polygonized_raster[gpd_polygonized_raster.raster_val == 2].index, inplace = True)
    
    
    saveGJSON = output_grid
    
    gpd_polygonized_raster.to_file(saveGJSON, driver="GeoJSON")

    
# Function 8
def create_ID_gdf(input_grid,name_column):
    """
     Creates a field with unique identifier for each row but not used as index
    
    :param input_grid: path to the polygon grid
    :type input_grid: geoDataFrame
    
    :param name_column: name of the column 
    :type name_column: str

    :return: geodataframe with the new column
    :rtype: GeoDataFrame
    
    """
    
    gdf = gpd.read_file(input_grid)
    
    gdf.insert(0,name_column,range(0,len(gdf)))
    
    #deletes the field with raster values
    del gdf['raster_val']
    
    return gdf    


# Function 9
def remove_edgeEffects(input_gdf,study_area):
    """
    Removes the grids out of the study area. Also avoids edge efects
    The result grid is COMPLETELY WITHIN the study area , thus , no mixture with out boundaries
    
    :param input_gdf: geodataframe to remove the grids from
    :type input_raster: GeoDataFrame
    
    :param study_area: path to bounding box of the study area
    :type study_area: str

    :return: void. writes a geojson.
    
    """
    
    study_area = gpd.read_file(study_area)
     
    # Change projection to EPSG 32628 
    study_area['geometry'] = study_area['geometry'].to_crs(epsg=32628)
    
    #Manage edge effects. "Traditional" way
    study_area_squares = gpd.overlay(input_gdf,study_area, how='difference',keep_geom_type = True)
    study_area_squares['Erase'] = "Yes"
    
    edges =pd.merge(input_gdf, study_area_squares,how = 'outer',on = "New_ID")
    
    edges.drop(edges.index[edges['Erase'] == 'Yes'], inplace = True)
    
    edges.rename(columns={'geometry_x':'geometry'}, inplace=True)
    
    edges = gpd.GeoDataFrame(edges, geometry = 'geometry')
                      
    del edges['geometry_y']    
    
    del edges['Erase']
    edges.to_file("grids_within_area.geojson", driver="GeoJSON" )    

# Function 10  
def extract_pixel_values(input_raster, AOI, output_data, name_fields):
    """
    Extracts pixel values to polygons or interest by aggregation
    
    :param input_raster: path to the original raster to extract values from
    :type input_raster: str
    
    :param AOI: polygons to aggregate the values
    :type AOI: GeoDataFrame
    
    :param name_fields: list of string to name the new fields with values of raster.
    :type name_fields: list str

    :return: void. writes a geojson.
    
    """
    
    from rasterstats import zonal_stats
    
    # Change projection: 
    AOI['geometry'] = AOI['geometry'].to_crs(epsg=32628)
    
    dataset = rio.open(input_raster)
    
    #get number of bands to use for iteration
    n_bands = dataset.count 
    
    for zstat in range(n_bands):
        print("BAND --",zstat)
        # zs is a dict containing information of the statistic
        zs = zonal_stats(AOI, input_raster, stats=['mean'], band=zstat+1)
        
        name_field = name_fields[zstat]
        
        # a datafrrame from dictionary
        bstats_df = pd.DataFrame(zs)
        bstats_df.rename(columns={'mean':name_fields[zstat]+"_m"}, inplace=True)
        
        #Concatenate extracted information with the main polygon
        AOI = pd.concat([AOI, bstats_df], axis=1)
    
    # store the whole dataframe
    
    AOI.to_file(output_data, driver="GeoJSON")
    
    print("AOI with raster values generated")
    


# Function 11
def save_to_csv(input_gdf, output_csv):
    """
    Stores geodataframe in a csv file
    
    :param input_gdf: path of geojson (or any geodata) with information.
    :type input_gdf: str
    
    :param output_csv: path to write the csv
    :type output_csv: str

    :return: void. writes a csv.
    
    """
    gdf = gpd.read_file(input_gdf)
    gdf.to_csv(output_csv)
    
    
# Function 12
def calculate_VI(band1, band2):
    """
    Calculates Vegetation Normalizaed Ratio indices using columns as input
    
    :param band1: column with band (Near Infrarred) values
    :type band1: pandas.Series
    
    :param band2: column with band (Red) values
    :type band2: pandas.Series

    :return: Vegetation index values
    :rtype: float
    
    """
    
    return ((band1 - band2) / (band1 + band2))
    
    
# Function 13
def run_MLalgorithm(training_dataset, predict_dataset, algorithm, test_size):
    """
    Runs a classification of the dataset using a Machine Learning algorithm
    Retrieving several metrics (see returns) 
    
    :param training_dataset: path of csv with training information
    :type training_dataset: str
    
    :param predict_dataset: path of csv with dataset to predict
    :type predict_dataset: str
    
    :param algorithm: algorithm object that will be used for classification with parameters defined. 
    :type output_csv: sklearn object

    :param test_size: decimal number to indicate the split in training and test of input dataset 
    :type test_size: float
    
    :return: Accuracy of model, accuracy with training, confusion matrix, Cohen's Kappa and predicted dataset
    :rtype: float, confusion_matrix, cohen_kappa_score, array
    
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.metrics import cohen_kappa_score
    
    
    training_dataset = pd.read_csv(training_dataset,sep = ",")
    predict_dataset = pd.read_csv(predict_dataset,sep = ",")
    
    # remove the column "Unnamed:0"
    training_dataset = training_dataset.iloc[:,1:]
    predict_dataset = predict_dataset.iloc[:,1:]

    # calculate NDVI in training dataset
    training_dataset['NDVI'] = training_dataset.apply(lambda x: calculate_VI(x['nir_m'], x['red_m']),axis = 1)
    
    # calculate GNDVI in training dataset
    training_dataset['GNDVI'] = training_dataset.apply(lambda x: calculate_VI(x['nir_m'], x['green_m']),axis = 1)
    
    # calculate NDVI in problem dataset
    predict_dataset['NDVI'] = predict_dataset.apply(lambda x: calculate_VI(x['nir_m'], x['red_m']),axis = 1)
    
    # calculate GNDVI in problem dataset
    predict_dataset['GNDVI'] = predict_dataset.apply(lambda x: calculate_VI(x['nir_m'], x['green_m']),axis = 1)
   
        # select in training dataset  the columns and prepare for algorithm: 
    
    # explanatory variables in a numpy array
    X = np.array(training_dataset[['NDVI', 'GNDVI','nir_m']]) # excluded visible bands

    # assign numbers to categories, as the ML algorithm uses number codes.
    #LC_types = {'Bare soil':1, 'Built-up surface':2 , 'Cropland': 3, 'Natural vegetation': 4, 'Water body': 5}
    LC_types = {'Cropland': 1, 'Natural vegetation': 2, 'Water body': 3}
    
    training_dataset['LC_cod'] = training_dataset['LandCover'].map(LC_types)
    
    # dependent variable
    y = training_dataset['LC_cod'].values.flatten()  # Categories

    seed = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    
    # fit and validate the model 
    fit_model = algorithm.fit(X_train, y_train)
    
    train_model = fit_model.predict(X_test)
    
    # compute accuracy of the model
    acc_model = accuracy_score(y_test, train_model)
    

        # predict on the whole dataset:
    
    dataset_pred = np.array(predict_dataset[['NDVI', 'GNDVI','nir_m']])
    
    whole_prediction = fit_model.predict(dataset_pred)
    
    
    # return the accuracy, confusion matrix and prediction
    
    return acc_model, fit_model.score(X_train, y_train), confusion_matrix(y_test,train_model), cohen_kappa_score(y_test, train_model), whole_prediction
    
    
# Function 14
def plot_confusion_matrix(input_confusionMatrix,kappa_value,algo_used):
    """
    shows the confusion matrix from the results
    
    :param input_confusionMatrix: confusion matrix
    :type input_confusionMatrix: confusion_matrix object
    
    :param kappa_value: kappa value
    :type kappa_value: float
    
    :param algo_used: name of the algorithm used to show on the title of the plot
    :type algo_used: str
    
    :return: void. Shows the confusion matrix
    
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")
    
    fig = plt.figure(figsize=[10, 8])
    
    
    
    ax = sns.heatmap(input_confusionMatrix, annot=True, cmap='Blues')

    ax.set_title(f'Confusion Matrix. Classifier: {algo_used} Kappa value: {round(kappa_value,2)}\n');
    ax.set_xlabel('\nPredicted Values');
    ax.set_ylabel('Actual Values ');
    
    ## Ticket labels 
    #ax.xaxis.set_ticklabels(['False','True'])
    #ax.yaxis.set_ticklabels(['False','True'])
    #name_labs = ['Bare soil', 'Built-up surface', 'Cropland', 'Natural vegetation', 'Water body']
    name_labs = [ 'Cropland', 'Natural vegetation', 'Water body']
    
    labels_ticks = [item.get_text() for item in ax.get_xticklabels()]
    labels_ticks[0] = name_labs[0]
    labels_ticks[1] = name_labs[1]
    
    labels_ticks[2] = name_labs[2]
    
    
    ax.set_xticklabels(labels_ticks)
    ax.set_yticklabels(labels_ticks)
    
    plt.savefig("Confusion_matrix_"+algo_used+".jpg",dpi = 250)
    plt.tight_layout()
    plt.show()

# Function 15
def LC_array_to_df(input_prediction, df,output_final_df):    
    """
    Assign the predicted codes to the main dataframe and trasnform them to the proper description
     
    :param input_prediction: predicted values
    :type input_prediction: np.array
    
    :param df: original polygon grid with "New_ID" field and removed the edges. 
    :type df: GeoDataFrame
    
    :param tile_name: tile of satellite footprint in the study area
    :type tile_name:  str
    
    :return: void. writes the final GeoDataFrame

    """
    #LC_types_rev = {1: 'Bare soil',2:'Built-up surface', 3:'Cropland', 4:'Natural vegetation', 5:'Water body'}
    
    LC_types_rev = {1: 'Cropland',2:'Natural vegetation', 3:'Water body'}
    
    df['Id_LC'] = input_prediction
    
    df['LandCover'] = df['Id_LC'].map(LC_types_rev)
    
    df.drop(['Id_LC'],axis = 1, inplace = True)
    df.to_file(output_final_df,driver = "GeoJSON")
    