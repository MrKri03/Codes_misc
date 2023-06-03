import os


###################################  
# get connection credentials
def extract_credentials(the_json):
    """
    Extracts credentials from a json file
    Retrieving the user and password
   
    :param the_json: path to JSON file where information is shown
    :type output_csv: str
    
    :return: user and password
    :rtype: str, str
    
    """
    import json
    with open(the_json, "r") as f:
    
        my_dict = json.load(f)
        f.close()
    
    credentials_user = my_dict["credentials_user"]
    credentials_password = my_dict["credentials_password"]
    
    return credentials_user, credentials_password



###################################  
def connect_landsatxplore():
     """
    Connects to Earth Explorer of USGS
   
    no input
    
    :return: connection API to EE if sucessful; otherwise, 
    message if it was successful to connect, otherwise, will retrieve an exception message
    :rtype: str
    
    """
    import sys
    from landsatxplore.api import API
    from landsatxplore.earthexplorer import EarthExplorer
    
    # Function to get  connection credentials
    user, password = extract_credentials("creds.json")
    
    
    # Connects to the landsatexplore API and EarthExplorer
    
    try: 
        connect_to_api = API(user, password)
        print("Connection to API successful")
        connect_to_ee = EarthExplorer(user, password)
        print("Connection to E E successful")
        return connect_to_api, connect_to_ee
    
    
    except Exception as e: 
        """
        The error related to INDEX OUT OF RANGE can be fixed with: 
        https://github.com/yannforget/landsatxplore/issues/82
        """
        print("\n ***THERE IS A CONNECTION ERROR to api*** \n")
        print(e)
        print(sys.exc_info()[2])

        
    
###################################      
def check_days_in_month(year,month):
    """
    This function retrieves the number of days of a month given a year and its month.
       
    :param year: year to analyse
    :type year: str
    
    :param month: month to analyse
    :type month: str
    
    :return: monthrange (number of days in the month)
    :rtype: int
    
    """
    
    from calendar import monthrange
    
    return monthrange(year,int(month))[1]
    


###################################     
def period_time(init_year,finit_year,months):
    """
    This extracts calculates the initial and finish month with specific number of days to end the month
       
    :param init_year: initial year of period
    :type init_year: str
    
    :param finit_year: last year of period
    :type finit_year: str
    
    :param months: month to analyse
    :type months: str
    
    :return: init_month, fin_month
    :rtype: str, str
    
    """
        
    days_month = check_days_in_month(init_year,months[1])
            
    init_month = str(init_year)+"-"+months[0]+"-01"
    
    fin_month = str(init_year)+"-"+months[1]+"-"+str(days_month)
    
    return init_month, fin_month

###################################
def generate_period_df(sensor, period_t,n_scenes):
    """
    Generates a dataframe with the information of the image with sensor and the number of images captured in that period of time
       
    :param sensor: name of the sensor
    :type sensor: str
    
    :param period_t: time where the image was captured
    :type period_t: str
    
    :param n_scenes: number of scenes found
    :type n_scenes: str
    
    :return: df
    :rtype: pandas DataFrame
    """
    
    import pandas as pd
    
    df = pd.DataFrame(columns=['sensor','period','n_scenes'])
    
    df.loc[len(df)]= pd.Series([sensor, period_t,n_scenes ], index=['sensor','period','n_scenes'])
    
    return df

#########################################    
def generate_specific_scene_df(display_id,cloud_cover):
    """
    Generates a dataframe with specific information about the information of cloud cover and the id of image to be used
       
    :param display_id: attribute "display_id" from metadata in Landsat
    :type display_id: str
    
    :param cloud_cover: % cloud cover
    :type cloud_cover: int
       
    :return: df
    :rtype: pandas DataFrame
    """
    
    import pandas as pd
    
    
    df = pd.DataFrame({
        
        'id_image': display_id,
        
        'date': [date.split("_")[3][0:4] +\
                 "-" +date.split("_")[3][4:6] +\
                 "-" + date.split("_")[3][6:8] for date in display_id],
        
        'cloudCover': cloud_cover
    })
    
    
    return df
    
#########################################
def retrieve_images_period_years(connection_api,landsat_data, init, fin, max_cloudCover, 
               study_area_extent, period ):
    """
    Retrieves all useful information from previous functions:
    df of number of images found in a period of time, 
    df of specific images found and cloud cover
    and list of images
       
    :param connection_api: connection api with credentials already satistied
    :type connection_api: connection API object
    
    :param landsat_data: id of landsat data found in the metadata
    :type landsat_data: str
    
    :param init: initial date
    :type init: str
    
    :param fin: final date
    :type fin: str    
    
    :param max_cloudCover: % cloud cover
    :type max_cloudCover: int    
        
    :param study_area_extent: tuple of coordinates (total_bounds from shape)
    :type study_area_extent: tuple
    
    :param period: period of time
    :type period: int 
    
    :return: list of landsat images found, df_period, df_specific
    :rtype: list, pandas dataframe, pandas dataframe
    """
    
    all_scenes = []
    
    #controls the sensor to show in the period of year
    
    landsat_sensor = list(landsat_data.keys())[period]
    
    print("Sensor:",landsat_sensor.upper(),"dates",init," -- ",fin)

    scenes = connection_api.search(
                    dataset= landsat_data.get(landsat_sensor),
                    bbox = study_area_extent,
                    start_date= init,
                    end_date= fin,
                    max_cloud_cover=max_cloudCover)
            
            
    all_scenes.append([i.get("display_id") for i in scenes])
    
    print(f"{len(scenes)} scenes found.")
    
    df_period = generate_period_df(landsat_sensor.upper(), "May-Sep "+init.split("-")[0], len(scenes))
    
    df_specific = generate_specific_scene_df([i.get("display_id") for i in scenes],
                                            [i.get("cloud_cover") for i in scenes])
    

    return [indv_scene for scenes in all_scenes for indv_scene in scenes], df_period, df_specific

#########################################
def flatten_list(input_list):
    """
    generates a plain list of elements
   
    :param input_list: input list with elements
    :type input_list: list
    
    :return: flattened list
    :rtype: list
    """
    
    return [one for el in input_list for one in el]


#####################################
def create_path_download(main_path,date):
    """
    creates a specific local folder to download the requested images to API
   
    :param main_path: root path to create the folder
    :type main_path: str
    
    :param date: period of time identifying the image
    :type date: str
    
    :return: path of new folder
    :rtype: str
    """
    
    path = os.path.join(main_path,"_"+date)
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    return path

#####################################
def extract_landsat_from_tar(path_tar, path_output,img_id,conditions):
    """
    Extracts requested files from downloaded .tar file
       
    :param path_tar: Location of original tar file
    :type path_tar: str
    
    :param path_output: Destination of extracted files from tar
    :type path_output: str
    
    :param img_id: identifier of landsat image
    :type img_id: str
    
    
    :return: no return, extracts files from tar and removes the initial tar file
    :rtype: None
    
    
    """
    
    import tarfile

    
    the_tar_file = os.path.join(path_tar,img_id+".tar")
    
    
    with tarfile.open(the_tar_file, mode="r") as tar:
        for file in tar.getnames():
            if any(x in file for x in list(conditions.values()) ):
                
               # print("extracting",file)
                tar.extract(file,path= path_output)
                
        tar.close()            
    # remove the tar file
    os.remove(the_tar_file)
        
#####################################        
def create_not_valid_IDs(img_id,error_message):
    
    """
    For or some reason, identifiers cannot be found in EE API. Creates a dataframe with error messages
       
    :param img_id: image identifier
    :type img_id: str
    
    :param error_message: Error message to write in the in the dataframe
    :type error_message: str
    
    :return: dataframe with error messages
    :rtype: pandas DataFrame
    
    
    """
    
    import pandas as pd
    
    df_not_valid_ids = pd.DataFrame({
        
        
        "id_img": img_id,
        "error_msg": [error_message]

    })
        
        
    return df_not_valid_ids
    
#####################################  
def download_scenes(main_path,erthexp_object, df_scenes,conditions):
    """
    Function that uses all previous functions. Downloads images, stores images requested and removes tar files downloaded, not necessary
    Outputs csvs with information and error from images 
       
    :param main_path: path where files will be stored
    :type main_path: str
    
    :param erthexp_object: Earth Explorer generated object from API
    :type erthexp_object: EE API object
    
    :param df_scenes: pandas dataframe with all specific image identifiers
    :type df_scenes: pandas dataframe
    
    :param conditions: string containing the name of bands requested to extract them
    :type conditions: str
    
    :return: number (counter) of images downloaded.
    :rtype: int
    
    """
    import pandas as pd
    
    count_success = 0
    df_all_error = pd.DataFrame()
    
    for row in range(0,len(df_scenes)):
        print(df_scenes.date.iloc[row])
        
        
        if not os.path.exists(os.path.join(main_path,"_"+df_scenes.date.iloc[row].replace("-","_"))):

        
            #get image ID
            img_id = df_scenes.id_image.iloc[row]
        
            try:
                erthexp_object.download(img_id, output_dir=main_path)
                
                            #Call function to create a folder with name of that date and get the path
                path = create_path_download(main_path,
                                    df_scenes.date.iloc[row].replace("-","_"))
            
                extract_landsat_from_tar(main_path,path,img_id,conditions)
            
                count_success = count_success +1 
    
            except Exception as e:
                print(e)
                df_error_message = create_not_valid_IDs(img_id,e)
                df_all_error = pd.concat([df_all_error,df_error_message],axis = 0,ignore_index = True) # Will work?
                df_all_error.to_csv(r".\error_images.csv")
                continue
    
        else:
            print(os.path.join(main_path,"_"+df_scenes.date.iloc[row].replace("-","_"))," exists")
            continue
        
        #df_all_error.to_csv(r".\error_images.csv")
    return count_success
