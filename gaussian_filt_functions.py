import math
from scipy.signal.windows import gaussian
from scipy.ndimage import gaussian_filter
from scipy import signal, stats
from numpy import average,mean
import numpy as np
import pandas as pd

###
## Whole process to generate a 2D gaussian filter on input arrays from sections of drone images within greater spatial units ##
###

## 1 ##
def gkern(kern_length, std):
    
    """Returns a 2D Gaussian kernel array.
    https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    import matplotlib.pyplot as plt
    
    gkern1d = signal.gaussian(kern_length, std=std).reshape(kern_length, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    plt.plot(signal.gaussian(kern_length, std=std))
    plt.show()
    #gkern2d = gkern1d[:,None] @ gkern1d[None]
  
    return gkern2d

## 2 ##
def make_simetric_arrays(array, shape0, shape1):
    
    if shape0 > shape1:
        return array[:-(shape0-shape1),:]
    elif shape0 < shape1:
        return array[:,:-(shape1-shape0)]
    else:
        return array

    
## 3 ##
def stat_function(x):
 
    import matplotlib.pyplot as plt
    
    new_array = make_simetric_arrays(x, x.shape[0], x.shape[1])
    
    #Calculate standard deviation of UAV values in each pixel
    
   # sd = np.nanstd(new_array.flatten(),dtype=np.float64)
    #print(sd*3)
    
    #print(math.pow(sd*3, 2))
        
    #Creates gkern variable
    #99.7% of the data in the gaussian distribution
    
    # simple reference: https://www.scribbr.com/statistics/standard-deviation/
    # second argument is standard deviation, which one to use? <----
    
    gauss = gkern(new_array.shape[0],45) # Changing the standard deviation of gauss bell 
    
    plt.imshow(gauss, interpolation='none', cmap = "gist_gray")
    plt.show()
    
    # Here is the KEY, where I perform 2D weighted sum
    np_average = average(new_array,axis = None, weights = gauss) 
    
    return np_average

## 4 ##
def crop_pixels_in_grid(df_geometry, input_raster):
    """ Input raster here is a individual raster because
    I iterate through a list of glob before
    """
    import rasterio as rio
    import rasterio.mask
   
    

    with rio.open(input_raster) as src:

        out_image, out_transform = rio.mask.mask(src, df_geometry, all_touched = False, crop=True) # False because we need perfectly alligned
        out_meta = src.meta
        src.close()
    
    out_meta.update({"driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform})
    
    # I do not need the following:  out_meta
    return out_image, out_transform

    
## 5 ##
def weight_average_2D_Gaussian(grid_cell, input_raster):
    
    from rasterstats import zonal_stats
    import matplotlib.pyplot as plt
    
    #
    
    cropped_image,transform = crop_pixels_in_grid(grid_cell["geometry"],input_raster)
    
    cropped_image = np.squeeze(cropped_image, axis=0)
    
    
    plt.imshow(cropped_image, interpolation='none',cmap='GnBu_r',vmin=0, vmax=np.amax(cropped_image))
    plt.show()
    
    zs = zonal_stats(grid_cell,
                     cropped_image, affine = transform,
                     stats= ['mean','std'],
                     add_stats={'gauss':stat_function},band=1)
    
    stats = pd.DataFrame(zs)
 
    stats['grid_ID'] = grid_cell["New_ID"].values[0].astype(int)# This is the number of New ID
     
    end_string = "bandX" ## Only to test
    
    stats.rename(columns={'mean':"mea_"+end_string, 'std':'std_'+end_string,'gauss':"gss_"+end_string}, inplace=True)
                
    # Returns a pandas dataframe with only stats for ONE CELL. 
    # Then, you should run this function in a loop in the main script.
    return stats


    
    
    