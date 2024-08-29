import xarray as xr
import numpy as np
import scipy.ndimage
from skimage.measure import regionprops 
from skimage.measure import label as label_np
import dask.array as dsa
from sklearn.metrics.pairwise import haversine_distances
import pandas as pd

def _apply_mask(binary_images, mask):
    binary_images_with_mask = binary_images.where(mask==1, drop=False, other=0)
    return binary_images_with_mask

class Tracker:
        
    def __init__(self, da, mask, radius, min_size_quartile, timedim, xdim = 'xh', ydim = 'yh', positive=True):
        
        self.da = da
        self.mask = mask
        self.radius = radius
        self.min_size_quartile = min_size_quartile
        self.timedim = timedim
        self.xdim = xdim
        self.ydim = ydim   
        self.positive = positive
        
        if ((timedim, ydim, xdim) != da.dims):
            try:
                da = da.transpose(timedim, ydim, xdim) 
            except:
                raise ValueError(f'Ocetrac currently only supports 3D DataArrays. The dimensions should only contain ({timedim}, {xdim}, and {ydim}). Found {list(da.dims)}')

            
    def track(self):
        '''
        Label and track image features.
        
        Parameters
        ----------
        da : xarray.DataArray
            The data to label.

        mask : xarray.DataArray
            The mask of ponts to ignore. Must be binary where 1 = true point and 0 = background to be ignored. 

        radius : int
            The size of the structuring element used in morphological opening and closing. Radius specified by the number of grid units.

        min_size_quartile : float
            The quantile used to define the threshold of the smallest area object retained in tracking. Value should be between 0 and 1.

        timedim : str
            The name of the time dimension
        
        xdim : str
            The name of the x dimension

        ydim : str
            The namne of the y dimension
            
        positive : bool
            True if da values are expected to be positive, false if they are negative. Default argument is True

        Returns
        -------
        labels : xarray.DataArray
            Integer labels of the connected regions.
        '''

        if (self.mask == 0).all():
            raise ValueError('Found only zeros in `mask` input. The mask should indicate valid regions with values of 1')

        # Convert data to binary, define structuring element, and perform morphological closing then opening
        binary_images = self._morphological_operations()

        # Apply mask
        binary_images_with_mask  = _apply_mask(binary_images,self.mask) # perhaps change to method? JB

        # Filter area
        area, min_area, binary_labels, N_initial = self._filter_area(binary_images_with_mask)

        # Label objects
        labels, num = self._label_either(binary_labels, return_num= True, connectivity=3)

        # Wrap labels
        grid_res = abs(self.da[self.xdim][1]-self.da[self.xdim][0])
        if self.da[self.xdim][-1]-self.da[self.xdim][0] >= 360-grid_res:
            labels_wrapped, N_final = self._wrap(labels)
        else:
            labels_wrapped = labels
            N_final = np.max(labels)
                
        # Final labels to DataArray
        new_labels = xr.DataArray(labels_wrapped, dims=self.da.dims, coords=self.da.coords)   
        new_labels = new_labels.where(new_labels!=0, drop=False, other=np.nan)

        ## Metadata

        # Calculate Percent of total object area retained after size filtering
        sum_tot_area = int(np.sum(area.values))

        reject_area = area.where(area<=min_area, drop=True)
        sum_reject_area = int(np.sum(reject_area.values))
        percent_area_reject = (sum_reject_area/sum_tot_area)

        accept_area = area.where(area>min_area, drop=True)
        sum_accept_area = int(np.sum(accept_area.values))
        percent_area_accept = (sum_accept_area/sum_tot_area)

        new_labels = new_labels.rename('labels')
        new_labels.attrs['inital objects identified'] = int(N_initial)
        new_labels.attrs['final objects tracked'] = int(N_final)
        new_labels.attrs['radius'] = self.radius
        new_labels.attrs['size quantile threshold'] = self.min_size_quartile
        new_labels.attrs['min area'] = min_area
        new_labels.attrs['percent area reject'] = percent_area_reject
        new_labels.attrs['percent area accept'] = percent_area_accept

        print('inital objects identified \t', int(N_initial))
        print('final objects tracked \t', int(N_final))

        return new_labels

    def collect_surface_stats(self, labels):
        """Collects surface statistics"""
    
        ids = np.unique(labels)
        ids = np.array([id for id in ids if ~np.isnan(id)])
        
        dataframes = []
        num_events = len(ids)
    
        for i in range(num_events):
            event = labels.where(labels == ids[i], drop=True)
            df = self._get_surface_stats(event) #self.da or sst
            dataframes.append(df)
            #break
    
        dff = pd.concat(dataframes, ignore_index=True)
        return dff

    ### PRIVATE METHODS - not meant to be called by user ###

    def _compute_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
        """Compute the x and y resolution in km for a geographic degree resolution
    
        :param clat: the point latitude at which to compute the resolution
        :param clon: the point longitude at which to compute the resolution
        :param res: the geographic degree resolution
        :return: distance in km between two points
        """
        
        dist = haversine_distances(
            np.deg2rad(np.array([(lat1, lon1)])),
            np.deg2rad(np.array([(lat2, lon2)])),
        )
        
        dist = dist * 6371000 / 1000  # multiply by Earth radius to get kilometers
        dist
        return dist[0][0]

    def _get_surface_stats(self, event):
        """ get surface stats for one event """
        
        # Initialize dictionary 
        mhw = {}
        mhw['id'] = [] # event label
        mhw['dates'] = [] # datetime format
        mhw['coords'] = [] # (lat, lon)
        mhw['centroid'] = []  # (lat, lon)
        mhw['duration'] = [] # [months]
        mhw['intensity_max'] = [] # [deg C]
        mhw['intensity_mean'] = [] # [deg C]
        mhw['intensity_min'] = [] # [deg C]
        mhw['intensity_cumulative'] = [] # [deg C]
        mhw['total_area'] = [] # [km2]
        mhw['distance'] = [] 
        
        # TO ADD:
        # mhw['rate_onset'] = [] # [deg C / month]
        # mhw['rate_decline'] = [] # [deg C / month]
        
        mhw["id"].append(int(np.nanmedian(event.values)))
        mhw["dates"].append(event.time.values)
        mhw["duration"] = event.time.shape[0]
        mhw["total_area"] = int(event.sum("time").count((self.ydim,self.xdim)).values)
        
        centroid_list = []
        distance_list = []
        
        prev_coords = None  # Initialize previous coordinates as None
        
        for time in event.time.values:
            mhw_slice = event.sel(time=time)
            
            # Create latitude and longitude grids
            lon_grid, lat_grid = np.meshgrid(np.arange(0, mhw_slice[self.xdim].shape[0]), np.arange(0, mhw_slice[self.ydim].shape[0]))
            lon_flat = lon_grid.flatten()
            lat_flat = lat_grid.flatten()
            data_flat = mhw_slice.values.flatten()
        
            # Compute the centroid
            centroid_lon = np.nansum(lon_flat * data_flat) / np.nansum(data_flat)
            centroid_lat = np.nansum(lat_flat * data_flat) / np.nansum(data_flat)

            coords = [int(mhw_slice[self.xdim][round(centroid_lon)].values), int(mhw_slice[self.ydim][round(centroid_lat)].values)]
            centroid_list.append(coords)
            
            # Extract lat and lon from the current coordinates
            lon2, lat2 = coords  # lon2 is the first element, lat2 is the second
            
            # Calculate distance from previous coordinates, if available
            if prev_coords is not None:
                lon1, lat1 = prev_coords  
                distance = self._compute_distance(lat1, lon1, lat2, lon2)
                distance_list.append(distance)
            
            prev_coords = coords  
        
        # Add a None or NaN for the first distance since it's undefined
        distance_list.insert(0, np.nan)
        #distance_list.insert(0, None)  # or use `np.nan` if you prefer
        
        mhw['centroid'].append(centroid_list)
        mhw['distance'].append(distance_list)
        
        # Process intensity metrics using try-except to handle ValueError
        event_ssta = self.da.where(event > 0, drop=True)
        
        try:
            mhw['intensity_mean'].append(event_ssta.mean((self.ydim, self.xdim)).thetao.values)
        except ValueError:
            mhw['intensity_mean'].append(np.nan)
        try:
            mhw['intensity_max'].append(event_ssta.max((self.ydim, self.xdim)).thetao.values)
        except ValueError:
            mhw['intensity_max'].append(np.nan)
        try:
            mhw['intensity_min'].append(event_ssta.min((self.ydim, self.xdim)).thetao.values)
        except ValueError:
            mhw['intensity_min'].append(np.nan)
        try:
            mhw['intensity_cumulative'].append(np.nansum(event_ssta.to_array()))
        except ValueError:
            mhw['intensity_cumulative'].append(np.nan)
            
        coords = event.stack(z=(self.ydim, self.xdim))
        coord_pairs = [(coords.isel(time=t[0]).dropna(dim='z', how='any').z[self.ydim].values, 
                          coords.isel(time=t[0]).dropna(dim='z', how='any').z[self.xdim].values) for t in enumerate(event.time)]
        
        mhw['coords'].append(coord_pairs)
        
        mhw = pd.DataFrame(dict([(name, pd.Series(data)) for name,data in mhw.items()]))
        return mhw

    def _morphological_operations(self): 
        '''Converts xarray.DataArray to binary, defines structuring element, and performs morphological closing then opening.
        Parameters
        ----------
        da     : xarray.DataArray
                The data to label
        radius : int
                Length of grid spacing to define the radius of the structing element used in morphological closing and opening.

        '''

        # Convert images to binary. All positive values == 1, otherwise == 0
        if self.positive == True:
            bitmap_binary = self.da.where(self.da>0, drop=False, other=0)
        
        elif self.positive == False:
            bitmap_binary = self.da.where(self.da<0, drop=False, other=0)
    
        bitmap_binary = bitmap_binary.where(bitmap_binary==0, drop=False, other=1)

        # Define structuring element
        diameter = self.radius*2
        x = np.arange(-self.radius, self.radius+1)
        x, y = np.meshgrid(x, x)
        r = x**2+y**2 
        se = r<self.radius**2

        def binary_open_close(bitmap_binary):
            bitmap_binary_padded = np.pad(bitmap_binary,
                                          ((diameter, diameter), (diameter, diameter)),
                                          mode='wrap')
            s1 = scipy.ndimage.binary_closing(bitmap_binary_padded, se, iterations=1)
            s2 = scipy.ndimage.binary_opening(s1, se, iterations=1)
            unpadded= s2[diameter:-diameter, diameter:-diameter]
            return unpadded

        mo_binary = xr.apply_ufunc(binary_open_close, bitmap_binary,
                                   input_core_dims=[[self.ydim, self.xdim]],
                                   output_core_dims=[[self.ydim, self.xdim]],
                                   output_dtypes=[bitmap_binary.dtype],
                                   vectorize=True,
                                   dask='parallelized')
        return mo_binary


    def _filter_area(self, binary_images):
        '''calculatre area with regionprops'''

        def get_labels(binary_images):
            blobs_labels = self._label_either(binary_images, background=0)
            return blobs_labels

        labels = xr.apply_ufunc(get_labels, binary_images,
                                input_core_dims=[[self.ydim, self.xdim]],
                                output_core_dims=[[self.ydim, self.xdim]],
                                output_dtypes=[binary_images.dtype],
                                vectorize=True,
                                dask='parallelized')


        labels = xr.DataArray(labels, dims=binary_images.dims, coords=binary_images.coords)
        labels = labels.where(labels>0, drop=False, other=np.nan)  

        # The labels are repeated each time step, therefore we relabel them to be consecutive
        for i in range(1, labels.shape[0]):
            labels[i,:,:] = labels[i,:,:].values + labels[i-1,:,:].max().values

        labels = labels.where(labels>0, drop=False, other=0)  
        labels_wrapped, N_initial = self._wrap(np.array(labels))

        # Calculate Area of each object and keep objects larger than threshold
        props = regionprops(labels_wrapped.astype('int'))
        
        labelprops = [p.label for p in props]
        labelprops = xr.DataArray(labelprops, dims=['label'], coords={'label': labelprops}) 
        
        area = xr.DataArray([p.area for p in props], dims=['label'], coords={'label': labelprops})  # Number of pixels of the region.

        if area.size == 0:
            raise ValueError(f'No objects were detected. Try changing radius or min_size_quartile parameters.')
        
        min_area = np.percentile(area, self.min_size_quartile*100)
        print(f'minimum area: {min_area}') 
        
        keep_labels = labelprops.where(area>=min_area, drop=True)
        keep_where = np.isin(labels_wrapped, keep_labels)
        out_labels = xr.DataArray(np.where(keep_where==False, 0, labels_wrapped), dims=binary_images.dims, coords=binary_images.coords)

        # Convert images to binary. All positive values == 1, otherwise == 0
        binary_labels = out_labels.where(out_labels==0, drop=False, other=1)

        return area, min_area, binary_labels, N_initial


    def _label_either(self, data, **kwargs):
        if isinstance(data, dsa.Array):
            try:
                from dask_image.ndmeasure import label as label_dask
                def label_func(a, **kwargs):
                    ids, num = label_dask(a, **kwargs)
                    return ids
            except ImportError:
                raise ImportError(
                    "Dask_image is required to use this function on Dask arrays. "
                    "Either install dask_image or else call .load() on your data."
                )
        else:
            label_func = label_np
        return label_func(data, **kwargs)


    def _wrap(self, labels):
        ''' Impose periodic boundary and wrap labels'''
        first_column = labels[..., 0]
        last_column = labels[..., -1]

        unique_first = np.unique(first_column[first_column>0])

        # This loop iterates over the unique values in the first column, finds the location of those values in 
        # the first columnm and then uses that index to replace the values in the last column with the first column value
        for i in enumerate(unique_first):
            first = np.where(first_column == i[1])
            last = last_column[first[0], first[1]]
            bad_labels = np.unique(last[last>0])
            replace = np.isin(labels, bad_labels)
            labels[replace] = i[1]

        labels_wrapped = np.unique(labels, return_inverse=True)[1].reshape(labels.shape)

        # recalculate the total number of labels 
        N = np.max(labels_wrapped)

        return labels_wrapped, N