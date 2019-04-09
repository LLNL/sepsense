# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 18:23:23 2017

@author: Michael Zelinski
"""

import matplotlib.pyplot as plt
import numpy as np
import os.path as p
import json

from osgeo import gdal

def circle(dim,r):
    """
    This function defines the aperture array that is converted into a psf
    and mtf
    
    Parameters
    ----------
    'dim': integer, dimension of circle image
    
    'r': integer, radius of circle
    
    Returns
    -------
    'y': array with dimensions [dim,dim] with circle in center    
    """
    
    coordsx = np.dot( np.expand_dims(np.arange(-dim/2,dim/2), axis=1), np.ones((1,dim)))
    coordsy = coordsx.transpose()
    y = (np.sqrt( coordsx**2 + coordsy**2 )<r).astype('float')
    return y


class imageSystem:
    """
    This is a class that for defining a downward looking satellite system. 
    
    Parameters
    ----------
    
    focal length, primary/secondary aperture diameter, pixel pitch, altitude
    wavelength center, gaussian noise, integration time, optic transmission, 
    detector quantum efficiency, detector electron well depth, detector bit
    depth, detector delta wavelength.  And then several other parameters that 
    calculated from these initial parameters.
    """    
    
    def __init__( 
            self, focal_length=10, p_diameter=.7, s_diameter=.10, pixel_pitch=8E-6, alt=400E3, wvl=np.array([450,550,650])*1E-9, 
            noise = 1, Tint = .001, opticTransmission = 1, QE = np.array([1,1,1]), 
            election_well_depth = 20000, bit_depth = 12, read_noise = 20, add_poissonNoise = 0, delta_wvl = np.array([100,100,100])*1E-9,
            force_no_aliasing=0):
        
        self.f = focal_length #focal length
        self.p_diameter = p_diameter #primary aperture diameter [meters]
        self.s_diameter = s_diameter #secondary aperture diameter [meters]
        self.pixel_pitch = pixel_pitch #width between pixel centers on fpa [meters]
        self.alt = alt #altitude [meters]
        self.wvl = wvl #wavelength [meters]
        self.delta_wvl = delta_wvl #spectral bandwidth [meters]
        self.read_noise = read_noise #[electrons]
        self.Tint = Tint #integration time
        self.opticTransmission = opticTransmission #optical transmission
        self.QE = QE #quantum efficiency
        self.nyquist = 1/(2*pixel_pitch) #Nyquist frequency        
        self.fNum = self.f/p_diameter #f number
        self.F_fill = 1 - (np.pi*(s_diameter/2)**2)/(np.pi*(p_diameter/2)**2) #optics fill factor percentage
        self.GSS = wvl * alt/p_diameter #ground sample size
        self.optCut = 1/(wvl * self.fNum) #optical cutoff frequency
        self.gnd_optCut = .5/self.GSS # .5/( 1/self.optCut * self.alt / self.f  )
        self.gsd = pixel_pitch * (alt/self.f) #ground sample distance
        self.gnd_nyquist = 1./(2*self.gsd) #nyquist sampling frequency
        self.nbands = len(wvl) #number of bands
        self.gain = election_well_depth/2**float(bit_depth) #gain
        self.election_well_depth=election_well_depth #number of elections a pixel can hold
        self.add_poissonNoise = add_poissonNoise #boolean to add or not add Poisson noise
        self.bit_depth=bit_depth #bit depth
        self.Q = wvl/pixel_pitch * self.f/p_diameter #detector Q.
        if self.GSS[0] < self.gsd:
            self.requiredGSD = self.GSS[0] #required gsd to be sampled by detector and optics
        else:
            self.requiredGSD = self.gsd
        self.force_no_aliasing = force_no_aliasing


def convert_to_radiance(im):
    """"
    Parameters
    ----------
    'im' object: image object
    
    
    Returns
    -------
    'im' object: image object that is updated with radiance values.
    
    For calculating radiance from digital numbers
    #############################
    Gains, offsets, and calibration equations provided here: 
    https://calval.cr.usgs.gov/wordpress/wp-content/uploads/JACIE2015_Kuester_V3.pdf
    Delta values taken from here:
    https://dg-cms-uploads-production.s3.amazonaws.com/uploads/document/file/207/Radiometric_Use_of_WorldView-3_v2.pdf
    Look closely at comments as I had to search around for additional 
    information - the urls are provided and assumed accurate. DigitalGlobe does
    not do a very good job defining these parameters online. Usually this
    information is provided in the .IMD and it should have been provide in the
    .json files by the fMoW folks, but it wasn't. 
    """
    
    if im.meta['sensor_platform_name'] == 'WORLDVIEW03_VNIR':
        g = np.array([1.028, 1.082, 1.070]) #RGB
        o = np.array([1.807, 2.633, 4.253]) #RGB
        delta = np.array([0.0585, 0.0618, 0.0540]) #RGB
        e_sun = np.array([1535.33, 1830.18, 2004.61]) #RGB
    
    # Delta values taken from here:
    # http://www.pancroma.com/downloads/Radiometric_Use_of_WorldView-2_Imagery.pdf
    if im.meta['sensor_platform_name'] == 'WORLDVIEW02':
        g = np.array([1.042, 1.063, 1.004]) #RGB
        o = np.array([1.752, 4.455, 9.809]) #RGB
        delta = np.array([0.0574, 0.0630, 0.0543]) #RGB 
        e_sun = np.array([1538.85, 1829.62, 2007.27]) #RGB
    
    # Reusing delta values from WV-II because they were not found online
    if im.meta['sensor_platform_name'] == 'GEOEYE01':
        g = np.array([.998, .994, 1.054]) #RGB
        o = np.array([-3.754, -4.175, -4.537]) #RGB
        delta = np.array([0.0574, 0.0630, 0.0543]) #RGB 
        e_sun = np.array([1491.49, 1828.83, 1993.18]) #RGB
    
    # https://apollomapping.com/wp-content/user_uploads/2011/09/Radiance_Conversion_of_QuickBird_Data.pdf    
    if im.meta['sensor_platform_name'] == 'QUICKBIRD02':
        g = np.array([1.060, 1.071, 1.105]) #RGB
        o = np.array([-2.954, -3.338, -2.820]) #RGB
        delta = np.array([0.071, 0.099, 0.068]) #RGB 
        e_sun = np.array([1553.78, 1823.64, 1949.59]) #RGB
    
    # https://github.com/NikosAlexandris/i.ikonos.toar/blob/master/i.ikonos.toar.py
    if im.meta['sensor_platform_name'] == 'IKONOS':
        g = np.array([.940, .990, 1.073]) #RGB
        o = np.array([-4.767, -7.937, -9.699]) #RGB
        delta = np.array([0.0658, 0.0886, 0.0713]) #RGB
        e_sun = np.array([1517.76, 1803.28, 1921.26]) #RGB
    
        
    # retrieving absolute calibration values.
    for i in range(0,len(im.meta['abs_cal_factors'])):
        if im.meta['abs_cal_factors'][i]['band']=='red':
            acf_red = im.meta['abs_cal_factors'][i]['value']
        if im.meta['abs_cal_factors'][i]['band']=='green':
            acf_green = im.meta['abs_cal_factors'][i]['value']
        if im.meta['abs_cal_factors'][i]['band']=='blue':
            acf_blue = im.meta['abs_cal_factors'][i]['value']
    
    acf = np.array([acf_red, acf_green, acf_blue])
    
    # radiance calibration as well as sensor drift correction.
    im.im = im.im * acf/delta * (2-g) - o
    im.meta.update({'e_sun': e_sun})
    im.meta.update({'units': 1})
    return im

       
class dg_rgb_im:
    """ 
    This is the Digital Globe RGB image class. 
    
    A DG RGB image object contains the dataset, meta data, and several useful 
    functions for padding, adjusting the bounding box after resampling, and 
    displaying. 
    
    """
    def __init__(self, meta={}, im=np.empty(0)):
        self.meta=meta
        self.im = im.astype('float')
        
    def makeSquare(self):
        """
        minDim = np.argmin(self.im.shape[0:1])
        padWidth = int(np.round(np.abs(self.im.shape[np.argmin(self.im.shape[0:1])] - self.im.shape[np.argmax(self.im.shape)])/2))
        if minDim == 0:
            self.im = np.concatenate((np.zeros((padWidth, self.im.shape[1], self.im.shape[2])), self.im, np.zeros((padWidth, self.im.shape[1], self.im.shape[2])) ), axis=0)            
        else:
            self.im = np.concatenate((np.zeros((self.im.shape[1], padWidth, self.im.shape[2])), self.im, np.zeros(( self.im.shape[1], padWidth, self.im.shape[2])) ), axis=1)
        self.meta['img_width'] = self.im.shape[1]
        """
        # Make image square
        h, w, c = self.im.shape
        mx = max(w, h)
        mn = min(w, h)
        sq_im = np.zeros((mx, mx, c))
        d = mx - mn
        lb = d//2
        ub = mx - d//2 - d%2
        if w > h:
            sq_im[lb:ub, :, :] = self.im
        elif h > w:
            sq_im[:, lb:ub, :] = self.im
        else:
            sq_im = self.im
        self.im = sq_im
        self.meta['square'] = lb, ub, w, h

    def adjustBoundingBox(self):
        """
        scaleFraction = self.im.shape[1] / self.meta.get('img_width')
        for i in self.meta.get('bounding_boxes'):   #Wow python is nice.  This block updates the bounding boxes.  Look closely at how.  
            i.update({'box': np.round(np.array(i.get('box')) * scaleFraction).astype('int').tolist()})
        """
        if 'square' in self.meta:
            lb, ub, w, h = self.meta['square']
            if w >= h:
                f = self.im.shape[1] / w
            elif h > w:
                f = self.im.shape[0] / h
            self.meta['square'] = round(lb*f), round(ub*f), round(w*f), round(h*f)
        #if 'pc' in self.meta:
        #    print(self.meta['pc'])
        #    self.meta['pc'] = max(1, round(self.meta['pc']*f))
        #    print(self.meta['pc'])
            
    def restoreDimensions(self):
        """
        minDim = np.argmin([self.meta.get('img_height'), self.meta.get('img_width')])
        if minDim == 0:
            fraction = ((self.meta.get('img_width') - self.meta.get('img_height'))/2)/self.meta.get('img_width')
            self.im = self.im[   int(np.round(self.im.shape[0] * fraction)) : int(np.round(self.im.shape[0] - self.im.shape[0] * fraction)) , :, :]
        else:
            fraction = ((self.meta.get('img_height') - self.meta.get('img_width'))/2)  /self.meta.get('img_height')
            self.im = self.im[:, int(np.round(self.im.shape[0] * fraction)) : int(np.round(self.im.shape[0] - self.im.shape[0] * fraction)) , :]

        self.meta.update({'img_height': self.im.shape[0]})
        self.meta.update({'img_width': self.im.shape[1]})
        """
        lb, ub, w, h = self.meta['square']
        if w > h:
            self.im = self.im[lb:ub, :, :]
        elif h > w:
            self.im = self.im[:, lb:ub, :]

    def restoreContext(self):
        if 'pc' in self.meta:
            pc = self.meta['pc']
            self.im = self.im[pc:-pc, pc:-pc, :]
               
    def displayIm(self):
        plt.figure()
        plt.imshow(self.im.astype('uint8'), interpolation = 'nearest')
        
    def displayIm_DN(self):
        plt.figure()
        #plt.imshow((self.im * [255,255,255] / self.im.max(axis=0).max(axis=0)).astype('uint8'), interpolation = 'nearest')
        plt.imshow( (self.im * [255,255,255] / self.im.max()).astype('uint8'), interpolation = 'nearest')
        
    def displayBB(self):
        bb = self.meta.get('bounding_boxes')
        scalar =  [255,255,255] / self.im.max() #self.im.max(axis=0).max(axis=0)
        for i in bb:
            coords = i.get('box')
            plt.figure()
            plt.imshow((self.im[ coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2], :] * scalar).astype('uint8'), interpolation='nearest')


def readDG_tif(fn, ms=False):
    """
    This function reads the RGB Digital Globe images provided in the fMoW 
    dataset as wells as the meta data for each file.  
    
    Parameters
    ----------
    'fn': string, full filepath
    
    Returns
    -------
    'im': object, an object that contains the im.im NxMx3 radiance dataset.  
    And the im.meta which is the meta data for the image.    
    """
    im1_fn = p.abspath(fn)
    
    if not p.exists(im1_fn):
        print('Image does not exist.  Check filepath: {}'.format(fn))
        return

    ds = gdal.Open(im1_fn)
    im = ds.GetRasterBand(1).ReadAsArray()
    im = np.stack((im, ds.GetRasterBand(2).ReadAsArray()))
    
    for bi in range(ds.RasterCount-2):
        im = np.vstack((im, np.expand_dims(ds.GetRasterBand(bi+3).ReadAsArray(), axis = 0)))

    with open(fn.replace('.tif','.json')) as data_file:
        meta = json.load(data_file)
    
    meta.update( {'units': 0} ) # 0, digital numbers; 1, radiance [W/(m^2 sr um)]; 2, top of the atmosphere reflectance
    meta.update( {'e_sun': [0,0,0]} ) # default solar irradiance. this is updated later in the code.
    im = np.transpose( im.astype('float32'), (1,2,0) )
    im = dg_rgb_im( meta=meta, im=im)

    if ms:
        if im.meta['sensor_platform_name'] in ('WORLDVIEW03_VNIR', 'WORLDVIEW02'):
            try:
                im.im = im.im[:, :, [4, 2, 1]]
                im.meta['approximate_wavelengths'] = [im.meta['approximate_wavelengths'][i] for i in [4,2,1]]
            except IndexError:
                print('Platform: {}, shape: {}'.format(im.meta['sensor_platform_name'], im.im.shape))
                raise
        elif im.meta['sensor_platform_name'] in ('QUICKBIRD02', 'GEOEYE01'):
            im.im = im.im[:, :, [2, 1, 0]]
            im.meta['approximate_wavelengths'] = [im.meta['approximate_wavelengths'][i] for i in [2,1,0]]
        else:
            raise Exception('Invalid platform name: {}'.format(im.meta['sensor_platform_name']))

    return im

def cos_im_centered(dim, freq):
    """
    This is a useful funtion for generating test data.  It is useful for 
    generating experimental imagery to verify the aliasing code is working 
    correctly. 
    
    Parameters
    ----------
    'dim' int, dimension desired square shaped dataset.
    
    'freq' int, frequency of cosine.
    
    Returns
    -------
    Returns array 
    
    """
    coordsx = np.dot( np.expand_dims(np.arange(-dim/2,dim/2), axis=1), np.ones((1,dim))) * 1/dim * 2 *np.pi * freq
    coordsy = coordsx.transpose()
    oneCycle = np.sqrt( coordsx**2 + coordsy**2 )
    return np.stack(  (np.cos(oneCycle), np.cos(oneCycle), np.cos(oneCycle))).transpose((1,2,0))


def d_earth_sun(y,m,d,H,M,S):
    """
    https://dg-cms-uploads-production.s3.amazonaws.com/uploads/document/file/207/Radiometric_Use_of_WorldView-3_v2.pdf
    the distance from the earth to the sun is dependent on the time of year. This function takes as input the 
    date and converts it to earth-sun distance.
    
    Parameters
    ----------
    'y':  year 
    'm':  month
    'd':  day
    'H':  hour
    'M':  minute
    'S':  second
    
    Returns
    -------
    'd_es': float, Returns the Earth-Sun distance in astronomical units. 
    """
    
    #print('Printing date vector: ' + str([y,m,d,H,M,S]))
    UT = H + M/60. + S/3600.
    
    if m == 1:
        y = y-1
        m = m+12
    
    A = int(y/100)
    B = 2 - A + int(A/4)
    JD = int( 365.25 * (y + 4716) ) + int( 30.6001 * (m+1)) + d + UT/24. + B - 1524.5
    D = JD-2451545.
    g = 357.529 + 0.98560028*D
    d_es = 1.00014 - 0.01671*np.cos(g*np.pi/180) - 0.00014*np.cos(2*g*np.pi/180)
    return d_es # distance value should be in astronomical units (AU)


def convert_to_toa_reflectance(im, meta):
    """
    Parameters
    ----------
    'im': array, shape: [M,N,lambda], top of the atmosphere radiance image array with image. 
    'meta': dict, with all metadata on the image.  
    
    Reference
    ---------
    https://dg-cms-uploads-production.s3.amazonaws.com/uploads/document/file/207/Radiometric_Use_of_WorldView-3_v2.pdf
    
    Returns
    -------
    'im': image array, with image data in top of the atmosphere reflectance 
    units.
    """    
    
    d_es = d_earth_sun(
    float(meta['timestamp'][0:4]), 
    float(meta['timestamp'][5:7]),
    float(meta['timestamp'][8:10]), 
    float(meta['timestamp'][11:13]),
    float(meta['timestamp'][14:16]),
    float(meta['timestamp'][17:19]))
    
    # compute reflectance
    return (im * d_es**2 * np.pi)/(meta['e_sun'] * np.cos(np.pi/180 * (90 - meta['sun_elevation_dbl'])))

    


def degradeIm(im, sat, displayImages=0, verbose=0, im_out_units=0):
    """
    This function applies a satellite image system to a image. The satellite
    degrades the imagery in a realistic manner. This tool can be used to 
    study the impact of different parameterizations of the satellite on the 
    imagery. The tool does two things.  1) Takes at aperture radiance and
    determines how many digital numbers will be generated from the data. 
    Gaussian noise is included and the effects can be observed.  
    2) Incorporates the effects of aliasing and can allow detector sampling
    above or below the optical cutoff frequency. 
    
    Constraints on this software are that there
    
    Parameters
    ----------
    'im' : object, image object where 'im.im' is the MxNx3 data array that is
    in units of radiance [W/(m^2 sr um)]. 'im.meta' contains the im.im metadata
    provided in the fMoW dataset.
    
    'sat': object, satellite object with all parameters defined.
    
    'force_no_aliasing': '0' or '1' with default '0'. Setting this to '1' you 
    can force the code to NOT include aliasing even if the sampling included 
    effects of aliasing. This is useful only when you want to study the effects
    of aliasing. 
    
    'displayImages': '0' or '1' with default '0'. Setting this to '1' will 
    cause this code to display images related to the psf, mtf, Fourier image, 
    and aliased Fourier image.
    
    'verbose': '0' or '1' with default '1'. Setting this will allow for print 
    out of many useful values.
    
    'im_out_units': '0', '1', or '2'. 0 will keep the data in units of digital 
    numbers. 1 will convert it to radiance values, this can be useful for 
    further analysis such as atmospheric compensation. 2 will convert the 
    to top of the atmosphere reflectance. 
    
    Returns
    -------
    A degraded image object that is in units of digital numbers or radiance if 
    specified.    
    """
    pixArea = sat.pixel_pitch**2
    h = 6.62606896e-34; #[J * s] Planck's constant
    c = 299792458 ; #[m/s] speed of light
    ns, nf, nb = im.im.shape
    flux_scalar = sat.delta_wvl * sat.opticTransmission * pixArea * sat.F_fill * np.pi / (1 + 4 * sat.fNum**2) # [W]
    electron_scalar = sat.QE * sat.Tint * flux_scalar * 1/(h*c/sat.wvl)  # [converts to number of electrons]

    if verbose:
        print('Image GSD: ' + str(im.meta.get('gsd')))
        print('Sensor GSD: ' + str(sat.gsd))

    wv_gnd_nyquist = .5/im.meta.get('gsd')
    if wv_gnd_nyquist < sat.gnd_nyquist:
        print('WARNING: CRASH ALERT! primary pixel pitch, altitude, or focal length should be adjusted so that the pixel pitch cut off frequency does not exceed WorldView ground sampling frequency.')
        return(dg_rgb_im(meta=im.meta, im=np.zeros((0))))
    
    if wv_gnd_nyquist < sat.gnd_optCut[0]:
        print('WARNING: CRASH ALERT! primary diameter, altitude, or shortest wavelength should be adjusted so that the optical cut off frequency does not exceed WorldView ground sampling frequency.')
        return(dg_rgb_im(meta=im.meta, im=np.zeros((0))))
    
    Xi_wv = wv_gnd_nyquist * np.expand_dims( np.arange(-1, 1+2/ns, 2/ns), axis=1 ) # define nyquist sampling of data.

    if sat.gnd_nyquist < sat.gnd_optCut[0]:  # if the highest spatial frequency passed is limited by the detector then it is detector limited and there will be aliasing.
        if verbose:
            print('Aliasing used')
        aliasing = 1 
        freq_limit = np.array([np.abs(Xi_wv + sat.gnd_optCut[0]).argmin(), np.abs(Xi_wv - sat.gnd_optCut[0]).argmin()]) # use blue band in calculation       
    else:
        if verbose:
            print('No aliasing used')
        aliasing = 0
        freq_limit = np.array([np.abs(Xi_wv + sat.gnd_nyquist).argmin(), np.abs(Xi_wv - sat.gnd_nyquist).argmin()])
    
    if sat.force_no_aliasing: 
        aliasing = 0
        if verbose:
            print('Forcing no aliasing')
    
    #print(freq_limit)
    Xi_wv = Xi_wv[freq_limit[0]:freq_limit[1]]
    #print(Xi_wv)
    
    if verbose:
        print('Frequency limited image length: ' + str(Xi_wv.shape))
        print('Frequency limits: ' + str(freq_limit))
        
    #if sat.gnd_nyquist < sat.gnd_optCut[0]:  # if the highest spatial frequency passed is limited by the detector then determine where lowest frequency is.
    #    nyq_i = np.array([np.abs(Xi_wv + sat.gnd_nyquist).argmin(), np.abs(Xi_wv - sat.gnd_nyquist).argmin()]) # nyquist optics and focalplane intersection.  Use blue band.
    #else:
    #    #nyq_i = np.array([np.abs(Xi_wv + sat.gnd_optCut[0]).argmin(), np.abs(Xi_wv - sat.gnd_optCut[0]).argmin()])   
    #    nyq_i = np.array([np.abs(Xi_wv + sat.gnd_nyquist).argmin(), np.abs(Xi_wv - sat.gnd_nyquist).argmin()])

    nyq_i = np.array([np.abs(Xi_wv + sat.gnd_nyquist).argmin(), np.abs(Xi_wv - sat.gnd_nyquist).argmin()]) # determine the sampling intersection between the ground and detector.
    
    if verbose:
        print('Nyquist intersection: ' + str(nyq_i))
    
    if aliasing:
        apertures = np.zeros((ns,ns,nb))
        psf = np.zeros((ns,ns,nb), dtype=np.complex_)
        mtf = np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb), dtype=np.complex_)
        ftim= np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb), dtype=np.complex_)
        fim = np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb), dtype=np.complex_)
                    
        ftim_alias_1 = np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb),dtype=np.complex_)
        ftim_alias_2 = np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb),dtype=np.complex_)

        ftim_alias   = np.zeros((nyq_i[1]-nyq_i[0],nyq_i[1]-nyq_i[0],nb),dtype=np.complex_)
        im_blur_orig = np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb),dtype=np.complex_)
        im_blur = np.zeros((nyq_i[1]-nyq_i[0],nyq_i[1]-nyq_i[0],nb))
    else:
        if sat.force_no_aliasing:
            apertures = np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb))
            psf = np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb), dtype=np.complex_)
            mtf = np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb), dtype=np.complex_)
            ftim= np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb), dtype=np.complex_)
            fim = np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb), dtype=np.complex_)
            im_blur = np.zeros((nyq_i[1]-nyq_i[0]+0,nyq_i[1]-nyq_i[0]+0,nb))
        else:
            #apertures = np.zeros((nyq_i[1]-nyq_i[0]+0,nyq_i[1]-nyq_i[0]+0,nb))
            #psf = np.zeros((nyq_i[1]-nyq_i[0]+0,nyq_i[1]-nyq_i[0]+0,nb), dtype=np.complex_)
            #ftim= np.zeros((nyq_i[1]-nyq_i[0]+0,nyq_i[1]-nyq_i[0]+0,nb), dtype=np.complex_)
            #fim = np.zeros((nyq_i[1]-nyq_i[0]+0,nyq_i[1]-nyq_i[0]+0,nb), dtype=np.complex_)
            #mtf = np.zeros((nyq_i[1]-nyq_i[0]+0,nyq_i[1]-nyq_i[0]+0,nb), dtype=np.complex_)
            apertures = np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb))
            psf = np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb), dtype=np.complex_)
            mtf = np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb), dtype=np.complex_)
            ftim= np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb), dtype=np.complex_)
            fim = np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb), dtype=np.complex_)
            im_blur = np.zeros((Xi_wv.shape[0],Xi_wv.shape[0],nb))
        
    nyq_opt_gnd_int = np.zeros((nb,2))

    for i in range(0,nb):
        
        if aliasing:
            nyq_opt_gnd_int[i,:] = np.array([np.abs(Xi_wv + sat.gnd_optCut[i]).argmin(), np.abs(Xi_wv - sat.gnd_optCut[i]).argmin()]) # nyquist optics and ground intersection
            #print(ns)
            #print( nyq_opt_gnd_int )
            apertures[:,:,i] = np.fft.fftshift(circle(ns, (nyq_opt_gnd_int[i,1]-nyq_opt_gnd_int[i,0])/4)) # create aperatures and fftshift them.

            psf[:,:,i] = np.abs(np.fft.fft2(apertures[:,:,i]))**2 # compute point spread function
            psf[:,:,i] = psf[:,:,i]/np.sum(psf[:,:,i]) # normalize psf

            mtf[:,:,i] = np.fft.fftshift(np.abs(np.fft.fft2(psf[:,:,i])))[freq_limit[0]:freq_limit[1],freq_limit[0]:freq_limit[1]] # crop at optical cutoff
            ftim[:,:,i] = 1/(im.im.shape[0]*im.im.shape[1]) * np.fft.fftshift(np.fft.fft2(im.im[:,:,i]))[freq_limit[0]:freq_limit[1],freq_limit[0]:freq_limit[1]] # crop at optical cutoff
#            print(nyq_i)
#            print( 'Upper bounds1: ' + str(nyq_i[1] - nyq_i[0]) )
#            print( 'Upper bounds2: ' + str( nyq_i[1] ) )
#            print( 'Lower bounds1: ' + str( nyq_i[0] ) )
#            print( 'Lower bounds2: ' + str( nyq_i[0]+nyq_i[0] ) )
#            print( 'Image Sample 1, 0 and : ' + str( nyq_i[0] ) )
#            print( 'Image Sample 2, ' + str(nyq_i[1]) + ' and ' + str( nyq_i[1]+nyq_i[0] ) )
            
            fim[:,:,i] = mtf[:,:,i] * ftim[:,:,i]
            ftim_alias_1[ nyq_i[1] - nyq_i[0] : nyq_i[1],:,i] = fim[0:nyq_i[0],:,i] # calculate aliased frequencies and fold back into aliased signal
            ftim_alias_1[ nyq_i[0] : nyq_i[0]+nyq_i[0],  :,i] = ftim_alias_1[ nyq_i[0] : nyq_i[0]+nyq_i[0],  :,i] + fim[nyq_i[1]:nyq_i[1]+nyq_i[0],:,i]
            ftim_alias_2[nyq_i[0]:nyq_i[1],nyq_i[0]:nyq_i[1],i] = ftim_alias_1[nyq_i[0]:nyq_i[1],nyq_i[0]:nyq_i[1],i]
            ftim_alias_2[:,nyq_i[1]-nyq_i[0]:nyq_i[1],i] =   ftim_alias_2[:,nyq_i[1]-nyq_i[0]:nyq_i[1],i] + fim[:,0:nyq_i[0],i] + ftim_alias_1[:,0:nyq_i[0],i]
            ftim_alias_2[:,nyq_i[0]:nyq_i[0]+nyq_i[0],i] = ftim_alias_2[:,nyq_i[0]:nyq_i[0]+nyq_i[0],i] + fim[:,nyq_i[1]:nyq_i[1]+nyq_i[0],i] + ftim_alias_1[:,nyq_i[1]:nyq_i[1]+nyq_i[0],i]
            
            ftim_alias[:,:,i] =  fim[nyq_i[0]:nyq_i[1],nyq_i[0]:nyq_i[1],i] + ftim_alias_2[nyq_i[0]:nyq_i[1],nyq_i[0]:nyq_i[1],i]
            im_blur[:,:,i] = ftim_alias.shape[0] * ftim_alias.shape[1] * np.abs(  np.fft.ifft2( ( ftim_alias[:,:,i] ) ) )
            im_blur_orig[:,:,i] = fim.shape[0] * fim.shape[1] * np.abs( np.fft.ifft2( ( fim[:,:,i] + ftim_alias_2[:,:,i] )))
        else:
            nyq_opt_gnd_int[i,:] = np.array([np.abs(Xi_wv + sat.gnd_optCut[i]).argmin(), np.abs(Xi_wv - sat.gnd_optCut[i]).argmin()]) # nyquist optics and ground intersection
            apertures[:,:,i] = np.fft.fftshift(circle(len(Xi_wv), (nyq_opt_gnd_int[i,1]-nyq_opt_gnd_int[i,0])/4))
            psf[:,:,i] = np.abs(np.fft.fft2(apertures[:,:,i]))**2
            psf[:,:,i] = psf[:,:,i]/np.sum(psf[:,:,i])
            mtf[:,:,i] = np.fft.fftshift(np.abs(np.fft.fft2(psf[:,:,i])))
            ftim[:,:,i] = 1/(im.im.shape[0]*im.im.shape[1]) * np.fft.fftshift(np.fft.fft2(im.im[:,:,i]))[freq_limit[0]:freq_limit[1],freq_limit[0]:freq_limit[1]]
            if sat.force_no_aliasing:
                im_blur[:,:,i] = (nyq_i[1]+1 - nyq_i[0])**2 * np.abs( np.fft.ifft2( np.fft.fftshift((mtf[:,:,i] * ftim[:,:,i])[nyq_i[0]:nyq_i[1], nyq_i[0]:nyq_i[1]]  )))#[nyq_opt_gnd_int[i,0]:nyq_opt_gnd_int[i,1]+1, nyq_opt_gnd_int[i,0]:nyq_opt_gnd_int[i,1]+1]
            else:
                im_blur[:,:,i] = ftim.shape[0] * ftim.shape[1] * np.abs( np.fft.ifft2( np.fft.fftshift((mtf[:,:,i] * ftim[:,:,i]) ) ) )
                #zeroFreq = (mtf[:,:,i]==mtf[:,:,i].max()).nonzero()
                #print('zero frequency: ' + str(zeroFreq[0]) + ' ' + str(zeroFreq[1]))
                #print('image mean: ' + str(im.im[:,:,i].mean()) )
                #print('maximum band mtf value: ' + str( mtf[:,:,i].max() ) )
                #print('maximum band image value: ' + str( ftim[:,:,i].max())) 
                #print('blur image mean: ' + str(im_blur[:,:,i].mean()) )
                #print( (mtf[:,:,i]==mtf[:,:,i].max()).nonzero() )
                #print( (ftim[:,:,i]==ftim[:,:,i].max()).nonzero() )
                

    #print('image mean: ' + str(im.im.mean()) )
    #print('maximum band value: ' + str( mtf.max() ) )
    #print('blur image mean: ' + str(im_blur.mean()) )
    
    if displayImages:
        plt.figure()
        plt.imshow( np.abs(mtf[:,:,0]>.00001 ) , interpolation = 'nearest')
        plt.colorbar()
        plt.title('mtf')
                
        plt.figure()
        plt.imshow( np.log(np.abs( ftim[:,:,0] )) , interpolation = 'nearest')
        plt.title('abs fourier image')
    
        if aliasing:
            plt.figure()
            plt.imshow( np.abs(fim[:,:,0] ) , interpolation = 'nearest')    
            plt.title('abs mtf*fourier image')
            
            plt.figure()
            plt.imshow( np.abs(ftim_alias_1[:,:,0] ) )
            plt.title('alias 1')
        
            plt.figure()
            plt.imshow( np.abs(ftim_alias_2[:,:,0] ) )
            plt.title('alias 2')
            
            plt.figure()
            plt.imshow( np.abs(ftim_alias[:,:,0] ) , interpolation = 'nearest')
            plt.title('alias')
            
            plt.figure()
            plt.imshow( np.abs(  np.fft.ifft2( ( ftim_alias[:,:,i] ) ) ) )
            plt.title('ifft of aliased signal')
           
    if sat.add_poissonNoise: # Poisson noise is used to simulate random photon arrival rate.  This is kind of inherent to DigitalGlobe data, so I'm not sure it is actually correct to include it here ... 
        poissonNoise = np.random.randn(im_blur.shape[0],im_blur.shape[1],im_blur.shape[2]) * np.sqrt(electron_scalar * im_blur) # expressed in electrons
    else:
        poissonNoise = 0
    
    readNoise = np.random.randn(im_blur.shape[0],im_blur.shape[1],im_blur.shape[2]) * sat.read_noise # expressed in electrons
    #noise = np.sqrt( readNoise**2 + poissonNoise**2 )
    noise = readNoise + poissonNoise # See thesis. System noise adds in quadrature. Noise per-pixel simply sums. 
    
    im_out_meta = im.meta.copy()
    im_out_meta.update({'gsd': sat.pixel_pitch * sat.alt/sat.f }) 
    im_out = np.round(1/sat.gain * (electron_scalar * im_blur + noise)) # convert to DN    
    im_out[(im_out<0).nonzero()]=0 # if any noise values cause the radiance to be less than 0 then this corrects it.
    
    # This converts the image back to radiance values 
    if im_out_units == 1:
        im_out = sat.gain/electron_scalar * im_out
        im_out_meta.update({'units': 1})

    if im_out_units == 2:
        im_out = sat.gain/electron_scalar * im_out # first convert back to radiance from digital numbers
        im_out = convert_to_toa_reflectance(im_out, im_out_meta)
        im_out_meta.update({'units': 2})
    
    if verbose:
        print('Output image dimension is: ', str(im_out.shape))
    
    return( dg_rgb_im(meta=im_out_meta, im=im_out) )
    #return( dg_rgb_im(meta=im.meta, im=np.abs( np.fft.ifft2( fim[:,:,0] + ftim_alias_2[:,:,0]))))



