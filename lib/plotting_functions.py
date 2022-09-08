"""
This module contains generic
plotting helper functions that
can be used across multiple
different user-provided
plotting scripts.
"""

#Need to split-up this module, for now
#just ignore the total number of lines
#during testing:
# pylint: disable=too-many-lines

#import statements:
from typing import Optional
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
#nice formatting for tick labels
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
import geocat.comp as gcomp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#Import ADF objects:
from adf_diag import AdfDiag
from adf_base import AdfError

#Set non-X-window backend for matplotlib:
mpl.use('Agg')

#################
#HELPER FUNCTIONS
#################

def get_difference_colors(values):
    """Provide a color norm and colormap assuming this is a difference field.

       Values can be either the data field or a set of specified contour levels.

       Uses 'OrRd' colormap for positive definite, 'BuPu_r' for negative definite,
       and 'RdBu_r' centered on zero if there are values of both signs.
    """
    dmin = np.min(values)
    dmax = np.max(values)
    # color normalization for difference
    if dmin < 0 < dmax:
        dnorm = mpl.colors.TwoSlopeNorm(vmin=np.min(values),
                                        vmax=np.max(values),
                                        vcenter=0.0)
        cmap = mpl.cm.RdBu_r
    else:
        dnorm = mpl.colors.Normalize(vmin=np.min(values), vmax=np.max(values))
        if dmin >= 0:
            cmap = mpl.cm.OrRd
        elif dmax <= 0:
            cmap = mpl.cm.BuPu_r
        else:
            dnorm = mpl.colors.TwoSlopeNorm(vmin=dmin, vcenter=0, vmax=dmax)
    return dnorm, cmap


def get_central_longitude(*args):
    """Determine central longitude for maps.
       Can provide multiple arguments.
       If any of the arguments is an instance of AdfDiag,
       then check whether it has a central_longitude in `diag_basic_info`.
       --> This takes precedence.
       --> Else: if any of the arguments are scalars in [-180, 360],
                 assumes the FIRST ONE is the central longitude.
       There are no other possible conditions, so if none of those are met,
       RETURN the default value of 180.

       This allows a script to, for example, allow a config file to specify,
       but also have a preference:
       get_central_longitude(AdfObj, 30.0)
    """
    chk_for_adf = [isinstance(arg, AdfDiag) for arg in args]
    # preference is to get value from AdfDiag:
    if any(chk_for_adf):
        for arg in args:
            if isinstance(arg, AdfDiag):
                result = arg.get_basic_info('central_longitude', required=False)
                if isinstance(result, (int, float)) and (-180 <= result <= 360):
                    return result
                #End if

                #If result exists but doesn't match requirements,
                #then write info to debug log:
                if result:
                    msg = f"central_lngitude of type '{type(result).__name__}'"
                    msg += f" and value '{result}', which is not a valid longitude"
                    msg += " for the ADF."
                    arg.debug_log(msg)
                #End if

                #There is only one ADF object per ADF run, so if its
                #not present or configured correctly then no
                #reason to keep looking:
                break
            #End if
        #End for
    #End if

    # 2nd pass through arguments, look for numbers:
    for arg in args:
        if isinstance(arg, (int, float)) and (-180 <= result <= 360):
            return arg
        #End if
    #End for

    #If none of the arguments meet the criteria, do this:
    print("No valid central longitude specified. Defaults to 180.")
    return 180

#######

def global_average(fld, wgt, verbose=False):
    """
    A simple, pure numpy global average.
    fld: an input ndarray
    wgt: a 1-dimensional array of weights
    wgt should be same size as one dimension of fld
    """

    for i in range(len(fld.shape)):
        if np.size(fld, i) == len(wgt):
            axis_index = i
            break
        #End if
    #End for
    fld2 = np.ma.masked_invalid(fld)
    if verbose:
        #pylint: disable=no-member
        mask_frac = np.count_nonzero(fld2.mask) / np.size(fld2)
        #pylint: enable=no-member
        print(f"(global_average)-- fraction of mask that is True: {mask_frac}")
        print(f"(global_average)-- apply ma.average along axis = {axis_index}" + \
               f" // validate: {fld2.shape}")
    #End if
    avg1, _ = np.ma.average(fld2, axis=axis_index, weights=wgt, returned=True)

    return np.ma.average(avg1)


def wgt_rmse(fld1, fld2, wgt):
    """Calculated the area-weighted RMSE.

    Inputs are 2-d spatial fields, fld1 and fld2 with the same shape.
    They can be xarray DataArray or numpy arrays.

    Input wgt is the weight vector, expected to be 1-d,
    matching length of one dimension of the data.

    Returns a single float value.
    """
    assert len(fld1.shape) == 2,     "Input fields must have exactly two dimensions."
    assert fld1.shape == fld2.shape, "Input fields must have the same array shape."
    # in case these fields are in dask arrays, compute them now.
    if hasattr(fld1, "compute"):
        fld1 = fld1.compute()
    #End if
    if hasattr(fld2, "compute"):
        fld2 = fld2.compute()
    #End if
    if isinstance(fld1, xr.DataArray) and isinstance(fld2, xr.DataArray):
        return (np.sqrt(((fld1 - fld2)**2).weighted(wgt).mean())).values.item()
    #End if

    check = [len(wgt) == s for s in fld1.shape]
    if ~np.any(check):
        emsg = f"Sorry, weight array has shape {wgt.shape} which is not"
        emsg += f" compatible with data of shape {fld1.shape}"
        raise IOError(emsg)
    #End if

    check = [len(wgt) != s for s in fld1.shape]
    # want to get the dimension length for the dim that does not match the size of wgt:
    dimsize = fld1.shape[np.argwhere(check).item()]
    # May need more logic to ensure shape is correct:
    warray = np.tile(wgt, (dimsize, 1)).transpose()
    warray = warray / np.sum(warray) # normalize
    wmse = np.sum(warray * (fld1 - fld2)**2)
    return np.sqrt( wmse ).item()

#######

#Polar Plot funcctions

def domain_stats(data, domain):
    """Provide statistics (mean, max, and min) for a specified (sub)domain."""

    x_region = data.sel(lat=slice(domain[2],domain[3]), lon=slice(domain[0],domain[1]))
    x_region_mean = x_region.weighted(np.cos(x_region['lat'])).mean().item()
    x_region_min = x_region.min().item()
    x_region_max = x_region.max().item()
    return x_region_mean, x_region_max, x_region_min

def make_polar_plot(wks,data1:xr.DataArray, data2:xr.DataArray,
                    difference:Optional[xr.DataArray]=None,
                    domain:Optional[list]=None, hemisphere:Optional[str]=None,
                    **kwargs):
    '''
    Make a stereographic polar plot for the given data and hemisphere.
    - Uses contourf. No contour lines (yet).
    data1, data2 -> the data to be plotted. Any tranformations/operations should be done,
                    and dimensions should be [lat, lon]
    difference -> optional, the difference between the data (d2 - d1).
                  If not supplied, it will be derived as d2 - d1.
    domain -> optional, a list of [west_lon, east_lon, south_lat, north_lat] that defines
              the domain to be plotted. If not provided, defaults to all longitudes,
              45deg to pole of the given hemisphere
    hemisphere -> must be provided as NH or SH to determine which hemisphere to plot
    kwargs -> expected to be variable-dependent options for plots.
    '''
    if difference is None:
        dif = data2 - data1
    else:
        dif = difference

    if hemisphere.upper() == "NH":
        proj = ccrs.NorthPolarStereo()
    elif hemisphere.upper() == "SH":
        proj = ccrs.SouthPolarStereo()
    else:
        emsg = '[make_polar_plot] hemisphere not specified,'
        emsg += f' must be NH or SH; hemisphere set as {hemisphere}'
        raise AdfError(emsg)

    if domain is None:
        if hemisphere.upper() == "NH":
            domain = [-180, 180, 45, 90]
        else:
            domain = [-180, 180, -90, -45]

    # statistics for annotation (these are scalars):
    d1_region_mean, d1_region_max, d1_region_min = domain_stats(data1, domain)
    d2_region_mean, d2_region_max, d2_region_min = domain_stats(data2, domain)
    dif_region_mean, dif_region_max, dif_region_min = domain_stats(dif, domain)

    #downsize to the specified region; makes plotting/rendering/saving much faster
    data1 = data1.sel(lat=slice(domain[2],domain[3]))
    data2 = data2.sel(lat=slice(domain[2],domain[3]))
    dif = dif.sel(lat=slice(domain[2],domain[3]))

    # add cyclic point to the data for better-looking plot
    d1_cyclic, lon_cyclic = add_cyclic_point(data1, coord=data1.lon)
    # since we can take difference, assume same longitude coord:
    d2_cyclic, _ = add_cyclic_point(data2, coord=data2.lon)
    dif_cyclic, _ = add_cyclic_point(dif, coord=dif.lon)

    # -- deal with optional plotting arguments that might provide variable-dependent choices

    # determine levels & color normalization:
    minval    = np.min([np.min(data1), np.min(data2)])
    maxval    = np.max([np.max(data1), np.max(data2)])
    absmaxdif = np.max(np.abs(dif))

    #Get contour color map:
    cmap1 = kwargs.get("colormap", "coolwarm")

    if 'contour_levels' in kwargs:
        levels1 = kwargs['contour_levels']
        norm1 = mpl.colors.Normalize(vmin=min(levels1), vmax=max(levels1))
    elif 'contour_levels_range' in kwargs:
        assert len(kwargs['contour_levels_range']) == 3, \
               "contour_levels_range must have exactly three entries: min, max, step"
        levels1 = np.arange(*kwargs['contour_levels_range'])
        norm1 = mpl.colors.Normalize(vmin=min(levels1), vmax=max(levels1))
    else:
        levels1 = np.linspace(minval, maxval, 12)
        norm1 = mpl.colors.Normalize(vmin=minval, vmax=maxval)

    if ('colormap' not in kwargs) and ('contour_levels' not in kwargs):
        # maybe these are better defaults if nothing else is known.
        norm1, cmap1 = get_difference_colors(levels1)

    if "diff_contour_levels" in kwargs:
        levelsdiff = kwargs["diff_contour_levels"]  # a list of explicit contour levels
    elif "diff_contour_range" in kwargs:
        assert len(kwargs['diff_contour_range']) == 3, \
               "diff_contour_range must have exactly three entries: min, max, step"
        levelsdiff = np.arange(*kwargs['diff_contour_range'])
    else:
        # set levels for difference plot (with a symmetric color bar):
        levelsdiff = np.linspace(-1*absmaxdif, absmaxdif, 12)
    #End if

    #NOTE: Sometimes the contour levels chosen in the defaults file
    #can result in the "contourf" software stack generating a
    #'TypologyException', which should manifest itself as a
    #"PredicateError", but due to bugs in the stack itself
    #will also sometimes raise an AttributeError.

    #To prevent this from happening, the polar max and min values
    #are calculated, and if the default contour values are significantly
    #larger then the min-max values, then the min-max values are used instead:
    #-------------------------------
    if max(levels1) > 10*maxval:
        levels1 = np.linspace(minval, maxval, 12)
        norm1 = mpl.colors.Normalize(vmin=minval, vmax=maxval)
    elif minval < 0 and min(levels1) < 10*minval:
        levels1 = np.linspace(minval, maxval, 12)
        norm1 = mpl.colors.Normalize(vmin=minval, vmax=maxval)
    #End if

    if max(abs(levelsdiff)) > 10*absmaxdif:
        levelsdiff = np.linspace(-1*absmaxdif, absmaxdif, 12)
    #End if
    #-------------------------------

    # Difference options -- Check in kwargs for colormap and levels
    if "diff_colormap" in kwargs:
        cmapdiff = kwargs["diff_colormap"]
        dnorm, _ = get_difference_colors(levelsdiff)  # color map output ignored
    else:
        dnorm, cmapdiff = get_difference_colors(levelsdiff)
    #End if

    # -- end options

    lons, lats = np.meshgrid(lon_cyclic, data1.lat)

    fig = plt.figure(figsize=(10,10))
    grid_spec = mpl.gridspec.GridSpec(2, 4, wspace=0.9)

    ax1 = plt.subplot(grid_spec[0, :2], projection=proj)
    ax2 = plt.subplot(grid_spec[0, 2:], projection=proj)
    ax3 = plt.subplot(grid_spec[1, 1:3], projection=proj)
    ax_list = [ax1, ax2, ax3]

    empty_message = "No Valid\nData Points"
    props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.9}
    levs = np.unique(np.array(levels1))
    if len(levs) < 2:
        img1 = ax1.contourf(lons, lats, d1_cyclic, transform=ccrs.PlateCarree(),
                            colors="w", norm=norm1)
        ax1.text(0.4, 0.4, empty_message, transform=ax1.transAxes, bbox=props)

        ax2.contourf(lons, lats, d2_cyclic, transform=ccrs.PlateCarree(),
                     colors="w", norm=norm1)
        ax2.text(0.4, 0.4, empty_message, transform=ax2.transAxes, bbox=props)

        img3 = ax3.contourf(lons, lats, dif_cyclic, transform=ccrs.PlateCarree(),
                            colors="w", norm=dnorm)
        ax3.text(0.4, 0.4, empty_message, transform=ax3.transAxes, bbox=props)
    else:
        img1 = ax1.contourf(lons, lats, d1_cyclic, transform=ccrs.PlateCarree(),
                            cmap=cmap1, norm=norm1, levels=levels1)
        ax2.contourf(lons, lats, d2_cyclic, transform=ccrs.PlateCarree(),
                     cmap=cmap1, norm=norm1, levels=levels1)
        img3 = ax3.contourf(lons, lats, dif_cyclic, transform=ccrs.PlateCarree(),
                            cmap=cmapdiff, norm=dnorm, levels=levelsdiff)
    #end if

    ax1.text(-0.2, -0.10,
             f"Mean: {d1_region_mean:5.2f}\nMax: {d1_region_max:5.2f}\nMin: {d1_region_min:5.2f}",
             transform=ax1.transAxes)
    ax1.set_title(f"{data1.name} [{data1.units}]")
    ax2.text(-0.2, -0.10,
             f"Mean: {d2_region_mean:5.2f}\nMax: {d2_region_max:5.2f}\nMin: {d2_region_min:5.2f}",
             transform=ax2.transAxes)
    ax2.set_title(f"{data2.name} [{data2.units}]")
    ax3.text(-0.2, -0.10,
             f"Mean: {dif_region_mean:5.2f}\nMax: {dif_region_max:5.2f}\n"+\
             f"Min: {dif_region_min:5.2f}", transform=ax3.transAxes)
    ax3.set_title(f"Difference [{dif.units}]", loc='left')

    for sub_plot in ax_list:
        sub_plot.set_extent(domain, ccrs.PlateCarree())
        sub_plot.coastlines()
    #End for

    # __Follow the cartopy gallery example to make circular__:
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpl.path.Path(verts * radius + center)
    for sub_plot in ax_list:
        sub_plot.set_boundary(circle, transform=sub_plot.transAxes)
    #End for

    # __COLORBARS__
    cb_mean_ax = inset_axes(ax2,
                    width="5%",  # width = 5% of parent_bbox width
                    height="90%",  # height : 50%
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0.05, 1, 1),
                    bbox_transform=ax2.transAxes,
                    borderpad=0,
                    )
    fig.colorbar(img1, cax=cb_mean_ax)

    cb_diff_ax = inset_axes(ax3,
                    width="5%",  # width = 5% of parent_bbox width
                    height="90%",  # height : 50%
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0.05, 1, 1),
                    bbox_transform=ax3.transAxes,
                    borderpad=0,
                    )
    fig.colorbar(img3, cax=cb_diff_ax)

    # Save files
    fig.savefig(wks, bbox_inches='tight', dpi=300)

    # Close figures to avoid memory issues:
    plt.close(fig)

#######

def plot_map_vect_and_save(wks, plev, umdlfld, vmdlfld, uobsfld, vobsfld,
                           udiffld, vdiffld, **kwargs):
    """
    This plots a vector plot. bringing in two variables which respresent a vector pair

    kwargs -> optional dictionary of plotting options
             ** Expecting this to be a variable-specific section,
                possibly provided by an ADF Variable Defaults YAML file.**
    """

    # specify the central longitude for the plot:
    cent_long = kwargs.get('central_longitude', 180)

    # generate progjection:
    proj = ccrs.PlateCarree(central_longitude=cent_long)

    # extract lat/lon values:
    lons, lats = np.meshgrid(umdlfld['lon'], umdlfld['lat'])

    # create figure:
    fig = plt.figure(figsize=(14,10))

    # LAYOUT WITH GRIDSPEC
    grid_spec = mpl.gridspec.GridSpec(3, 6, wspace=0.5, hspace=0.05)
    grid_spec.tight_layout(fig)
    ax1 = plt.subplot(grid_spec[0:2, :3], projection=proj)
    ax2 = plt.subplot(grid_spec[0:2, 3:], projection=proj)
    ax3 = plt.subplot(grid_spec[2, 1:5], projection=proj)
    subplot_list = [ax1,ax2,ax3]

    # formatting for tick labels
    lon_formatter = LongitudeFormatter(number_format='0.0f',
                                        degree_symbol='',
                                        dateline_direction_label=False)
    lat_formatter = LatitudeFormatter(number_format='0.0f',
                                        degree_symbol='')

    # too many vectors to see well, so prune by striding through data:
    skip=(slice(None,None,5),slice(None,None,8))

    title_string = "Missing title!"
    title_string_base = title_string
    var_name = kwargs.get("var_name", "missing VAR name")

    if "case_name" in kwargs:
        case_name = kwargs["case_name"]
        if plev:
            title_string = f"{case_name} {var_name} [{plev} hPa]"
        else:
            title_string = f"{case_name} {var_name}"
        #End if
    #End if
    if "baseline" in kwargs:
        data_name = kwargs["baseline"]
        if plev:
            title_string_base = f"{data_name} {var_name} [{plev} hPa]"
        else:
            title_string_base = f"{data_name} {var_name}"
        #End if
    #End if

    # We should think about how to do plot customization and defaults.
    # Here I'll just pop off a few custom ones, and then pass the rest into mpl.
    if 'tiFontSize' in kwargs:
        title_font_size = kwargs.pop('tiFontSize')
    else:
        title_font_size = 8
    #End if

    # Calculate vector magnitudes.
    # Please note that the difference field needs
    # to be calculated from the model and obs fields
    # in order to get the correct sign:
    mdl_mag  = np.sqrt(umdlfld**2 + vmdlfld**2)
    obs_mag  = np.sqrt(uobsfld**2 + vobsfld**2)
    diff_mag = mdl_mag - obs_mag

    # Get difference limits, in order to plot the correct range:
    min_diff_val = np.min(diff_mag)
    max_diff_val = np.max(diff_mag)

    # Color normalization for difference
    if min_diff_val < 0 < max_diff_val:
        normdiff = mpl.colors.TwoSlopeNorm(vmin=min_diff_val, vmax=max_diff_val, vcenter=0.0)
    else:
        normdiff = mpl.colors.Normalize(vmin=min_diff_val, vmax=max_diff_val)
    #End if

    # Generate vector plot:
    #  - contourf to show magnitude w/ colorbar
    #  - vectors (colored or not) to show flow -->
    # subjective (?) choice for how to thin out vectors to be legible
    ax1.contourf(lons, lats, mdl_mag, cmap='Greys', transform=ccrs.PlateCarree())
    ax1.quiver(lons[skip], lats[skip], umdlfld[skip], vmdlfld[skip], mdl_mag.values[skip],
               transform=ccrs.PlateCarree(central_longitude=cent_long), cmap='Reds')
    ax1.set_title(title_string, fontsize=title_font_size)

    img2 = ax2.contourf(lons, lats, obs_mag, cmap='Greys', transform=ccrs.PlateCarree())
    ax2.quiver(lons[skip], lats[skip], uobsfld[skip], vobsfld[skip], obs_mag.values[skip],
               transform=ccrs.PlateCarree(central_longitude=cent_long), cmap='Reds')
    ax2.set_title(title_string_base, fontsize=title_font_size)

    #Set Main title for figure:
    fig_title = fig.suptitle(wks.stem[:-19].replace("_"," - "), fontsize=12)
    fig_title.set_y(0.81)

    for subplot in subplot_list:
        subplot.spines['geo'].set_linewidth(1.5) #cartopy's recommended method
        subplot.coastlines()                     # Add coastlines
        subplot.set_xticks(np.linspace(-180, 120, 6), crs=ccrs.PlateCarree())
        subplot.set_yticks(np.linspace(-90, 90, 7), crs=ccrs.PlateCarree())
        subplot.tick_params('both', length=5, width=1.5, which='major')
        subplot.tick_params('both', length=5, width=1.5, which='minor')
        subplot.xaxis.set_major_formatter(lon_formatter)
        subplot.yaxis.set_major_formatter(lat_formatter)
    #End for

    ## Add colorbar to vector plot:
    cb_c2_ax = inset_axes(ax2,
                   width="5%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0, 1, 1),
                   bbox_transform=ax2.transAxes,
                   borderpad=0,
                   )

    fig.colorbar(img2, cax=cb_c2_ax)

    # Plot vector differences:
    img3 = ax3.contourf(lons, lats, diff_mag, transform=ccrs.PlateCarree(),
                        norm=normdiff, cmap='PuOr', alpha=0.5)
    ax3.quiver(lons[skip], lats[skip], udiffld[skip], vdiffld[skip],
               transform=ccrs.PlateCarree(central_longitude=cent_long))

    ax3.set_title(f"Difference in {var_name}", loc='left', fontsize=title_font_size)
    cb_d_ax = inset_axes(ax3,
                   width="5%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0, 1, 1),
                   bbox_transform=ax3.transAxes,
                   borderpad=0
                   )
    fig.colorbar(img3, cax=cb_d_ax)

    # Write final figure to file
    fig.savefig(wks, bbox_inches='tight', dpi=300)

    #Close plots:
    plt.close()


#######

def plot_map_and_save(wks, mdlfld, obsfld, diffld, **kwargs):
    """This plots mdlfld, obsfld, diffld in a 3-row panel plot of maps.


    kwargs -> optional dictionary of plotting options
             ** Expecting this to be variable-specific section,
                possibly provided by ADF Variable Defaults YAML file.**
    - colormap -> str, name of matplotlib colormap
    - contour_levels -> list of explict values or a tuple: (min, max, step)
    - diff_colormap
    - diff_contour_levels
    - tiString -> str, Title String
    - tiFontSize -> int, Title Font Size
    - mpl -> dict, This should be any matplotlib kwargs that should be passed along. Keep reading:
        + Organize these by the mpl function. In this function (`plot_map_and_save`)
          we will check for an entry called `subplots`, `contourf`, and `colorbar`.
          So the YAML might looks something like:
          ```
           mpl:
             subplots:
               figsize: (3, 9)
             contourf:
               levels: 15
               cmap: Blues
             colorbar:
               shrink: 0.4
          ```
        + This is experimental, and if you find yourself doing much with this,
          you probably should write a new plotting script that does not rely on this module.


    When these are not provided, colormap is set to 'coolwarm' and
    limits/levels are set by data range.
    """

    # preprocess
    # - assume all three fields have same lat/lon
    lat = obsfld['lat']
    wgt = np.cos(np.radians(lat))
    mwrap, lon = add_cyclic_point(mdlfld, coord=mdlfld['lon'])
    owrap, _ = add_cyclic_point(obsfld, coord=obsfld['lon'])
    dwrap, _ = add_cyclic_point(diffld, coord=diffld['lon'])
    wrap_fields = (mwrap, owrap, dwrap)
    # mesh for plots:
    lons, lats = np.meshgrid(lon, lat)

    # get statistics (from non-wrapped)
    fields = (mdlfld, obsfld, diffld)
    area_avg = [global_average(x, wgt) for x in fields]

    d_rmse = wgt_rmse(mdlfld, obsfld, wgt)  # correct weighted RMSE for (lat,lon) fields.

    # We should think about how to do plot customization and defaults.
    # Here I'll just pop off a few custom ones, and then pass the rest into mpl.
    title_string = kwargs.get("tiString", None)

    if 'tiFontSize' in kwargs:
        title_font_size = kwargs.pop('tiFontSize')
    else:
        title_font_size = 8
    #End if

    # generate dictionary of contour plot settings:
    cp_info = prep_contour_plot(mdlfld, obsfld, diffld, **kwargs)

    # specify the central longitude for the plot
    central_longitude = kwargs.get('central_longitude', 180)

    # create figure object
    fig = plt.figure(figsize=(14,10))

    # LAYOUT WITH GRIDSPEC
    # 2 rows, 4 columns, but each map will take up 2 columns:
    grid_spec = mpl.gridspec.GridSpec(3, 6, wspace=0.5, hspace=0.05)
    grid_spec.tight_layout(fig)
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    ax1 = plt.subplot(grid_spec[0:2, :3], projection=proj, **cp_info['subplots_opt'])
    ax2 = plt.subplot(grid_spec[0:2, 3:], projection=proj, **cp_info['subplots_opt'])
    ax3 = plt.subplot(grid_spec[2, 1:5], projection=proj,  **cp_info['subplots_opt'])
    subplot_list = [ax1,ax2,ax3]

    img = [] # contour plots

    # formatting for tick labels
    lon_formatter = LongitudeFormatter(number_format='0.0f',
                                        degree_symbol='',
                                        dateline_direction_label=False)
    lat_formatter = LatitudeFormatter(number_format='0.0f',
                                        degree_symbol='')

    for i, field in enumerate(wrap_fields):

        #Set title if one hasn't been specified:
        if title_string is None:
            title_string = f"AVG: {area_avg[i]:.3f}"
        #End if

        if i == len(wrap_fields)-1:
            levels = cp_info['levelsdiff']
            cmap = cp_info['cmapdiff']
            norm = cp_info['normdiff']
        else:
            levels = cp_info['levels1']
            cmap = cp_info['cmap1']
            norm = cp_info['norm1']
        #End if

        empty_message = "No Valid\nData Points"
        props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.9}
        levs = np.unique(np.array(levels))
        if len(levs) < 2:
            img.append(subplot_list[i].contourf(lons,lats,field,colors="w",
                       transform=ccrs.PlateCarree()))
            subplot_list[i].text(0.4, 0.4, empty_message,
                                 transform=subplot_list[i].transAxes, bbox=props)
        else:
            img.append(subplot_list[i].contourf(lons, lats, field, levels=levels, cmap=cmap,
                       norm=norm, transform=ccrs.PlateCarree(), **cp_info['contourf_opt']))
        #End if
        subplot_list[i].set_title(title_string, loc='right', fontsize=title_font_size)
    #End for

    # set rmse title:
    subplot_list[-1].set_title(f"RMSE: {d_rmse:.3f}", fontsize=title_font_size)

    for subplot in subplot_list:
        subplot.spines['geo'].set_linewidth(1.5) #cartopy's recommended method
        subplot.coastlines()
        subplot.set_xticks(np.linspace(-180, 120, 6), crs=ccrs.PlateCarree())
        subplot.set_yticks(np.linspace(-90, 90, 7), crs=ccrs.PlateCarree())
        subplot.tick_params('both', length=5, width=1.5, which='major')
        subplot.tick_params('both', length=5, width=1.5, which='minor')
        subplot.xaxis.set_major_formatter(lon_formatter)
        subplot.yaxis.set_major_formatter(lat_formatter)
    #End for

    # __COLORBARS__
    cb_mean_ax = inset_axes(ax2,
                    width="5%",  # width = 5% of parent_bbox width
                    height="100%",  # height : 50%
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0, 1, 1),
                    bbox_transform=ax2.transAxes,
                    borderpad=0,
                    )
    fig.colorbar(img[1], cax=cb_mean_ax, **cp_info['colorbar_opt'])

    cb_diff_ax = inset_axes(ax3,
                    width="5%",  # width = 5% of parent_bbox width
                    height="100%",  # height : 50%
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0, 1, 1),
                    bbox_transform=ax3.transAxes,
                    borderpad=0,
                    )
    fig.colorbar(img[2], cax=cb_diff_ax, **cp_info['colorbar_opt'])

    # Write final figure to file
    fig.savefig(wks, bbox_inches='tight', dpi=300)

    #Close plots:
    plt.close()

#
#  -- vertical interpolation code --
#

def pres_from_hybrid(psfc, hya, hyb, ref_pres=100000.):
    """
    Converts a hybrid level to a pressure
    level using the forumla:

    p = a(k)*p0 + b(k)*ps

    """
    return hya*ref_pres + hyb*psfc

#####

def vert_remap(x_mdl, p_mdl, plev):
    """
    Apply simple 1-d interpolation to a field, x
    given the pressure p and the new pressures plev.
    x_mdl, p_mdl are numpy arrays of shape (nlevel, spacetime).

    Andrew G.: changed to do interpolation in log pressure
    """

    #Determine array shape of output array:
    out_shape = (plev.shape[0], x_mdl.shape[1])

    #Initialize interpolated output numpy array:
    output = np.full(out_shape, np.nan)

    #Perform 1-D interpolation in log-space:
    for i in range(out_shape[1]):
        output[:,i] = np.interp(np.log(plev), np.log(p_mdl[:,i]), x_mdl[:,i])
    #End for

    #Return interpolated output:
    return output

#####

def lev_to_plev(data, psfc, hyam, hybm, ref_pres=100000., new_levels=None,
                convert_to_mb=False):
    """
    Interpolate model hybrid levels to specified pressure levels.

    new_levels-> 1-D numpy array (ndarray) containing list of pressure levels
                 in Pascals (Pa).

    If "new_levels" is not specified, then the levels will be set
    to the GeoCAT defaults, which are (in hPa):

    1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50,
    30, 20, 10, 7, 5, 3, 2, 1

    If "convert_to_mb" is True, then vertical (lev) dimension will have
    values of mb/hPa, otherwise the units are Pa.

    The function "interp_hybrid_to_pressure" used here is dask-enabled,
    and so can potentially be sped-up via the use of a DASK cluster.
    """

    #Temporary print statement to notify users to ignore warning messages.
    #This should be replaced by a debug-log stdout filter at some point:
    print("Please ignore the interpolation warnings that follow!")

    #Apply GeoCAT hybrid->pressure interpolation:
    if new_levels is not None:
        data_interp = gcomp.interpolation.interp_hybrid_to_pressure(data, psfc,
                                                                    hyam,
                                                                    hybm,
                                                                    p0=ref_pres,
                                                                    new_levels=new_levels
                                                                   )
    else:
        data_interp = gcomp.interpolation.interp_hybrid_to_pressure(data, psfc,
                                                                    hyam,
                                                                    hybm,
                                                                    p0=ref_pres
                                                                   )

    # data_interp may contain a dask array, which can cause
    # trouble downstream with numpy functions, so call compute() here.
    if hasattr(data_interp, "compute"):
        data_interp = data_interp.compute()

    #Rename vertical dimension back to "lev" in order to work with
    #the ADF plotting functions:
    data_interp_rename = data_interp.rename({"plev": "lev"})

    #Convert vertical dimension to mb/hPa, if requested:
    if convert_to_mb:
        data_interp_rename["lev"] = data_interp_rename["lev"] / 100.0

    return data_interp_rename

#####

def pmid_to_plev(data, pmid, new_levels=None, convert_to_mb=False):
    """
    Interpolate data from hybrid-sigma levels to isobaric levels.

    data : DataArray with a 'lev' coordinate
    pmid : like data array but the pressure  of each point (Pa)
    new_levels : the output pressure levels (Pa)
    """

    # determine pressure levels to interpolate to:
    if new_levels is None:
        pnew = 100.0 * np.array([1000, 925, 850, 700, 500, 400,
                                 300, 250, 200, 150, 100, 70, 50,
                                 30, 20, 10, 7, 5, 3, 2, 1])  # mandatory levels, converted to Pa
    else:
        pnew = new_levels
    #End if

    # save name of DataArray:
    data_name = data.name

    # reshape data and pressure assuming "lev" is the name of the coordinate
    zdims = [i for i in data.dims if i != 'lev']
    dstack = data.stack(z=zdims)
    pstack = pmid.stack(z=zdims)
    output = vert_remap(dstack.values, pstack.values, pnew)
    output = xr.DataArray(output, name=data_name, dims=("lev", "z"),
                          coords={"lev":pnew, "z":pstack['z']})
    output = output.unstack()

    # convert vertical dimension to mb/hPa, if requested:
    if convert_to_mb:
        output["lev"] = output["lev"] / 100.0
    #End if

    #Return interpolated output:
    return output

#
#  -- zonal & meridional mean code --
#

def zonal_mean_xr(fld):
    """Average over all dimensions except `lev` and `lat`."""
    if isinstance(fld, xr.DataArray):
        davgovr = [dim for dim in fld.dims if dim not in ('lev','lat')]
    else:
        raise IOError("zonal_mean_xr requires Xarray DataArray input.")
    return fld.mean(dim=davgovr)


def validate_dims(fld, list_of_dims):
    """Generalized function to check if specified dimensions are in a DataArray.

    input
        fld -> DataArray with named dimensions (fld.dims)
        list_of_dims -> a list of strings that specifiy the dimensions to check for

    return
        dict with keys that are "has_{x}" where x is the name from `list_of_dims` and
        values that are boolean

    """
    if not isinstance(list_of_dims, list):
        list_of_dims = list(list_of_dims)
    return { "_".join(["has",f"{v}"]):(v in fld.dims) for v in list_of_dims}


def lat_lon_validate_dims(fld):
    """
    Check if input field has the correct
    dimensions needed to plot on lat/lon map.
    """
    # note: we can only handle variables that reduce to (lat,lon)
    if len(fld.dims) > 3:
        return False
    #End if
    validate = validate_dims(fld, ['lat','lon'])
    if not all(validate.values()):
        return  False
    #End if

    #If all checks pass, then return True:
    return True


def zm_validate_dims(fld):
    """
    Check if input field has the correct
    dimensions needed to zonally average.
    """
    # note: we can only handle variables that reduce to (lev, lat) or (lat,)
    if len(fld.dims) > 4:
        print(f"Sorry, too many dimensions: {fld.dims}")
        return None
    #End if
    validate = validate_dims(fld, ['lev','lat'])
    has_lev, has_lat = validate['has_lev'], validate['has_lat']
    if not has_lat:
        return None
    #End if

    return has_lat, has_lev


def _plot_line(plot_obj, xdata, ydata, **kwargs):
    """Create a generic line plot and check for some ways to annotate."""
    plot_obj.plot(xdata, ydata, **kwargs)

    #Set Y-axis label:
    if hasattr(ydata, "units"):
        plot_obj.set_ylabel(f"[{getattr(ydata,'units')}]")
    elif "units" in kwargs:
        plot_obj.set_ylabel(f"[{kwargs['units']}]")
    #End if

    #Set plot title:
    if hasattr(ydata, "long_name"):
        plot_obj.set_title(getattr(ydata,"long_name"), loc="left")
    elif hasattr(ydata, "name"):
        plot_obj.set_title(getattr(ydata,"name"), loc="left")
    #End if

    return plot_obj


def _meridional_plot_line(plot_obj, lon, data, **kwargs):
    """Create line plot with longitude as the X-axis."""
    plot_obj = _plot_line(plot_obj, lon, data, **kwargs)
    plot_obj.set_xlim([lon.min(), lon.max()])
    #
    # annotate
    #
    plot_obj.set_xlabel("LONGITUDE")
    return plot_obj

def _zonal_plot_line(plot_obj, lat, data, **kwargs):
    """Create line plot with latitude as the X-axis."""
    plot_obj = _plot_line(plot_obj, lat, data, **kwargs)
    plot_obj.set_xlim([max([lat.min(), -90.]), min([lat.max(), 90.])])
    #
    # annotate
    #
    plot_obj.set_xlabel("LATITUDE")
    return plot_obj

def _zonal_plot_preslat(plot_obj, lat, lev, data, **kwargs):
    """Create plot with latitude as the X-axis, and pressure as the Y-axis."""
    mlev, mlat = np.meshgrid(lev, lat)
    if 'cmap' in kwargs:
        cmap = kwargs.pop('cmap')
    else:
        cmap = 'Spectral_r'
    #End if

    img = plot_obj.contourf(mlat, mlev, data.transpose('lat', 'lev'), cmap=cmap, **kwargs)

    minor_locator = mpl.ticker.FixedLocator(lev)
    plot_obj.yaxis.set_minor_locator(minor_locator)
    plot_obj.tick_params(which='minor', length=4, color='r')
    plot_obj.set_ylim([np.max(lev), np.min(lev)])
    return img, plot_obj


def _meridional_plot_preslon(plot_obj, lon, lev, data, **kwargs):
    """Create plot with longitude as the X-axis, and pressure as the Y-axis."""

    mlev, mlon = np.meshgrid(lev, lon)
    if 'cmap' in kwargs:
        cmap = kwargs.pop('cmap')
    else:
        cmap = 'Spectral_r'

    img = plot_obj.contourf(mlon, mlev, data.transpose('lon', 'lev'), cmap=cmap, **kwargs)

    minor_locator = mpl.ticker.FixedLocator(lev)
    plot_obj.yaxis.set_minor_locator(minor_locator)
    plot_obj.tick_params(which='minor', length=4, color='r')
    plot_obj.set_ylim([np.max(lev), np.min(lev)])
    return img, plot_obj


def zonal_plot(lat, data, plot_obj=None, **kwargs):
    """
    Determine which kind of zonal plot is needed based
    on the input variable's dimensions.
    """
    if plot_obj is None:
        plot_obj = plt.gca()
    #End if

    #If there is a vertical dimension, then
    #create a latitude/height plot:
    if 'lev' in data.dims:
        img, plot_obj = _zonal_plot_preslat(plot_obj, lat, data['lev'], data, **kwargs)
        return img, plot_obj
    #End if

    #Otherwise create a latitude line plot:
    plot_obj = _zonal_plot_line(plot_obj, lat, data, **kwargs)
    return plot_obj


def meridional_plot(lon, data, plot_obj=None, **kwargs):
    """
    Determine which kind of meridional plot is needed based
    on the input variable's dimensions.
    """
    if plot_obj is None:
        plot_obj = plt.gca()
    #End if

    #If there is a vertical dimension, then
    #create a longitude/height plot:
    if 'lev' in data.dims:
        img, plot_obj = _meridional_plot_preslon(plot_obj, lon, data['lev'], data, **kwargs)
        return img, plot_obj
    #End if

    #Otherwise create a longitude line plot:
    plot_obj = _meridional_plot_line(plot_obj, lon, data, **kwargs)
    return plot_obj

def prep_contour_plot(adata, bdata, diffdata, **kwargs):
    """
    Prepares for making contour plots of adata, bdata, and diffdata, which is
    presumably the difference between adata and bdata.
    - set colormap from kwargs or defaults to coolwarm
    - set contour levels from kwargs or 12 evenly spaced levels to span the data
    - normalize colors based on specified contour levels or data range
    - set option for linear or log pressure when applicable
    - similar settings for difference, defaults to symmetric about zero
    - separates Matplotlib kwargs into their own dicts

    return
        a dict with the following:
            'subplots_opt': mpl kwargs for subplots
            'contourf_opt': mpl kwargs for contourf
            'colorbar_opt': mpl kwargs for colorbar
            'normdiff': color normalization for difference panel
            'cmapdiff': colormap for difference panel
            'levelsdiff': contour levels for difference panel
            'cmap1': color map for a and b panels
            'norm1': color normalization for a and b panels
            'levels1' : contour levels for a and b panels
            'plot_log_p' : true/false whether to plot log(pressure) axis
    """
    # determine levels & color normalization:
    minval = np.min([np.min(adata), np.min(bdata)])
    maxval = np.max([np.max(adata), np.max(bdata)])
    cmap1  = kwargs.get('colormap', 'coolwarm')

    if 'contour_levels' in kwargs:
        levels1 = kwargs['contour_levels']
        norm1 = mpl.colors.Normalize(vmin=min(levels1), vmax=max(levels1))
    elif 'contour_levels_range' in kwargs:
        assert len(kwargs['contour_levels_range']) == 3, \
        "contour_levels_range must have exactly three entries: min, max, step"

        levels1 = np.arange(*kwargs['contour_levels_range'])
        norm1 = mpl.colors.Normalize(vmin=min(levels1), vmax=max(levels1))
    else:
        levels1 = np.linspace(minval, maxval, 12)
        norm1 = mpl.colors.Normalize(vmin=minval, vmax=maxval)
    #End if

    #Check if the minval and maxval are actually different.  If not,
    #then set "levels1" to be an empty list, which will cause the
    #plotting scripts to add a label instead of trying to plot a variable
    #with no contours:
    if minval == maxval:
        levels1 = []
    #End if

    if ('colormap' not in kwargs) and ('contour_levels' not in kwargs):
        if minval < 0 < maxval:
            norm1 = mpl.colors.TwoSlopeNorm(vmin=minval, vmax=maxval, vcenter=0.0)
        else:
            norm1 = mpl.colors.Normalize(vmin=minval, vmax=maxval)
        #End if
    #End if

    # Difference options -- Check in kwargs for colormap and levels
    cmapdiff = kwargs.get('diff_colormap', 'coolwarm')

    if "diff_contour_levels" in kwargs:
        levelsdiff = kwargs["diff_contour_levels"]  # a list of explicit contour levels
    elif "diff_contour_range" in kwargs:
        assert len(kwargs['diff_contour_range']) == 3, \
        "diff_contour_range must have exactly three entries: min, max, step"

        levelsdiff = np.arange(*kwargs['diff_contour_range'])
    else:
        # set a symmetric color bar for diff:
        absmaxdif = np.max(np.abs(diffdata))
        # set levels for difference plot:
        levelsdiff = np.linspace(-1*absmaxdif, absmaxdif, 12)
    #End if

    #Detrmine if vertical coordinate is pressure or log(pressure):
    plot_log_p = kwargs.get('plot_log_pressure', False)

    # color normalization for difference
    if np.min(levelsdiff) < 0 < np.max(levelsdiff):
        normdiff = mpl.colors.TwoSlopeNorm(vmin=np.min(levelsdiff),
                                           vmax=np.max(levelsdiff),
                                           vcenter=0.0)
    else:
        normdiff = mpl.colors.Normalize(vmin=np.min(levelsdiff), vmax=np.max(levelsdiff))
    #End if

    subplots_opt = {}
    contourf_opt = {}
    colorbar_opt = {}

    # extract any MPL kwargs that should be passed on:
    if 'mpl' in kwargs:
        subplots_opt.update(kwargs['mpl'].get('subplots',{}))
        contourf_opt.update(kwargs['mpl'].get('contourf',{}))
        colorbar_opt.update(kwargs['mpl'].get('colorbar',{}))
    #End if
    return {'subplots_opt': subplots_opt,
            'contourf_opt': contourf_opt,
            'colorbar_opt': colorbar_opt,
            'normdiff': normdiff,
            'cmapdiff': cmapdiff,
            'levelsdiff': levelsdiff,
            'cmap1': cmap1,
            'norm1': norm1,
            'levels1': levels1,
            'plot_log_p': plot_log_p
            }


def plot_zonal_mean_and_save(wks, adata, bdata, has_lev, **kwargs):
    """This is the default zonal mean plot:
        adata: data to plot ([lev], lat, [lon]).
               The vertical coordinate (lev) must be pressure levels.
        bdata: baseline or observations to plot adata against.
               It must have the same dimensions and vertical levels as adata.

        - For 2-d variables (reduced to (lat,)):
          + 2 panels: (top) zonal mean, (bottom) difference
        - For 3-D variables (reduced to (lev,lat)):
          + 3 panels: (top) zonal mean adata, (middle) zonal mean bdata, (bottom) difference
          + pcolormesh/contour plot
    kwargs -> optional dictionary of plotting options
             ** Expecting this to be variable-specific section,
                possibly provided by ADF Variable Defaults YAML file.**
    - colormap -> str, name of matplotlib colormap
    - contour_levels -> list of explict values or a tuple: (min, max, step)
    - diff_colormap
    - diff_contour_levels
    - tiString -> str, Title String
    - tiFontSize -> int, Title Font Size
    - mpl -> dict, This should be any matplotlib kwargs that should be passed along. Keep reading:
        + Organize these by the mpl function. In this function (`plot_map_and_save`)
          we will check for an entry called `subplots`, `contourf`, and `colorbar`.
          So the YAML might looks something like:
          ```
           mpl:
             subplots:
               figsize: (3, 9)
             contourf:
               levels: 15
               cmap: Blues
             colorbar:
               shrink: 0.4
          ```


    """
    if has_lev:

        # calculate zonal average:
        azm = zonal_mean_xr(adata)
        bzm = zonal_mean_xr(bdata)

        # calculate difference:
        diff = azm - bzm

        # generate dictionary of contour plot settings:
        cp_info = prep_contour_plot(azm, bzm, diff, **kwargs)

        # Generate zonal plot:
        fig, subplots = plt.subplots(nrows=3, constrained_layout=True, sharex=True,
                               sharey=True,**cp_info['subplots_opt'])
        levs = np.unique(np.array(cp_info['levels1']))
        if len(levs) < 2:
            empty_message = "No Valid\nData Points"
            props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.9}
            img0, subplots[0] = zonal_plot(adata['lat'], azm, plot_obj=subplots[0])
            subplots[0].text(0.4, 0.4, empty_message, transform=subplots[0].transAxes, bbox=props)
            img1, subplots[1] = zonal_plot(bdata['lat'], bzm, plot_obj=subplots[1])
            subplots[1].text(0.4, 0.4, empty_message, transform=subplots[1].transAxes, bbox=props)
            img2, subplots[2] = zonal_plot(adata['lat'], diff, plot_obj=subplots[2])
            subplots[2].text(0.4, 0.4, empty_message, transform=subplots[2].transAxes, bbox=props)
        else:
            img0, subplots[0] = zonal_plot(adata['lat'], azm, plot_obj=subplots[0],
                                     norm=cp_info['norm1'], cmap=cp_info['cmap1'],
                                     levels=cp_info['levels1'], **cp_info['contourf_opt'])
            img1, subplots[1] = zonal_plot(bdata['lat'], bzm, plot_obj=subplots[1],
                                     norm=cp_info['norm1'], cmap=cp_info['cmap1'],
                                     levels=cp_info['levels1'], **cp_info['contourf_opt'])
            img2, subplots[2] = zonal_plot(adata['lat'], diff, plot_obj=subplots[2],
                                     norm=cp_info['normdiff'], cmap=cp_info['cmapdiff'],
                                     levels=cp_info['levelsdiff'], **cp_info['contourf_opt'])

            #Add colorbars to contour plots:
            fig.colorbar(img0, ax=subplots[0], location='right',**cp_info['colorbar_opt'])
            fig.colorbar(img1, ax=subplots[1], location='right',**cp_info['colorbar_opt'])
            fig.colorbar(img2, ax=subplots[2], location='right',**cp_info['colorbar_opt'])
        #End if

        # style the plot:
        subplots[-1].set_xlabel("LATITUDE")
        if cp_info['plot_log_p']:
            for subplot in subplots:
                subplot.set_yscale("log")
            #End for
        #End if
        fig.text(-0.03, 0.5, 'PRESSURE [hPa]', va='center', rotation='vertical')
    else:
        azm = zonal_mean_xr(adata)
        bzm = zonal_mean_xr(bdata)
        diff = azm - bzm
        fig, subplots = plt.subplots(nrows=2, constrained_layout=True)
        zonal_plot(adata['lat'], azm, plot_obj=subplots[0])
        zonal_plot(bdata['lat'], bzm, plot_obj=subplots[0])
        zonal_plot(adata['lat'], diff, plot_obj=subplots[1])
        for subplot in subplots:
            try:
                subplot.label_outer()
            except AttributeError:
                pass
            #End except
        #End for
    #End if

    #Write the figure to provided workspace/file:
    fig.savefig(wks, bbox_inches='tight', dpi=300)

    #Close plots:
    plt.close()


def plot_meridional_mean_and_save(wks, adata, bdata, has_lev, latbounds=None, **kwargs):
    """This is the default meridional mean plot:
        adata: data to plot ([lev], [lat], lon).
               The vertical coordinate (lev) must be pressure levels.
        bdata: baseline or observations to plot adata against.
               It must have the same dimensions and vertical levels as adata.

        - For 2-d variables (reduced to (lon,)):
          + 2 panels: (top) meridional mean, (bottom) difference
        - For 3-D variables (reduced to (lev,lon)):
          + 3 panels: (top) meridonal mean adata, (middle) meridional mean bdata,
                      (bottom) difference
          + pcolormesh/contour plot

        has_lev: boolean whether 'lev' is a dimension

        latbounds: the latitude bounds to average, defaults to 5S to 5N;
                   - if it is a number, assume symmetric about equator
                   - otherwise, can be a slice object. E.g., slice(-10, 20)
                   - if not a number and not a slice, print warning and skip plotting.


        kwargs -> optional dictionary of plotting options
                ** Expecting this to be variable-specific section,
                   possibly provided by ADF Variable Defaults YAML file.**
        - colormap             -> str, name of matplotlib colormap
        - contour_levels       -> list of explicit values or a tuple: (min, max, step)
        - diff_colormap        -> str, name of matplotlib colormap used for different plot
        - diff_contour_levels  -> list of explicit values or a tuple: (min, max, step)
        - tiString             -> str, Title String
        - tiFontSize           -> int, Title Font Size
        - mpl -> dict,
                 This should be any matplotlib kwargs that should be passed along. Keep reading:
            + Organize these by the mpl function. In the function `plot_meridional_mean_and_save`
            we will check for an entry called `subplots`, `contourf`, and `colorbar`.
            So the YAML might looks something like:
            ```
            mpl:
                subplots:
                figsize: (3, 9)
                contourf:
                levels: 15
                cmap: Blues
                colorbar:
                shrink: 0.4
            ```
        """
    # apply averaging:
    import numbers  # built-in; just checking on the latbounds input
    if latbounds is None:
        latbounds = slice(-5, 5)
    elif isinstance(latbounds, numbers.Number):
        latbounds = slice(-1*np.absolute(latbounds), np.absolute(latbounds))
    elif not isinstance(latbounds, slice):  #If not a slice object, then quit this routine.
        emsg = "ERROR: plot_meridonal_mean_and_save - "
        emsg += f"received an invalid value for latbounds ({latbounds})."
        emsg += " Must be a number or a slice."
        print(emsg)
        return
    #End if

    # possible that the data has time, but usually it won't
    if len(adata.dims) > 4:
        print(f"ERROR: plot_meridonal_mean_and_save - too many dimensions: {adata.dims}")
        return

    if 'time' in adata.dims:
        adata = adata.mean(dim='time', keep_attrs=True)
    if 'lat' in adata.dims:
        latweight = np.cos(np.radians(adata.lat))
        adata = adata.weighted(latweight).mean(dim='lat', keep_attrs=True)
    if 'time' in bdata.dims:
        adata = bdata.mean(dim='time', keep_attrs=True)
    if 'lat' in bdata.dims:
        latweight = np.cos(np.radians(bdata.lat))
        bdata = bdata.weighted(latweight).mean(dim='lat', keep_attrs=True)
    # If there are other dimensions, they are still going to be there:
    if len(adata.dims) > 2:
        emsg = "ERROR: plot_meridonal_mean_and_save - "
        emsg += f"AFTER averaging, there are too many dimensions: {adata.dims}"
        print(emsg)
        return

    diff = adata - bdata

    # plot-controlling parameters:
    xdim = 'lon' # the name used for the x-axis dimension
    pltfunc = meridional_plot  # the plotting function ...
                               # maybe we can generalize to get
                               # zonal/meridional into one function.

    if has_lev:
        # generate dictionary of contour plot settings:
        cp_info = prep_contour_plot(adata, bdata, diff, **kwargs)

        # generate plot objects:
        fig, subplots = plt.subplots(nrows=3, constrained_layout=True,
                               sharex=True, sharey=True,**cp_info['subplots_opt'])
        levs = np.unique(np.array(cp_info['levels1']))
        if len(levs) < 2:
            empty_message = "No Valid\nData Points"
            props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.9}
            img0, subplots[0] = pltfunc(adata[xdim], adata, plot_obj=subplots[0])
            subplots[0].text(0.4, 0.4, empty_message, transform=subplots[0].transAxes,
                             bbox=props)
            img1, subplots[1] = pltfunc(bdata[xdim], bdata, plot_obj=subplots[1])
            subplots[1].text(0.4, 0.4, empty_message, transform=subplots[1].transAxes,
                             bbox=props)
            img2, subplots[2] = pltfunc(adata[xdim], diff, plot_obj=subplots[2])
            subplots[2].text(0.4, 0.4, empty_message, transform=subplots[2].transAxes, bbox=props)
        else:
            img0, subplots[0] = pltfunc(adata[xdim], adata, plot_obj=subplots[0],
                                  norm=cp_info['norm1'], cmap=cp_info['cmap1'],
                                  levels=cp_info['levels1'], **cp_info['contourf_opt'])
            img1, subplots[1] = pltfunc(bdata[xdim], bdata, plot_obj=subplots[1],
                                  norm=cp_info['norm1'], cmap=cp_info['cmap1'],
                                  levels=cp_info['levels1'], **cp_info['contourf_opt'])
            img2, subplots[2] = pltfunc(adata[xdim], diff, plot_obj=subplots[2],
                                  norm=cp_info['normdiff'], cmap=cp_info['cmapdiff'],
                                  levels=cp_info['levelsdiff'], **cp_info['contourf_opt'])

            #Add colorabar to figure object:
            fig.colorbar(img0, plot_obj=subplots[0], location='right',**cp_info['colorbar_opt'])
            fig.colorbar(img1, plot_obj=subplots[1], location='right',**cp_info['colorbar_opt'])
            fig.colorbar(img2, plot_obj=subplots[2], location='right',**cp_info['colorbar_opt'])
        #End if

        # style the plot:
        subplots[-1].set_xlabel("LONGITUDE")
        if cp_info['plot_log_p']:
            for subplot in subplots:
                subplot.set_yscale("log")
            #End for
        #End if

        fig.text(-0.03, 0.5, 'PRESSURE [hPa]', va='center', rotation='vertical')

    else:
        fig, subplots = plt.subplots(nrows=2, constrained_layout=True)
        pltfunc(adata[xdim], adata, plot_obj=subplots[0])
        pltfunc(bdata[xdim], bdata, plot_obj=subplots[0])
        pltfunc(adata[xdim], diff, plot_obj=subplots[1])
        for subplot in subplots:
            try:
                subplot.label_outer()
            except AttributeError:
                pass
            #End except
        #End for
    #End if

    #Write the figure to provided workspace/file:
    fig.savefig(wks, bbox_inches='tight', dpi=300)

    #Close plots:
    plt.close()

#
#  -- zonal mean annual cycle --
#

def square_contour_difference(fld1, fld2, **kwargs):
    """Produce a figure with square axes that show fld1, fld2,
       and their difference as filled contours.

       Intended use is latitude-by-month to show the annual cycle.
       Example use case: use climo files to get data, take zonal averages,
       rename "time" to "month" if desired,
       and then provide resulting DataArrays to this function.

       Input:
           fld1 and fld2 are 2-dimensional DataArrays with same shape
           kwargs are optional keyword arguments
               this function checks kwargs for `case1name`, `case2name`

       Returns:
           figure object

       Assumptions:
           fld1.shape == fld2.shape
           len(fld1.shape) == 2

       Annnotation:
           Will try to label the cases by looking for
           case1name and case2name in kwargs,
           and then fld1['case'] & fld2['case'] (i.e., attributes)
           If no case name, will proceed with empty strings.
           ** IF THERE IS A BETTER CONVENTION WE SHOULD USE IT.

           Each panel also puts the Min/Max values into the title string.

           Axis labels are upper-cased names of the coordinates of fld1.
           Ticks are automatic with the exception that if the
           first coordinate is "month" and is length 12, use np.arange(1,13).

    """
    # 2-Dimension => plot contourf
    assert len(fld1.shape) == 2,     "Input fields must have exactly two dimensions."
    # Same shape => allows difference
    assert fld1.shape == fld2.shape, "Input fields must have the same array shape."


    if "case1name" in kwargs:
        case1name = kwargs.pop("case1name")
    elif hasattr(fld1, "case"):
        case1name = getattr(fld1, "case")
    else:
        case1name = ""
    #End if

    if "case2name" in kwargs:
        case2name = kwargs.pop("case2name")
    elif hasattr(fld2, "case"):
        case2name = getattr(fld2, "case")
    else:
        case2name = ""
    #End if

    # Geometry of the figure is hard-coded
    fig = plt.figure(figsize=(10,10))

    rows = 5
    columns = 5
    grid = mpl.gridspec.GridSpec(rows, columns, wspace=1, hspace=1,
                            width_ratios=[1,1,1,1,0.2],
                            height_ratios=[1,1,1,1,0.2])
    # plt.subplots_adjust(wspace= 0.01, hspace= 0.01)
    ax1 = plt.subplot(grid[0:2, 0:2])
    ax2 = plt.subplot(grid[0:2, 2:4])
    ax3 = plt.subplot(grid[2:4, 1:3])
    # color bars / means share top bar.
    cbax_top = plt.subplot(grid[0:2, -1])
    cbax_bot = plt.subplot(grid[-1, 1:3])

    # determine color normalization for means:
    max_val = np.max([fld1.max(), fld2.max()])
    min_val = np.min([fld1.min(), fld2.min()])
    mpl.colors.Normalize(min_val, max_val)

    coord1, coord2 = fld1.coords  # ASSUMES xarray WITH coords AND 2-dimensions
    print(f"{coord1}, {coord2}")
    xmesh, ymesh = np.meshgrid(fld1[coord2], fld1[coord1])
    print(f"shape of meshgrid: {xmesh.shape}")

    img1 = ax1.contourf(xmesh, ymesh, fld1.transpose())
    if (coord1 == 'month') and (fld1.shape[0] ==12):
        ax1.set_xticks(np.arange(1,13))
    ax1.set_ylabel(coord2.upper())
    ax1.set_xlabel(coord1.upper())
    ax1.set_title(f"{case1name}\nMIN:{fld1.min().values:.2f}  MAX:{fld1.max().values:.2f}")

    ax2.contourf(xmesh, ymesh, fld2.transpose())
    if (coord1 == 'month') and (fld1.shape[0] ==12):
        ax2.set_xticks(np.arange(1,13))
    ax2.set_xlabel(coord1.upper())
    ax2.set_title(f"{case2name}\nMIN:{fld2.min().values:.2f}  MAX:{fld2.max().values:.2f}")


    diff = fld1 - fld2
    ## USE A DIVERGING COLORMAP CENTERED AT ZERO
    ## Special case is when min > 0 or max < 0
    dmin = diff.min()
    dmax = diff.max()
    if dmin > 0:
        dnorm = mpl.colors.Normalize(dmin, dmax)
        cmap = mpl.cm.OrRd
    elif dmax < 0:
        dnorm = mpl.colors.Normalize(dmin, dmax)
        cmap = mpl.cm.BuPu_r
    else:
        dnorm = mpl.colors.TwoSlopeNorm(vmin=dmin, vcenter=0, vmax=dmax)
        cmap = mpl.cm.RdBu_r
    #End if

    img3 = ax3.contourf(xmesh, ymesh, diff.transpose(), cmap=cmap, norm=dnorm)
    if (coord1 == 'month') and (fld1.shape[0] ==12):
        ax3.set_xticks(np.arange(1,13))
    ax3.set_ylabel(coord2.upper())
    ax3.set_xlabel(coord1.upper())
    ax3.set_title(f"DIFFERENCE (= a - b)\nMIN:{diff.min().values:.2f}  MAX:{diff.max().values:.2f}")


    # Try to construct the title:
    if hasattr(fld1, "long_name"):
        tstr = getattr(fld1, "long_name")
    elif hasattr(fld2, "long_name"):
        tstr = getattr(fld2, "long_name")
    elif hasattr(fld1, "short_name"):
        tstr = getattr(fld1, "short_name")
    elif hasattr(fld2, "short_name"):
        tstr = getattr(fld2, "short_name")
    elif hasattr(fld1, "name"):
        tstr = getattr(fld1, "name")
    elif hasattr(fld2, "name"):
        tstr = getattr(fld2, "name")
    else:
        tstr = ""
    #End if

    if hasattr(fld1, "units"):
        tstr = tstr + f" [{getattr(fld1, 'units')}]"
    elif hasattr(fld2, "units"):
        tstr = tstr + f" [{getattr(fld2, 'units')}]"
    else:
        tstr = tstr + "[-]"
    #End if

    #Add title to figure:
    fig.suptitle(tstr, fontsize=18)

    #Add colorabar to figure object:
    fig.colorbar(img1, cax=cbax_top)
    fig.colorbar(img3, cax=cbax_bot, orientation='horizontal')

    #Return figure object:
    return fig

#####################
#END HELPER FUNCTIONS
