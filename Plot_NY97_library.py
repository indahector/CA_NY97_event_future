#!/global/homes/i/indah/.conda/envs/climate_py39/bin/python

from copy import deepcopy
import matplotlib.gridspec as gridspec
from cartopy.util import add_cyclic_point
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cmasher as cmr
from scipy.ndimage import gaussian_filter as gf
import pyinterp.backends.xarray
import pyinterp.fill
from PIL import Image
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import TwoSlopeNorm
import matplotlib
import matplotlib as mpl
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)
import cftime
import datetime
global xc,yc
global ivt_levs

#----------------------------------------------------------------------------------------------------------
def plot_z_dIVT(ds, date, lev=850):
    
    cases = ds.case.values
    time = cftime.DatetimeNoLeap(date[0], date[1], date[2], date[3], has_year_zero=True)
    d = datetime.datetime(date[0], date[1], date[2], date[3])
    date = np.datetime_as_string(np.datetime64(d)).split(":")[0]

    ds = ds.sel(time=time, lev=lev)
    dIVT  = [(ds.isel(case=0) - ds.isel(case=1)).IVT,
             (ds.isel(case=0) - ds.isel(case=2)).IVT,
             (ds.isel(case=0) - ds.isel(case=3)).IVT,
             (ds.isel(case=0) - ds.isel(case=4)).IVT,
             (ds.isel(case=0) - ds.isel(case=5)).IVT,
            ]

    Zextr = [fill_lev_nans(ds.isel(case=0).Z3),
             fill_lev_nans(ds.isel(case=1).Z3),
             fill_lev_nans(ds.isel(case=2).Z3),
             fill_lev_nans(ds.isel(case=3).Z3),
             fill_lev_nans(ds.isel(case=4).Z3),
             fill_lev_nans(ds.isel(case=5).Z3),
            ]
    
    titles = ["Control - Pre-industrial", "Control- Contemporary", "Control - +1.7$^\circ$C",
              "Control - +2.5$^\circ$C", "Control - +3.5$^\circ$C"]
    
    source_proj = ccrs.PlateCarree()
    target_proj = ccrs.PlateCarree(central_longitude=-170)

    ivt_levs = np.arange(-270,300,30)
    z_levs = np.linspace(900,1600,8)

    fig = plt.figure(figsize=(10,11.7), linewidth=1, edgecolor="black")
    spec = gridspec.GridSpec(ncols=22, nrows=5, figure=fig)

    axs00 = fig.add_subplot(spec[0, 0:8], projection=target_proj)
    axs10 = fig.add_subplot(spec[1, 0:8], projection=target_proj)
    axs20 = fig.add_subplot(spec[2, 0:8], projection=target_proj)
    axs30 = fig.add_subplot(spec[3, 0:8], projection=target_proj)
    axs40 = fig.add_subplot(spec[4, 0:8], projection=target_proj)

    axs01 = fig.add_subplot(spec[0, 8:13], projection=target_proj)
    axs11 = fig.add_subplot(spec[1, 8:13], projection=target_proj)
    axs21 = fig.add_subplot(spec[2, 8:13], projection=target_proj)
    axs31 = fig.add_subplot(spec[3, 8:13], projection=target_proj)
    axs41 = fig.add_subplot(spec[4, 8:13], projection=target_proj)

    axs02 = fig.add_subplot(spec[0, 13:22], projection=target_proj)
    axs12 = fig.add_subplot(spec[1, 13:22], projection=target_proj)
    axs22 = fig.add_subplot(spec[2, 13:22], projection=target_proj)
    axs32 = fig.add_subplot(spec[3, 13:22], projection=target_proj)
    axs42 = fig.add_subplot(spec[4, 13:22], projection=target_proj)

    axs = np.array([axs00,axs10,axs20,axs30,axs40])
    for i,ax in enumerate(axs):
        cf = ax.contourf(ds.lon, ds.lat, dIVT[i], levels=ivt_levs, \
                         transform=source_proj, cmap=bwr_cmap, extend="both")
        custimize_ax(ax=ax, proj=source_proj, ylab=titles[i], xlab=None, \
                     extent=[-126.5,-115.5,31.5,43.5], states=True, fontsize=7)
    cb0 = fig.colorbar(cf, ax=axs, orientation="vertical", aspect=50, shrink=0.90)
    cb0.ax.tick_params(labelsize=7, labelrotation=90)
    axs[0].set_title(r"IVT [kg m$^{-1}$s$^{-1}$]", loc="right", pad=0.01, \
                     fontdict={"fontsize":8, "fontweight":"normal"})

    axs = np.array([axs01,axs11,axs21,axs31,axs41])
    for i,ax in enumerate(axs):
        cf = ax.contourf(ds.lon, ds.lat, Zextr[0], levels=z_levs, \
                         transform=source_proj, cmap=mpl.cm.Greys, alpha=0.5)
        c = ax.contour  (ds.lon, ds.lat, Zextr[0], levels=z_levs, \
                         transform=source_proj, colors="black", alpha=0.8, linewidths=0.75)
        c = ax.contour  (ds.lon, ds.lat, Zextr[i+1], levels=z_levs, \
                         transform=source_proj, cmap=mpl.cm.RdYlBu, linewidths=2.5)
        custimize_ax(ax=ax, proj=source_proj, yrlab=titles[i], xlab=None, \
                     extent=[-128.5,-115.5,31.5,43.5], states=True, fontsize=8)

    axs[0].set_title(r"z [m]", loc="right", \
                     fontdict={"fontsize":8, "fontweight":"normal"})

    axs = np.array([axs02,axs12,axs22,axs32,axs42])
    for i,ax in enumerate(axs):
        cf = ax.contourf(ds.lon, ds.lat, gf(Zextr[0], sigma=2), levels=z_levs, \
                         transform=source_proj, cmap=mpl.cm.Greys, alpha=0.5)
        c = ax.contour  (ds.lon, ds.lat, gf(Zextr[0], sigma=2), levels=z_levs, \
                         transform=source_proj, colors="black", alpha=0.8, linewidths=0.75)
        c = ax.contour  (ds.lon, ds.lat, gf(Zextr[i+1], sigma=2), levels=z_levs, \
                         transform=source_proj, cmap=mpl.cm.RdYlBu, linewidths=2.5)
        custimize_ax(ax=ax, proj=source_proj, extent=[180, 250, 11.5, 63.5], states=True, fontsize=8)

    cb = fig.colorbar(cf, ax=axs, orientation="vertical", aspect=40, shrink=0.9)
    cb.add_lines(c)
    for l in cb.lines:
        l.set_linewidth(5)
    cb.ax.tick_params(labelsize=7, labelrotation=90)
    axs[0].set_title(r"z [m]", loc="right", \
                     fontdict={"fontsize":8, "fontweight":"normal"})

    return fig

#----------------------------------------------------------------------------------------------------------
def plot_ivt(ds, date, ylabels):

    source_proj = ccrs.PlateCarree()
    target_proj = ccrs.PlateCarree(central_longitude=-170)

    # ivt_levs = np.linspace(0,1200,nlevs)
    z_levs = np.linspace(900,1600,8)

    time = cftime.DatetimeNoLeap(date[0], date[1], date[2], date[3], has_year_zero=True)
    d = datetime.datetime(date[0], date[1], date[2], date[3])
    date = np.datetime_as_string(np.datetime64(d)).split(":")[0]
    
    fig,axs = plt.subplots(6,1,figsize=(3.5,9.5), subplot_kw={"projection":target_proj})
    for i,ax,res in zip(np.arange(len(ds.case)),axs,ds.case.values):

        ivt = ds.isel(case=i).sel(time=time).IVT
        z   = ds.isel(case=i).sel(time=time, lev=850).Z3

        cf = ax.contourf(ds.lon, ds.lat, ivt, levels=ivt_levs, \
                         transform=source_proj, extend="max", cmap=sat_cmap)
        
        cs = ax.contour(ds.lon, ds.lat, z,                       
                        levels=z_levs, transform=source_proj, \
                        linestyles=":", linewidths=1.5, \
                        colors="black", alpha=0.9999)
        cl = ax.clabel(cs,inline=1,fontsize=6)
        custimize_ax(ax=ax, proj=source_proj, ylab=ylabels[i], xlab=None, fontsize=10)

    fig.subplots_adjust(right=0.9)
    axs[0].set_title(date, fontsize=10, pad=4)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.68])
    cb = fig.colorbar(cf, cax=cbar_ax, orientation="vertical", )
    cb.ax.tick_params(labelsize=10)
    cb.set_label(r"IVT [kg m$^{-1}$s$^{-1}$]", weight='bold', fontsize=10)

    return fig, str(ds.time.values).split(":")[0]

#----------------------------------------------------------------------------------------------------------
def smooth_cmap(rgb, method="cubic", s=2):

    from scipy.interpolate import splev, splrep
    from scipy.interpolate import CubicSpline
    import numpy as np
    
    new = []
    
    if method=="smooth":
        for i in range(3):
            y = rgb_colours[:,i]
            x = np.arange(len(y))
            xp = np.linspace(x[0], x[-1], 266)

            spl = splrep(x, y, s=2)
            yp = splev(xp, spl)[5:-5]
            yp = np.clip(yp,0,1)
            new.append(yp)
    elif method=="cubic":
        for i in range(3):
            y = rgb_colours[:,i]
            x = np.arange(len(y))
            xp = np.linspace(x[0], x[-1], 266)

            cs = CubicSpline(x, y)
            yp = cs(xp)[5:-5]
            yp = np.clip(yp,0,1)
            new.append(yp)
    return np.vstack(new).T

#----------------------------------------------------------------------------------------------------------
img = Image.open('./BW_BWR.png')
data = img.load()

# Loop through pixels and extract rgb value
rgb_colours = []
for i in range(img.size[1]):
    rgb = [x/255 for x in data[0, i]]  # scale values 0-1
    rgb_colours.append(rgb)

rgb_colours = np.array(rgb_colours)[::-1,:4]
rgb_colours = smooth_cmap(rgb_colours, s=3, method="smooth")
bwr_cmap = mpl.colors.ListedColormap(rgb_colours, name="bw_bwr")

#----------------------------------------------------------------------------------------------------------
ivt_levs = np.linspace(0,1300,14)
xc,yc = 237.7779906860763-360, 40.059411677627395

#----------------------------------------------------------------------------------------------------------
img = Image.open('./Satellite.png')
data = img.load()

# Loop through pixels and extract rgb value
rgb_colours = []
for i in range(img.size[1]):
    rgb = [x/255 for x in data[0, i]]  # scale values 0-1
    rgb_colours.append(rgb)

rgb_colours = np.array(rgb_colours)[::-1,:4]
rgb_colours = smooth_cmap(rgb_colours, s=3, method="smooth")
sat_cmap = mpl.colors.ListedColormap(rgb_colours, name="satellite")

#----------------------------------------------------------------------------------------------------------
def custimize_ax(ax=None, proj=None, ylab=None, xlab=None, yrlab=None, fontsize=8, extent=[-200,-100,10,65],\
                 states=False):
    
    ax.text(-0.02, 0.5, ylab, va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=fontsize)
    ax.text(0.5, -0.1, xlab, va='bottom', ha='center',
            rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=fontsize)
    ax.text(1.15, 0.5, yrlab, va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=fontsize)
    ax.coastlines()
    ax.set_extent(extent, crs=proj)
    if states:
        ax.add_feature(cfeature.STATES,linewidth=0.5,edgecolor="black")
        ax.add_feature(cfeature.BORDERS,linewidth=0.5,edgecolor="black")
    
    return

#----------------------------------------------------------------------------------------------------------
def sample_sierra_jet(ds,res):

    topo = xr.open_dataset("topo_wus30x{}.nc".format(res)).sel(lon=slice(234-360, 244-360),lat=slice(34, 42))
    topo = topo.assign_coords(lon=topo.lon+360)
    topo = topo.z
    topo.values[np.where(topo<0)]=0.
    topo = topo.to_dataset()
    ds = ds.sel(lon=slice(234, 244),lat=slice(34, 42))
    
    if res>=8:
        U = pyinterp.backends.xarray.Grid3D(ds.drop('time', dim=None).U, geodetic=False)
        U = pyinterp.fill.gauss_seidel(U)[1]
        U = np.rollaxis(np.rollaxis(U, axis=1),axis=-1)
        V = pyinterp.backends.xarray.Grid3D(ds.drop('time', dim=None).V, geodetic=False)
        V = pyinterp.fill.gauss_seidel(V)[1]
        V = np.rollaxis(np.rollaxis(V, axis=1),axis=-1)
        ds.U.values = U#.swapaxes(-1,0)#.swapaxes(-1,1)
        ds.V.values = V#.swapaxes(-1,0)#.swapaxes(-1,1)
    
    dp = 1/res
    N = 40

    theta=0.6333548955340397
    x0 = xc - 3.0*np.cos(theta)
    x1 = xc + 2.0*np.cos(theta)
    y0 = yc - 3.0*np.sin(theta)
    y1 = yc + 2.0*np.sin(theta)
    dx,dy = y1-y0,x1-x0
    r = dy/dx
    NN = 1/dp*((x1-x0)**2+(y1-y0)**2)**0.5
    NN = int(np.floor(NN))
    
    # ACROSS
    x = xr.DataArray(np.linspace(x0, x1, NN)+360, dims="topo")
    y = xr.DataArray(np.linspace(y0, y1, NN), dims="topo")
    
    dx = np.diff(x.values)[0]
    dy = np.diff(y.values)[0]
    nleft = np.where(x<=xc+360)[0][-1]+1
    x0,x1 = xc-dx*nleft,xc+dx*(NN-nleft-1)
    y0,y1 = yc-dy*nleft,yc+dy*(NN-nleft-1)
    x = xr.DataArray(np.linspace(x0, x1, NN)+360, dims="topo")
    y = xr.DataArray(np.linspace(y0, y1, NN), dims="topo")
    
    topo_across = topo.sel(lat=y, lon=x, method="nearest")
    topo_across = topo_across.drop_vars(["lon","lat"])
    topo_across = topo_across.assign(xt=x, yt=y)
    topo_across =  1e5*np.exp(-1*topo_across.z/8000)
    topo_across.name = "H"
    xacross = np.linspace(x0, x1, N)+360
    yacross = np.linspace(y0, y1, N)
    lat_across = xr.DataArray(yacross, dims="points")
    lon_across = xr.DataArray(xacross, dims="points")
    across =  ds.sel(lat=lat_across, lon=lon_across, method="nearest")
    across = across.assign(lon=lon_across, lat=lat_across)
    across = across.assign(H=topo_across)
    across = across.assign(xt=x)    
    across = across.assign(yt=y)
    
    # ALONG
    x0 = xc - 4.5*np.cos(theta+np.pi/2)
    x1 = xc + 2.0*np.cos(theta+np.pi/2)
    y0 = yc - 4.5*np.sin(theta+np.pi/2)
    y1 = yc + 2.0*np.sin(theta+np.pi/2)

    x = xr.DataArray(np.linspace(x0, x1, NN)+360, dims="topo")
    y = xr.DataArray(np.linspace(y0, y1, NN), dims="topo")
        
    dx = np.diff(x.values)[0]
    dy = np.diff(y.values)[0]
    nleft = np.where(x>=xc+360)[0][-1]+1
    x0,x1 = xc-dx*nleft,xc+dx*(NN-nleft-1)
    y0,y1 = yc-dy*nleft,yc+dy*(NN-nleft-1)
    x = xr.DataArray(np.linspace(x0, x1, NN)+360, dims="topo")
    y = xr.DataArray(np.linspace(y0, y1, NN), dims="topo")
    
    topo_along = topo.sel(lat=y, lon=x, method="nearest")
    topo_along = topo_along.drop_vars(["lon","lat"])
    topo_along = topo_along.assign(xt=x, yt=y)
    topo_along  =  1e5*np.exp( -1*topo_along.z/8000)
    topo_along.name = "H"    
    
    xalong = np.linspace(x0, x1, N)+360
    yalong = np.linspace(y0, y1, N)
    lat_along = xr.DataArray(yalong, dims="points")
    lon_along = xr.DataArray(xalong, dims="points")
    along =  ds.sel(lat=lat_along, lon=lon_along, method="nearest")
    along = along.assign(lon=lon_along, lat=lat_along)
    along = along.assign(xt=x)    
    along = along.assign(yt=y)    
    along = along.assign(H=topo_along)

    along.U.values  = -1*(along.U.values*np.cos(theta) - along.V.values*np.sin(theta)                   )  
    along.V.values  =  1*(along.U.values*np.sin(theta) + along.V.values*np.cos(theta)                   ) 
    across.U.values =  1*(across.U.values*np.cos(-theta+np.pi/2) - across.V.values*np.sin(-theta+np.pi/2) )
    across.V.values = -1*(across.U.values*np.sin(-theta+np.pi/2) + across.V.values*np.cos(-theta+np.pi/2) )
    
    newlev = np.arange(600,1000,10)
    along = along.interp(lev=newlev, method="cubic")
    across = across.interp(lev=newlev, method="cubic")
    
    return along,across

#----------------------------------------------------------------------------------------------------------
def create_AA(ds, date):

    time = cftime.DatetimeNoLeap(date[0], date[1], date[2], date[3], has_year_zero=True)
    d = datetime.datetime(date[0], date[1], date[2], date[3])
    date = np.datetime_as_string(np.datetime64(d)).split(":")[0]
    ds = ds.isel(lev=slice(None,None,-1)).sel(time=time, lev=slice(600,1000))
    
    Along,Across=[],[]
    for i in range(6):
        along,across = sample_sierra_jet(ds.isel(case=i), res=32)
        Along.append(along)
        Across.append(across)
    
    return Along,Across

#----------------------------------------------------------------------------------------------------------
def plot_transect_location(Along, Across, Resolutions, time=None, max=False):
    
    if max:
        j1,j2 = 33,26
    else:
        j1,j2 = 27,24
        
    lats = Along[0].lat.values
    lats,lats=np.meshgrid(lats,lats)
    
    colors = sat_cmap(np.linspace(0,1,40))
    
    fig,ax = plt.subplots()
    CS3 = ax.contourf(lats, cmap=sat_cmap)
    plt.close(fig)
    
    xlims =           [-18,33]  
    xticks = np.arange(-15,35,10)    
    xlabels = r"u [m/s] (S-N)"
    fig,axs = plt.subplots(1,4,figsize=(8.5, 3.0))

    #===========================================================================================================================
    for i,ax in enumerate(axs):
        for j in range(len(Along[i].points)):
            ax.plot(Along[i].U.isel(points=j), Along[i].lev, color=colors[j], label=Along[i].lat.isel(points=j).values, lw=0.85)
    #===========================================================================================================================
    #Plot parameters
    for i,ax in enumerate(axs):
        ax.invert_yaxis()
        ax.tick_params(labelsize=7)
        ax.text(0, 1.03, Resolutions[i], fontdict={"weight":"normal", "fontsize":"7"}, \
                    ha="left", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabels, fontdict={"weight":"normal", "fontsize":"8"})
        ax.set_xticks(xticks)
        ax.set_xlim(xlims)
        if i==0:    
            ax.text(-0.26, 0.5, r"p [hPa]", fontdict={"weight":"normal", "fontsize":"7"}, \
                    ha="center", va="center", transform=ax.transAxes, rotation=90)
        else:
            ax.set_yticklabels([])
    cbar_ax = fig.add_axes([0.915, 0.15, 0.013, 0.7])
    cb = fig.colorbar(CS3, cax=cbar_ax)
    cb.set_label(r"u [m/s] (S-N)", fontsize=7.5)
    fig.show()
    
    return fig

#----------------------------------------------------------------------------------------------------------
def plot_profile(Along, Across, date, labels, max=False, fs=10, move=False):

    time = cftime.DatetimeNoLeap(date[0], date[1], date[2], date[3], has_year_zero=True)
    d = datetime.datetime(date[0], date[1], date[2], date[3])
    date = np.datetime_as_string(np.datetime64(d)).split(":")[0]
    
    j1 = 29
    colors = ("#010080", "#01688B", "#FFA500", "#EE3F01", "#B12122", "#B12122")
    xlims =  (         [-13,24.5],            [-3,38]  )
    xticks = (np.arange(-15,30,5), np.arange(-5,40,5))
    
    fig,axs = plt.subplots(1,2,figsize=(5.0, 3.0))
    LW = 2
    lw = 0.25
    #===========================================================================================================================
    #Along
    xlabels = [r"C-D wind [m/s]", r"A-B wind [m/s]", r"u [m/s]", r"v [m/s] (S-N)"]
    ax = axs[0]
    for i in range(6):
        ax.plot(Along[i].U.isel(points=j1), Along[i].lev, color=colors[i], lw=LW, label="{}".format(labels[i]), alpha=0.9)    
    ax.text(0.96, 1.05, '{}'.format(date), \
            transform=ax.transAxes, horizontalalignment='center', size=fs)
    ax = axs[1]
    for i in range(6):
        ax.plot(Along[i].V.isel(points=j1), Along[i].lev, color=colors[i], lw=LW, label="{}".format(labels[i]))
    #===========================================================================================================================
    #Plot parameters
    for i,ax in enumerate(axs):
        ax.invert_yaxis()
        ax.tick_params(labelsize=fs)
        ax.set_xlabel(xlabels[i], fontdict={"weight":"normal", "fontsize":fs})
        ax.set_xticks(xticks[i])
        ax.set_xticklabels(xticks[i], fontdict={"weight":"normal", "fontsize":fs})
        ax.set_xlim(xlims[i])
        if i==0:    
            ax.text(-0.22, 0.5, r"p [hPa]", fontdict={"weight":"normal", "fontsize":fs}, \
                    ha="center", va="center", transform=ax.transAxes, rotation=90)
        if i==1:
            if move:
                ax.legend(fontsize=fs-1.5, loc='upper right', framealpha=0.8, \
                          bbox_to_anchor=(0.00, 0.99), handlelength=0.75, handleheight=1.5)
            else:
                ax.legend(fontsize=fs-1.5, loc='upper right', framealpha=0.8, \
                          bbox_to_anchor=(0.14, 0.99), handlelength=0.75, handleheight=1.5)
            ax.yaxis.tick_right()
            ax.set_yticklabels([])
    fig.show()
    
    return fig

#----------------------------------------------------------------------------------------------------------
def fill_lev_nans(x):
    """
    """
    x2 = x.rename({"lon":"x", "lat":"y"}).rio.write_crs("epsg:4326", inplace=True)
    x3 = x2.rio.write_crs("epsg:4326", inplace=True)
    x4 = x3.rio.write_nodata(np.nan, inplace=True)
    x5 = x4.rio.interpolate_na().rename({"x":"lon", "y":"lat"})
    return x5

#----------------------------------------------------------------------------------------------------------
def plot_transect(along, across,res=None, time=0, lev=950):
        
    f = xr.open_dataset("data/topo_wus30x{}.nc".format(res)).sel(lon=slice(232-360, 245-360), lat=slice(32, 44))
    
    if res==4:  RES,tit = "ERA5","ERA5"
    if res==8:  RES,tit = "ne0wus30x8","RRM-E3SM(14 km)"
    if res==16: RES,tit = "ne0wus30x16","RRM-E3SM(7 km)"
    if res==32: RES,tit = "ne0wus30x32","RRM-E3SM(3.5 km)"
    
    if res==4:
        ds = xr.open_dataset("data/era5_p.nc").isel(time=time).sel(lev=lev,lon=slice(228, 255), lat=slice(28, 48), res=RES)
    else:        
        ds = xr.open_dataset("data/e3sm_p.nc").isel(time=time).sel(lev=lev,lon=slice(228, 255), lat=slice(28, 48), res=RES)
    u = ds.U.values
    v = ds.V.values
    lon = ds.lon.values
    lat = ds.lat.values
    
    source_proj = ccrs.PlateCarree()
    target_proj = ccrs.PlateCarree(central_longitude=-170)
    
    cmap = cmr.get_sub_cmap('gist_earth', 0.05, 1.00)
    cm0 = cmr.get_sub_cmap('RdYlBu_r', 0.0, 1.0)
    cm1 = cmr.get_sub_cmap('RdYlBu_r', 0.0, 1.0)

    u_levs0 = np.arange(0,30,5)
    v_levs0 = np.arange(-10,8,2)
    norm0 = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=6)

    u_levs1 = np.arange(-10,25,5)
    v_levs1 = np.arange(0,24,4)
    norm1 = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=25)
    
    fig = plt.figure(figsize=(10,2.2))
    axs = [fig.add_subplot(1,3, 1),  fig.add_subplot(1,3, 2), fig.add_subplot(1,3, 3, projection=target_proj)]
    ax0,ax1,ax2 = axs
    
    ax=ax0
    cf = ax.contourf(across.lon, across.lev, across.V, levels=v_levs0, cmap=cm0, norm=norm0, extend="both")
    ax.fill_between(across.xt, y1=np.ones_like(across.xt)*1000, y2=across.H/100, color="black", zorder=101)
    cs = ax.contour(across.lon, across.lev, -across.U, levels=u_levs0, colors="black", linewidths=0.8)
    cl = ax.clabel(cs,inline=1, fontsize=8)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(r"v [m/s] (S-N)", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    ax.plot(np.ones_like(across.lev)*xc+360, across.lev, color="white", linestyle=":", alpha=0.65)
    ax.plot(across.lon, np.ones_like(across.lon)*950, color="white", linestyle=":", alpha=0.65)
    ax.set_ylabel("p [hPa]", labelpad=1.5, fontdict={"weight":"normal", "size":7})    
    ax.text(-0.25, 0.5, tit, fontsize=8, rotation="vertical", va='center', transform=ax.transAxes)
    ax.tick_params(axis='y', rotation=90)
    ax.set_xlabel("Lon", fontdict={"weight":"normal", "size":8})
    ax.set_title("Sierra-perpendicular", fontsize=8, loc="left")
    
    ax=ax1
    cf = ax.contourf(along.lat,along.lev, along.U, norm=norm1, cmap=cm1, extend="both", levels=u_levs1)
    ax.fill_between(along.yt, y1=np.ones_like(along.yt)*1000, y2=along.H/100, color="black", zorder=101)
    cs = ax.contour(along.lat, along.lev, along.V, levels=v_levs1, colors="black", linewidths=0.8)
    cl = ax.clabel(cs, inline=1, fontsize=8)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(r"u [m/s] (S-N)", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    ax.plot(np.ones_like(along.lev)*yc, along.lev, color="white", linestyle=":", alpha=0.65)
    ax.plot(along.lat, np.ones_like(along.lat)*950, color="white", linestyle=":", alpha=0.65)
    ax.set_xlabel("Lat", fontdict={"weight":"normal", "size":8})
    ax.set_title("Sierra-parallel", fontsize=8, loc="left")
    ax.set_yticklabels([])
    ax.set_ylabel(None)
    
    for ax in (ax0,ax1):
        ax.tick_params(labelsize=7)
        ax.set_ylim(600,1000)
        ax.invert_yaxis()

    levels=np.arange(0,3750,50)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    ax=ax2
    xskip,yskip=4,4
    lon_along,lat_along,lon_across,lat_across = along.lon,along.lat,across.lon,across.lat
    LON,LAT = np.meshgrid(lon,lat)
    
    x,y = LON[::xskip,::yskip],LAT[::xskip,::yskip]
    u,v = u[::xskip,::yskip],v[::xskip,::yskip]
    
    u = fill_lev_nans(u)
    v = fill_lev_nans(v)
    
    cf = ax.pcolormesh(f.lon, f.lat, f.z, transform=source_proj, cmap=cmap, norm=norm)#, vmax=3300)#, levels=np.linspace(0,3500,10), )
    cb = fig.colorbar(cf, ax=ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label(r"Topography [m]", fontsize=8)
    ax.coastlines(edgecolor="white")
    ax.add_feature(cfeature.STATES)
    ax.set_extent([232, 244, 34, 42])
    ax.plot( lon_along,  lat_along, transform=source_proj, lw=1, color="cyan")
    ax.plot(lon_across, lat_across, transform=source_proj, lw=1, color="cyan")
    # q = ax.quiver(x, y, u, v, transform=source_proj, color="white", angles="uv", scale=250, headwidth=1000)
    q = ax.quiver(x, y, u, v, transform=source_proj, color="white", angles="uv", headwidth=6)
    qk = ax.quiverkey(q, 0.50, 1.06, 10, r'$\vec{u}_{950}$ [10 m s$^{-1}$]',
                      labelpos='W', transform=ccrs.PlateCarree(),
                      color='k', fontproperties={"size":8})
    fig.tight_layout()
    
    return fig

#----------------------------------------------------------------------------------------------------------
def plot_transect_inFig(ds, along, across, J, fig, 
                        tit=None, time=0, lev=950, fs=10, 
                        qkey=False, skip=4):
        
    f = xr.open_dataset("topo_wus30x32.nc").sel(lon=slice(232-360, 245-360), lat=slice(32, 44))
    ds = ds.sel(lev=lev, lon=slice(228, 255), lat=slice(28, 48))
    extent = [228,255,28,48]
    u = ds.U.values
    v = ds.V.values
    lon = ds.lon.values
    lat = ds.lat.values
    
    source_proj = ccrs.PlateCarree()
    target_proj = ccrs.PlateCarree(central_longitude=-170)
    
    cmap = cmr.get_sub_cmap('gist_earth', 0.05, 1.00)
    cm0 = cmr.get_sub_cmap('RdYlBu_r', 0.0, 1.0)
    cm1 = cmr.get_sub_cmap('RdYlBu_r', 0.0, 1.0)

    u_levs0 = np.arange(0,30,5)
    v_levs0 = np.arange(-10,8,2)
    norm0 = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=6)

    u_levs1 = np.arange(-10,25,5)
    v_levs1 = np.arange(0,24,4)
    norm1 = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=25)
    
    axs = [fig.add_subplot(6,3, int(3*J + 1)),  
           fig.add_subplot(6,3, int(3*J + 2)), 
           fig.add_subplot(6,3, int(3*J + 3), projection=target_proj)]
    ax0,ax1,ax2 = axs
    
    ax=ax0
    cf = ax.contourf(across.lon, across.lev, across.V, levels=v_levs0, cmap=cm0, norm=norm0, extend="both")
    ax.fill_between(across.xt, y1=np.ones_like(across.xt)*1000, y2=across.H/100, color="black", zorder=98)
    cs = ax.contour(across.lon, across.lev, -across.U, levels=u_levs0, colors="black", linewidths=0.8)
    cl = ax.clabel(cs,inline=1, fontsize=fs)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(r"C-D wind [m/s]", fontsize=fs)
    cb.ax.tick_params(labelsize=fs)
    ax.plot(np.ones_like(across.lev)*xc+360, across.lev, color="white", linestyle=":", alpha=0.75, zorder=99)
    ax.plot(across.lon, np.ones_like(across.lon)*950, color="white", linestyle=":", alpha=0.75, zorder=99)
    ax.set_ylabel("p [hPa]", labelpad=1.5, fontdict={"weight":"normal", "size":fs})    
    ax.text(-0.25, 0.5, tit, fontsize=fs, rotation="vertical", va='center', transform=ax.transAxes)
    ax.tick_params(axis='y', rotation=90)
    
    ax=ax1
    cf = ax.contourf(along.lat,along.lev, along.U, norm=norm1, cmap=cm1, extend="both", levels=u_levs1)
    ax.fill_between(along.yt, y1=np.ones_like(along.yt)*1000, y2=along.H/100, color="black", zorder=98)
    cs = ax.contour(along.lat, along.lev, along.V, levels=v_levs1, colors="black", linewidths=0.8)
    cl = ax.clabel(cs, inline=1, fontsize=fs)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(r"A-B wind [m/s]", fontsize=fs)
    cb.ax.tick_params(labelsize=fs)
    ax.plot(np.ones_like(along.lev)*yc, along.lev, color="white", linestyle=":", alpha=0.75, zorder=99)
    ax.plot(along.lat, np.ones_like(along.lat)*950, color="white", linestyle=":", alpha=0.75, zorder=99)
    ax.set_yticklabels([])
    ax.set_ylabel(None)
    
    for ax in (ax0,ax1):
        ax.tick_params(labelsize=fs)
        ax.set_ylim(600,1000)
        ax.invert_yaxis()

    levels=np.arange(0,3750,50)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    ax=ax2
    xskip,yskip=skip,skip
    lon_along,lat_along,lon_across,lat_across = along.lon,along.lat,across.lon,across.lat
    LON,LAT = np.meshgrid(lon,lat)
    
    x,y = LON[::xskip,::yskip],LAT[::xskip,::yskip]
    u,v = u[::xskip,::yskip],v[::xskip,::yskip]
    
    cf = ax.pcolormesh(f.lon, f.lat, f.z, transform=source_proj, cmap=cmap, norm=norm)#, vmax=3300)#, levels=np.linspace(0,3500,10), )
    
    cb = fig.colorbar(cf, ax=ax)
    cb.ax.tick_params(labelsize=fs)
    cb.set_label(r"Topography [m]", fontsize=fs)
    ax.coastlines(edgecolor="white")
    ax.add_feature(cfeature.STATES)
    ax.set_extent([360-126, 360-116, 33, 43])

    ax.plot( lon_along,  lat_along, transform=source_proj, lw=1.5, color="cyan")
    ax.plot(lon_across, lat_across, transform=source_proj, lw=1.5, color="cyan")
    ax.text( lon_across[0]+0.15,   lat_across[0]-0.5,  "A", transform=source_proj, size=fs, weight="bold", color="cyan")
    ax.text( lon_across[-1],  lat_across[-1], "B", transform=source_proj, size=fs, weight="bold", color="cyan")
    ax.text( lon_along[0],   lat_along[0]-0.5,  "C", transform=source_proj, size=fs, weight="bold", color="cyan")
    ax.text( lon_along[-1]-0.75,  lat_along[-1]-0.25, "D", transform=source_proj, size=fs, weight="bold", color="cyan")
    q = ax.quiver(x, y, u, v, transform=source_proj, color="white", angles="uv", scale=250,  headwidth=4.5, width=0.004, headlength=6.5)
    
    if qkey:
        qk = ax.quiverkey(q, 0.50, 1.06, 10, r'$\vec{u}_{950}$ [10 m s$^{-1}$]',
                          labelpos='W', transform=ccrs.PlateCarree(),
                          color='k', fontproperties={"size":fs-2})
    return axs

#----------------------------------------------------------------------------------------------------------
def plot_along_across_transect(ds, date=None, fs=10, figsize=(10,13.5), skip=4):
    
    time = cftime.DatetimeNoLeap(date[0], date[1], date[2], date[3], has_year_zero=True)
    d = datetime.datetime(date[0], date[1], date[2], date[3])
    date = np.datetime_as_string(np.datetime64(d)).split(":")[0]
    ds = ds.isel(lev=slice(None,None,-1)).sel(time=time, lev=slice(600,1000))
    
    along0,across0 = sample_sierra_jet(ds.isel(case=0), res=32)
    along1,across1 = sample_sierra_jet(ds.isel(case=1), res=32)
    along2,across2 = sample_sierra_jet(ds.isel(case=2), res=32)
    along3,across3 = sample_sierra_jet(ds.isel(case=3), res=32)
    along4,across4 = sample_sierra_jet(ds.isel(case=4), res=32)
    along5,across5 = sample_sierra_jet(ds.isel(case=5), res=32)
    
    fig = plt.figure(figsize=figsize)
    axs0 = plot_transect_inFig(ds.isel(case=0), along0, across0, 0, fig, tit="Control", time=time, fs=fs, qkey=True, skip=skip)
    axs1 = plot_transect_inFig(ds.isel(case=1), along1, across1, 1, fig, tit="Pre-industrial", time=time, fs=fs, skip=skip)
    axs2 = plot_transect_inFig(ds.isel(case=2), along2, across2, 2, fig, tit="Contemporary",   time=time, fs=fs, skip=skip)
    axs3 = plot_transect_inFig(ds.isel(case=3), along3, across3, 3, fig, tit="+1.7$^\circ$C",  time=time, fs=fs, skip=skip)
    axs4 = plot_transect_inFig(ds.isel(case=4), along4, across4, 4, fig, tit="+2.5$^\circ$C",  time=time, fs=fs, skip=skip)
    axs5 = plot_transect_inFig(ds.isel(case=5), along5, across5, 5, fig, tit="+3.5$^\circ$C",  time=time, fs=fs, skip=skip)
        
    return fig

#----------------------------------------------------------------------------------------------------------
def plot_along_across_transect_diff(ds, date=None, fs=10, figsize=(10,13.5)):
    
    time = cftime.DatetimeNoLeap(date[0], date[1], date[2], date[3], has_year_zero=True)
    d = datetime.datetime(date[0], date[1], date[2], date[3])
    date = np.datetime_as_string(np.datetime64(d)).split(":")[0]
    ds = ds.isel(lev=slice(None,None,-1)).sel(time=time, lev=slice(600,1000))
    
    along0,across0 = sample_sierra_jet(ds.isel(case=0), res=32)
    along1,across1 = sample_sierra_jet(ds.isel(case=1), res=32)
    along2,across2 = sample_sierra_jet(ds.isel(case=2), res=32)
    along3,across3 = sample_sierra_jet(ds.isel(case=3), res=32)
    along4,across4 = sample_sierra_jet(ds.isel(case=4), res=32)
    along5,across5 = sample_sierra_jet(ds.isel(case=5), res=32)
    
    fig = plt.figure(figsize=figsize)
    
    axs1 = plot_transect_inFig_diff(along0, across0, along1, across1, 1, fig, tit="Control - Pre-industrial", time=time, fs=fs)
    axs2 = plot_transect_inFig_diff(along0, across0, along2, across2, 2, fig, tit="Control - Contemporary",   time=time, fs=fs)
    axs3 = plot_transect_inFig_diff(along0, across0, along3, across3, 3, fig, tit="Control - +1.7$^\circ$C",  time=time, fs=fs)
    axs4 = plot_transect_inFig_diff(along0, across0, along4, across4, 4, fig, tit="Control - +2.5$^\circ$C",  time=time, fs=fs)
    axs5 = plot_transect_inFig_diff(along0, across0, along5, across5, 5, fig, tit="Control - +3.5$^\circ$C",  time=time, fs=fs)
        
    return fig

#----------------------------------------------------------------------------------------------------------
def plot_transect_inFig_diff(along0, across0, along, across, J, fig, 
                        tit=None, time=0, lev=950, fs=10):
    
    across = across0-across
    along = along0-along
    cm0 = cmr.get_sub_cmap('coolwarm', 0.0, 1.0)
    cm1 = cmr.get_sub_cmap('coolwarm', 0.0, 1.0)

    v_levs0 = np.arange(-2, 2.1, 0.5)
    u_levs0 = None
    norm0 = None

    u_levs1 = np.arange(-5,6,1)
    v_levs1 = None
    norm1 = None
    
    axs = [fig.add_subplot(6,3, int(3*J + 1)),  
           fig.add_subplot(6,3, int(3*J + 2)), 
          ]
    ax0,ax1 = axs
    
    ax=ax0
    cf = ax.contourf(across0.lon, across0.lev, across.V, levels=v_levs0, cmap=cm0, norm=norm0, extend="both")
    ax.fill_between(across0.xt, y1=np.ones_like(across0.xt)*1000, y2=across0.H/100, color="black", zorder=98)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(r"C-D wind [m/s]", fontsize=fs)
    cb.ax.tick_params(labelsize=fs)
    ax.plot(np.ones_like(across0.lev)*xc+360, across0.lev, color="white", linestyle=":", alpha=0.75, zorder=99)
    ax.plot(across0.lon, np.ones_like(across0.lon)*950, color="white", linestyle=":", alpha=0.75, zorder=99)
    ax.set_ylabel("p [hPa]", labelpad=1.5, fontdict={"weight":"normal", "size":fs})    
    ax.text(-0.25, 0.5, tit, fontsize=fs, rotation="vertical", va='center', transform=ax.transAxes)
    ax.tick_params(axis='y', rotation=90)
    
    ax=ax1
    cf = ax.contourf(along0.lat,along0.lev, along.U, norm=norm1, cmap=cm1, extend="both", levels=u_levs1)
    ax.fill_between(along0.yt, y1=np.ones_like(along0.yt)*1000, y2=along0.H/100, color="black", zorder=98)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(r"A-B wind [m/s]", fontsize=fs)
    cb.ax.tick_params(labelsize=fs)
    ax.plot(np.ones_like(along0.lev)*yc, along0.lev, color="white", linestyle=":", alpha=0.75, zorder=99)
    ax.plot(along0.lat, np.ones_like(along0.lat)*950, color="white", linestyle=":", alpha=0.75, zorder=99)
    ax.set_yticklabels([])
    ax.set_ylabel(None)
    
    for ax in (ax0,ax1):
        ax.tick_params(labelsize=fs)
        ax.set_ylim(600,1000)
        ax.invert_yaxis()

    return axs
