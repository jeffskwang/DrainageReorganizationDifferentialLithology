import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors, colorbar, cm
from matplotlib.collections import LineCollection
from landlab.components import ChannelProfiler #plot channel profile
from landlab.io.esri_ascii import read_esri_ascii
from landlab.io.esri_ascii import write_esri_ascii
import math
import numba
    
def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier

def save_lithology(lith_prefix,ids,attrs,thicknesses):
    np.save(lith_prefix+'_ids.npy',ids)
    np.save(lith_prefix+'_attrs.npy',attrs)
    np.save(lith_prefix+'_thicknesses.npy',thicknesses)
  
def run_steady_state(grid,landlab_flow,landlab_erode,uplift,dt,tol,filename,mode):
    if os.path.exists(filename) and mode == 1:
        read_esri_ascii(filename,grid = grid, name = 'steady_state_topographic__elevation')
        grid.at_node['topographic__elevation'] = grid.at_node['steady_state_topographic__elevation']
        print ('loading... ' + filename) 
    else:
        difference = 1.0
        time = 0.0
        old_eta = np.zeros_like(grid.at_node['topographic__elevation'])
        while difference > tol: 
            old_eta = grid.at_node['topographic__elevation'].copy()
            landlab_flow.run_one_step() #find drainage area
            landlab_erode.run_one_step(dt=dt) #calculate erosion rate and subtract from the topography
            grid.at_node['topographic__elevation'][grid.core_nodes] += uplift * dt #add uplift
            difference = np.average(np.absolute(old_eta - grid.at_node['topographic__elevation']))
            time += dt
        print (str(time) + ' yrs to reach steady state')
        ss_eta = grid.add_zeros('steady_state_topographic__elevation', at = 'node')
        grid.at_node['steady_state_topographic__elevation'] = grid.at_node['topographic__elevation']
        write_esri_ascii(filename, grid, names = 'steady_state_topographic__elevation')
    
def add_fault_lines(grid,field,slope,spacing,horizontal_line_boolean,vertical_line_boolean,thickness):
    if horizontal_line_boolean == True and vertical_line_boolean == True:
        sys.exit("Cannot have horizontal_line_boolean == True and vertical_line_boolean == True.")   
    elif vertical_line_boolean == True:  
        for k in range(0,int(grid.extent[1]/spacing)):
            x = spacing * (0.5+float(k))
            for j in range(0,grid.shape[0]):
                grid.at_node[field].reshape(grid.shape)[j,round_half_up(x/grid.dx)] = thickness
    elif horizontal_line_boolean == True:  
        for k in range(0,int(grid.extent[0]/spacing)):
            y = spacing * (0.5+float(k))
            for i in range(0,grid.shape[1]):
                grid.at_node[field].reshape(grid.shape)[round_half_up(y/grid.dy),i] = thickness
    else:
        spacing_y = spacing/np.cos(np.arctan(abs(slope)))
        if slope > 0.0:
            lower_index = -int(round_half_up((grid.extent[1]*slope/spacing_y)))
            upper_index = int(round_half_up((grid.extent[0]/spacing_y)) + 1)
        elif slope < 0.0:
            lower_index = 0
            upper_index = int(round_half_up((grid.extent[0]/spacing_y)) - round_half_up((grid.extent[1]*slope/spacing_y)) + 1)

        for k in range(lower_index,upper_index):   
            intercept = spacing_y * (0.5+float(k))
            if np.abs(slope) > 1.:
                for j in range(0,grid.shape[0]):
                    y = float(j) * grid.dy
                    x = (y - intercept) / slope
                    if round_half_up(x/grid.dx) >= 0.0 and round_half_up(x/grid.dx) < grid.shape[1]:
                        grid.at_node[field].reshape(grid.shape)[j,int(round_half_up(x/grid.dx))] = thickness
            else:
                for i in range(0,grid.shape[1]):
                    x = float(i) * grid.dx
                    y = slope * x + intercept
                    if round_half_up(y/grid.dy) >= 0.0 and round_half_up(0.5+y/grid.dy) < grid.shape[0]:
                        grid.at_node[field].reshape(grid.shape)[int(round_half_up(y/grid.dy)),i] = thickness
                        
def add_folds(grid,field,slope,wavelength,amplitude,horizontal_line_boolean,vertical_line_boolean):
    if horizontal_line_boolean == True and vertical_line_boolean == True:
        sys.exit("Cannot have horizontal_line_boolean == True and vertical_line_boolean == True.")   
    elif horizontal_line_boolean == True:
        for i in range(0,grid.shape[0]):
            for j in range(0,grid.shape[1]):
                y = float(i) * grid.dy
                grid.at_node[field].reshape(grid.shape)[i,j] = amplitude * (1. - np.cos(2.0 * np.pi * y / wavelength))
    elif vertical_line_boolean == True:  
        for i in range(0,grid.shape[0]):
            for j in range(0,grid.shape[1]):
                x = float(j) * grid.dx
                grid.at_node[field].reshape(grid.shape)[i,j] = amplitude * (1. - np.cos(2.0 * np.pi * x / wavelength))
    else:
        if slope == 0.0:
            sys.exit("Slope cannot be zero. Try using horizontal_line_boolean == True or vertical_line_boolean == True instead.") 
        else:
            #origin at the center
            for i in range(0,grid.shape[0]):
                for j in range(0,grid.shape[1]):
                    x = float(j) * grid.dx
                    y = float(i) * grid.dy
                    x_o = (slope ** 2.0 * grid.extent[1] / 2.0 - slope * grid.extent[0] / 2.0 + slope * y + x) / (1. + slope ** 2.)
                    y_o = slope * (x_o - grid.extent[1] / 2.0) + grid.extent[0] / 2.0
                    dist = np.sqrt((x-x_o)**2.0 + (y-y_o)**2.0)
                    grid.at_node[field].reshape(grid.shape)[i,j] = amplitude * (1. - np.cos(2.0 * np.pi * dist / wavelength))
                            
def jeff_plot(grid,field,log_boolean,cmap,zlabel,name,add_river):
    fig1 = plt.figure(1,figsize=(6,4),facecolor='white')
    ax1 = fig1.add_subplot(111)
    holder = np.flipud(grid.at_node[field].reshape(grid.shape))
    if log_boolean == True:
        holder = np.log10(holder)
    im = ax1.imshow(holder,
               cmap = cmap,
               extent=[0,grid.number_of_node_columns*grid.dx,0,grid.number_of_node_rows*grid.dx])
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    if grid.number_of_node_rows < grid.number_of_node_columns:
        cbar = fig1.colorbar(im, ax=ax1, label=zlabel,location='bottom')
    else:
        cbar = fig1.colorbar(im, ax=ax1, label=zlabel,location='right')
        
    if add_river == 1:
        chi_boolean = 1
        river_area_threshold = grid.dx * grid.dy * 10.
        num_lines = np.sum(grid.at_node['drainage_area'] >  river_area_threshold)
        x_1 = grid.x_of_node[grid.at_node['drainage_area'] >  river_area_threshold] + grid.dx * 0.5
        y_1 = grid.y_of_node[grid.at_node['drainage_area'] >  river_area_threshold] + grid.dy * 0.5
        x_2 = grid.x_of_node[grid.at_node['flow__receiver_node'][grid.at_node['drainage_area'] >  river_area_threshold]] + grid.dx * 0.5
        y_2 = grid.y_of_node[grid.at_node['flow__receiver_node'][grid.at_node['drainage_area'] >  river_area_threshold]] + grid.dy * 0.5
        lines_color = grid.at_node['channel__chi_index'][grid.at_node['drainage_area'] >  river_area_threshold]
        
        points_1 = np.array([x_1, y_1]).T.reshape(-1, 1, 2)
        points_2 = np.array([x_2, y_2]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points_1, points_2], axis=1)
        if chi_boolean == 1:
            lines = LineCollection(segments,array=lines_color,linewidths=1)
        else:
            lines = LineCollection(segments,color='r',linewidths=1)
        #ax1.autoscale()
        ax1.add_collection(lines)
    fig1.tight_layout()
    plt.savefig(name,dpi=300)   
    plt.close(fig1)
    
def jeff_text(grid,lith,name):
    np.savetxt(name+'_elevation.txt',grid.at_node['topographic__elevation'].reshape(grid.shape))
    np.savetxt(name+'_top_layer.txt',lith.dz[-1,:].reshape(grid.shape))
    np.savetxt(name+'_bottom_layer.txt',lith.dz[-2,:].reshape(grid.shape))
    
def multirock_plot(grid,top_layer,time_int):
    contrast = 0.5
    cmap = cm.terrain
    cmap.set_bad(color='sandybrown')
    
    dem = grid.at_node['topographic__elevation'].reshape(grid.shape)
        
    #get min and max value (integer values)
    ele_min,ele_max = int(np.nanmin(dem)),int(np.nanmax(dem))+1

    #make grid based on pixel locations
    X = grid.x_of_node.reshape(grid.shape)
    Y = grid.y_of_node.reshape(grid.shape)

    #initiate figure
    fig2 = plt.figure(2,figsize=(6,4),facecolor='white')
    ax2 = fig2.add_subplot(111, projection='3d')
    #fig, ax = plt.subplots(,subplot_kw={"projection": "3d"})
    # Make facecolor data
    dem_temp = dem.copy()
    dem_temp[grid.at_node['K_sp'].reshape(grid.shape)==top_layer] = np.nan
    
    norm = colors.Normalize(np.nanmin(dem),np.nanmax(dem))
    m = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
    m.set_array([])
    fcolors = m.to_rgba(dem_temp)

    # Plot the surface
    surf = ax2.plot_surface(X, Y, dem, cmap=cmap, facecolors=fcolors,rcount=grid.number_of_node_rows,ccount=grid.number_of_node_columns,shade=True)
    
    # set axes
    ax2.set_zlim(ele_min,ele_max)
    ax2.set_xlim(0,grid.number_of_node_columns*grid.dx)
    ax2.set_ylim(0,grid.number_of_node_rows*grid.dx)
    ax2.set_zticks(np.linspace(ele_min,ele_max,3))
    ax2.set_xlabel(r'x [m]',labelpad=20)
    ax2.set_ylabel(r'y [m]',labelpad=10) #add padding so the labels don't overlap the tick labels. YMMV
    ax2.set_zlabel(r'$\eta$ [m]')
    ax2.set_box_aspect((np.ptp(X), np.ptp(Y), (ele_max-ele_min)))
            
    #draw colorbar
    #cax = fig.add_axes([.2,0.075,.6,.03])
    #colorbar.ColorbarBase(cax, cmap=cmap,norm=norm,orientation='horizontal',label='K')  
    fig2.tight_layout()   
    plt.savefig('3d_plot_'+ '%06d' % time_int + '.png' ,dpi=300)
    plt.close(fig2)  
    
def jeff_bc_open_or_closed(grid, right_boolean,top_boolean,left_boolean,bottom_boolean):
    
    for i in range(0,grid.number_of_node_rows):
        if left_boolean == True:
            grid.status_at_node[i*grid.number_of_node_columns] = 1
        else:
            grid.status_at_node[i*grid.number_of_node_columns] = 4

        if right_boolean == True:
            grid.status_at_node[(1+i)*grid.number_of_node_columns-1] = 1
        else:
            grid.status_at_node[(1+i)*grid.number_of_node_columns-1] = 4
     
    for i in range(0,grid.number_of_node_columns):   
        if top_boolean == True:
            grid.status_at_node[i] = 1
        else:
            grid.status_at_node[i] = 4
        if bottom_boolean == True:
            grid.status_at_node[grid.number_of_node_rows * grid.number_of_node_columns - i - 1] = 1
        else:       
            grid.status_at_node[grid.number_of_node_rows * grid.number_of_node_columns - i - 1] = 4

def multirock_profiler(grid,lith,ksp,x_or_y,time_int,uplift,dt):#,uplift,rock_spatial_reference_boolean
    dem = grid.at_node['topographic__elevation'].reshape(grid.shape)
    status = grid.status_at_node.reshape(grid.shape)
    dem[status==4]=np.nan
    
    if x_or_y == 0: 
        distance = np.linspace(0,grid.extent[1],grid.shape[1])
        min_eta = np.zeros(grid.shape[1])
        max_eta = np.zeros(grid.shape[1])
        rock_elevation = np.zeros((grid.shape[1],lith.shape[0]))
        for i in range(0,grid.shape[1]):
            min_eta[i] = np.nanmin(dem[1:-1,i])
            max_eta[i] = np.nanmax(dem[1:-1,i])   
            for j in range(0,lith.shape[0]):
                rock_elevation[i,j] = min(max_eta[i],lith[j].reshape(grid.shape)[int(grid.shape[0]/2),i] + uplift * time_int * dt)
    elif x_or_y == 1:
        distance = np.linspace(0,grid.extent[0],grid.shape[0])
        min_eta = np.zeros(grid.shape[0])
        max_eta = np.zeros(grid.shape[0])
        rock_elevation = np.zeros((grid.shape[0],lith.shape[0]))
        for i in range(0,grid.shape[0]):
            min_eta[i] = np.nanmin(dem[i,1:-1])
            max_eta[i] = np.nanmax(dem[i,1:-1])
            for j in range(0,lith.shape[0]):
                rock_elevation[i,j] = min(max_eta[i],lith[j].reshape(grid.shape)[i,int(grid.shape[1]/2)] + uplift * time_int * dt)

    fig3 = plt.figure(3,figsize=(8,4),facecolor='white')
    ax3 = fig3.add_subplot(111)
    for i in range(lith.shape[0]-1,0,-1):
        ax3.fill_between(distance,rock_elevation[:,i],rock_elevation[:,i-1],color= cm.coolwarm((ksp[i]-min(ksp))/(max(ksp)-min(ksp))))
    ax3.fill_between(distance,rock_elevation[:,0],max_eta,color= cm.coolwarm((ksp[0]-min(ksp))/(max(ksp)-min(ksp))))
    ax3.plot(distance,min_eta,color='k')
    ax3.plot(distance,max_eta,color='k',linestyle='--')
    ax3.set_xlim(distance[0],distance[-1])
    ax3.set_xlim(distance[0],distance[-1])
    ax3.set_ylim(-500,2500)
    ax3.set_xlabel('distance [m]')
    # ax3.set_xlabel('chi-predicted elevation [m]')
    # ax3.set_ylabel('elevation [m]')
    # ax3.axis('equal')
    fig3.tight_layout()
    plt.savefig('rock_profile'+ '%06d' % time_int + '.png' ,dpi=300)
    plt.close(fig3)  

# def multirock_profiler(mg,lith,uplift,time_int):
#     n=1.0
#     fig3 = plt.figure(3,figsize=(4,4),facecolor='white')
#     ax3 = fig3.add_subplot(111)
#     profiler = ChannelProfiler(mg)
#     profiler.run_one_step()    
#     profile_data = list(profiler.data_structure.values())
#     channel_extent = list(profile_data[0].keys())
#     channel_ids = profile_data[0][channel_extent[0]]['ids']
#     channel_distances = profile_data[0][channel_extent[0]]['distances']
#     channel_elevations =  mg.at_node['topographic__elevation'][channel_ids]
#     channel_chi =  mg.at_node['channel__chi_index'][channel_ids]
#     #channel_ksp =  mg.at_node['K_sp'][channel_ids]
#     # for i in range(lith.dz.shape[0]):
#     #     channel_bedrock_thickness=lith.dz[i][channel_ids]
#     #     ax3.plot(channel_distances,channel_elevations-channel_bedrock_thickness-time_int*0.1)
#     #ax3.plot(channel_distances,channel_elevations-time_int*0.1,color='k')
#     # chi_change = channel_chi[1:] - channel_chi[:-1]
#     # fixed_chi = np.zeros_like(channel_chi)
#     # for i in range(1,len(channel_chi)):
#     #     fixed_chi[i] = fixed_chi[i-1] + chi_change[i-1] * (uplift / channel_ksp[i]) ** (1./n)
#     ax3.plot(channel_chi,channel_elevations,color='k')
#     #ax3.plot(fixed_chi,channel_elevations,color='r')
#     ax3.set_xlim(0.0,)
#     ax3.set_ylim(0.0,)
#     #ax3.set_xlabel('distance [m]')
#     ax3.set_xlabel('chi-predicted elevation [m]')
#     ax3.set_ylabel('elevation [m]')
#     ax3.axis('equal')
#     fig3.tight_layout()
#     plt.savefig('river_profile'+ '%06d' % time_int + '.png' ,dpi=300)
#     plt.close(fig3)  

def reorganization_map(grid,initial_flow,cmap,zlabel,name):
    fig4 = plt.figure(4,figsize=(6,4),facecolor='white')
    ax4 = fig4.add_subplot(111)
    final_flow = grid.at_node['flow__receiver_node']
    holder = np.zeros_like(final_flow)
    holder[final_flow!=initial_flow] = 1   
    
    im = ax4.imshow(np.flipud(holder.reshape(grid.shape)),
               cmap = cmap,
               extent=[0,grid.number_of_node_columns*grid.dx,0,grid.number_of_node_rows*grid.dx],
               vmin=0,vmax=1)
    ax4.set_xlabel('x [m]')
    ax4.set_ylabel('y [m]')
    if grid.number_of_node_rows < grid.number_of_node_columns:
        cbar = fig4.colorbar(im, ax=ax4, label=zlabel,location='bottom')
    else:
        cbar = fig4.colorbar(im, ax=ax4, label=zlabel,location='right')
        
    fig4.tight_layout()
    plt.savefig(name,dpi=300)   
    plt.close(fig4)