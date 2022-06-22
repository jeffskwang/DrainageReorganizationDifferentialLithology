#Importing the usual suspects
import numpy as np
import sys
import os
import time
start_time = time.time()
np.set_printoptions(threshold=sys.maxsize)

input_filename = 'ss_topo.txt'
output_folder = 'test'

parent = os.getcwd()
input_folder = parent +'/input_topography'
output_main = os.path.dirname(parent) +'/' + 'DRDL_output' 
if os.path.exists(output_main+'/'+output_folder)==False:
    os.makedirs(output_main+'/'+output_folder)
os.chdir(output_main+'/'+output_folder)

import warnings
warnings.simplefilter(action='ignore') #stops annoying pandas warning
from landlab import RasterModelGrid #sets up grid
from landlab.components import FlowAccumulator #finds drainage area
from landlab.components import FastscapeEroder #employs SPIM
from landlab.components import Lithology
from landlab.components import DepressionFinderAndRouter
from landlab.components import ChiFinder
from layer_functions import *

np.random.seed(12345)
multiplier = 0.2
base_Ksp = 0.00001
rows = 50
columns = 100
dx = 100. #meters
dt = 100. #time step, yrs
T = 5. * 10. ** 6. #simulation time, yrs
uplift = 0.001 #uplift rate, m/yr

mg = RasterModelGrid((rows,columns),dx) #The Grid
eta = mg.add_zeros('topographic__elevation', at = 'node')
mg.at_node['topographic__elevation'][mg.core_nodes] = np.random.rand(len(mg.core_nodes)) #resetting topography
jeff_bc_open_or_closed(mg, False, False, True, False)

#D8 is the method of finding flow direction
flow = FlowAccumulator(mg, flow_director='D8')#,depression_finder=DepressionFinderAndRouter)
#K_sp is K in the SPIM equation
erode = FastscapeEroder(mg, K_sp=base_Ksp)
cf = ChiFinder(mg,min_drainage_area=0.0,reference_concavity=.5,heterogenous=True,uplift=uplift)#,use_true_dx=True)

run_steady_state(mg,flow,erode,uplift,dt,1E-5,input_folder+'/'+input_filename)

'''
fault = mg.add_zeros('fault', at = 'node')
add_fault_lines(mg,'fault',-2.,mg.dx * 25.,False,False,T/6.*uplift)
add_fault_lines(mg,'fault',0.5,mg.dx * 25.,False,False,T/6.*uplift)
#add_fault_lines(mg,'fault',0.0,mg.dx * 20.,True,False,T/6.*uplift)
#add_fault_lines(mg,'fault',0.0,mg.dx * 20.,False,True,T/6.*uplift)
jeff_plot(mg,'fault',False,'inferno','fault','Fault.png',0)  
non_fault = np.zeros_like(fault)
non_fault[fault==0.0] = 10000.
thicknesses = [mg.at_node['topographic__elevation']+non_fault,fault,100000]
'''                      
fold = mg.add_zeros('fold', at = 'node')
fold_thickness = T/10.*uplift
#add_folds(mg,'fold',0.0,mg.extent[0]/3.,T/6.*uplift,True,False)
#jeff_plot(mg,'fold',False,'inferno','fold','Fold_horiz.png',0)       
add_folds(mg,'fold',0.0,mg.extent[1]/2.0,T/10.*uplift,False,True)
jeff_plot(mg,'fold',False,'inferno','fold','Fold_vert.png',0)        
#add_folds(mg,'fold',2.0,mg.extent[1]/5.0,T/2.*uplift,False,False)
#jeff_plot(mg,'fold',False,'inferno','fold','Fold_sloped.png',0)     
thicknesses = [mg.at_node['topographic__elevation']+fold,fold_thickness,100000]

ids = [1, 2, 1]
attrs = {'K_sp': {1: base_Ksp, 2: base_Ksp*multiplier}}

save_lithology('test',ids,attrs,thicknesses)
lith = Lithology(mg, thicknesses, ids, attrs, dz_advection = uplift * dt)

lith_elevation = np.zeros_like(lith.dz)
lith_rock_ksp = np.zeros(lith.dz.shape[0])
lith_elevation[0,:] = mg.at_node['topographic__elevation'] - lith.dz[-1,:]
for i in range(0,lith.dz.shape[0]):
    lith_rock_ksp[i] = attrs['K_sp'][ids[i]]
j = 1
for i in range(lith.dz.shape[0]-2,-1,-1):
    lith_elevation[j,:] = lith_elevation[j-1,:] - lith.dz[i,:]
    j+=1
lith.run_one_step()
flow.run_one_step()

t_set_initial_flow = 0
check_difference = 0
for t in range(0,int(T/dt)+1):   
    if t == t_set_initial_flow:
        check_difference=1
        initial_flow_direction = mg.at_node['flow__receiver_node'].copy()
    if t%(int(T/dt)/20) == 0: #progress bar
    #if t >= T/dt/3 and  t <= T/dt/2 and t%100 == 0:
        #multirock_plot(mg,attrs['K_sp'][1],t) 
        cf.calculate_chi()     
        #multirock_profiler(mg,lith,uplift,t)
        jeff_plot(mg,'channel__chi_index',False,'viridis',r'$\chi$ [m]','Chi_'+ '%06d' % t + '.png',0)    
        jeff_plot(mg,'topographic__elevation',False,'viridis',r'$\eta$ [m]','Topography_'+ '%06d' % t + '.png',0)   
        jeff_plot(mg,'drainage_area',True,'inferno',r'log(A) [log m$^2$]','Drainage_Area_'+ '%06d' % t + '.png',0)
        jeff_plot(mg,'K_sp',False,'gray',r'K [$yr^{-1}$]','Erodibility_'+ '%06d' % t + '.png',1)
        if check_difference == 1:
            reorganization_map(mg,initial_flow_direction,'viridis','0 - same, 1 - reorganized','Flow_diff_'+ '%06d' % t + '.png')
        multirock_profiler(mg,lith_elevation,lith_rock_ksp,0,t,uplift,dt)
        #jeff_text(mg,lith,'_'+ '%06d' % t)
        print (str(round(float(t)/float(T/dt)*100,1)) +'% ', end = '')
    erode = FastscapeEroder(mg, K_sp = mg.at_node['K_sp'], m_sp = 0.5, n_sp = 1.0)
    flow.run_one_step() #find drainage area
    erode.run_one_step(dt=dt) #calculate erosion rate and subtract from the topography
    mg.at_node['topographic__elevation'][mg.core_nodes] += uplift * dt #add uplift
    lith.run_one_step()

#jeff_plot(mg,'topographic__elevation',False,'viridis',r'$\eta$ [m]','Topography_'+ str(multiplier) + '.png')   
#jeff_plot(mg,'drainage_area',True,'inferno',r'log(A) [log m$^2$]','Drainage_Area_'+ str(multiplier) +'.png')
print ('done!') 
print ((time.time()-start_time)/60.)
