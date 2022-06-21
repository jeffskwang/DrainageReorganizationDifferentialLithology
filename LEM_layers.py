#Importing the usual suspects
import numpy as np
import sys
import os
import time
np.set_printoptions(threshold=sys.maxsize)

input_filename = 'ss_topo.txt'
output_folder = 'test_vscode'

parent = os.getcwd()
input_folder = parent +'\\input_topography'


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
'''
from multiprocessing import Process

print ('parallel')

time_start = time.time()
if __name__ == "__main__":  # confirms that the code is under main function
    procs = []
    for i in range(1,3):
        proc = Process(target=run_steady_state, args=(mg,flow,erode,uplift,dt,1E-5,input_folder+'\\'+input_filename+'_'+str(i),0))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()
    print ((time.time()-time_start)/60.)


'''
print ('serial')
time_start = time.time()
run_steady_state(mg,flow,erode,uplift,dt,1E-5,input_folder+'\\'+input_filename+'_3',0)
print ((time.time()-time_start)/60.)
run_steady_state(mg,flow,erode,uplift,dt,1E-5,input_folder+'\\'+input_filename+'_4',0)
print ((time.time()-time_start)/60.)
