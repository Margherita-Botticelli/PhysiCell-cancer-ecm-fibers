import xml.etree.ElementTree as ET
import os, sys
from pathlib import Path
import numpy as np
import subprocess
from itertools import product

#### Script to run multiple simulations automatically by giving the parameter values

#### Project name corresponding to folder in user_projects
proj = 'ecm_fibers' # 'test' # 

#### True if we want to repeat an existing simulation with different random seeds or ribose concentrations 
repeat_simulations = False

if repeat_simulations:
    seeds = [0,1,2,3,4,5,6,7,8,9]
    riboses = [0]
    simulations = [0]

    return_code = os.system(f'make reset; make clean; make load PROJ={proj}')
    if return_code != 0:
        print(f'Failed with exit code: {return_code}')
        sys.exit()
    # sys.exit()

else:
    
    #### Simulations ID number: write number manually or write negative number to find last simulation number in the data folder 
    simulation_id = 120 # 114 # 108 # 0 # 

    #### Define random seeds and ribose concentrations
    random_seed_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  #[0] #, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
 
    # ribose_concentration_values = [0] # [0,50,200] # 
    ecm_orientation_setup_values = ['random'] # ['random','radial','tangential'] # ['split'] # 

    #### Define parameter values for the simulations
    cell_cell_adhesion_strength_values = ['4.0']
    cell_cell_repulsion_strength_values =  ['10']
    prolif_rate_values = ['0.00072'] # ['0'] # 
    mot_speed_values = ['0.4'] 
    fiber_reorientation_rate_values = ['0.01'] # ['0.0','0.002','0.01','0.05','0.25','1.25'] # ['1.25'] # 
    ecm_displacement_rate_values = ['0.0'] # ['0.01'] # 
    ecm_density_rate_values = ['0.0'] # ['0.01'] # 
    fiber_realignment_rate_values = ['0.001']
    initial_ecm_density_values = ['0.5'] # ['0.7'] # 
    density_target_values = ['0.5'] 
    initial_anisotropy_values = ['0.0'] # ['0.5'] # ['0.5','0.2','0.8'] # 
    tumor_radius_values = ['40'] # ['100'] # 
    ecm_sensitivity_values = ['0.0','0.2','0.4','0.6','0.8','1.0'] # ['0.8'] # ['0.0','0.4','0.6'] # 
    chemotaxis_bias_values = ['0.0','0.2','0.4','0.6','0.8','1.0'] # ['0.2'] # ['0.2','1.0'] # ['0.0','0.2','0.4'] # ['0.2'] # 
    substrate_density_threshold_values = ['15'] 
    cell_setup_values = ['random_strip_bottom'] # ['spheroid'] # ['random_strip'] # 

    simulation_parameters = list(product(
    cell_cell_adhesion_strength_values,
    cell_cell_repulsion_strength_values,
    prolif_rate_values,
    mot_speed_values,
    fiber_reorientation_rate_values,
    ecm_displacement_rate_values,
    ecm_density_rate_values,
    fiber_realignment_rate_values,
    initial_ecm_density_values,
    density_target_values,
    initial_anisotropy_values,
    tumor_radius_values,
    ecm_sensitivity_values,
    chemotaxis_bias_values,
    substrate_density_threshold_values,
    cell_setup_values
    ))

    #### Get total number of simulations
    num_simulations = len(simulation_parameters)

    #### Reset, clean and initiate the project (make sure the project folder exists in data folder too)
    return_code = os.system(f'make reset; make clean; make load PROJ={proj}')
    if return_code != 0:
        print(f'Failed with exit code: {return_code}')
        sys.exit()

    #### If the simulation_id number is negative find the last simulation number in the data folder
    if simulation_id < 0:
        directories = []

        #### Check what is the last simulation number in the data folder
        for p in Path(f'./data/{proj}').iterdir():
            if p.is_dir():
                directories.append(p.name)
        if not directories:
            simulation_id = 0
        else:
            # simulation_ids = [int(name.split('_')[2]) for name in directories]
            simulation_ids = [int(name.split('_')[1]) for name in directories]
            simulation_ids.sort()
            simulation_id = max(simulation_ids) + 1

    #### List of all simulations
    simulations = range(simulation_id, simulation_id + num_simulations)
    print(f'simulations: {simulation_id} to {simulation_id + num_simulations -1} ')

#### List to check when all of the simulations are finished
results = []

i = 0

for sim in simulations:
    # for rib in ribose_concentration_values: 
    for orientation in ecm_orientation_setup_values: 
        for seed in random_seed_values:
            # print(f'\n####Simulation {sim}, ribose {rib}, random seed {seed}', flush=True)
            
            #### Create the output folder for the current simulation
            # return_code = os.system(f'mkdir -p ./data/{proj}/output_rib{rib}_{sim}_{seed}')
            return_code = os.system(f'mkdir -p ./data/{proj}/output_{orientation}_{sim}_{seed}')
            if return_code != 0:
                print(f'Failed with exit code: {return_code}')
                sys.exit()  

            if repeat_simulations:
                ### Start the experiment with the given ribose and seed in the settings file ###
                
                #### Copy PhysiCell_settings file from existing simulation with ribose 0 into config folder
                return_code = os.system(f'cp ./data/{proj}/output_rib50_{sim}_{seed}/PhysiCell_settings.xml ./config/')
                if return_code != 0:
                    print(f'Failed with exit code: {return_code}')
                    sys.exit()
                
                #### Copy custom.cpp file from existing simulation with ribose 0 into custom_modules folder
                return_code = os.system(f'cp ./user_projects/{proj}/custom_modules/* ./custom_modules/ ')
                if return_code != 0:
                    print(f'Failed with exit code: {return_code}')
                    sys.exit()
            else:
                # print(f'#### {motility_speed=}, {proliferation=}, {adhesion=}, {repulsion=}, {r_density=} ####\n',flush=True)

                print(f'\nSimulation {sim} out of {simulation_id + num_simulations -1}, {orientation=}, {seed=}') # type: ignore

            #### Select PhysiCell_settings file from config folder 
            tree = ET.parse('./config/PhysiCell_settings.xml')  
            root = tree.getroot()

            #### Build the PhysiCell_settings file with the given parameter values
            options = root.find('options')
            user_parameters = root.find('user_parameters')
            save = root.find('save')
            cell_definitions = root.find('cell_definitions')
            virtual_wall_at_domain_edge = options.find('virtual_wall_at_domain_edge') # type: ignore
            random_seed = user_parameters.find('random_seed') # type: ignore
            ribose_concentration = user_parameters.find('ribose_concentration') # type: ignore
            ecm_orientation_setup = user_parameters.find('ecm_orientation_setup') # type: ignore
            initial_anisotropy = user_parameters.find('initial_anisotropy') # type: ignore
            initial_ecm_density = user_parameters.find('initial_ecm_density') # type: ignore
            tumor_radius = user_parameters.find('tumor_radius') # type: ignore
            folder = save.find('folder')    # type: ignore
            cell_definition = cell_definitions.find('cell_definition') # type: ignore
            phenotype = cell_definition.find('phenotype') # type: ignore
            custom_data = cell_definition.find('custom_data') # type: ignore
            mechanics = phenotype.find('mechanics') # type: ignore
            cycle = phenotype.find('cycle') # type: ignore
            volume = phenotype.find('volume') # type: ignore
            motility = phenotype.find('motility') # type: ignore
            total = volume.find('total') # type: ignore
            cell_cell_adhesion_strength = mechanics.find('cell_cell_adhesion_strength') # type: ignore
            cell_cell_repulsion_strength = mechanics.find('cell_cell_repulsion_strength') # type: ignore
            phase_transition_rates = cycle.find('phase_transition_rates') # type: ignore
            prolif_rate = phase_transition_rates.find('rate') # type: ignore
            mot_speed = motility.find('speed') # type: ignore
            fiber_reorientation_rate = custom_data.find('fiber_reorientation_rate') # type: ignore
            ecm_density_rate = custom_data.find('ecm_density_rate') # type: ignore
            ecm_displacement_rate = custom_data.find('ecm_displacement_rate') # type: ignore
            fiber_realignment_rate = custom_data.find('fiber_realignment_rate') # type: ignore
            ecm_sensitivity = custom_data.find('ecm_sensitivity') # type: ignore
            sigma = user_parameters.find('sigma') # type: ignore
            delta = user_parameters.find('delta') # type: ignore

            chemotaxis_bias = custom_data.find('chemotaxis_bias') # type: ignore
            ecm_orientation_setup = user_parameters.find('ecm_orientation_setup') # type: ignore
            substrate_density_threshold = custom_data.find('substrate_density_threshold') # type: ignore
            cell_setup = user_parameters.find('cell_setup') # type: ignore

            density_target = user_parameters.find('density_target') # type: ignore
        
            if(repeat_simulations):
                virtual_wall_at_domain_edge.text = 'true' # type: ignore
                random_seed.text = str(seed)  # type: ignore
                ribose_concentration.text = str(rib) # type: ignore
                folder.text = f'data/{proj}/output_rib{rib}_{sim}_{seed}/' # type: ignore

            else:
                # folder.text = f'data/{proj}/output_rib{rib}_{sim}_{seed}/' # type: ignore
                folder.text = f'data/{proj}/output_{orientation}_{sim}_{seed}/' # type: ignore

                # ribose_concentration.text = str(rib) # type: ignore
                random_seed.text = str(seed)  # type: ignore

                ecm_orientation_setup.text = orientation # type: ignore

                cell_cell_adhesion_strength.text = simulation_parameters[i][0] # type: ignore
                cell_cell_repulsion_strength.text = simulation_parameters[i][1] # type: ignore
                prolif_rate.text = simulation_parameters[i][2] # type: ignore
                mot_speed.text = simulation_parameters[i][3] # type: ignore
                fiber_reorientation_rate.text = simulation_parameters[i][4] # type: ignore
                ecm_displacement_rate.text = simulation_parameters[i][5] # type: ignore
                ecm_density_rate.text = simulation_parameters[i][6] # str(r_density) # type: ignore
                fiber_realignment_rate.text = simulation_parameters[i][7] # type: ignore
                initial_ecm_density.text = simulation_parameters[i][8] # type: ignore
                density_target.text = simulation_parameters[i][9] # type: ignore
                initial_anisotropy.text = simulation_parameters[i][10] # type: ignore
                tumor_radius.text = simulation_parameters[i][11] # type: ignore
                ecm_sensitivity.text = simulation_parameters[i][12] # type: ignore
                chemotaxis_bias.text = simulation_parameters[i][13] # type: ignore
                substrate_density_threshold.text = simulation_parameters[i][14] # type: ignore
                cell_setup.text = simulation_parameters[i][15] # type: ignore

                # sigma.text = str(sigma_text) # type: ignore
                # delta.text = str(delta_text) # type: ignore
            
            #### Write the xml file for the current simulation
            tree.write('./config/PhysiCell_settings.xml')    
            print(f'#### Xml settings file has been written\n',flush=True)

            #### Copy custom.cpp file from custom_modules folder into new simulation folder
            # os.system(f'cp ./custom_modules/custom.cpp ./data/{proj}/output_rib{rib}_{sim}_{seed}/')
            os.system(f'cp ./custom_modules/custom.cpp ./data/{proj}/output_{orientation}_{sim}_{seed}/')

            #### Copy PhysiCell_settings file from config folder into new simulation folder
            # os.system(f'cp ./config/PhysiCell_settings.xml ./data/{proj}/output_rib{rib}_{sim}_{seed}/')
            os.system(f'cp ./config/PhysiCell_settings.xml ./data/{proj}/output_{orientation}_{sim}_{seed}/')

            #### Start simulation (simulations run in parallel)
            os.system('make') 
            if len(results)==0:
                return_code = subprocess.Popen(f'./project')
                results.append(return_code)
            elif len(results)%19==0:
                return_code = subprocess.run(f'./project')
                results.append(return_code)
            else:
                return_code = subprocess.Popen(f'./project')
                results.append(return_code)

            os.system('sleep 5') 

    i += 1 

# Iterate through all return codes, only exit when all are done (subprocess.run)
# returns True because we know it waits until finishing
while True:
    temp_results = []
    for x in results:
        if isinstance(x,subprocess.CompletedProcess):
            res = x.returncode
        else:
            res = x.poll()

        # If encounter a return code of not 0, exit program immedietly
        if not(res == None or res == 0):
            sys.exit(1)
            print('Non-zero exit code found, exiting...', flush=True)

        temp_results.append(res == 0)
    if all(temp_results):
        break

