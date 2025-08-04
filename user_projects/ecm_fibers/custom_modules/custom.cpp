/*
###############################################################################
# If you use PhysiCell in your project, please cite PhysiCell and the version #
# number, such as below:                                                      #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1].    #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# See VERSION.txt or call get_PhysiCell_version() to get the current version  #
#     x.y.z. Call display_citations() to get detailed information on all cite-#
#     able software used in your PhysiCell application.                       #
#                                                                             #
# Because PhysiCell extensively uses BioFVM, we suggest you also cite BioFVM  #
#     as below:                                                               #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1],    #
# with BioFVM [2] to solve the transport equations.                           #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# [2] A Ghaffarizadeh, SH Friedman, and P Macklin, BioFVM: an efficient para- #
#     llelized diffusive transport solver for 3-D biological simulations,     #
#     Bioinformatics 32(8): 1256-8, 2016. DOI: 10.1093/bioinformatics/btv730  #
#                                                                             #
###############################################################################
#                                                                             #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)     #
#                                                                             #
# Copyright (c) 2015-2021, Paul Macklin and the PhysiCell Project             #
# All rights reserved.                                                        #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
# this list of conditions and the following disclaimer.                       #
#                                                                             #
# 2. Redistributions in binary form must reproduce the above copyright        #
# notice, this list of conditions and the following disclaimer in the         #
# documentation and/or other materials provided with the distribution.        #
#                                                                             #
# 3. Neither the name of the copyright holder nor the names of its            #
# contributors may be used to endorse or promote products derived from this   #
# software without specific prior written permission.                         #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" #
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   #
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  #
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   #
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  #
# POSSIBILITY OF SUCH DAMAGE.                                                 #
#                                                                             #
###############################################################################
*/

#include "./custom.h"
#include "./extracellular_matrix.h"
#include <cmath>

using namespace BioFVM; 
using namespace PhysiCell;

// extern ECM ecm;

ECM ecm;

void create_cell_types( void )
{
	// set the random seed 
	SeedRandom( parameters.ints("random_seed") );  
	
	/* 
	   Put any modifications to default cell definition here if you 
	   want to have "inherited" by other cell types. 
	   
	   This is a good place to set default functions. 
	*/ 

	initialize_default_cell_definition(); 
	
	cell_defaults.phenotype.secretion.sync_to_microenvironment( &microenvironment ); 
	
	cell_defaults.functions.update_velocity = custom_update_cell_velocity;
	cell_defaults.functions.custom_cell_rule = custom_function; 
	cell_defaults.functions.update_migration_bias = cell_ecm_adhesion_direction; 

	cell_defaults.functions.update_phenotype = NULL;

	cell_defaults.functions.volume_update_function = standard_volume_update_function;
	// cell_defaults.functions.update_velocity = standard_update_cell_velocity;

	// cell_defaults.functions.update_migration_bias = NULL; 
	// cell_defaults.functions.update_phenotype = NULL; // update_cell_and_death_parameters_O2_based; 
	// cell_defaults.functions.custom_cell_rule = NULL; 
	cell_defaults.functions.contact_function = NULL; 
	
	cell_defaults.functions.add_cell_basement_membrane_interactions = NULL; 
	cell_defaults.functions.calculate_distance_to_membrane = NULL; 
	
	/*
	   This parses the cell definitions in the XML config file. 
	*/
	
	initialize_cell_definitions_from_pugixml(); 

	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/
		
	build_cell_definitions_maps(); 

	/*
	   This intializes cell signal and response dictionaries 
	*/

	setup_signal_behavior_dictionaries(); 	

	/* 
	   Put any modifications to individual cell definitions here. 
	   
	   This is a good place to set custom functions. 
	*/ 
	
	// cell_defaults.functions.update_phenotype = phenotype_function; 
	// cell_defaults.functions.custom_cell_rule = custom_function; 
	// cell_defaults.functions.contact_function = contact_function; 
	// cell_defaults.functions.update_migration_bias = cell_ecm_adhesion_direction; 

	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/

	// cell_defaults.functions.update_velocity = custom_update_cell_velocity;
	
		
	// display_cell_definitions( std::cout ); 
	
	return; 
}

void setup_microenvironment( void )
{
	SeedRandom( parameters.ints("random_seed") );  

	// set domain parameters 
	
	// put any custom code to set non-homogeneous initial conditions or 
	// extra Dirichlet nodes here. 
	
	// initialize BioFVM 

	initialize_microenvironment(); 

	return; 
}
void setup_extracellular_matrix(void)
{
	SeedRandom( parameters.ints("random_seed") );  

	// DEPENDS ON MICROENVIRONMENT - CALL SETUP MICROENVIRONEMNT FIRST!!!!!

    // Resize the ECM mesh according to the microenvironment options
    ecm.ecm_mesh.resize(default_microenvironment_options.X_range[0], default_microenvironment_options.X_range[1], 
                        default_microenvironment_options.Y_range[0], default_microenvironment_options.Y_range[1], 
                        default_microenvironment_options.Z_range[0], default_microenvironment_options.Z_range[1], 
                        parameters.doubles("ecm_dx"), parameters.doubles("ecm_dy"), parameters.doubles("ecm_dz"));
    // Adjust the ECM units based on the resized ECM mesh
	ecm.resize_ecm_units_from_ecm_mesh();

	// ecm.ecm_mesh.display_information(std::cout );

    // Set up ECM alignment

    // <ecm_orientation_setup description="Specifies the initial ECM orientation: random, tangential, starburt, oriented to the right, or oriented to the top" type="string" units="NA">tangential</ecm_orientation_setup> parameters.string("ecm_orientation_setup")
	// std::vector<double> fiber_direction = { 1.0 , 0.0, 0.0 }; 
    // ecm_fiber_alignment.resize(microenvironment.mesh.voxels.size(), fiber_direction);  

    // Retrieve parameters for spacing and tumor radius
	double cell_radius = cell_defaults.phenotype.geometry.radius;
    double spacing = parameters.doubles("ecm_dx");
    double tumor_radius = parameters.doubles("tumor_radius") - 2*cell_radius;


    for(int n = 0; n < ecm.ecm_mesh.voxels.size(); n++)
	{
	
		double x = spacing / 2; // Initialize x-coordinate
		double y = spacing / 2; // Initialize y-coordinate
		double x_outer; // Outer boundary of x-coordinate within the tumor

		double ecm_density = parameters.doubles("initial_ecm_density");

		if (parameters.strings("cell_setup") == "spheroid")
		{
			ecm_density = 0; 
		}

		int inner_n = 0; // Counter for loop iterations
		int voxel_index; // Index of the voxel in the microenvironment
		std::vector<double> pos; // Position vector for voxel

		// Iterate over y-coordinates within the tumor radius
		while(y < tumor_radius)
		{
			x = spacing / 2; // Reset x-coordinate for each y-level
		
			// Adjust x-coordinate step for odd/even iterations
			if(inner_n % 2 == 1)
			{ 
				x = 0.5 * spacing; 
			}
			x_outer = sqrt(tumor_radius * tumor_radius - y * y); // Calculate outer boundary for x-coordinate
			
			// Iterate over x-coordinates within the current y-level
			while(x < x_outer)
			{
				// Set ECM density for the current position
				pos = {x, y, 0.0};
				voxel_index = microenvironment.nearest_voxel_index(pos);
				// Check if the voxel index is valid before accessing
				if (voxel_index >= 0 && voxel_index < ecm.ecm_voxels.size())
				{
				ecm.ecm_voxels[voxel_index].density = ecm_density;
				}
				
				// Set ECM density for the mirrored y-position if needed
				if(fabs(y) > 0.01)
				{
					pos = {x, -y, 0.0};
					voxel_index = microenvironment.nearest_voxel_index(pos);
					if (voxel_index >= 0 && voxel_index < ecm.ecm_voxels.size())
					{
					ecm.ecm_voxels[voxel_index].density = ecm_density;
				}
				}
				
				// Set ECM density for the mirrored x-position if needed
				if(fabs(x) > 0.01)
				{ 
					pos = {-x, y, 0.0};
					voxel_index = microenvironment.nearest_voxel_index(pos);
					if (voxel_index >= 0 && voxel_index < ecm.ecm_voxels.size())
					{
					ecm.ecm_voxels[voxel_index].density = ecm_density;	
					}
					
					// Set ECM density for the mirrored x and y positions if needed
					if(fabs(y) > 0.01)
					{
						pos = {-x, -y, 0.0};
						voxel_index = microenvironment.nearest_voxel_index(pos);
						if (voxel_index >= 0 && voxel_index < ecm.ecm_voxels.size())
						{
						ecm.ecm_voxels[voxel_index].density = ecm_density;
						}
					}
				}
				x += spacing; // Move to the next x-coordinate
			}
			y += spacing; // Move to the next y-coordinate
			inner_n++; // Increment the counter

		}

        // For random 2-D initialization
        if(parameters.strings("ecm_orientation_setup") == "random")
		{
            double theta = 6.2831853071795864769252867665590 * UniformRandom(); // Random angle in radians
            ecm.ecm_voxels[n].ecm_fiber_alignment = {cos(theta), sin(theta), 0.0}; // Set ECM fiber alignment
        }
		// for radial initialization 
		else if(parameters.strings( "ecm_orientation_setup") == "radial")
		{
			std::vector<double> position = ecm.ecm_mesh.voxels[n].center; 
			normalize( &position ); 
			ecm.ecm_voxels[n].ecm_fiber_alignment =  { position[0],position[1],0}; // oriented out (perpindeicular to concentric circles)
			normalize(&ecm.ecm_voxels[n].ecm_fiber_alignment);
		}
		// for tangential initialization 
		else if(parameters.strings( "ecm_orientation_setup") == "tangential")
		{
			std::vector<double> position = ecm.ecm_mesh.voxels[n].center;; 
			normalize( &position );
			ecm.ecm_voxels[n].ecm_fiber_alignment =  { position[1],-position[0],0}; // oriented in cirlce
			normalize(&ecm.ecm_voxels[n].ecm_fiber_alignment);
		}

		else if(parameters.strings( "ecm_orientation_setup") == "horizontal")
		{
			ecm.ecm_voxels[n].ecm_fiber_alignment =  { 1.0, 0.0, 0.0}; 
			normalize(&ecm.ecm_voxels[n].ecm_fiber_alignment);
		}
		else if(parameters.strings( "ecm_orientation_setup") == "vertical")
		{
			ecm.ecm_voxels[n].ecm_fiber_alignment=  { 0.0, 1.0, 0.0}; 
			normalize(&ecm.ecm_voxels[n].ecm_fiber_alignment);
		}
		else if(parameters.strings( "ecm_orientation_setup") == "split")
		{
			std::vector<double> position = ecm.ecm_mesh.voxels[n].center; 
			// normalize( &position ); // pretty sure I shouldn't normalize this - unless I really NEED a unit vector. The position itself isn't actuall normal. 
			double ecm_dx = parameters.doubles("ecm_dx");
			
			if(position[1]<=-ecm_dx)
			{
				ecm.ecm_voxels[n].ecm_fiber_alignment =  { 1.0,0.0,0.0}; // oriented horizontally
				normalize(&ecm.ecm_voxels[n].ecm_fiber_alignment);
			}
			else if(position[1]>=ecm_dx)
			{
				ecm.ecm_voxels[n].ecm_fiber_alignment =  { 0.0,1.0,0.0}; // oriented vertically 
				normalize(&ecm.ecm_voxels[n].ecm_fiber_alignment);				
			}
			else
			{
				// Random orientation for the rest of the ECM
				double theta = 6.2831853071795864769252867665590 * UniformRandom(); // Random angle in radians
				ecm.ecm_voxels[n].ecm_fiber_alignment = {cos(theta), sin(theta), 0.0}; // Set ECM fiber alignment
				normalize(&ecm.ecm_voxels[n].ecm_fiber_alignment);
			}
		}
		else
		{
            // If the orientation setup is not specified, halt the program
            std::cout << "WARNING: NO ECM ORIENTATION SPECIFIED. FIX THIS!!!" << std::endl;
            std::cout << "Halting program!!!" << std::endl;
			abort();
			return;
		}

        // Ensure index is within bounds before accessing
        if(n >= 0 && n < ecm.ecm_voxels.size())
        {
		ecm.ecm_voxels[n].density = parameters.doubles("initial_ecm_density");
		ecm.ecm_voxels[n].anisotropy = parameters.doubles("initial_anisotropy");
        }

        // Check if the initial density and anisotropy values are within valid bounds
        if(parameters.doubles("initial_ecm_density") > 1 || parameters.doubles("initial_anisotropy") > 1 ||
           parameters.doubles("initial_ecm_density") < 0 || parameters.doubles("initial_anisotropy") < 0)
		{
            std::cout << "WARNING: INITIAL DENSITY OR ANISOTROPY OUT OF BOUNDS! FIX THIS!" << std::endl;
            std::cout << "Halting program!!!" << std::endl;
			abort();
			return;
		}		
	}
}

void setup_tissue(void)
{
	// double Xmin = microenvironment.mesh.bounding_box[0]; 
	// double Ymin = microenvironment.mesh.bounding_box[1]; 
	// double Zmin = microenvironment.mesh.bounding_box[2]; 

	// double Xmax = microenvironment.mesh.bounding_box[3]; 
	// double Ymax = microenvironment.mesh.bounding_box[4]; 
	// double Zmax = microenvironment.mesh.bounding_box[5]; 
	
    // If simulating in 2D, adjust the Z bounds to be zero
    // if(default_microenvironment_options.simulate_2D == true)
	// {
    //     Zmin = 0.0; 
    //     Zmax = 0.0; 
	// }
	
	// double Xrange = Xmax - Xmin; 
	// double Yrange = Ymax - Ymin; 
	// double Zrange = Zmax - Zmin; 
	
    // Create cells of each type defined
	// Cell* pC;
	
    // for(int k = 0; k < cell_definitions_by_index.size(); k++)
	// {
    //     Cell_Definition* pCD = cell_definitions_by_index[k]; 
    //     std::cout << "Placing cells of type " << pCD->name << " ... " << std::endl; 
    //     for(int n = 0; n < parameters.ints("number_of_cells"); n++)
    //     {
    //         std::vector<double> position = {0, 0, 0}; 
    //         position[0] = Xmin + UniformRandom() * Xrange; 
    //         position[1] = Ymin + UniformRandom() * Yrange; 
    //         position[2] = Zmin + UniformRandom() * Zrange; 
    //         
    //         pC = create_cell(*pCD); 
    //         pC->assign_position(position);
    //     }
	// }
	// std::cout << std::endl; 
	
    // Load cells from your CSV file (if enabled)
	// load_cells_from_pugixml(); 	
	
    // Setting seed so cells always start with the same initial configuration
    SeedRandom(parameters.ints("random_seed"));  

    /******************************Single cell initialization*****************************/
	if(parameters.strings("cell_setup") == "single")
	{
		Cell* pCell;
		pCell = create_cell(cell_defaults);
		pCell->assign_position(0.0, 0.0, 0.0);
        // std::cin.get(); 
	}

	/******************************************2D Spheroid initialization***************************************/
	else if(parameters.strings("cell_setup") == "spheroid")
	{
        // Place a cluster of tumor cells at the center
	
        // Get tumor radius from XML parameters
		double tumor_radius = parameters.doubles("tumor_radius"); 

        // Calculate cell spacing based on adhesion and repulsion parameters
		double cell_radius = cell_defaults.phenotype.geometry.radius;
		double relative_maximum_adhesion_distance = cell_defaults.phenotype.mechanics.relative_maximum_adhesion_distance;
		double cell_adhesion = cell_defaults.phenotype.mechanics.cell_cell_adhesion_strength;
		double cell_repulsion = cell_defaults.phenotype.mechanics.cell_cell_repulsion_strength;
        double sqrt_adhesion_to_repulsion_ratio = sqrt(cell_adhesion / cell_repulsion);
		double cell_spacing = (1 - sqrt_adhesion_to_repulsion_ratio);
        cell_spacing /= (0.5 * 1 / cell_radius - 0.5 * sqrt_adhesion_to_repulsion_ratio / (relative_maximum_adhesion_distance * cell_radius));

		Cell* pCell = NULL; 
		
        double x = 0.0; // Initialize x-coordinate
        double x_outer = tumor_radius; // Outer boundary for x-coordinate within the tumor
        double y = 0.0; // Initialize y-coordinate

        int n = 0; // Counter for loop iterations
        int voxel_index; // Index of the voxel in the microenvironment
        std::vector<double> pos; // Position vector for voxel

        // Iterate over y-coordinates within the tumor radius
        while(y < tumor_radius)
		{
			x = 0.0; 
            if(n % 2 == 1)
            { 
                x = 0.5 * cell_spacing; 
            }
            x_outer = sqrt(tumor_radius * tumor_radius - y * y); // Calculate outer boundary for x-coordinate
            
            // Iterate over x-coordinates within the current y-level
            while(x < x_outer)
			{
				pCell = create_cell(cell_defaults);
                pCell->assign_position(x, y, 0.0);
                set_single_behavior(pCell, "cycle entry", 0); // Set initial cell behavior
                
                // Set cells at mirrored y-position if needed
                if(fabs(y) > 0.01)
				{
					pCell = create_cell(cell_defaults);
                    pCell->assign_position(x, -y, 0.0);
                    set_single_behavior(pCell, "cycle entry", 0);
				}
				
                // Set cells at mirrored x-position if needed
                if(fabs(x) > 0.01)
				{ 
					pCell = create_cell(cell_defaults); 
                    pCell->assign_position(-x, y, 0.0);
                    set_single_behavior(pCell, "cycle entry", 0);
					
                    // Set cells at mirrored x and y positions if needed
                    if(fabs(y) > 0.01)
					{
						pCell = create_cell(cell_defaults);
                        pCell->assign_position(-x, -y, 0.0);
                        set_single_behavior(pCell, "cycle entry", 0);
					}
				}

                x += cell_spacing; // Move to the next x-coordinate
			}
			
            y += cell_spacing * sqrt(3.0) / 2.0; // Adjust y-coordinate for hexagonal packing
            n++; // Increment the counter
		}

        std::cout << "Cells placed in 2D spheroid at the center of domain" << std::endl;
	}

	else if(parameters.strings("cell_setup") == "square")
	{
        // Place a cluster of tumor cells at the center
	
        // Get tumor radius from XML parameters
		double tumor_radius = parameters.doubles("tumor_radius"); 

        // Calculate cell spacing based on adhesion and repulsion parameters
		double cell_radius = cell_defaults.phenotype.geometry.radius;
		double relative_maximum_adhesion_distance = cell_defaults.phenotype.mechanics.relative_maximum_adhesion_distance;
		double cell_adhesion = cell_defaults.phenotype.mechanics.cell_cell_adhesion_strength;
		double cell_repulsion = cell_defaults.phenotype.mechanics.cell_cell_repulsion_strength;
        double sqrt_adhesion_to_repulsion_ratio = sqrt(cell_adhesion / cell_repulsion);
		double cell_spacing = (1 - sqrt_adhesion_to_repulsion_ratio);
        cell_spacing /= (0.5 * 1 / cell_radius - 0.5 * sqrt_adhesion_to_repulsion_ratio / (relative_maximum_adhesion_distance * cell_radius));

		Cell* pCell = NULL; 
		
        double x = 0.0; // Initialize x-coordinate
        double x_outer = tumor_radius; // Outer boundary for x-coordinate within the tumor
        double y = 0.0; // Initialize y-coordinate

        int n = 0; // Counter for loop iterations
        int voxel_index; // Index of the voxel in the microenvironment
        std::vector<double> pos; // Position vector for voxel

        // Iterate over y-coordinates within the tumor radius
        while(y < tumor_radius)
		{
			x = 0.0; 
            if(n % 2 == 1)
            { 
                x = 0.5 * cell_spacing; 
            }
            // x_outer = sqrt(tumor_radius * tumor_radius - y * y); // Calculate outer boundary for x-coordinate
            
            // Iterate over x-coordinates within the current y-level
            while(x < x_outer)
			{
				pCell = create_cell(cell_defaults);
                pCell->assign_position(x, y, 0.0);
                set_single_behavior(pCell, "cycle entry", 0); // Set initial cell behavior
                
                // Set cells at mirrored y-position if needed
                if(fabs(y) > 0.01)
				{
					pCell = create_cell(cell_defaults);
                    pCell->assign_position(x, -y, 0.0);
                    set_single_behavior(pCell, "cycle entry", 0);
				}
				
                // Set cells at mirrored x-position if needed
                if(fabs(x) > 0.01)
				{ 
					pCell = create_cell(cell_defaults); 
                    pCell->assign_position(-x, y, 0.0);
                    set_single_behavior(pCell, "cycle entry", 0);
					
                    // Set cells at mirrored x and y positions if needed
                    if(fabs(y) > 0.01)
					{
						pCell = create_cell(cell_defaults);
                        pCell->assign_position(-x, -y, 0.0);
                        set_single_behavior(pCell, "cycle entry", 0);
					}
				}

                x += cell_spacing; // Move to the next x-coordinate
			}
			
            y += cell_spacing * sqrt(3.0) / 2.0; // Adjust y-coordinate for hexagonal packing
            n++; // Increment the counter
		}

        std::cout << "Cells placed in 2D square at the center of domain" << std::endl;
	}

	else if(parameters.strings("cell_setup") == "random_strip")
	{

		double height = parameters.doubles("tumor_radius"); 
	
        // Get tumor radius from XML parameters
		double number_of_cells = parameters.ints("number_of_cells"); 

		double x_min = default_microenvironment_options.X_range[0];

		Cell* pCell = NULL; 

        // Iterate over cells
        for (unsigned int i = 0; i<number_of_cells; ++i)
		{
				pCell = create_cell(cell_defaults);
                pCell->assign_position(x_min * (1 - 2 *UniformRandom()), height * (UniformRandom() - 0.5), 0.0);
                set_single_behavior(pCell, "cycle entry", 0); // Set initial cell behavior
		}
        std::cout << "Cells placed in 2D random strip at the center of domain" << std::endl;
	}

	else if(parameters.strings("cell_setup") == "random_strip_bottom")
	{

		double height = parameters.doubles("tumor_radius"); 
	
        // Get tumor radius from XML parameters
		double number_of_cells = parameters.ints("number_of_cells"); 

		double x_min = default_microenvironment_options.X_range[0];
		double y_min = default_microenvironment_options.Y_range[0]; 

		// Adjust height to be half of the specified tumor radius
		height /= 2.0;

		Cell* pCell = NULL; 

        // Iterate over cells
        for (unsigned int i = 0; i<number_of_cells; ++i)
		{
				pCell = create_cell(cell_defaults);
                pCell->assign_position(x_min * (1 - 2 *UniformRandom()), y_min + height * UniformRandom(), 0.0);
                set_single_behavior(pCell, "cycle entry", 0); // Set initial cell behavior
		}
        std::cout << "Cells placed in 2D random strip at the center of domain" << std::endl;
	}

	else if(parameters.strings("cell_setup") == "boundary")
	{
        // Calculate cell spacing based on adhesion and repulsion parameters
		double cell_radius = cell_defaults.phenotype.geometry.radius;
		double relative_maximum_adhesion_distance = cell_defaults.phenotype.mechanics.relative_maximum_adhesion_distance;
		double cell_adhesion = cell_defaults.phenotype.mechanics.cell_cell_adhesion_strength;
		double cell_repulsion = cell_defaults.phenotype.mechanics.cell_cell_repulsion_strength;
        double sqrt_adhesion_to_repulsion_ratio = sqrt(cell_adhesion / cell_repulsion);
		double cell_spacing = (1 - sqrt_adhesion_to_repulsion_ratio);
        cell_spacing /= (0.5 * 1 / cell_radius - 0.5 * sqrt_adhesion_to_repulsion_ratio / (relative_maximum_adhesion_distance * cell_radius));

		Cell* pCell = NULL; 
		
        double y = 0.0; // Initialize x-coordinate
        // double x_outer = tumor_radius; // Outer boundary for x-coordinate within the tumor
        double x = -500.0 + cell_radius; // Initialize y-coordinate

        int n = 0; // Counter for loop iterations
        int voxel_index; // Index of the voxel in the microenvironment
        std::vector<double> pos; // Position vector for voxel

        // Iterate over y-coordinates within the tumor radius
        while(y < 500)
		{
			pCell = create_cell(cell_defaults);
			pCell->assign_position(x, y, 0.0);
			set_single_behavior(pCell, "cycle entry", 0);

			pCell = create_cell(cell_defaults);
			pCell->assign_position(x, -y, 0.0);
			set_single_behavior(pCell, "cycle entry", 0);
			
            y += cell_spacing * sqrt(3.0) / 2.0; // Adjust y-coordinate for hexagonal packing
            n++; // Increment the counter
		}

        std::cout << "Cells placed in 2D square at the center of domain" << std::endl;
	}

	else
	{
        // If no cell setup is specified, halt the program
        std::cout << "WARNING!!! NO CELL SETUP SPECIFIED. SEE DOCUMENTATION and FIX" << std::endl;
        std::cout << "Halting program!!!" << std::endl;
		abort();
		return;
	}

	return;
}


std::vector<std::string> AMIGOS_invasion_coloring_function( Cell* pCell )
{
	// leaders are blue 
	std::vector< std::string > output( 4, "black" ); 
	
	if( pCell->type == 1 )
	{ 
		output[0] = "blue"; 
		output[2] = "blue"; 
		return output; 
	} 

	// followers are yellow except for the marker cells when running testing mode (then 20 % of cells are red)
    
	if( pCell->type == 2 )
    {
        output[2] = "yellow";	
		output[0] = "yellow";
		
		// Return yellow for followers and exit statement

		if(parameters.ints("unit_test_setup")==0)
		{return output;}

		// Return red for 20% of followers if unit test is called for	

		if( pCell->ID % 5 == 0)
		{
			output[0] = "red";
        	output[2] = "red";
        	return output;
		}

		// If doing unit testing AND cell not selected as marker cell, return blue. 
		output[2] = "blue";	
		output[0] = "blue";


        return output;
    }
	
	// dead are red
	if( pCell->phenotype.death.dead == false )
	{
		output[0] = "red";
		output[2] = "red";
		
		return output; 
	}

	// if not, dead colors 
	
	if (pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::apoptotic )  // Apoptotic - Red
	{
		output[0] = "rgb(255,0,0)";
		output[2] = "rgb(125,0,0)";
	}
	
	// Necrotic - Brown
	if( pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::necrotic_swelling || 
		pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::necrotic_lysed || 
		pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::necrotic )
	{
		output[0] = "rgb(250,138,38)";
		output[2] = "rgb(139,69,19)";
	}	
	
	return output; 
}

double dot_product_ext(const std::vector<double>& v, const std::vector<double>& w)
{
    // Initialize the result of the dot product to zero
	double out = 0.0; 
    
    // Ensure both vectors are of the same size
    if (v.size() != w.size()) {
        std::cerr << "Error: Vectors must be of the same size." << std::endl;
        return 0.0; // Return zero or handle the error as needed
    }

    // Compute the dot product of vectors v and w
    for (unsigned int i = 0; i < v.size(); i++)
    {
        out += (v[i] * w[i]);
    }

    // If the absolute value of the result is very small, set it to zero
    if (fabs(out) < 1e-10)
    {
        out = 0.0;
    }

    return out; // Return the computed dot product
}

double sign_function (double number)
{
	// double sign = 0.0
	if (number<0)
	{ return -1.0;}

	else
	{ return 1.0;}
}

void cell_ecm_adhesion_speed(Cell* pCell, Phenotype& phenotype, double dt)
{
    // Function that computes the cell-ECM adhesion speed and stores it as motility speed (migration speed)
	
    // Check if the cell is dead
    if (phenotype.death.dead == true)
	{
		phenotype.motility.is_motile = false;
		pCell->functions.update_migration_bias = NULL;
		pCell->functions.update_phenotype = NULL;
        std::cout << "Cell is dead" << std::endl;
        return;
	}

    // Determine which voxel to use based on the specified method
    int voxel_index = 0;

	if (parameters.strings("nearest_voxel_speed") == "front")
	{
        // Pick voxel closest to the cell's membrane in the direction of movement
        std::vector<double> direction = {pCell->custom_data["total_velocity_x"], 
                                         pCell->custom_data["total_velocity_y"], 
                                         pCell->custom_data["total_velocity_z"]};
        normalize(&direction); // Normalize the direction vector
		std::vector<double> scaled_direction = pCell->phenotype.geometry.radius * direction;

        // Calculate the position of the membrane in the direction of the cell
        std::vector<double> position_membrane = pCell->position;
        for (size_t i = 0; i < position_membrane.size(); ++i) {
            position_membrane[i] += scaled_direction[i];
        }

        // Compute the index of the voxel at the position of the membrane
        voxel_index = microenvironment.nearest_voxel_index(position_membrane);
	}
	else if (parameters.strings("nearest_voxel_speed") == "position")
	{
        // Compute the index of the voxel at the cell's current position
        voxel_index = microenvironment.nearest_voxel_index(pCell->position);
	}
	else
	{
        // Handle case where the voxel selection method is not defined
        std::cout << "Nearest voxel for computing migration speed not defined!" << std::endl;
        abort(); // Terminate the program if voxel method is not defined
		return;
	}

    // Retrieve ECM density at the computed voxel index
	double ecm_density = ecm.ecm_voxels[voxel_index].density;

    // If the motility vector is to be normalized, set migration speed to 1.0
    if (parameters.bools("normalize_ecm_influenced_motility_vector") == true)
	{
		pCell->phenotype.motility.migration_speed = 1.0;
	}
	else
	{
        // Otherwise, set migration speed based on the magnitude of the motility vector
        pCell->phenotype.motility.migration_speed = norm(phenotype.motility.migration_bias_direction);
        // std::cout << "Magnitude of motility vector is " << pCell->phenotype.motility.migration_speed << std::endl;
	}
	
    // Set migration bias to 1.0 to ensure no additional random motion is added
	phenotype.motility.migration_bias = 1.0; 

    // Get the maximum migration speed
    double max_migration_speed = get_single_base_behavior(pCell, "migration speed");

    // Compute cell-ECM adhesion speed, which is stored in migration speed
	pCell->phenotype.motility.migration_speed *= 4 * max_migration_speed * ecm_density;

	return; 
}


void cell_ecm_adhesion_direction( Cell* pCell, Phenotype& phenotype, double dt )
{
	// Function to compute cell-ECM adhesion direction, stored in motility direction

	if(phenotype.death.dead == true)
	{
		phenotype.motility.is_motile = false;
		pCell->functions.update_migration_bias = NULL;
		pCell->functions.update_phenotype = NULL;
		std::cout<<"Cell is dead"<<std::endl;
				
		return; 
	}

	// Find nearest voxel to cell's front or position
    int voxel_index = 0;
    if (parameters.strings("nearest_voxel_direction") == "front")
    {
        //Pick voxel closest to cell's membrane in direction of movement
        std::vector<double> direction = {pCell->custom_data["total_velocity_x"], pCell->custom_data["total_velocity_y"], pCell->custom_data["total_velocity_z"]};
		normalize(&direction); // Normalize the direction vector
        std::vector<double> scaled_direction = pCell->phenotype.geometry.radius * direction;

        // Calculate the position of the membrane in the direction of the cell
        std::vector<double> position_membrane = pCell->position;
        for (size_t i = 0; i < position_membrane.size(); ++i) {
            position_membrane[i] += scaled_direction[i];
        }

        // Computing the index of the voxel at that position
        voxel_index = microenvironment.nearest_voxel_index( position_membrane );
    }
    
    else if (parameters.strings("nearest_voxel_direction") == "position")
    {
        // Computing the index of the voxel at cell position
        voxel_index = microenvironment.nearest_voxel_index( pCell->position );
    }

    else
    {
        std::cout<<"Nearest voxel for computing motility direction not defined!"<<std::endl;
        abort();
        return;
    }

	// Get random vector 
    std::vector<double> d_random(3,0.0);
    if( phenotype.motility.restrict_to_2D == true )
    { 
        d_random = UniformOnUnitCircle(); 
    }
    else
    { 
        d_random = UniformOnUnitSphere(); 
    }

	// Initiate preferred direction 
	std::vector<double> d_pref;

	// Preferred direction given by random vector with no chemotaxis
	double chemotaxis_bias = pCell->custom_data["chemotaxis_bias"];

	if(chemotaxis_bias == 0)
	{
		d_pref = d_random;
	}

	// Preferred direction with chemotaxis
	else
	{

		// Get vector for chemotaxis 
		static int substrate_index = microenvironment.find_density_index( "substrate" ); 
		std::vector<double> chemotaxis_grad = pCell->nearest_gradient(substrate_index);
		normalize( &chemotaxis_grad ); 

		// combine cell chosen random direction and chemotaxis direction (like standard update_motility function)
		d_pref = (1 - chemotaxis_bias) * d_random + chemotaxis_bias * chemotaxis_grad;
	}

	normalize( &d_pref ); 

	// Compute fiber orientation and anisotropy contribution to motility preferred direction
	std::vector<double> motility_direction;

    /************************* cell-ECM adhesion direction with density ************************/
	if (parameters.strings("ecm_definition") == "ecm_density")
	{

		motility_direction = d_pref;
	}

	/************************* cell-ECM adhesion direction with fibres ************************/
	else if(parameters.strings("ecm_definition") == "ecm_fibers")
	{

		// Get cell sensitivity for ECM 
		double ecm_sensitivity = pCell->custom_data["ecm_sensitivity"];

		double anisotropy = ecm.ecm_voxels[voxel_index].anisotropy; 
		std::vector<double> fiber_orientation = ecm.ecm_voxels[voxel_index].ecm_fiber_alignment;

		std::vector<double> d_random_fiber_orientation(3,0.0);
		if( phenotype.motility.restrict_to_2D == true )
		{ 
			d_random_fiber_orientation = UniformOnUnitCircle(); 
		}
		else
		{ 
			d_random_fiber_orientation = UniformOnUnitSphere(); 
		}
		// d_random_fiber_orientation = UniformOnUnitSphere(); 

		fiber_orientation = anisotropy * fiber_orientation + (1 - anisotropy) * d_random_fiber_orientation;

		double ddotf;
		// std::vector<double> norm_cell_velocity = pCell->phenotype.motility.motility_vector;
		// normalize(&norm_cell_velocity);
		// ddotf = dot_product_ext(fiber_orientation, norm_cell_motility);

		ddotf = dot_product_ext(fiber_orientation, d_pref);

		fiber_orientation = sign_function(ddotf) * fiber_orientation;
		normalize( &fiber_orientation ); 

		motility_direction = (1.0 - ecm_sensitivity) * d_pref + ecm_sensitivity * fiber_orientation;
		normalize( &motility_direction );
	}
	else
	{
		std::cout<<"ECM definition not declared!!!"<<std::endl;
		abort();
		return;
	}

	pCell->phenotype.motility.migration_bias_direction = motility_direction;

	return;
}


void cell_ecm_repulsion(Cell* pCell , Phenotype& phenotype , double dt) 
{ 
	if(phenotype.death.dead == true)
	{
		phenotype.motility.is_motile = false;
		pCell->functions.update_migration_bias = NULL;
		pCell->functions.update_phenotype = NULL;
		std::cout<<"Cell is dead"<<std::endl;
				
		return; 
	}
    // Function that computes the cell-ECM repulsion velocity

    // Get cell's direction of movement
    std::vector<double> direction = pCell->velocity;
    normalize(&direction); // Normalize the direction vector

    double cell_radius = pCell->phenotype.geometry.radius;

    // Pick voxel closest to cell's front
    std::vector<double> scaled_direction = cell_radius * direction;
    
    // Calculate the position of the membrane in the direction of the cell
    std::vector<double> position_membrane = pCell->position;
    for (size_t i = 0; i < position_membrane.size(); ++i) 
	{
        position_membrane[i] += scaled_direction[i];
    }

    // Compute the index of the voxel at that position
    int voxel_index_memb = microenvironment.nearest_voxel_index(position_membrane);

    // Ensure voxel index is valid
    if (voxel_index_memb < 0 || voxel_index_memb >= ecm.ecm_voxels.size()) {
        std::cerr << "Invalid voxel index: " << voxel_index_memb << std::endl;
        return;
    }

    double ecm_density = ecm.ecm_voxels[voxel_index_memb].density;
    ecm_density = std::min(ecm_density, 1.0); 
    // std::cout<<"ecm_density: "<<ecm_density<<std::endl;

	// Get total cell's velocity
	std::vector<double> velocity = pCell->velocity;

    // If ECM density is positive, compute repulsion
    if (ecm_density > 0.000001)
    {
        // Assign resulting velocity to cell's velocity
        for (size_t i = 0; i < pCell->velocity.size(); ++i) {
            pCell->velocity[i] -= ecm_density * velocity[i];    
        }
    }

	// if (pCell->velocity[0] == 0 && pCell->velocity[1] == 0)
	// {
	// 	std::cout<<"ecm_density: ";
	// 	std::cout<<ecm_density<<std::endl;
	// 	std::cout<<"direction: ";
	// 	std::cout<<direction<<std::endl;
	// 	std::cout<<"velocity: ";
	// 	std::cout<<velocity<<std::endl;
	// 	std::cout<<std::endl;
	// }

    return; 
}


void ecm_degradation(Cell* pCell , Phenotype& phenotype , double dt) 
{
	/**************** Change in ECM density: degradation ***************/

	// Position of the cell
	std::vector<double> position = pCell->position;

	// Get nearest voxel to cell's position
	int voxel_index = microenvironment.nearest_voxel_index( position );

	//Pick voxel closest to cell's membrane in direction of movement
	std::vector<double> direction = {pCell->custom_data["total_velocity_x"], pCell->custom_data["total_velocity_y"], pCell->custom_data["total_velocity_z"]};
	normalize(&direction);

	std::vector<double> scaled_direction;
	scaled_direction = pCell->phenotype.geometry.radius * direction;

	// Calculating the position of the membrane in the direction of the cell
	std::vector<double> position_membrane = position + scaled_direction;

	if(parameters.strings("nearest_voxel_remodeling") == "front")
	{
		// Computing the index of the voxel at cell front
		voxel_index = microenvironment.nearest_voxel_index( position_membrane );
	}

	// Cell-ECM density interaction
	double ecm_density = ecm.ecm_voxels[voxel_index].density;

	// Get rate of ECM degradation 
	double r_density = pCell->custom_data["ecm_density_rate"]; 

	// Set density target 
	double density_target = parameters.doubles("density_target");

	if (ecm_density > density_target) 
	{
		ecm.ecm_voxels[voxel_index].density = ecm_density + r_density * dt  * (density_target - ecm_density);
	}

    return;

}


void ecm_displacement(Cell* pCell , Phenotype& phenotype , double dt) 
{
	/**************** Change in ECM density: displacement ***************/

	// Get displacement rate of ECM  
	double r_displacement = pCell->custom_data["ecm_displacement_rate"];

	// Cell's velocity vector
	std::vector<double> velocity = {pCell->custom_data["total_velocity_x"], pCell->custom_data["total_velocity_y"], pCell->custom_data["total_velocity_z"]};
	// std::vector<double> velocity = phenotype.motility.motility_vector;

	// Position of the cell
	std::vector<double> position = pCell->position;

	// Get nearest voxel to cell's position
	int voxel_index = microenvironment.nearest_voxel_index(position);

	double ecm_density = ecm.ecm_voxels[voxel_index].density;

	std::vector<double> voxel_center = ecm.ecm_voxels[voxel_index].center;
	
	// Set density target 
	double density_target = parameters.doubles("density_target");

	// Initialize remainder ECM density at position of the cell
	int adjacent_voxel_index;
	double initial_adjacent_ecm_density = 0;
	double adjacent_ecm_density = 0;
	double remainder_ecm_density = 0;

	int ecm_mesh_length = sqrt(ecm.ecm_voxels.size());		

	// Check if ECM density at cell's position is zero and that we are not at the edge of the domain
	if (ecm_density > 0 && 
	voxel_index > ecm_mesh_length && 							// exclude top row
	voxel_index < (ecm.ecm_voxels.size() - ecm_mesh_length) &&  // exclude bottom row
	voxel_index % ecm_mesh_length > 0 && 						// exclude LHS
	voxel_index % ecm_mesh_length < ecm_mesh_length - 1)		// exclude RHS
	{
		double m = velocity[1] / velocity[0];
		double y = 0;
		double ecm_dx = parameters.doubles("ecm_dx");
		// voxel_index zero is on the bottom left
		// Adjacent voxel indices
		// voxel_index + ecm_mesh_length,	// top
		// voxel_index - 1,					// left
		// voxel_index + 1,					// right
		// voxel_index - ecm_mesh_length,	// bottom

		if (velocity[0] > 0)
		{
			// y - y_c = m * (x - x_c)
			// y = y_c + m * (x_v + ecm_dx/2 - x_c)
			y  = position[1] + m * (voxel_center[0] + ecm_dx / 2.0 - position[0]);

			if ( (voxel_center[1] - ecm_dx / 2.0) < y && y < (voxel_center[1] + ecm_dx / 2.0))
			{
				adjacent_voxel_index = voxel_index + 1; // right
			}
			else if (velocity[1] > 0)
			{
				adjacent_voxel_index = voxel_index + ecm_mesh_length; // top
			}
			else if (velocity[1] < 0)
			{
				adjacent_voxel_index = voxel_index - ecm_mesh_length; // bottom
			}
		}
		else if (velocity[0] < 0)
		{			
			y  = position[1] + m * (voxel_center[0] - ecm_dx / 2.0 - position[0]);

			if ( (voxel_center[1] - ecm_dx / 2.0) < y && y < (voxel_center[1] + ecm_dx / 2.0))
			{
				adjacent_voxel_index = voxel_index - 1; // left
			}
			else if (velocity[1] > 0)
			{
				adjacent_voxel_index = voxel_index + ecm_mesh_length; // top
			}
			else if (velocity[1] < 0)
			{
				adjacent_voxel_index = voxel_index - ecm_mesh_length; // bottom
			}
		}
		else if (velocity[1] > 0 || velocity[1] < 0)
		{
			if (velocity[1] > 0)
			{
				adjacent_voxel_index = voxel_index + ecm_mesh_length; // top
			}
			else if (velocity[1] < 0)
			{
				adjacent_voxel_index = voxel_index - ecm_mesh_length; // bottom
			}
		}

		// Make list of adjacent voxel to push the ECM density to
		std::vector<int> adjacent_voxel_indices = {adjacent_voxel_index};

		// std::cout<<"Cell position: "<<position[0]<<" "<<position[1]<<std::endl;
		// std::cout<<"Voxel index: "<<voxel_index<<std::endl;
		// std::cout<<"Voxel center: "<<voxel_center[0]<<" "<<voxel_center[1]<<std::endl;
		// std::cout<<"Cell velocity: "<<velocity[0]<<" "<<velocity[1]<<std::endl;
		// std::cout<<"Adjacent voxel index: "<<adjacent_voxel_index<<std::endl;
		// std::cout<<"Adjacent voxel center: "<<ecm.ecm_voxels[adjacent_voxel_index].center[0]<<" "<<ecm.ecm_voxels[adjacent_voxel_index].center[1]<<std::endl<<std::endl;

		// Iterate through adjacent voxel indices
		for (auto i = 0; i < adjacent_voxel_indices.size(); i++) 
		{
			adjacent_voxel_index = adjacent_voxel_indices[i];

			// Find current ECM density at adjacent voxel 
			initial_adjacent_ecm_density = ecm.ecm_voxels[adjacent_voxel_index].density;
			adjacent_ecm_density = initial_adjacent_ecm_density + ecm_density / adjacent_voxel_indices.size();

			if (adjacent_ecm_density > 1)
			{
				remainder_ecm_density += adjacent_ecm_density - 1;
				adjacent_ecm_density = 1;
			}
		
			ecm.ecm_voxels[adjacent_voxel_index].density = initial_adjacent_ecm_density + r_displacement * dt * (adjacent_ecm_density - initial_adjacent_ecm_density);
		}
		ecm.ecm_voxels[voxel_index].density = ecm_density + r_displacement * dt * (remainder_ecm_density - ecm_density);
	}
	
    return;

}


void ecm_reorientation(Cell* pCell, Phenotype& phenotype, double dt) 
{
    /**************** Cell-ECM Fiber Realignment ***************/

    // Compute the index of the voxel at the cell's position
    int voxel_index = microenvironment.nearest_voxel_index(pCell->position);

    // Get the fiber orientation and anisotropy at the voxel
    std::vector<double> fiber_orientation = ecm.ecm_voxels[voxel_index].ecm_fiber_alignment;
    double anisotropy = ecm.ecm_voxels[voxel_index].anisotropy;

    // Get the cell's migration speed
    double migration_speed = pCell->phenotype.motility.migration_speed;

    // Compute the fiber reorientation rate
    double r_f0 = pCell->custom_data["fiber_reorientation_rate"];
    double r_fiber = r_f0 * migration_speed * (1 - anisotropy);

    // Get the normalized cell motility vector
    std::vector<double> norm_cell_velocity = pCell->phenotype.motility.motility_vector;
    normalize(&norm_cell_velocity);

	// Check if the fiber should be flipped
	double ddotf;
	ddotf = dot_product_ext(fiber_orientation, norm_cell_velocity);
	fiber_orientation = sign_function(ddotf) * fiber_orientation;

    // Update the fiber orientation to align closer to the cell motility vector
    for (int i = 0; i < 3; i++) {
        fiber_orientation[i] += dt * r_fiber * (norm_cell_velocity[i] - fiber_orientation[i]);
    }

    // Normalize the updated fiber orientation
    normalize(&fiber_orientation);

    // Update the ECM voxel's fiber alignment
    ecm.ecm_voxels[voxel_index].ecm_fiber_alignment = fiber_orientation;


	// // Old update method

	// for(int i = 0; i < 3; i++)
	// {
	// 	// if (ddotf<0.0)
	// 	// {
	// 	// 	fiber_orientation = -1.0 * fiber_orientation;
	// 	// }
	// 	ecm.ecm_voxels[voxel_index].ecm_fiber_alignment[i] += dt * r_fiber * (norm_cell_motility[i] - fiber_orientation[i]); 
	// }
	
	// normalize(&(ecm.ecm_voxels[voxel_index].ecm_fiber_alignment)); 

    return;
}


void ecm_anisotropy_increase(Cell* pCell , Phenotype& phenotype , double dt) 
{
	/**************************** Cell-ECM Anisotropy Modification ******************/
	
	// Computing the index of the voxel at cell position
	int voxel_index = microenvironment.nearest_voxel_index( pCell->position );

	// Get anisotrpy at cell's position
	double anisotropy = ecm.ecm_voxels[voxel_index].anisotropy;

	// Get cell migration speed
	double migration_speed = pCell->phenotype.motility.migration_speed;

	// Compute anisotropy update rate 
	double r_a0 = pCell->custom_data["fiber_realignment_rate"]; // min-1
	
	double r_anisotropy = r_a0 * migration_speed;
	
	ecm.ecm_voxels[voxel_index].anisotropy = anisotropy + r_anisotropy * dt * (1- anisotropy);

    return;
}


void proliferation_inhibition( Cell* pCell, Phenotype& phenotype, double dt )
{
	if(phenotype.death.dead == true)
	{
		phenotype.motility.is_motile = false;
		pCell->functions.update_migration_bias = NULL;
		pCell->functions.update_phenotype = NULL;
		//std::cout<<2<<std::endl;
		std::cout<<"Cell is dead"<<std::endl;
	}

	// Proliferation depends on number of cells in contact
	double proliferation_rate = get_single_base_behavior(pCell,"cycle entry");

	// Get threshold neighbours parameter
	double overcrowding_threshold = pCell->custom_data["overcrowding_threshold"];

	// Get cell's neighbours 
	double n_attached =  pCell->state.neighbors.size();

	static int substrate_index = microenvironment.find_density_index( "substrate" ); 

	double substrate_density = pCell->nearest_density_vector()[substrate_index];

	double substrate_density_threshold = pCell->custom_data["substrate_density_threshold"];

	proliferation_rate *= Hill_response_function(substrate_density, 21.5, 4);

	double pressure = get_single_signal( pCell , "pressure");

	// Calculate cell spacing based on adhesion and repulsion parameters
	double cell_radius = cell_defaults.phenotype.geometry.radius;
	double relative_maximum_adhesion_distance = cell_defaults.phenotype.mechanics.relative_maximum_adhesion_distance;
	double cell_adhesion = cell_defaults.phenotype.mechanics.cell_cell_adhesion_strength;
	double cell_repulsion = cell_defaults.phenotype.mechanics.cell_cell_repulsion_strength;
	double sqrt_adhesion_to_repulsion_ratio = sqrt(cell_adhesion / cell_repulsion);
	double cell_spacing = (1 - sqrt_adhesion_to_repulsion_ratio);
	cell_spacing /= (0.5 * 1 / cell_radius - 0.5 * sqrt_adhesion_to_repulsion_ratio / (relative_maximum_adhesion_distance * cell_radius));

	double distance;
	distance = cell_spacing; // 2 * phenotype.geometry.nuclear_radius;
	double R = 2 * phenotype.geometry.radius;
	double simple_pressure_scale = 0.027288820670331;

	double temp_r = -distance; // -d
	temp_r /= R; // -d/R
	temp_r += 1.0; // 1-d/R
	temp_r *= temp_r; // (1-d/R)^2 

	// Pressure equivalent to 6 cells at equilibrium distance
	// pressure = 6 * (1 - d/R)^2 / simple_pressure_scale
	double pressure_threshold = 6*(temp_r/simple_pressure_scale);

	// Arrest proliferation 
	if( pressure > pressure_threshold)
	{
		proliferation_rate = 0;
	}
	
	// Set new proliferation rate
	set_single_behavior(pCell,"cycle entry", proliferation_rate);
}


void custom_update_cell_velocity( Cell* pCell, Phenotype& phenotype, double dt)
{

	// Update cell-ECM adhesion speed
	cell_ecm_adhesion_speed(pCell, phenotype, dt);

	// Assign speed to motility vector with same motility direction
	pCell->phenotype.motility.motility_vector = pCell->phenotype.motility.migration_bias_direction; 
	normalize( &(pCell->phenotype.motility.motility_vector) ); 

	pCell->phenotype.motility.motility_vector *= pCell->phenotype.motility.migration_speed;

	// Calling the standard update velocity of PhysiCell (updates cell-ECM adhesion direction and cell-cell adhesion and repulsion)
	standard_update_cell_velocity(pCell, phenotype, dt);

	// Update cell-ECM repulsion
	cell_ecm_repulsion(pCell, phenotype, dt);

	/************************* SAVE TOTAL SPEED AND VELOCITY DATA **************************/
	// Compute total cell's speed and velocity
	double total_speed = sqrt(pow(pCell->velocity[0],2) + pow(pCell->velocity[1],2) + pow(pCell->velocity[2],2));
	// std::cout<<"total_speed: "<<total_speed<<std::endl;
	pCell->custom_data["total_speed"] = total_speed;
	pCell->custom_data["total_velocity_x"] = pCell->velocity[0];
	pCell->custom_data["total_velocity_y"] = pCell->velocity[1];
	pCell->custom_data["total_velocity_z"] = pCell->velocity[2];

    pCell->custom_data["attached_cells"] = pCell->state.neighbors.size();

	// Remodel the ECM
	ecm_remodelling(pCell,phenotype,dt);
	
	return; 
}

void ecm_remodelling(Cell* pCell , Phenotype& phenotype , double dt) 
{
	if(phenotype.death.dead == true)
	{
		phenotype.motility.is_motile = false;
		pCell->functions.update_migration_bias = NULL;
		pCell->functions.update_phenotype = NULL;
		std::cout<<"Cell is dead"<<std::endl;
				
		return; 
	}

	ecm_degradation(pCell,phenotype,dt);
	ecm_displacement(pCell,phenotype,dt);

	/************************* ECM remodeling with fibers ************************/
	if(parameters.strings("ecm_definition") == "ecm_fibers")
	{
		ecm_reorientation(pCell,phenotype,dt);
		ecm_anisotropy_increase(pCell,phenotype,dt);
	}

    return;

}

void phenotype_function( Cell* pCell, Phenotype& phenotype, double dt )
{ return; }

void custom_function( Cell* pCell, Phenotype& phenotype , double dt )
{ 

	proliferation_inhibition(pCell,phenotype,dt);
	
	return;
} 

void contact_function( Cell* pMe, Phenotype& phenoMe , Cell* pOther, Phenotype& phenoOther , double dt )
{ return; } 

void write_ecm_data_matlab( std::string filename )
{
    int number_of_data_entries = ecm.ecm_mesh.voxels.size();
    int size_of_each_datum = 8;

	// static int ecm_anisotropy_index = microenvironment.find_density_index( "ECM anisotropy" ); 
	// static int ecm_density_index = microenvironment.find_density_index( "ECM" ); 

    FILE* fp = write_matlab_header( size_of_each_datum, number_of_data_entries,  filename, "ECM_Data" );  // Note - the size of datum needs to correspond exaectly to the lines of output or there is an error upon importing.

    for( int i=0; i < number_of_data_entries ; i++ )
    {
	    fwrite( (char*) &( ecm.ecm_mesh.voxels[i].center[0] ) , sizeof(double) , 1 , fp ); // 1
        fwrite( (char*) &( ecm.ecm_mesh.voxels[i].center[1] ) , sizeof(double) , 1 , fp ); // 2
        fwrite( (char*) &( ecm.ecm_mesh.voxels[i].center[2] ) , sizeof(double) , 1 , fp ); //3
		fwrite( (char*) &( ecm.ecm_voxels[i].anisotropy), sizeof(double) , 1 , fp ); // 4
        fwrite( (char*) &( ecm.ecm_voxels[i].density), sizeof(double) , 1 , fp ); // 5
        fwrite( (char*) &( ecm.ecm_voxels[i].ecm_fiber_alignment[0]), sizeof(double) , 1 , fp ); // 6
        fwrite( (char*) &( ecm.ecm_voxels[i].ecm_fiber_alignment[1]), sizeof(double) , 1 , fp ); // 7
        fwrite( (char*) &( ecm.ecm_voxels[i].ecm_fiber_alignment[2]), sizeof(double) , 1 , fp ); // 8
		// This will only work if the diffusion and ECM meshes are the same size. Commenting out for actualrunning. To do a direct comparison, leave them in and change length. Will have to change the vizualization to get this from the regular BioFVM outputs.
		// fwrite( (char*) &( microenvironment.gradient_vector(i)[0][0]), sizeof(double) , 1 , fp ); // 9
		// fwrite( (char*) &( microenvironment.gradient_vector(i)[0][1]), sizeof(double) , 1 , fp ); // 10
		// fwrite( (char*) &( microenvironment.gradient_vector(i)[0][2]), sizeof(double) , 1 , fp ); // 11
    }

    fclose( fp );

    return;
}