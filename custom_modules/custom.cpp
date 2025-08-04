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

using namespace BioFVM; 
using namespace PhysiCell;

// extern ECM ecm;

ECM ecm;

void create_cell_types( void )
{
	// set the random seed 
	//SeedRandom( parameters.ints("random_seed") );  
	
	/* 
	   Put any modifications to default cell definition here if you 
	   want to have "inherited" by other cell types. 
	   
	   This is a good place to set default functions. 
	*/ 

	initialize_default_cell_definition(); 
	
	cell_defaults.phenotype.secretion.sync_to_microenvironment( &microenvironment ); 
	
	cell_defaults.functions.update_velocity = custom_update_cell_velocity;
	cell_defaults.functions.custom_cell_rule = custom_function; 
	cell_defaults.functions.update_migration_bias = cell_ecm_interaction_motility_direction; 

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

	cell_defaults.custom_data.add_variable( "ribose_concentration", "mM" , parameters.doubles( "ribose_concentration") );
	
	// cell_defaults.functions.update_phenotype = phenotype_function; 
	// cell_defaults.functions.custom_cell_rule = custom_function; 
	// cell_defaults.functions.contact_function = contact_function; 
	// cell_defaults.functions.update_migration_bias = cell_ecm_interaction_motility_direction; 

	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/

	// cell_defaults.functions.update_velocity = custom_update_cell_velocity;
	
		
	// display_cell_definitions( std::cout ); 
	
	return; 
}

// CHECKED
void setup_microenvironment( void )
{
	// set domain parameters 
	
	// put any custom code to set non-homogeneous initial conditions or 
	// extra Dirichlet nodes here. 
	
	// initialize BioFVM 

	initialize_microenvironment(); 

	return; 
}

// TO CHECK
void setup_extracellular_matrix( void )
{
	// DEPENDS ON MICROENVIRONMENT - CALL SETUP MICROENVIRONEMNT FIRST!!!!!

	ecm.ecm_mesh.resize(default_microenvironment_options.X_range[0], default_microenvironment_options.X_range[1] , 
	default_microenvironment_options.Y_range[0], default_microenvironment_options.Y_range[1],default_microenvironment_options.Z_range[0], default_microenvironment_options.Z_range[1], \
	parameters.doubles("ecm_dx"), parameters.doubles("ecm_dy"),parameters.doubles("ecm_dz"));
	ecm.resize_ecm_units_from_ecm_mesh();

	// ecm.ecm_mesh.display_information(std::cout );

	// set up ECM alignment 

	// <ecm_orientation_setup description="Specifies the initial ECM orientation: random, circular, starburt, oriented to the right, or oriented to the top" type="string" units="NA">circular</ecm_orientation_setup> parameters.string( "ecm_orientation_setup")
	// std::vector<double> fiber_direction = { 1.0 , 0.0, 0.0 }; 
	// ecm_fiber_alignment.resize( microenvironment.mesh.voxels.size() , fiber_direction );  

	for( int n = 0; n < ecm.ecm_mesh.voxels.size() ; n++ )
	{
		// For random 2-D initalization 
		if(parameters.strings( "ecm_orientation_setup") == "random")
		{
			double theta = 6.2831853071795864769252867665590 * uniform_random(); 
			// ecm.ecm_data[i].ecm_orientation[0] = cos(theta);
			// ecm.ecm_data[i].ecm_orientation[1] = sin(theta);
			// ecm.ecm_data[i].ecm_orientation[2] = 0.0;
			ecm.ecm_voxels[n].ecm_fiber_alignment = {cos(theta), sin(theta), 0.0};
		}

		// for starburst initialization 
		else if(parameters.strings( "ecm_orientation_setup") == "starburst")
		{
			std::vector<double> position = ecm.ecm_mesh.voxels[n].center; 
			normalize( &position ); 
			ecm.ecm_voxels[n].ecm_fiber_alignment =  { position[0],position[1],0}; // oriented out (perpindeicular to concentric circles)
			normalize(&ecm.ecm_voxels[n].ecm_fiber_alignment);
		}

		// for circular initialization 
		else if(parameters.strings( "ecm_orientation_setup") == "circular")
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
			normalize( &position ); 

			if(position[1]<=0)
			{
				ecm.ecm_voxels[n].ecm_fiber_alignment =  { 1,-1,0}; // oriented out (perpindeicular to concentric circles)
				normalize(&ecm.ecm_voxels[n].ecm_fiber_alignment);
			}

			else
			{
				ecm.ecm_voxels[n].ecm_fiber_alignment =  { 1,1,0}; // oriented out (perpindeicular to concentric circles)
				normalize(&ecm.ecm_voxels[n].ecm_fiber_alignment);				
			}

		}

		else
		{
			std::cout<<"WARNING: NO ECM ORIENTATION SPECIFIED. FIX THIS!!!"<<std::endl;
			std::cout<<"Halting program!!!"<<std::endl;
			abort();
			return;
		}

		ecm.ecm_voxels[n].density = parameters.doubles("initial_ecm_density");
		ecm.ecm_voxels[n].anisotropy = parameters.doubles("initial_anisotropy");
		if(parameters.doubles("initial_ecm_density")>1 || parameters.doubles("initial_anisotropy")>1 || parameters.doubles("initial_ecm_density")<0 || parameters.doubles("initial_anisotropy")<0)
		{
			std::cout<<"WARNING: INITIAL DENSITY OR ANISOTROPY OUT OF BOUNDS! FIX THIS!"<<std::endl;
			std::cout<<"Halting program!!!"<<std::endl;
			abort();
			return;
		}		
	}
	
	return;
}


// TO CHECK
void setup_tissue( void )
{
	// double Xmin = microenvironment.mesh.bounding_box[0]; 
	// double Ymin = microenvironment.mesh.bounding_box[1]; 
	// double Zmin = microenvironment.mesh.bounding_box[2]; 

	// double Xmax = microenvironment.mesh.bounding_box[3]; 
	// double Ymax = microenvironment.mesh.bounding_box[4]; 
	// double Zmax = microenvironment.mesh.bounding_box[5]; 
	
	// if( default_microenvironment_options.simulate_2D == true )
	// {
	// 	Zmin = 0.0; 
	// 	Zmax = 0.0; 
	// }
	
	// double Xrange = Xmax - Xmin; 
	// double Yrange = Ymax - Ymin; 
	// double Zrange = Zmax - Zmin; 
	
	// // create some of each type of cell 
	
	// Cell* pC;
	
	// for( int k=0; k < cell_definitions_by_index.size() ; k++ )
	// {
	// 	Cell_Definition* pCD = cell_definitions_by_index[k]; 
	// 	std::cout << "Placing cells of type " << pCD->name << " ... " << std::endl; 
	// 	for( int n = 0 ; n < parameters.ints("number_of_cells") ; n++ )
	// 	{
	// 		std::vector<double> position = {0,0,0}; 
	// 		position[0] = Xmin + UniformRandom()*Xrange; 
	// 		position[1] = Ymin + UniformRandom()*Yrange; 
	// 		position[2] = Zmin + UniformRandom()*Zrange; 
			
	// 		pC = create_cell( *pCD ); 
	// 		pC->assign_position( position );
	// 	}
	// }
	// std::cout << std::endl; 
	
	// // load cells from your CSV file (if enabled)
	// load_cells_from_pugixml(); 	
	

	// Setting seed so cells always start with same initial configuration
	SeedRandom(parameters.ints("random_seed") );  
	

	if(parameters.strings("cell_setup") == "single")
	{
		Cell* pCell;

		pCell = create_cell(cell_defaults);

		pCell->assign_position(0.0, 0.0, 0.0);
		// std::cin.get();
		

	}

	/*******************************************Random initialization****************************************/
		
	else if(parameters.strings("cell_setup") == "random")
	{
		std::cout<<"string worked"<<std::endl;
	
		Cell* pC;
		
		for( int n = 0 ; n < 200 ; n++ )
		{
			pC = create_cell(); 
			pC->assign_position( -450 + 900*UniformRandom() , -450 + 900*UniformRandom() , 0.0 );
		}

		std::cout<<"Cell's placed randomly on domain - this function uses a HARD CODED domain size!!!! WARNING!!!!!"<<std::endl;
		std::cout<<" hit Enter to continue:"<<std::flush;
		std::cin.get();

	}
	
	/******************************************2D Spheroid initialization***************************************/
	
	else if(parameters.strings("cell_setup") == "lesion")
	{
		// place a cluster of tumor cells at the center
	
		//Get tumor radius from XML parameters

		double tumor_radius = parameters.doubles("tumor_radius"); 
		// double number_of_cells = parameters.ints("number_of_cells");
		//std::cout<<"number_of_cells: "<<number_of_cells<<std::endl;

		// if( parameters.ints("unit_test_setup") == 1)
		// {
		// 	tumor_radius = 150;
		// }

		// else
		// {
		// 	tumor_radius = parameters.doubles("tumor_radius");
		// }


		// these lines produce automatically calcuated equilibirum spacing for intiailizing cells, even when changing adh-rep parameters.

		double cell_radius = cell_defaults.phenotype.geometry.radius;
		// std::cout<<"cell_radius: "<<cell_radius<<std::endl;
		// tumor_radius = sqrt( pow(cell_radius,2) * number_of_cells );
		// std::cout<<"tumor_radius: "<<tumor_radius<<std::endl;

		double relative_maximum_adhesion_distance = cell_defaults.phenotype.mechanics.relative_maximum_adhesion_distance;
		// std::cout<<"relative_maximum_adhesion_distance: "<<relative_maximum_adhesion_distance<<std::endl; //relative_maximum_adhesion_distance;
		double cell_adhesion = cell_defaults.phenotype.mechanics.cell_cell_adhesion_strength;
		// std::cout<<"cell_adhesion: "<<cell_adhesion<<std::endl; 
		double cell_repulsion = cell_defaults.phenotype.mechanics.cell_cell_repulsion_strength;
		// std::cout<<"cell_repulsion: "<<cell_repulsion<<std::endl; 
		double sqrt_adhesion_to_repulsion_ratio = sqrt(cell_adhesion/cell_repulsion);
		
		double cell_spacing = (1 - sqrt_adhesion_to_repulsion_ratio);
		cell_spacing /= (0.5 * 1/cell_radius - 0.5 * sqrt_adhesion_to_repulsion_ratio/(relative_maximum_adhesion_distance * cell_radius));
		// std::cout<<"cell_spacing: "<<cell_spacing<<std::endl;

		
		Cell* pCell = NULL; 
		

		double x = 0.0;
		double x_outer = tumor_radius; 
		double y = 0.0;


		// if( parameters.ints("unit_test_setup") == 1 && parameters.ints("march_unit_test_setup") == 0)
		// {
		// 	cell_spacing = 1.90 * cell_radius;
		// }

		int n = 0; 
		while( y < tumor_radius )
		{
			x = 0.0; 
			if( n % 2 == 1 )
			{ x = 0.5*cell_spacing; }
			x_outer = sqrt( tumor_radius*tumor_radius - y*y ); 
			
			while( x < x_outer )
			{
				//std::cout<<"x: "<<x;
				//std::cout<<", y:"<<y<<std::endl;
				pCell = create_cell(cell_defaults);
					
				pCell->assign_position( x , y , 0.0 );
				set_single_behavior(pCell,"cycle entry", 0);
				
				if( fabs( y ) > 0.01 )
				{
					pCell = create_cell(cell_defaults);
					pCell->assign_position( x , -y , 0.0 );
					set_single_behavior(pCell,"cycle entry", 0);
				}
				
				if( fabs( x ) > 0.01 )
				{ 
					pCell = create_cell(cell_defaults); 
					pCell->assign_position( -x , y , 0.0 );
					set_single_behavior(pCell,"cycle entry", 0);
					
					if( fabs( y ) > 0.01 )
					{
						pCell = create_cell(cell_defaults);
						pCell->assign_position( -x , -y , 0.0 );
						set_single_behavior(pCell,"cycle entry", 0);
					}
				}
				x += cell_spacing; 
				
			}
			
			y += cell_spacing * sqrt(3.0)/2.0; 
			n++; 
		}

		std::cout<<"Cell's placed in 2-lesion at center of domain"<<std::endl;
	}

	

	/******************************************3D Spheroid initialization***************************************/

	/*To come later*/

	/************************************Circle of cells at R = 300 initialization***************************************/

	else if(parameters.strings("cell_setup") == "circle of cells")
	{
		double theta2 = 0.0;
		for (int a = 0; a<42; a++)
		{
			Cell* pCell = NULL;
			pCell = create_cell(cell_defaults); 
			pCell->assign_position( 300 * cos(theta2) , 300 * sin(theta2) , 0.0 );
			theta2 += 0.14959952;
		}

	}

	/************************************Line of cells at y = 0, x > 0 initialization***************************************/

	else if(parameters.strings("cell_setup") == "cells at y = 0")
	{

		Cell* pCell = NULL; 
		int n = default_microenvironment_options.X_range[0] + 10.0; 
		while( n <= default_microenvironment_options.X_range[1] )
		{
			pCell = create_cell(cell_defaults); 

			// To prevent droping cells in areas of high ECM curvature. 
			while(abs(n) < 70)
			{n = n + 30.0;}

			pCell->assign_position( n , 0.0 , 0.0 );
			n = n + 30.0;
		}
		std::cout<<"Cell's placed at y = 0"<<std::endl;
	}

	/******************************************Line of cells at x = left boundary + 10 initialization***************************************/
	else if(parameters.strings("cell_setup") == "cells at left boundary/march")
	{
		
		Cell* pCell = NULL; 
		int n = default_microenvironment_options.X_range[0] + 10.0; 
		while( n <= default_microenvironment_options.X_range[1] - 10.0 )
		{
			pCell = create_cell(cell_defaults);
			pCell->assign_position( default_microenvironment_options.X_range[0] + 10.0 , n , 0.0 );
			n = n + 10.0;
		}
		std::cout<<"Cell's placed at left boundary for march test"<<std::endl;
	}

	else
	{
		std::cout<<"WARNING!!! NO CELL SETUP SPECIFIED. SEE DOCUMENTATION and FIX"<<std::endl;
		std::cout<<"Halting program!!!"<<std::endl;
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

double dot_product_ext( const std::vector<double>& v , const std::vector<double>& w )
{
	double out = 0.0; 
	for( unsigned int i=0 ; i < v.size() ; i++ )
	{ out += ( v[i] * w[i] ); }

	if( fabs(out) < 1e-10)
	{out = 0.0;}

	return out; 
}

double sign_function (double number)
{
	// double sign = 0.0
	if (number<0)
	{ return -1.0;}

	else
	{ return 1.0;}

}

void cell_ecm_interaction_motility_speed( Cell* pCell, Phenotype& phenotype, double dt )
{
	// Check if cell is dead
	if(phenotype.death.dead == true)
	{
		phenotype.motility.is_motile = false;
		pCell->functions.update_migration_bias = NULL;
		pCell->functions.update_phenotype = NULL;
		//std::cout<<2<<std::endl;
		std::cout<<"Cell is dead"<<std::endl;
	}

	// Updates cell bias vector and cell speed based on the ECM density, anisotropy, and fiber direction

	int voxel_index = 0;
	if (parameters.strings("nearest_voxel_speed") == "membrane")
	{
		//Pick voxel closest to cell's membrane in direction of movement
		std::vector<double> direction = {pCell->custom_data["total_velocity_x"], pCell->custom_data["total_velocity_y"], pCell->custom_data["total_velocity_z"]};
		std::vector<double> scaled_direction = phenotype.geometry.radius * normalize(direction);
		
		// Calculating the position of the membrane in the direction of the cell
		std::vector<double> position_membrane = pCell->position + scaled_direction;
		pCell->custom_data["point_on_membrane_x"] = position_membrane[0];
		pCell->custom_data["point_on_membrane_y"] = position_membrane[1];
		pCell->custom_data["point_on_membrane_z"] = position_membrane[2];
		
		// Computing the index of the voxel at that position
		voxel_index = microenvironment.nearest_voxel_index( position_membrane );
	}

	else if (parameters.strings("nearest_voxel_speed") == "position")
	{
		// Computing the index of the voxel at cell position
		voxel_index = microenvironment.nearest_voxel_index( pCell->position );
	}

	else
	{
		std::cout<<"Nearest voxel for computing migration speed not defined!"<<std::endl;
		abort();
		return;
	}


	// Finding out the ECM density
	double ecm_density = ecm.ecm_voxels[voxel_index].density;

	if(parameters.bools("normalize_ecm_influenced_motility_vector") == true)
	{
		// if the vector is to be normalized, we, by definition, already know the magnitude will be 1.0
		pCell->phenotype.motility.migration_speed = 1.0;
	}

	else
	{
		pCell->phenotype.motility.migration_speed = norm( phenotype.motility.migration_bias_direction);
		// std::cout<<"Magnitude of motility vector is "<< pCell->phenotype.motility.migration_speed<<std::endl;
	}
	
	// std::cout<<"migration_speed: "<<pCell->phenotype.motility.migration_speed<<std::endl;

	// We MUST set migration bias at 1.0 so that standard update_motility function doesn't add random motion. 
	phenotype.motility.migration_bias = 1.0; 

	// ECM density values for speed function
	double rho_low = pCell->custom_data["rho_low"];
	double rho_high = pCell->custom_data["rho_high"];
	double rho_ideal = pCell->custom_data["rho_ideal"];

	// Relation between ribose concentration and density

	// Get ribose concentration
	double ribose_concentration = parameters.doubles("ribose_concentration");
	
	// // Ribose coefficient
	// double c_ribose = 1 - atan(ribose_concentration / 100) * 2 / pi;

	// Get value of max migration speed
	double max_migration_speed = get_single_base_behavior(pCell,"migration speed");
	// std::cout<<"max_migration_speed: "<<max_migration_speed<<std::endl;

	// Compute cell speed
	if (ecm_density <= rho_low)
	{
		pCell->phenotype.motility.migration_speed = 0.0;
	}

	else if (rho_low < ecm_density && ecm_density <= rho_ideal)
	{
		// for base speed: y - y_1 = m (x - x_1) or y = m (x - x_1) + y_1
		// Assuming that y_1 = 0 --> y = m (x - x_1)
		// m = rise/run = (speed(rho_ideal) - speed(rho_l)/(rho_ideal - rho_l)). Same for rho_h
		// Assuming that speed(rho_ideal) = 1.0 and speed(rho_l (or rho_h)) = 0.0, m = 1/(rho_ideal - rho_l)
		// y = 1/(x_2 - x_1) * (x - x_1) --> speed_base = 1/(rho_ideal - rho_l) * (rho - rho_l)
		// So finally: speed = normalized_value * max_speed * rules_modifier * (1/(rho_ideal - rho_l) * (rho - rho_l))

		// pCell->phenotype.motility.migration_speed = get_single_base_behavior( pCD, "migration speed" ) * ( 1/(rho_ideal - rho_high) * (ECM_density - rho_high)); 

		pCell->phenotype.motility.migration_speed *= max_migration_speed; 
		pCell->phenotype.motility.migration_speed *= ( 1/(rho_ideal - rho_low) * (ecm_density - rho_low)); 
	}

	else if (rho_ideal < ecm_density && ecm_density < rho_high )
	{

		// for base speed: y - y_1 = m (x - x_1) or y = m (x - x_1) + y_1
		// Assuming that y_1 = 0 --> y = m (x - x_1)
		// m = rise/run = (speed(rho_ideal) - speed(rho_l)/(rho_ideal - rho_l)). Same for rho_h
		// Assuming that speed(rho_ideal) = 1.0 and speed(rho_l (or rho_h)) = 0.0, m = 1/(rho_ideal - rho_l)
		// y = 1/(x_2 - x_1) * (x - x_1) --> speed_base = 1/(rho_ideal - rho_l) * (rho - rho_l)
		// So finally: speed = normalized_value * max_speed * rules_modifier * (1/(rho_ideal - rho_l) * (rho - rho_l))

		// pCell->phenotype.motility.migration_speed = get_single_base_behavior( pCD, "migration speed" ) * ( 1/(rho_ideal - rho_high) * (ECM_density - rho_high)); 
		
		pCell->phenotype.motility.migration_speed *= max_migration_speed; 
		pCell->phenotype.motility.migration_speed *= ( 1/(rho_ideal - rho_high) * (ecm_density - rho_high)); 
	}

	else //if (f_stiff >= rho_high)
	{
		pCell->phenotype.motility.migration_speed = 0.0;
	}	

	// std::cout<<"migration_speed: "<<pCell->phenotype.motility.migration_speed<<std::endl;

	return; 
}

void cell_ecm_interaction_motility_direction( Cell* pCell, Phenotype& phenotype, double dt )
{
	if(phenotype.death.dead == true)
	{
		phenotype.motility.is_motile = false;
		pCell->functions.update_migration_bias = NULL;
		pCell->functions.update_phenotype = NULL;
		//std::cout<<2<<std::endl;
		std::cout<<"Cell is dead"<<std::endl;
	}

	// Select random direction (for random portion of motility vector) and begin building updated motility direction vector
	// (note - there is NO memeory of previous direction in this model - previous ECM-based motility used the current 
	// velocity vector to build off, not a random one - this could produce divergent behaviors between models)

	// See lab note book for more notes - MUST start with random vector. In the old method I defintiely used the previous motility vector in the method, but makes no sense here!

	int voxel_index = 0;
	if (parameters.strings("nearest_voxel_direction") == "membrane")
	{
		//Pick voxel closest to cell's membrane in direction of movement
		std::vector<double> direction = {pCell->custom_data["total_velocity_x"], pCell->custom_data["total_velocity_y"], pCell->custom_data["total_velocity_z"]};
		std::vector<double> scaled_direction = phenotype.geometry.radius * normalize(direction);
		
		// Calculating the position of the membrane in the direction of the cell
		std::vector<double> position_membrane = pCell->position + scaled_direction;
		pCell->custom_data["point_on_membrane_x"] = position_membrane[0];
		pCell->custom_data["point_on_membrane_y"] = position_membrane[1];
		pCell->custom_data["point_on_membrane_z"] = position_membrane[2];
		
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
	

	// // Computing the index of the voxel at cell position
	// int voxel_index = microenvironment.nearest_voxel_index( pCell->position );

	// Finding out the ECM density, anisotropy and fibre orientation
	double ecm_density = ecm.ecm_voxels[voxel_index].density;
	double anisotropy = ecm.ecm_voxels[voxel_index].anisotropy; 
	std::vector<double> fiber_orientation = ecm.ecm_voxels[voxel_index].ecm_fiber_alignment;

	// Get ECM chemotaxis bias and cell sensitivity for ECM 
	double ecm_chemotaxis_bias = pCell->custom_data["ecm_chemotaxis_bias"];
	double ecm_sensitivity = pCell->custom_data["ecm_sensitivity"];

	// get random vector - cell's "intended" or chosen random direction
	std::vector<double> d_random(3,0.0);
	if( phenotype.motility.restrict_to_2D == true )
	{ 
		d_random = UniformOnUnitCircle(); 
	}
	else
	{ 
		d_random = UniformOnUnitSphere(); 
	}

	std::vector<double> d_pref;

	if(ecm_chemotaxis_bias == 0)
	{
		// Preferred direction given by random vector
		d_pref = d_random;
	}
	else
	{
		// get vector for chemotaxis (sample uE)
		static int substrate_index = microenvironment.find_density_index( "substrate" ); 
		std::vector<double> chemotaxis_grad = pCell->nearest_gradient(substrate_index);
		normalize( &chemotaxis_grad ); 

		//combine cell chosen random direction and chemotaxis direction (like standard update_motility function)
		d_pref = (1 - ecm_chemotaxis_bias) * d_random + ecm_chemotaxis_bias * chemotaxis_grad;
	}

	normalize( &d_pref ); 

	std::vector<double> motility_direction;
	if (parameters.strings("ecm_definition") == "ecm_density")
	{
		motility_direction = d_pref;
	}
	else if(parameters.strings("ecm_definition") == "ecm_fibers")
	{
		// to determine direction along fiber_orientation, find part of d_choice that is perpendicular to fiber_orientation; 
		std::vector<double> d_perp = d_pref - dot_product_ext(d_pref, fiber_orientation) * fiber_orientation; 
		normalize( &d_perp ); 
		
		// find constants to span d_choice with d_perp and fiber_orientation
		double c_1 = dot_product_ext( d_pref , d_perp ); 
		double c_2 = dot_product_ext( d_pref, fiber_orientation ); 

		// calculate bias away from directed motility - combination of sensitity to ECM and anisotropy
		double gamma = ecm_sensitivity * anisotropy; // at low values, directed motility vector is recoved. At high values, fiber direction vector is recovered.
		
		// Compute motility direction
		motility_direction = (1.0 - gamma) * c_1 * d_perp + c_2 * fiber_orientation;
	}
	else
	{
		std::cout<<"ECM definition no declared!!!"<<std::endl;
		abort();
		return;
	}

	pCell->phenotype.motility.migration_bias_direction = motility_direction;
	// normalize( &(phenotype.motility.motility_vector) );

	return;
}

void ecm_update_from_cell_velocity(Cell* pCell , Phenotype& phenotype , double dt) 
{
	int voxel_index = microenvironment.nearest_voxel_index( pCell->position );

	// Relation between ribose concentration and density
	double ribose_concentration = parameters.doubles("ribose_concentration");
	// Ribose coefficient
	double pi = 3.1415926535897932384;
	double c_ribose = 1 - atan(ribose_concentration / 50) * 2 / pi;


	/**************** Change in ECM density***************/

	// Get rate of ECM degradation (dependent on ribose concentration)
	double r_density = pCell->custom_data["ecm_density_rate"]; // * c_ribose; // 1.0;
	// if(PhysiCell_globals.current_time == 0)
	// 	std::cout<<"r_density = "<<r_density<<std::endl;

	// Set density target to zero
	double density_target = 0;
	// double density_target = pCell->custom_data["rho_ideal"];

	// Get threshold neighbours parameter
	int overcrowding_threshold = pCell->custom_data["overcrowding_threshold"];
	
	if(pCell->state.neighbors.size() < overcrowding_threshold)
	{
		density_target = pCell->custom_data["rho_ideal"];
	}

	//Pick voxel closest to cell's membrane in direction of movement
	std::vector<double> direction = {pCell->custom_data["total_velocity_x"], pCell->custom_data["total_velocity_y"], pCell->custom_data["total_velocity_z"]};
	std::vector<double> scaled_direction = phenotype.geometry.radius * normalize(direction);

	// Calculating the position of the membrane in the direction of the cell
	std::vector<double> position_membrane = pCell->position + scaled_direction;
	pCell->custom_data["point_on_membrane_x"] = position_membrane[0];
	pCell->custom_data["point_on_membrane_y"] = position_membrane[1];
	pCell->custom_data["point_on_membrane_z"] = position_membrane[2];

	if(parameters.strings("nearest_voxel_remodeling") == "membrane")
	{
		// Computing the index of the voxel at that position
		voxel_index = microenvironment.nearest_voxel_index( position_membrane );
	}

	// std::cout<<"neighbours = "<<pCell->state.neighbors.size()<<", density_target = "<<density_target<<std::endl;
	// std::cout<<"membrane voxel = "<<microenvironment.nearest_voxel_index( position_membrane )<<", position voxel = "<<microenvironment.nearest_voxel_index( pCell->position )<<" nearest voxel = "<<voxel_index<<std::endl;

    // Cell-ECM density interaction
    double ecm_density = ecm.ecm_voxels[voxel_index].density;
	
	ecm.ecm_voxels[voxel_index].density = ecm_density + r_density * dt  * (density_target - ecm_density);
	
	if(parameters.strings("ecm_definition") == "ecm_fibers")
	{

		/**************** Cell-ECM Fiber realingment***************/

		// Computing the index of the voxel at cell position
		voxel_index = microenvironment.nearest_voxel_index( pCell->position );

		// Get index for accessing the ecm_fiber_alignment data structure and then copy the correct value
		std::vector<double> fiber_orientation = ecm.ecm_voxels[voxel_index].ecm_fiber_alignment; 
		double anisotropy = ecm.ecm_voxels[voxel_index].anisotropy;

		// Get cell migration speed
		double migration_speed = pCell->phenotype.motility.migration_speed;

		// Compute fibre realignment rate
		double r_f0 = pCell->custom_data["fiber_realignment_rate"];// * c_ribose; //* (1 - 0.99 * c_ribose ); // * (2 - c_ribose) /2; // (2 + c_ribose) / 3; // 1.0;
		double r_fiber = r_f0 * migration_speed * (1 - anisotropy);

		double ddotf;
		std::vector<double> norm_cell_motility; // = pCell->velocity;
		norm_cell_motility.resize(3,0.0);
		norm_cell_motility = phenotype.motility.motility_vector;
		// std::cout<<"Motility vector: "<< norm_cell_motility<<std::endl;
		normalize(&norm_cell_motility);
		
		ddotf = dot_product_ext(fiber_orientation, norm_cell_motility);
		
		// if(PhysiCell_globals.current_time >600)
		// {
		// 	std::cout<<"Motility vector nomalized: "<< norm_cell_motility<<std::endl;
		// 	std::cout<<"ddotf: "<< ddotf<<std::endl;
		// 	std::cout<<"fiber orientation before: "<< ecm.ecm_voxels[voxel_index].ecm_fiber_alignment<<std::endl;
		// }

		// flips the orientation vector so that it is aligned correctly with the moving cell for proper reoirentation later.
		fiber_orientation = sign_function(ddotf) * fiber_orientation; 

		std::vector<double> f_minus_d;
		f_minus_d.resize(3,0.0);

		for(int i = 0; i < 3; i++)
		{
			// if (ddotf<0.0)
			// {
			// 	fiber_orientation = -1.0 * fiber_orientation;
			// }
			f_minus_d[i] = fiber_orientation[i] - norm_cell_motility[i]; 
			ecm.ecm_voxels[voxel_index].ecm_fiber_alignment[i] -= dt * r_fiber * f_minus_d[i]; 
		}
		
		normalize(&(ecm.ecm_voxels[voxel_index].ecm_fiber_alignment)); 
		// if(PhysiCell_globals.current_time >600)
		// {
		// 	std::cout<<"f_minus_d: "<< f_minus_d<<std::endl;
		// 	std::cout<<"fiber orientation after: "<< ecm.ecm_voxels[voxel_index].ecm_fiber_alignment<<std::endl<<std::endl;
		// }
	
		/**************************** Cell-ECM Anisotropy Modification ******************/

		// Compute anisotropy update rate (dependent on ribose concentration)
		double r_a0 = pCell->custom_data["anisotropy_increase_rate"];// * c_ribose; // * (2 + c_ribose) / 3; // min-1
		
		double r_anisotropy = r_a0 * migration_speed;
		
		ecm.ecm_voxels[voxel_index].anisotropy = anisotropy + r_anisotropy * dt * (1- anisotropy);
	}
	
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

	// Get threshold neighbours parameter
	int overcrowding_threshold = pCell->custom_data["overcrowding_threshold"];
	//std::cout<<"overcrowding_threshold = "<<overcrowding_threshold<<std::endl;

	// Get cell's neighbours 
	int n_attached =  pCell->state.neighbors.size();
	//std::cout<<"neighbours = "<<neighbors<<std::endl;
	
	// Proliferation depends on number of cells in contact
	double proliferation_rate = get_single_base_behavior(pCell,"cycle entry");
	// std::cout<<"base proliferation_rate = "<<proliferation_rate<<std::endl;
	proliferation_rate *= decreasing_linear_response_function(n_attached,0,overcrowding_threshold);
	// std::cout<<"proliferation rate: "<<proliferation_rate;
	// std::cout<<" attached cells: "<<n_attached<<std::endl;

	// Dependence on ECM density
	// Computing the index of the voxel at cell position
	int voxel_index = microenvironment.nearest_voxel_index( pCell->position );

	proliferation_rate *= 1 - ecm.ecm_voxels[voxel_index].density;
	// std::cout<<"ECM density: "<<ecm.ecm_voxels[voxel_index].density;
	// std::cout<<" attached cells: "<<n_attached;
	// std::cout<<" proliferation rate: "<<proliferation_rate<<std::endl;

	// // Relation between ribose concentration and density
	// double ribose_concentration = parameters.doubles("ribose_concentration");

	// Set new proliferation rate
	set_single_behavior(pCell,"cycle entry", proliferation_rate);
	//std::cout<<"proliferation_rate = "<<get_single_behavior(pCell,"cycle entry")<<std::endl;
}


void custom_update_cell_velocity( Cell* pCell, Phenotype& phenotype, double dt)
{
	// Update cell motility speed
	cell_ecm_interaction_motility_speed(pCell, phenotype, dt);

	// Assign speed to motility vector with same motility direction
	pCell->phenotype.motility.motility_vector = pCell->phenotype.motility.migration_bias_direction; 
	// std::cout<<"motility_vector: "<<phenotype.motility.motility_vector<<std::endl;
	normalize( &(pCell->phenotype.motility.motility_vector) ); 

	// pCell->phenotype.motility.migration_speed = 0;

	// std::cout<<"motility_vector after normalize: "<<phenotype.motility.motility_vector<<std::endl;
	pCell->phenotype.motility.motility_vector *= pCell->phenotype.motility.migration_speed;
	// std::cout<<"motility_vector times speed: "<<phenotype.motility.motility_vector<<std::endl;

	/********************* Update standard velocity ******************************/
	
	// Calling the standard update velocity of PhysiCell
	standard_update_cell_velocity(pCell, phenotype, dt);
	// std::cout<<"motility_vector after standard update: "<<phenotype.motility.motility_vector<<std::endl;

	/********************************************SAVE TOTAL SPEED AND VELOCITY DATA*******************************************/
	// Compute total cell's speed and velocity
	double total_speed = sqrt(pow(pCell->velocity[0],2) + pow(pCell->velocity[1],2) + pow(pCell->velocity[2],2));
	// std::cout<<"total_speed: "<<total_speed<<std::endl;
	pCell->custom_data["total_speed"] = total_speed;
	pCell->custom_data["total_velocity_x"] = pCell->velocity[0];
	pCell->custom_data["total_velocity_y"] = pCell->velocity[1];
	pCell->custom_data["total_velocity_z"] = pCell->velocity[2];

	return; 
}

void phenotype_function( Cell* pCell, Phenotype& phenotype, double dt )
{ return; }

void custom_function( Cell* pCell, Phenotype& phenotype , double dt )
{ 
	ecm_update_from_cell_velocity(pCell,phenotype,dt);

	proliferation_inhibition(pCell,phenotype,dt);
	
	pCell->custom_data["attached_cells"] = pCell->state.neighbors.size();

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