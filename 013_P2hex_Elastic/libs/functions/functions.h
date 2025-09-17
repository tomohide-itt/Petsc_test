#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <petscdmplex.h>
#include <petscksp.h>
#include <petscds.h>  
#include "node.h"
#include "elems.h"

PetscErrorCode read_gmsh( const std::string& mesh_path, DM &dm );

PetscErrorCode partition_mesh( DM &dm, DM &dm_dist );

PetscErrorCode show_DM_info( const DM& dm );

PetscErrorCode set_nodes( DM& dm, node_vec& nodes );

PetscErrorCode set_elems( DM& dm, node_vec& nodes, elem_vec& elems );

PetscErrorCode create_FE( DM dm, const bool debug=false );

PetscErrorCode cal_D_matrix( const DM& dm, const double E, const double nu, std::vector<PetscScalar>& D );

PetscErrorCode merge_Kuu_matrix( const DM& dm, const std::vector<PetscScalar>& D, const elem_vec& elems, Mat& A, const bool debug=false );

PetscErrorCode set_nodal_force( const DM& dm, const PetscInt phys_id, const PetscScalar F, const PetscInt dir, Vec& b );

#endif

