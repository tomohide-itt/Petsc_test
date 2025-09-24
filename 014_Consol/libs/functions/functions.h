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

PetscErrorCode show_DM_info( const DM& dm, const int iwat );

PetscErrorCode set_nodes( DM& dm, node_vec& nodes );

PetscErrorCode set_elems( DM& dm, node_vec& nodes, elem_vec& elems );

PetscErrorCode create_FE( DM dm, const int iwat, const bool debug=false );

PetscErrorCode cal_D_matrix( const DM& dm, const double E, const double nu, std::vector<PetscScalar>& D );

PetscErrorCode merge_Kuu_matrix( const DM& dm, const std::vector<PetscScalar>& D, const elem_vec& elems, Mat& A, const bool debug=false );

PetscErrorCode merge_Kuh_matrix( const DM& dm, const elem_vec& elems, Mat& A, const bool debug=false );

PetscErrorCode merge_Khu_matrix( const DM& dm, const elem_vec& elems, Mat& A, const bool debug=false );

PetscErrorCode merge_Khh_matrix( const DM& dm, const double k, const double gmw, const double beta, const double dt, const elem_vec& elems, Mat& A, const bool debug=false );

PetscErrorCode set_nodal_force( const DM& dm, const PetscInt phys_id, const PetscScalar F, const PetscInt dir, Vec& b, const bool debug=false );

PetscErrorCode set_GBC( const DM& dm, const PetscInt phys_id, Mat& A, Vec& b );

PetscErrorCode set_HBC( const DM& dm, const PetscInt phys_id, Mat& A, Vec& b );

PetscErrorCode set_displacement( const DM& dm, const Vec& sol, node_vec& nodes );

PetscErrorCode set_ex_pore_water_pressure( const DM& dm, const Vec& sol, node_vec& nodes );

PetscErrorCode show_displacement( const elem_vec& elems );

PetscErrorCode show_ex_pore_water_pressure( const elem_vec& elems );

void output_vtk( const std::string& vtk_path, const int iwat, const node_vec& nodes, const elem_vec& elems, const std::map<int,int>& lpid2ntag );

//PetscErrorCode set_Dirichlet_zero( const DM& dm, const PetscInt phys_id, Mat& A, Vec& b );
#endif

