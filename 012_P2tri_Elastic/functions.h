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
//#include "mesh.h"
#include "node.h"
#include "elem.h"

PetscErrorCode read_gmsh( const std::string& mesh_path, DM &dm );

PetscErrorCode partition_mesh( DM &dm, DM &dm_dist );

PetscErrorCode set_nodes( DM& dm, node_vec& nodes );
PetscErrorCode set_elems( DM& dm, node_vec& nodes, elem_vec& elems );

PetscErrorCode create_FE( const DM& dm, PetscFE& fe );

PetscErrorCode cal_D_matrix( const double E, const double nu, PetscScalar* D );
PetscErrorCode merge_Kuu_matrix( const DM& dm, const PetscScalar* D, const elem_vec& elems, Mat& A, const bool debug=false );

PetscErrorCode set_Dirichlet_zero( const DM& dm, const PetscInt phys_id, Mat& A, Vec& b );
PetscErrorCode set_nodal_force( const DM& dm, const PetscFE& fe,
  const PetscInt phys_id, const PetscScalar F, const PetscInt dir, Vec& b );

PetscErrorCode set_displacement( const DM& dm, const Vec& sol, node_vec& nodes );
PetscErrorCode show_displacement( const elem_vec& elems );

void output_vtk( const std::string& vtk_path, const node_vec& nodes, const elem_vec& elems,
  const std::map<int,int>& lpid2ntag );


PetscErrorCode get_coords_face( const DM& dm, const PetscInt p, std::vector<double>& xy );
PetscErrorCode get_coords_vertex( const DM& dm, const PetscInt p, std::vector<double>& xy );
PetscErrorCode get_coords( const DM& dm, const PetscInt p, std::vector<double>& xy );

PetscErrorCode show_vertexID_range( const DM& dm );
PetscErrorCode show_faceID_range(   const DM& dm );
PetscErrorCode show_cellID_range(   const DM& dm );
PetscErrorCode show_coords_each_cell( const DM& dm );
PetscErrorCode show_coords_boundary( const int rank, const DM& dm, const PetscInt f );

#endif

