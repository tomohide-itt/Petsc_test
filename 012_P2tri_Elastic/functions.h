#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <petscdmplex.h>
#include <petscksp.h>
#include "mesh.h"

PetscErrorCode read_gmsh( const std::string& mesh_path, DM &dm );

PetscErrorCode partition_mesh( DM &dm, DM &dm_dist );

PetscErrorCode get_elemID_map( const DM& dm, const node_vec& nodes, const elem_vec& elems,
    std::map<int,int>& eID2pID, std::map<int,int>& pID2eID, const bool debug=false );

PetscErrorCode create_FE( const DM& dm, PetscFE& fe );

PetscErrorCode cal_D_matrix( const double E, const double nu, PetscScalar* D );

PetscErrorCode set_Dirichlet_zero( const int rank, const DM& dm, const PetscInt phys_id, Mat& A, Vec& b );
PetscErrorCode set_nodal_force( const int rank, const DM& dm, const PetscFE& fe,
  const PetscInt phys_id, const PetscScalar F, const PetscInt dir, Vec& b );
PetscErrorCode get_coords_face( const int rank, const DM& dm, const PetscInt p, std::vector<double>& xy );
PetscErrorCode get_coords_vertex( const int rank, const DM& dm, const PetscInt p, std::vector<double>& xy );
PetscErrorCode show_displacement( const int rank, const DM& dm, const Vec& sol );

PetscErrorCode build_cell_dof_map( const int rank, const DM& dm, const PetscSection& sec, const PetscInt c,
  PetscInt& ncelldof, PetscInt* idx, PetscInt* pt, PetscInt* comp );


PetscErrorCode show_vertexID_range( const DM& dm );
PetscErrorCode show_faceID_range(   const DM& dm );
PetscErrorCode show_cellID_range(   const DM& dm );
PetscErrorCode show_coords_each_cell( const DM& dm );
PetscErrorCode show_coords_boundary( const int rank, const DM& dm, const PetscInt f );

#endif

