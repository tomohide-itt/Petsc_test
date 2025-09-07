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

PetscErrorCode read_gmsh( const std::string& mesh_path, DM &dm, PetscInt &dim );
PetscErrorCode get_coords( const int rank, const DM& dm, const PetscInt dim, const PetscScalar *a_loc, const PetscScalar *a_glob, const bool debug=false );
PetscErrorCode get_vertex_ID_range( const int rank, const DM& dm, PetscInt& ID_start, PetscInt& ID_end, const bool debug=false );
PetscErrorCode get_face_ID_range( const int rank, const DM& dm, PetscInt& ID_start, PetscInt& ID_end, const bool debug=false );
PetscErrorCode get_cell_ID_range( const int rank, const DM& dm, PetscInt& ID_start, PetscInt& ID_end, const bool debug=false );
PetscErrorCode get_num_vertex( const int rank, const DM& dm, PetscInt& num, const bool debug=false );
PetscErrorCode get_num_face( const int rank, const DM& dm, PetscInt& num, const bool debug=false );
PetscErrorCode get_num_cell( const int rank, const DM& dm, PetscInt& num, const bool debug=false );
PetscErrorCode get_num_boundary( const int rank, const DM& dm, PetscInt& num, const bool debug=false );
PetscErrorCode get_label_name( const int rank, const DM& dm, std::vector<std::string> &label_names, const bool debug=false );
PetscErrorCode get_label_num( const int rank, const DM& dm, const std::string& label_name, PetscInt& num, const bool debug=false );

PetscErrorCode get_elemID_map( const int rank, const DM& dm, const PetscInt dim, const node_vec& nodes, const elem_vec& elems,
    std::map<int,int>& eID2pID, std::map<int,int>& pID2eID, const bool debug=false );

PetscErrorCode get_map_nic2ni( const int rank, const DM& dm, std::vector<PetscInt>& map );

PetscErrorCode show_coords_each_cell( const int rank, const DM& dm );
PetscErrorCode show_coords_boundary( const int rank, const DM& dm, const PetscInt f );

PetscErrorCode set_Dirichlet_zero( const int rank, const DM& dm, const PetscInt phys_id, Mat& A, Vec& b );
PetscErrorCode set_nodal_force( const int rank, const DM& dm, const PetscFE& fe,
  const PetscInt phys_id, const PetscScalar F, const PetscInt dir, Vec& b );
PetscErrorCode get_coords_face( const int rank, const DM& dm, const PetscInt p, std::vector<double>& xy );
PetscErrorCode get_coords_vertex( const int rank, const DM& dm, const PetscInt p, std::vector<double>& xy );
PetscErrorCode show_displacement( const int rank, const DM& dm, const Vec& sol );

PetscErrorCode build_cell_dof_map( const int rank, const DM& dm, const PetscSection& sec, const PetscInt c,
  PetscInt& ncelldof, PetscInt* idx, PetscInt* pt, PetscInt* comp );


#endif

