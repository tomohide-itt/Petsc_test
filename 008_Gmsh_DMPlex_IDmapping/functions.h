#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <petscdmplex.h>
#include <petscksp.h>
#include "mesh.h"

PetscErrorCode read_gmsh( const char* mesh_path, DM &dm, PetscInt &dim );
PetscErrorCode get_coords( const DM& dm, const PetscInt dim, const PetscScalar *a_loc, const PetscScalar *a_glob, const bool debug=false );
PetscErrorCode get_vertex_ID_range( const DM& dm, PetscInt& ID_start, PetscInt& ID_end, const bool debug=false );
PetscErrorCode get_face_ID_range( const DM& dm, PetscInt& ID_start, PetscInt& ID_end, const bool debug=false );
PetscErrorCode get_cell_ID_range( const DM& dm, PetscInt& ID_start, PetscInt& ID_end, const bool debug=false );
PetscErrorCode get_num_vertex( const DM& dm, PetscInt& num, const bool debug=false );
PetscErrorCode get_num_face( const DM& dm, PetscInt& num, const bool debug=false );
PetscErrorCode get_num_cell( const DM& dm, PetscInt& num, const bool debug=false );
PetscErrorCode get_num_boundary( const DM& dm, PetscInt& num, const bool debug=false );
PetscErrorCode get_label_name( const int rank, const DM& dm, std::vector<std::string> &label_names, const bool debug=false );
PetscErrorCode get_label_num( const int rank, const DM& dm, const std::string& label_name, PetscInt& num, const bool debug=false );

PetscErrorCode get_elemID_map( const int rank, const DM& dm, const PetscInt dim, const node_vec& nodes, const elem_vec& elems,
    std::map<int,int>& eID2pID, const bool debug=false );


#endif

