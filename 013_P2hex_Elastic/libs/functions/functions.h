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
#include "node.h"
#include "elem.h"
#include "hexl27.h"

PetscErrorCode read_gmsh( const std::string& mesh_path, DM &dm );

PetscErrorCode partition_mesh( DM &dm, DM &dm_dist );

PetscErrorCode show_DM_info( const DM& dm );

PetscErrorCode set_nodes( DM& dm, node_vec& nodes );

#endif

