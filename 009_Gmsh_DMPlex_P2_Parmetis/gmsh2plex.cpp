#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <petscdmplex.h>
#include <petscksp.h>
#include "functions.h"
#include "mesh.h"

int main(int argc,char **argv)
{
  PetscCall(PetscInitialize(&argc,&argv,NULL,""));

  PetscMPIInt rank, nproc;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
  MPI_Comm_size( PETSC_COMM_WORLD, &nproc );

  // ======= 実行時のオプションで，mshのパス，vtkのパスを指定する =============================================
  // 実行時に -mesh <パス> または -mesh=<パス> が渡されていれば，その<パス>を mesh_path バッファにコピーして，hasMesh = PETSC_TRUE にする
  char mesh_path[PETSC_MAX_PATH_LEN] = "";
  PetscBool hasMesh = PETSC_FALSE;
  PetscCall( PetscOptionsGetString( NULL, NULL, "-mesh", mesh_path, sizeof(mesh_path), &hasMesh ) );
  //+++ check
  {
    PetscPrintf( PETSC_COMM_WORLD, "mesh_path = %s\n", mesh_path );
    PetscPrintf( PETSC_COMM_WORLD, "hasMesh = %d\n", (int)hasMesh );
  }
  //---
  // hasMesh が PETSC_FALSE のとき，PETSC_ERR_USER のエラーを発生させ，指定したメッセージを出力して処理を中断する
  PetscCheck( hasMesh, PETSC_COMM_WORLD, PETSC_ERR_USER, "-mesh <mesh.msh> を指定してください" );

  //=== .mshから節点を読込む ===============================================================================
  node_vec nodes;
  read_msh_nodes( mesh_path, nodes );
  nodes.show();

  //=== .mshから要素を読込む ===============================================================================
  elem_vec elems;
  read_msh_elems( mesh_path, elems );
  elems.show();

  //=== .mshをDMPlexで読み込む ==============================================================================
  DM dm = NULL; // 解析用DM
  PetscInt dim; // 次元
  read_gmsh( mesh_path, dm, dim );

  //=== 並列分割 ==============================================================================
  DM dmDist = NULL;
  if( nproc > 1 )
  {
    PetscPartitioner part;
    PetscCall( DMPlexGetPartitioner( dm, &part ) );
    PetscCall( PetscPartitionerSetType( part, PETSCPARTITIONERPARMETIS ) );
    PetscCall( DMPlexDistribute( dm, 0, NULL, &dmDist ) );
    if(dmDist)
    {
      PetscCall( DMDestroy( &dm ) );
      dm = dmDist;
    }
  }

  //=== 頂点，面，セルのIDの範囲を取得 ==============================================================================
  PetscInt vtx_start, vtx_end, face_start, face_end, cell_start, cell_end; // DMPlex内部のセルと頂点の最初のIDと最後のID
  get_vertex_ID_range( rank, dm, vtx_start,  vtx_end,  true );
  get_face_ID_range(   rank, dm, face_start, face_end, true );
  get_cell_ID_range(   rank, dm, cell_start, cell_end, true );

  //=== セル数，面数，境界面数，頂点数を取得 ==============================================================================
  PetscInt num_vtx, num_face, num_cell, num_boundary;
  get_num_vertex(   rank, dm, num_vtx,      true );
  get_num_face(     rank, dm, num_face,     true );
  get_num_cell(     rank, dm, num_cell,     true );
  get_num_boundary( rank, dm, num_boundary, true );

  //=== 座標（ローカル/グローバル）を取得 ==============================================================================
  const PetscScalar *a_loc  = NULL;
  const PetscScalar *a_glob = NULL;
  get_coords( rank, dm, dim, a_loc, a_glob, true );

  //=== DMPlexのセルpointIDとgmshのelementTagの紐づけ ========================================================
  std::map<int,int> eID2pID;
  std::map<int,int> pID2eID;
  PetscCall( get_elemID_map( rank, dm, dim, nodes, elems, eID2pID, pID2eID, false ) );
  //+++
  {
    for( const auto& [pID, eID] : pID2eID )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, pID=%7d, eID=%7d\n", rank, pID, eID );
    }
    PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  }
  //---
  
  

  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&dmDist));
  PetscCall(PetscFinalize());
  return 0;
}