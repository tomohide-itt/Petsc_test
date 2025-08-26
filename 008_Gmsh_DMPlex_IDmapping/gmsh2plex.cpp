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

  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

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

  //=== DMPlexのセルpointIDとgmshのelementTagの紐づけ ========================================================
  std::map<int,int> eID2pID;
  PetscCall( get_elemID_map( rank, dm, dim, nodes, elems, eID2pID, false ) );
  //+++
  {
    for( const auto& [eID, pID] : eID2pID )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, eID=%7d, pID=%7d\n", rank, eID, pID );
    }
    PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  }
  //---
  
  

  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}