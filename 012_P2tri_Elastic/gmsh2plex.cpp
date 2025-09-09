// mpiexec -n 2 ./gmsh2plex -mesh test2D_2.msh -vtk mesh.vtk -ksp_monitor

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
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
  // 実行時に -mesh <パス> または -mesh=<パス> が渡されていれば，その<パス>を mesh_path バッファにコピーする
  char cmesh_path[PETSC_MAX_PATH_LEN] = "";
  PetscCall( PetscOptionsGetString( NULL, NULL, "-mesh", cmesh_path, sizeof(cmesh_path), NULL ) );
  std::string mesh_path( cmesh_path );
  if( mesh_path.empty() )
  {
    PetscPrintf( PETSC_COMM_WORLD, "-mesh <***.msh> を指定してください\n" );
    exit(1);
  }

  // 実行時に -vtk <パス> または -vtk=<パス> が渡されていれば，その<パス>を vtk_path バッファにコピーする
  char cvtk_path[PETSC_MAX_PATH_LEN] = "";
  PetscCall( PetscOptionsGetString( NULL, NULL, "-vtk", cvtk_path, sizeof(cvtk_path), NULL ) );
  std::string vtk_path( cvtk_path );
  if( vtk_path.empty() )
  {
    PetscPrintf( PETSC_COMM_WORLD, "-vtk <***.vtk> を指定してください\n" );
    exit(1);
  }

  //=== .mshから節点を読込む ===============================================================================
  msh::node_vec nodes;
  read_msh_nodes( mesh_path, nodes );

  //=== .mshから要素を読込む ===============================================================================
  msh::elem_vec elems;
  read_msh_elems( mesh_path, elems );

  //=== .mshをDMPlexで読み込む ==============================================================================
  DM dm = NULL; // 解析用DM
  read_gmsh( mesh_path, dm );

  //=== 並列分割 ==============================================================================
  DM dm_dist = NULL;
  PetscCall( partition_mesh( dm, dm_dist ) );

  //=== 頂点，面，セルのIDの範囲を出力 ==============================================================================
  PetscCall( show_vertexID_range( dm ) );
  PetscCall( show_faceID_range(   dm ) );
  PetscCall( show_cellID_range(   dm ) );

  //=== 座標の表示 ==============================================================================
  PetscCall( show_coords_each_cell( dm ) );

  //=== DMPlexのセルpointIDとgmshのelementTagの紐づけ ========================================================
  std::map<int,int> eID2pID;
  std::map<int,int> pID2eID;
  PetscCall( get_elemID_map( dm, nodes, elems, eID2pID, pID2eID, true ) );

  //=== FE空間作成 ==============================================================================
  PetscFE fe;
  create_FE( dm, fe );

  //=== 係数行列，右辺ベクトルの作成 ==============================================================================
  Mat A;
  Vec b, sol;
  PetscCall( DMCreateMatrix( dm, &A ) );
  PetscCall( DMCreateGlobalVector( dm, &b ) );
  PetscCall( VecZeroEntries( b ) );
  PetscCall( VecDuplicate( b, &sol ) );

  //=== 材料定数, Dマトリクス計算 ==============================================================================
  double E = 1.0e3;
  double nu = 0.33;
  PetscScalar D[16];
  PetscCall( cal_D_matrix( E, nu, D ) );

  //=== Kuuマトリクスをマージ ==============================================================================
  PetscCall( merge_Kuu_matrix( dm, D, A, true ) );

  //=== 節点力 ==============================================================================
  PetscCall( set_nodal_force( dm, fe, 2, -10, 1, b ) );

  //=== Dirichlet境界条件 ==============================================================================
  PetscCall( set_Dirichlet_zero( dm, 4, A, b ) );

  //=== ソルバーで解く ==============================================================================
  KSP ksp;
  PetscCall( KSPCreate( PETSC_COMM_WORLD, &ksp ) );
  PetscCall( KSPSetOperators( ksp, A, A ) );
  PetscCall( KSPSetType( ksp, KSPCG ) );
  PetscCall( KSPSetFromOptions(ksp) );
  PetscCall( KSPSolve( ksp, b, sol ) );

  //=== 変位の出力 ==============================================================================
  PetscCall( show_displacement( dm, sol ) );

  PetscCall(VecDestroy(&sol));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall( DMDestroy( &dm ) );
  PetscCall( DMDestroy( &dm_dist ) );
  PetscCall( PetscFinalize() );
  return 0;
}