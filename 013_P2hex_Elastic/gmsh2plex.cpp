// mpiexec -n 2 ./gmsh2plex -mesh test2D_2.msh -vtk mesh.vtk -ksp_monitor
// mpiexec -n 1 ./gmsh2plex -mesh test3D_c.msh -vtk mesh.vtk
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <petscdmplex.h>
#include <petscksp.h>
#include "functions.h"
#include "gmsh.h"

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

  //=== .mshをDMPlexで読み込む ==============================================================================
  DM dm = NULL; // 解析用DM
  read_gmsh( mesh_path, dm );

  //=== 並列分割 ==============================================================================
  DM dm_dist = NULL;
  PetscCall( partition_mesh( dm, dm_dist ) );

  //=== DM の情報を出力
  PetscCall( show_DM_info( dm ) );

  //=== 節点の設定 ==============================================================================
  node_vec nodes;
  PetscCall( set_nodes( dm, nodes ) );
  nodes.show();

  //=== 要素の設定 ==============================================================================
  elem_vec elems;
  PetscCall( set_elems( dm, nodes, elems ) );
  elems.show();

  //=== tag - id - pid の関係を得る ==============================================================================
  std::map<int,int> ntag2gnid, gnid2ntag;
  std::map<int,int> etag2geid, geid2etag;
  std::map<int,int> etag2lpid, lpid2etag;
  std::map<int,int> ntag2lpid, lpid2ntag;
  PetscCall( get_mesh_info( mesh_path, dm, ntag2gnid, gnid2ntag, etag2geid, geid2etag, ntag2lpid, lpid2ntag, etag2lpid, lpid2etag ) );

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
  std::vector<PetscScalar> D;
  PetscCall( cal_D_matrix( dm, E, nu, D ) );
/*
  //=== Kuuマトリクスをマージ ==============================================================================
  PetscCall( merge_Kuu_matrix( dm, D, elems, A, false ) );

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
  PetscCall( set_displacement( dm, sol, nodes ) );
//  PetscCall( show_displacement( elems ) );

  //=== vtk ファイルの出力 ==============================================================================
  output_vtk( vtk_path, nodes, elems, lpid2ntag );

  PetscCall( VecDestroy( &sol ) );
  PetscCall( VecDestroy( &b ) );
  PetscCall( MatDestroy( &A ) );
  PetscCall( PetscFEDestroy( &fe ) );
*/
  PetscCall( DMDestroy( &dm ) );
  PetscCall( DMDestroy( &dm_dist ) );
  PetscCall( PetscFinalize() );
  return 0;
}