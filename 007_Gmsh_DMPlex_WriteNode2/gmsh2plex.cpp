#include <iostream>
#include <string>
#include <vector>
#include <petscdmplex.h>
#include <petscksp.h>
#include "functions.h"

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
    PetscPrintf( PETSC_COMM_WORLD, "--- check ---\n" );
    PetscPrintf( PETSC_COMM_WORLD, "mesh_path = %s\n", mesh_path );
    PetscPrintf( PETSC_COMM_WORLD, "hasMesh = %d\n", (int)hasMesh );
  }
  //---
  
  // hasMesh が PETSC_FALSE のとき，PETSC_ERR_USER のエラーを発生させ，指定したメッセージを出力して処理を中断する
  PetscCheck( hasMesh, PETSC_COMM_WORLD, PETSC_ERR_USER, "-mesh <mesh.msh> を指定してください" );

  //=== .mshをDMPlexで読み込む ==============================================================================
  DM dm = NULL; // 解析用DM
  PetscInt dim; // 次元
  read_gmsh( mesh_path, dm, dim );

  //=== 座標（ローカル/グローバル）を取得 ==============================================================================
  const PetscScalar *a_loc  = NULL;
  const PetscScalar *a_glob = NULL;
  get_coords( dm, dim, a_loc, a_glob, false );

  //=== 頂点，面，セルのIDの範囲を取得 ==============================================================================
  PetscInt vtx_start, vtx_end, face_start, face_end, cell_start, cell_end; // DMPlex内部のセルと頂点の最初のIDと最後のID
  get_vertex_ID_range( dm, vtx_start,  vtx_end,  true );
  get_face_ID_range(   dm, face_start, face_end, true );
  get_cell_ID_range(   dm, cell_start, cell_end, true );

  //=== セル数，面数，境界面数，頂点数を取得 ==============================================================================
  PetscInt num_vtx, num_face, num_cell, num_boundary;
  get_num_vertex(   dm, num_vtx,      true );
  get_num_face(     dm, num_face,     true );
  get_num_cell(     dm, num_cell,     true );
  get_num_boundary( dm, num_boundary, true );

  //=== ラベル（GmshのPhysical Group）を取得 ==============================================================================
  std::vector<std::string> label_names;
  get_label_name( rank, dm, label_names, true );

  //=== 各ラベルについて，値（=Physical ID）ごとの要素数を数える ==============================================================
  for( int i=0; i<label_names.size(); i++ )
  {
    PetscInt num;
    get_label_num( rank, dm, label_names[i], num, true );
  }


  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}