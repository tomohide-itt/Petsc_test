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
  node_vec nodes;
  read_msh_nodes( mesh_path, nodes );

  //=== .mshから要素を読込む ===============================================================================
  elem_vec elems;
  read_msh_elems( mesh_path, elems );

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

  //=== DMPlexのセルpointIDとgmshのelementTagの紐づけ ========================================================
  std::map<int,int> eID2pID;
  std::map<int,int> pID2eID;
  PetscCall( get_elemID_map( rank, dm, dim, nodes, elems, eID2pID, pID2eID, false ) );

  //=== vtkファイルを書き出す 
  output_vtk( vtk_path, rank, nproc, pID2eID, nodes, elems );

  //=== FE空間作成 ==============================================================================
  PetscFE fe;
  // 2D三角形要素（P2）で2成分ベクトル場（変位）に対する2次有限要素空間を作る
  // 引数は，コミュニケータ，次元，フィールド成分数，要素形状がsimplexか，P1/P2, 積分次数，出力先の有限要素オブジェクト
  PetscCall( PetscFECreateLagrange( PETSC_COMM_WORLD, dim, dim, PETSC_TRUE, 2, PETSC_DETERMINE, &fe ) );
  // dm にフィールド0を登録
  // 引数は，メッシュオブジェクト，フィールド番号，ラベル，そのフィールドに対応する有限要素オブジェクト
  PetscCall( DMSetField( dm, 0, NULL, (PetscObject)fe ) );
  // Discrete System (PetscDS) を dm から生成する （自由度構造の確定）
  // 内部では，dmに紐づいた PetscSection を構築，PDE用のオブジェクト PetscDS を作成，各フィールドごとに PetscFE の情報を関連づけを行っている
  PetscCall( DMCreateDS( dm ) );

  //=== 係数行列，右辺ベクトルの作成
  Mat A;
  Vec b, sol;
  PetscCall( DMCreateMatrix( dm, &A ) );
  PetscCall( DMCreateGlobalVector( dm, &b ) );
  PetscCall( VecDuplicate( b, &sol ) );

  //=== 材料定数
  double E = 1.0e3;
  double nu = 0.33;
  double lamb = (E*nu)/((1.0+nu)*(1.0-2.0*nu));
  double mu = E/(2.0*(1.0+nu));
  // Dマトリクス
  PetscScalar D[16];
  for( int i=0; i<16; i++ ) D[i] = 0.0;
  D[ 0] = lamb + 2.0*mu;  D[ 1] = lamb;  D[ 2] = lamb;  D[ 3] = 0.0;
  D[ 4] = lamb;  D[ 5] = lamb + 2.0*mu;  D[ 6] = lamb;  D[ 7] = 0.0;
  D[ 8] = lamb;  D[ 9] = lamb;  D[10] = lamb + 2.0*mu;  D[11] = 0.0;
  D[12] = 0.0;  D[13] = 0.0;  D[14] = 0.0;  D[15] = mu;

  //=== 要素でループ 
  PetscSection loc_section, glob_section;
  PetscCall( DMGetLocalSection( dm, &loc_section ) );
  PetscCall( DMGetGlobalSection( dm, &glob_section ) );
  PetscInt cell_start, cell_end;
  get_cell_ID_range( rank, dm, cell_start, cell_end, true );
  for( PetscInt c=cell_start; c<cell_end; c++ )
  {
    // 局所自由度インデックス（プロセス内のローカルVecの位置）を取得
    PetscInt* idx;
    PetscInt nidx;
    PetscCall( DMPlexGetClosureIndices( dm, loc_section, glob_section, c, PETSC_TRUE, &nidx, &idx, NULL, NULL ) );

    // 要素剛性マトリクスKeの計算
    PetscScalar Ke[144];
    for( int i=0; i<144; i++ ) Ke[i] = 0.0;

    // 積分点位置
    PetscScalar r[4];
    r[0] = 0.816847572980459; r[1] = 0.091576213509771;
    r[1] = 0.108103018168070; r[2] = 0.445948490915965;

    // 積分点重み
    PetscScalar w[2];
    w[0] = 0.109951743655322; w[1] = 0.223381589678011;

    // 座標マトリクス [xye] (6x2)
    PetscScalar xye[12];
    int eID  = pID2eID[c];
    int eidx = elems.idx_of_id(eID);
    for( int i=0; i<6; i++ )
    {
      int nID  = elems[eidx].nodeIDs[i];
      int nidx = nodes.idx_of_id(nID);
      xye[i*2+0] = nodes[nidx].x;
      xye[i*2+1] = nodes[nidx].y;
    }
    //+++
    if( 0 )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d pID=%5d eID=%5d xye: \n", rank, c, eID );
      for( int i=0; i<6; i++ )
      {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e%15.5e\n", xye[i*2+0], xye[i*2+1] );
      }
    }
    //---

    PetscInt num_gp = 6; //積分点数
    for( PetscInt gp=0; gp<num_gp; gp++ )
    {
      double L1, L2, L3;
      if( gp == 0 ){ L1 = r[0]; L2 = r[1]; }
      if( gp == 1 ){ L1 = r[1]; L2 = r[0]; }
      if( gp == 2 ){ L1 = r[1]; L2 = r[1]; }
      if( gp == 3 ){ L1 = r[2]; L2 = r[3]; }
      if( gp == 4 ){ L1 = r[3]; L2 = r[2]; }
      if( gp == 5 ){ L1 = r[3]; L2 = r[3]; }
      L3 = 1.0 - L1 - L2;
      //
      double wei;
      if( gp < 3 ) wei = w[0];
      else wei = w[1];
      // [dN/dr] (6x2)
      PetscScalar dNdr[12];
      dNdr[0*2+0] = 2.0*L1-1.0;
      dNdr[1*2+0] = 0.0;
      dNdr[2*2+0] =-4.0*L3+1.0;
      dNdr[3*2+0] =-4.0*L2;
      dNdr[4*2+0] = 4.0*(L3-L1);
      dNdr[5*2+0] = 4.0*L2;
      dNdr[0*2+1] = 0.0;
      dNdr[1*2+1] = 4.0*L2-1.0;
      dNdr[2*2+1] =-4.0*L3+1.0;
      dNdr[3*2+1] = 4.0*(L3-L2);
      dNdr[4*2+1] =-4.0*L1;
      dNdr[5*2+1] = 4.0*L1;
      //++++
      if( 0 )
      {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d pID=%5d eID=%5d gp=%5d dNdr: \n", rank, c, eID, gp );
        for( int i=0; i<6; i++ )
        {
          for( int j=0; j<2; j++ )
          {
            PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%12.3e", dNdr[i*2+j] );
          }
          PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
        }
      }
      //----
      // [J] (2x2)
      PetscScalar J[4];
      for( int i=0; i<2; i++ )
      {
        for( int j=0; j<2; j++ )
        {
          J[i*2+j] = 0.0;
          for( int k=0; k<6; k++ )
          {
            J[i*2+j] += dNdr[k*2+i]*xye[k*2+j];
          }
        }
      }
      // detJ
      double detJ = J[0*2+0]*J[1*2+1] - J[0*2+1]*J[1*2+0];
      // [J]-T (2x2)
      PetscScalar J_I_T[4];
      J_I_T[0*2+0] = J[1*2+1]/detJ;
      J_I_T[1*2+1] = J[0*2+0]/detJ;
      J_I_T[0*2+1] =-J[1*2+0]/detJ;
      J_I_T[1*2+0] =-J[0*2+1]/detJ;
      // [dN/dx] (6x2)
      PetscScalar derivN[12];
      for( int i=0; i<6; i++ )
      {
        for( int j=0; j<2; j++ )
        {
          derivN[i*2+j] = 0.0;
          for( int k=0; k<2; k++ )
          {
            derivN[i*2+j] += dNdr[i*2+k]*J_I_T[k*2+j];
          }
        }
      }
      // fac
      double fac = 0.5*wei*detJ;
      // B (4x12)
      PetscScalar B[48];
      for( int i=0; i<48; i++ ) B[i] = 0.0;
      for( int i=0; i<6; i++ )
      {
        B[0*12 + (2*i+0)] = -derivN[i*2+0];
        B[1*12 + (2*i+1)] = -derivN[i*2+1];
        B[3*12 + (2*i+0)] = -derivN[i*2+1]*0.5;
        B[3*12 + (2*i+1)] = -derivN[i*2+0]*0.5;
      }
      //++++
      if( 0 )
      {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d pID=%5d eID=%5d gp=%5d B: \n", rank, c, eID, gp );
        for( int i=0; i<4; i++ )
        {
          for( int j=0; j<12; j++ )
          {
            PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%12.3e", B[i*12+j] );
          }
          PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
        }
      }
      //----
      // BVOL (4x12)
      PetscScalar BVOL[48];
      for( int i=0; i<48; i++ ) BVOL[i] = B[i]*fac;
      // [B]T[D] (12x4)
      PetscScalar BTD[48];
      for( int i=0; i<12; i++ )
      {
        for( int j=0; j<4; j++ )
        {
          BTD[i*4+j] = 0.0;
          for( int k=0; k<4; k++ )
          {
            BTD[i*4+j] += B[k*12+i]*D[k*4+j];
          }
        }
      }
      // [B]T[D][B] (12x12)
      for( int i=0; i<12; i++ )
      {
        for( int j=0; j<12; j++ )
        {
          for( int k=0; k<4; k++ )
          {
            Ke[i*12+j] += BTD[i*4+k]*BVOL[k*12+j];
          }
        }
      }
    }
    //+++
    if( 0 )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d pID=%5d eID=%5d Ke: \n", rank, c, eID );
      for( int i=0; i<12; i++ )
      {
        for( int j=0; j<12; j++ )
        {
          PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%12.3e", Ke[i*12+j] );
        }
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
      }
    }
    //---
    
    // 要素既知ベクトルFeの計算
    PetscScalar Fe[12];
    for( int i=0; i<12; i++ ) Fe[i] = 0.0;

    // アセンブリ
    //MatSetValuesLocal( A, nidx, idx, nidx, idx, Ke, ADD_VALUES );
    //VecSetValuesLocal( b, nidx, idx, Fe, ADD_VALUES );
    PetscCall( DMPlexRestoreClosureIndices( dm, loc_section, glob_section, c, PETSC_TRUE, &nidx, &idx, NULL, NULL ) );
  }

  //PetscCall( MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY ) );
  //PetscCall( MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY ) );
  //PetscCall( VecAssemblyBegin( b ) );
  //PetscCall( VecAssemblyEnd( b ) );

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  
  PetscCall(VecDestroy(&sol));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&dmDist));
  PetscCall(PetscFinalize());
  return 0;
}