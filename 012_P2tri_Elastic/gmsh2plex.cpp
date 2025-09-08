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
  node_vec nodes;
  read_msh_nodes( mesh_path, nodes );

  //=== .mshから要素を読込む ===============================================================================
  elem_vec elems;
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
  
/*
  //=== 
  //{
  //  PetscSection sec;
  //  PetscCall( DMGetLocalSection( dm, &sec ) );
  //  PetscInt idx[12], pt[12], comp[12];
  //  PetscInt ncelldof = 0;
  //  for( PetscInt c=cell_start; c<cell_end; c++ )
  //  {
  //    PetscCall( build_cell_dof_map( rank, dm, sec, c, ncelldof, idx, pt, comp ) );
  //  }
  //}

  //=== 要素でループ ==============================================================================
  PetscSection loc_section, glob_section;
  PetscCall( DMGetLocalSection( dm, &loc_section ) );
  PetscCall( DMGetGlobalSection( dm, &glob_section ) );
  //PetscInt cell_start, cell_end;
  get_cell_ID_range( rank, dm, cell_start, cell_end, true );
  for( PetscInt c=cell_start; c<cell_end; c++ )
  {
    // 局所自由度インデックス（プロセス内のローカルVecの位置）を取得
    PetscInt* idx;
    PetscInt nidx;
    PetscCall( DMPlexGetClosureIndices( dm, loc_section, glob_section, c, PETSC_TRUE, &nidx, &idx, NULL, NULL ) );

    //+++
    if( 1 )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d idx: \n", rank, c );
      for( int i=0; i<nidx; i++ )
      {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "idx[%5d]=%5d\n", i, idx[i] );
      }
    }
    //---

    // 要素剛性マトリクスKeの計算
    PetscScalar Ke[144];
    for( int i=0; i<144; i++ ) Ke[i] = 0.0;

    // 積分点位置
    PetscScalar r[4];
    r[0] = 0.816847572980459; r[1] = 0.091576213509771;
    r[2] = 0.108103018168070; r[3] = 0.445948490915965;

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
      dNdr[0*2+0] = 4.0*L1-1.0;
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
      double fac = 0.5*wei*fabs(detJ);
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
            if( k != 3 ) BTD[i*4+j] += B[k*12+i]*D[k*4+j];
            else         BTD[i*4+j] += 2.0*B[k*12+i]*D[k*4+j];
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
            if( k != 3 ) Ke[i*12+j] += BTD[i*4+k]*BVOL[k*12+j];
            else         Ke[i*12+j] += 2.0*BTD[i*4+k]*BVOL[k*12+j];
          }
        }
      }
    }
    //+++
    if( 1 )
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

    // 並べ替え（とりあえずむりやり）
    PetscScalar Ke_tmp[144];
    for( int i=0; i<144; i++ ) Ke_tmp[i] = Ke[i];
    PetscInt pmt[6];
    pmt[0] = 5;
    pmt[1] = 3;
    pmt[2] = 4;
    pmt[3] = 0;
    pmt[4] = 1;
    pmt[5] = 2;

    for( int i=0; i<6; i++ )
    {
      int io = pmt[i];
      for( int k=0; k<2; k++ )
      {
        int ik = i*2+k;
        int iok = io*2+k;
        for( int j=0; j<6; j++ )
        {
          int jo = pmt[j];
          for( int l=0; l<2; l++ )
          {
            int jl = j*2+l;
            int jol = jo*2+l;
            Ke[ik*12+jl] = Ke_tmp[iok*12+jol];
          }
        }
      }
    }

    // 要素既知ベクトルFeの計算
    PetscScalar Fe[12];
    for( int i=0; i<12; i++ ) Fe[i] = 0.0;

    // アセンブリ
    PetscCall( DMPlexMatSetClosure( dm, loc_section, glob_section, A, c, Ke, ADD_VALUES ) );
    PetscCall( DMPlexVecSetClosure( dm, loc_section, b, c, Fe, ADD_VALUES ) );
    PetscCall( DMPlexRestoreClosureIndices( dm, loc_section, glob_section, c, PETSC_TRUE, &nidx, &idx, NULL, NULL ) );
  }

  PetscCall( MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY ) );
  PetscCall( MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY ) );
  PetscCall( VecAssemblyBegin( b ) );
  PetscCall( VecAssemblyEnd( b ) );

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );

  //=== 節点力 ==============================================================================
  PetscCall( set_nodal_force( rank, dm, fe, 2, -10, 1, b ) );

  //=== Dirichlet境界条件 ==============================================================================
  PetscCall( set_Dirichlet_zero( rank, dm, 4, A, b ) );

  //=== ソルバーで解く ==============================================================================
  KSP ksp;
  PetscCall( KSPCreate( PETSC_COMM_WORLD, &ksp ) );
  PetscCall( KSPSetOperators( ksp, A, A ) );
  PetscCall( KSPSetType( ksp, KSPCG ) );
  PetscCall( KSPSetFromOptions(ksp) );
  PetscCall( KSPSolve( ksp, b, sol ) );

  //=== 変位の出力 ==============================================================================
  PetscCall( show_displacement( rank, dm, sol ) );

  */
  PetscCall(VecDestroy(&sol));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall( DMDestroy( &dm ) );
  PetscCall( DMDestroy( &dm_dist ) );
  PetscCall( PetscFinalize() );
  return 0;
}