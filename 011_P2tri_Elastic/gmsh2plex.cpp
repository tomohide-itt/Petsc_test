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
    //PetscCall(DMPlexDistributeCoordinates(dm, NULL));
  }

  //=== DMPlexのセルpointIDとgmshのelementTagの紐づけ ========================================================
  std::map<int,int> eID2pID;
  std::map<int,int> pID2eID;
  PetscCall( get_elemID_map( rank, dm, dim, nodes, elems, eID2pID, pID2eID, false ) );
  //+++
  //{
  //  for( const auto& [pID, eID] : pID2eID )
  //  {
  //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, pID=%7d, eID=%7d\n", rank, pID, eID );
  //  }
  //  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  //}
  //---

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

  //=== 係数行列，右辺ベクトルの作成 ==============================================================================
  Mat A;
  Vec b, sol;
  PetscCall( DMCreateMatrix( dm, &A ) );
  PetscCall( DMCreateGlobalVector( dm, &b ) );
  PetscCall( VecDuplicate( b, &sol ) );

  //=== 頂点，面，セルのIDの範囲を取得 ==============================================================================
  PetscInt vtx_start, vtx_end, face_start, face_end, cell_start, cell_end; // DMPlex内部のセルと頂点の最初のIDと最後のID
  get_vertex_ID_range( rank, dm, vtx_start,  vtx_end,  true );
  get_face_ID_range(   rank, dm, face_start, face_end, true );
  get_cell_ID_range(   rank, dm, cell_start, cell_end, true );

  //=== 座標（ローカル/グローバル）を取得 ==============================================================================
  const PetscScalar *a_loc  = NULL;
  const PetscScalar *a_glob = NULL;
  get_coords( rank, dm, dim, a_loc, a_glob, true );

  //=== 材料定数 ==============================================================================
  double E = 1.0e3;
  double nu = 0.33;
  double lamb = (E*nu)/((1.0+nu)*(1.0-2.0*nu));
  double mu = E/(2.0*(1.0+nu));
  // Dマトリクス
  PetscScalar D[16];
  for( int i=0; i<16; i++ ) D[i] = 0.0;
  D[ 0] = lamb + 2.0*mu;  D[ 1] = lamb;           D[ 2] = lamb;           D[ 3] = 0.0;
  D[ 4] = lamb;           D[ 5] = lamb + 2.0*mu;  D[ 6] = lamb;           D[ 7] = 0.0;
  D[ 8] = lamb;           D[ 9] = lamb;           D[10] = lamb + 2.0*mu;  D[11] = 0.0;
  D[12] = 0.0;            D[13] = 0.0;            D[14] = 0.0;            D[15] = mu;

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
    PetscCall( DMPlexMatSetClosure( dm, loc_section, glob_section, A, c, Ke, ADD_VALUES ) );
    PetscCall( DMPlexVecSetClosure( dm, loc_section, b, c, Fe, ADD_VALUES ) );
    PetscCall( DMPlexRestoreClosureIndices( dm, loc_section, glob_section, c, PETSC_TRUE, &nidx, &idx, NULL, NULL ) );
  }

  PetscCall( MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY ) );
  PetscCall( MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY ) );
  PetscCall( VecAssemblyBegin( b ) );
  PetscCall( VecAssemblyEnd( b ) );

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );

  //=== Dirichlet境界条件 ==============================================================================
  {
    //++++
    DM cdm = NULL;
    Vec coords_loc = NULL;
    const PetscScalar *coords_loc_arr = NULL;
    PetscSection csec = NULL;
    PetscCall( DMGetCoordinateDM( dm, &cdm ) );
    PetscCall( DMGetCoordinatesLocal( dm, &coords_loc ) );
    PetscCall( VecGetArrayRead( coords_loc, &coords_loc_arr ) );
    PetscCall( DMGetCoordinateSection( dm, &csec ) );
    PetscInt depth=-1, pstart=-1, pend=-1, cdof=-1, coff=-1;
    //cdmの有効ポイント範囲
    PetscCall( DMPlexGetChart( cdm, &pstart, &pend ) );
    //+++
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d cdm ps, pe = [%5d,%5d)\n", rank, pstart, pend );
    for( PetscInt cp=pstart; cp<pend; cp++ )
    {
      PetscCall( DMPlexGetPointDepth( dm, cp, &depth ) );
      PetscCall( PetscSectionGetDof( csec, cp, &cdof ) );
      PetscCall( PetscSectionGetOffset( csec, cp, &coff ) );
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d cp=%5d depth=%5d cdof=%5d coff=%5d\n", rank, cp, depth, cdof, coff );
    }
    //---
    //----
    // Tri-P2: closure中のノード -> { V0, E01, V1, E02, E12, V2 }
    static const PetscInt V_ofs[3] = { 0, 2, 5 }; // V0, V1, V2の位置
    static const PetscInt E_ofs[3] = { 1, 4, 3 }; // E01, E12, E20

    // 底面の変位を固定
    PetscInt phys_id = 4;
    // Face Sets ラベル取得
    DMLabel fs_label = NULL;
    PetscCall( DMGetLabel( dm, "Face Sets", &fs_label ) );

    // 対象エッジ集合を取得
    IS faceIS = NULL;
    PetscCall( DMLabelGetStratumIS( fs_label, phys_id, &faceIS ) );

    // それらのエッジ上の DOF と，両端点の DOF を集める
    PetscSection section;
    PetscCall( DMGetLocalSection(dm, &section ) );

    std::vector<PetscInt> locRows;

    if( faceIS )
    {
      const PetscInt* faces = NULL;
      PetscInt nfaces = 0;
      PetscCall( ISGetLocalSize( faceIS, &nfaces ) );
      PetscCall( ISGetIndices( faceIS, &faces ) );

      for( PetscInt i=0; i<nfaces; i++ )
      {
        const PetscInt f = faces[i];

        // 境界辺に隣接するセル c を取得
        const PetscInt *supp = NULL;
        PetscInt nsupp = 0;
        PetscCall( DMPlexGetSupportSize(dm, f, &nsupp ) );
        PetscCall( DMPlexGetSupport( dm, f, &supp ) );
        const PetscInt c = supp[0];
        //+++
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d f=%5d c=%5d\n", rank, f, c );
        //---

        // セルの座標クロージャ(P2tri)
        PetscInt cdof_c = 0;
        PetscScalar *xc = NULL; //borrowing pointer
        PetscCall( DMPlexVecGetClosure(cdm, csec, coords_loc, c, &cdof_c, &xc ) );
        //+++
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d xc:\n", rank );
        for( int j=0; j<12; j++ )
        {
          PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d xc[%5d]=%15.5e\n", rank, j, xc[j] );
        }
        //---
        // cdof_c は 12を想定
        if( cdof_c != 6*dim )
        {
          PetscPrintf( PETSC_COMM_WORLD, "warning cell %d coordinate dof = %d (expected %d)\n",
            c, cdof_c, 6*dim );
        }

        // セル内での辺 f のローカル番号 ie を求める
        const PetscInt *ccone = NULL;
        PetscInt nccone = 0;
        PetscCall( DMPlexGetConeSize( dm, c, &nccone ) ); //3
        PetscCall( DMPlexGetCone( dm, c, &ccone ) );
        PetscInt ie = -1;
        for( PetscInt j=0; j<nccone; j++ )
        {
          if( ccone[j] == f )
          {
            ie = j;
            break;
          }
        }
        if( ie < 0 )
        {
          PetscPrintf(PETSC_COMM_WORLD, "Warn: edge %d not in cell %d cone\n", (int)f, (int)c);
          PetscCall(DMPlexVecRestoreClosure(cdm, csec, coords_loc, c, &cdof_c, &xc));
          continue;
        }

        // 辺 f の中点座標
        const PetscScalar *xmid = xc + dim*E_ofs[ie];

        // 辺 f の両端点座標
        static const PetscInt edgeVerts[3][2] = { {0,1}, {1,2}, {2,0} };
        const PetscInt lv0 = edgeVerts[ie][0];
        const PetscInt lv1 = edgeVerts[ie][1];
        const PetscScalar *xEnd0 = xc + dim * V_ofs[lv0];
        const PetscScalar *xEnd1 = xc + dim * V_ofs[lv1];
        //+++
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d f=%5d\n", rank, f );
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d (x y) =%15.5e%15.5e\n", rank, xmid[0], xmid[1] );
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d (x y) =%15.5e%15.5e\n", rank, xEnd0[0], xEnd0[1] );
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d (x y) =%15.5e%15.5e\n", rank, xEnd1[0], xEnd1[1] );
        //---

        // エッジ自体の DOF （P2の辺中点DOF）
        PetscInt dof = 0;
        PetscInt off = 0;
        PetscCall( PetscSectionGetDof( section, f, &dof ) );
        PetscCall( PetscSectionGetOffset( section, f, &off ) );
        for( PetscInt k=0; k<dof; k++ ) locRows.push_back( off + k );
        //+++
        //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d f=%5d  off=%5d  dof=%5d\n", rank, f, off, dof );
        //---

        // 両端点の DOF (P2の頂点DOF)
        const PetscInt* cone = NULL;
        PetscInt ncone = 0;
        PetscCall( DMPlexGetConeSize( dm, f, &ncone ) );
        PetscCall( DMPlexGetCone( dm, f, &cone ) );

        for( PetscInt cn=0; cn<ncone; cn++ )
        {
          PetscInt vdof = 0;
          PetscInt voff = 0;
          PetscCall( PetscSectionGetDof( section, cone[cn], &vdof ) );
          PetscCall( PetscSectionGetOffset( section, cone[cn], &voff ) );
          for( PetscInt k=0; k<vdof; k++ ) locRows.push_back( voff + k );
          //+++
          //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d f=%5d voff=%5d vdof=%5d\n", rank, f, voff, vdof );
          //---
        }

        // 後片付け
        PetscCall(DMPlexVecRestoreClosure(cdm, csec, coords_loc, c, &cdof_c, &xc));
      }

      PetscCall( ISRestoreIndices( faceIS, &faces ) );
      PetscCall( ISDestroy(&faceIS) );
    }

    // 重複除去
    std::sort( locRows.begin(), locRows.end() );
    locRows.erase( std::unique( locRows.begin(), locRows.end()), locRows.end() );
    //+++
    //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d locRows:\n", rank );
    //for( int i=0; i<locRows.size(); i++ )
    //{
    //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d locRows[%5d] = %5d\n", rank, i, locRows[i] );
    //}
    //---

    // Local -> Global
    IS is_loc = NULL;
    IS is_glob = NULL;
    PetscCall( ISCreateGeneral( PETSC_COMM_WORLD, (PetscInt)locRows.size(), locRows.data(), PETSC_COPY_VALUES, &is_loc ) );
    ISLocalToGlobalMapping l2g;
    PetscCall( DMGetLocalToGlobalMapping(dm, &l2g) );
    PetscCall( ISLocalToGlobalMappingApplyIS( l2g, is_loc, &is_glob ) );
    PetscCall( ISDestroy(&is_loc) );

    // 係数行列と右辺ベクトルへ Diriclet境界を適用 (u,v=0)
    PetscCall( MatZeroRowsColumnsIS( A, is_glob, 1.0, b, NULL ) );
    PetscCall( ISDestroy(&is_glob) );

    PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );

    //+++
    PetscCall( VecRestoreArrayRead( coords_loc, &coords_loc_arr ) );
    //---
  }

  //=== 節点力 ==============================================================================
  {
  // 上面に (0, -10) の節点力
  PetscInt phys_id = 2;
  PetscScalar Fy = -10.0;
  // Face Sets ラベル取得
  DMLabel fs_label = NULL;
  PetscCall( DMGetLabel( dm, "Face Sets", &fs_label ) );

  // 対象エッジ集合を取得
  IS faceIS = NULL;
  PetscCall( DMLabelGetStratumIS( fs_label, phys_id, &faceIS ) );

  if( faceIS )
  {
    PetscSection section;
    PetscCall( DMGetLocalSection(dm, &section ) );
    
    //成分数を取得(Nc=2を想定)
    PetscInt Nc = 0;
    PetscCall( PetscFEGetNumComponents( fe, &Nc ) );

    const PetscInt* faces = NULL;
    PetscInt nfaces = 0;
    PetscCall( ISGetLocalSize( faceIS, &nfaces ) );
    PetscCall( ISGetIndices( faceIS, &faces ) );

    std::vector<PetscInt> idx;
    std::vector<PetscScalar> val;

    for( PetscInt i=0; i<nfaces; i++ )
    {
      const PetscInt f = faces[i];

      // エッジ自体の DOF （P2の辺中点DOF）
      PetscInt dof = 0;
      PetscInt off = 0;
      PetscCall( PetscSectionGetDof( section, f, &dof ) );
      PetscCall( PetscSectionGetOffset( section, f, &off ) );
      if( dof > 0 )
      {
        PetscInt nbf = dof / Nc;
        for( PetscInt j=0; j<nbf; j++ )
        {
          idx.push_back( off + j*Nc + 1 );
          val.push_back( Fy );
        }
      }

      // 両端点の DOF (P2の頂点DOF)
      const PetscInt* cone = NULL;
      PetscInt ncone = 0;
      PetscCall( DMPlexGetConeSize( dm, f, &ncone ) );
      PetscCall( DMPlexGetCone( dm, f, &cone ) );
      for( PetscInt cn=0; cn<ncone; cn++ )
      {
        PetscInt vdof = 0;
        PetscInt voff = 0;
        PetscCall( PetscSectionGetDof( section, cone[cn], &vdof ) );
        PetscCall( PetscSectionGetOffset( section, cone[cn], &voff ) );
        if( vdof > 0 )
        {
          PetscInt nbf = vdof / Nc;
          for( PetscInt j=0; j<nbf; j++ )
          {
            idx.push_back( voff + j*Nc + 1 );
            val.push_back( Fy );
          }
        }
        //+++
        //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d f=%5d offset=%5d dof=%5d\n", rank, f, voff, vdof );
        //---
      }
    }

    PetscCall( ISRestoreIndices( faceIS, &faces ) );
    PetscCall( ISDestroy(&faceIS) );

    // 重複除去
    if( !idx.empty() )
    {
      std::vector< std::pair< PetscInt, PetscScalar > > pairs;
      for( int k=0; k<idx.size(); k++ )
      {
        pairs.push_back( std::make_pair( idx[k], val[k] ) );
      }
      std::sort( pairs.begin(), pairs.end() );
      idx.clear();
      val.clear();
      int k=0;
      while( k < pairs.size() )
      {
        PetscInt key = pairs[k].first;
        PetscScalar sum = 0.0;
        // 同じ key (DOF)　を合計
        do
        {
          sum += pairs[k].second;
          k++;
        } while (k<pairs.size() && pairs[k].first == key);
        idx.push_back( key );
        val.push_back( sum );
      }
    }

    // RHSに加算
    PetscCall( VecSetValuesLocal( b, (PetscInt)idx.size(), idx.data(), val.data(), ADD_VALUES ) );
  }

  // 追加アセンブリ
  PetscCall( VecAssemblyBegin( b ) );
  PetscCall( VecAssemblyEnd( b ) );

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  }

  //=== ソルバーで解く ==============================================================================
  KSP ksp;
  PetscCall( KSPCreate( PETSC_COMM_WORLD, &ksp ) );
  PetscCall( KSPSetOperators( ksp, A, A ) );
  PetscCall(KSPSetType(ksp,KSPCG));
  PetscCall( KSPSetFromOptions(ksp) );
  PetscCall( KSPSolve( ksp, b, sol ) );

  //=== 変位の出力 ==============================================================================
  {
    // グローバル解をローカルに
    Vec sol_loc;
    PetscCall( DMGetLocalVector( dm, &sol_loc ) );
    PetscCall( DMGlobalToLocalBegin( dm, sol, INSERT_VALUES, sol_loc ) );
    PetscCall( DMGlobalToLocalEnd  ( dm, sol, INSERT_VALUES, sol_loc ) );

    // ローカルセクションと座標系
    PetscSection lsec;
    PetscCall( DMGetLocalSection( dm, &lsec ) );

    DM cdm;
    Vec crd;
    PetscCall( DMGetCoordinateDM( dm, &cdm ) );
    PetscCall( DMGetCoordinatesLocal( dm, &crd ) );

    PetscSection csec;
    //PetscCall(DMGetLocalSection(cdm, &csec));
    PetscCall(DMGetCoordinateSection(dm, &csec));

    //++++
    PetscInt cs=0;
    PetscInt ce=0;
    PetscInt nloc = 0;
    PetscInt nglob = 0;
    PetscCall( PetscSectionGetChart( csec, &cs, &ce ) );
    PetscCall( VecGetLocalSize(crd, &nloc) );
    PetscCall( VecGetSize( crd, &nglob ) );
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,
      "rank = %3d [coord] crd=%p  size(local/global)=%d/%d  chart=[%d,%d)\n",
      rank, (void*)crd, (int)nloc, (int)nglob, (int)cs, (int)ce);

    // 範囲（セル）
    PetscInt c_start, c_end;
    PetscCall( DMPlexGetHeightStratum( dm, 0, &c_start, &c_end ) );

    // 配列ポインタを取得
    const PetscScalar* a_loc = NULL;
    const PetscScalar* a_crd = NULL;
    PetscCall( VecGetArrayRead( sol_loc, &a_loc ) );
    PetscCall( VecGetArrayRead( crd,     &a_crd ) );

    //+++
    //for( PetscInt i=0; i<nloc; i=i+2 )
    //{
    //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d a_crd[%5d], a_crd[%5d]=%15.5e%15.5e\n",
    //      rank, i, i+1, a_crd[i], a_crd[i+1] );
    //}
    //---

    PetscInt Nc = 0;
    PetscCall(PetscFEGetNumComponents(fe, &Nc));

    // 出力
    for( PetscInt c=c_start; c<c_end; c++ )
    {
      PetscInt npts = 0;
      PetscInt* pts = NULL;
      int eID = pID2eID[c];
      PetscCall( DMPlexGetTransitiveClosure( dm, c, PETSC_TRUE, &npts, &pts ) );
      for( PetscInt k=0; k<npts; k++ )
      {
        const PetscInt p = pts[2*k];
        PetscInt depth;
        PetscCall( DMPlexGetPointDepth( dm, p, &depth ) );
        if( depth == 2 ) continue;

        PetscInt dof = 0;
        PetscInt off = 0;
        PetscCall( PetscSectionGetDof( lsec, p, &dof ) );
        PetscCall( PetscSectionGetOffset( lsec, p, &off ) );

        // 解ベクトル（ローカル）のこの点の先頭アドレス
        const PetscScalar ux  = a_loc[off + 0];
        const PetscScalar uy  = a_loc[off + 1];

        const PetscScalar *cp = NULL;
        //PetscCall( DMPlexPointLocalRead( cdm, p, a_crd, &cp ) );
        PetscCall(DMPlexPointLocalRead(dm, p, a_crd, &cp));

        double x=0.0, y=0.0;
        if( cp )
        {
          x = (double)cp[0];
          y = (double)cp[1];
        }

        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c= %5d eID=%5d k=%5d p=%5d depth=%5d dof=%5d off=%5d (x y)=%15.5e%15.5e (u v)=%15.5e%15.5e\n",
          rank, c, eID, k, p, depth, dof, off, x, y, ux, uy );
      }
      PetscCall( DMPlexRestoreTransitiveClosure( dm, c, PETSC_TRUE, &npts, &pts ) );
    }

    PetscCall( VecRestoreArrayRead( sol_loc, &a_loc ) );
    PetscCall( VecRestoreArrayRead( crd,     &a_crd ) );
    PetscCall( DMRestoreLocalVector( dm,     &sol_loc ) );

    PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  }
  
  PetscCall(VecDestroy(&sol));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&dmDist));
  PetscCall(PetscFinalize());
  return 0;
}