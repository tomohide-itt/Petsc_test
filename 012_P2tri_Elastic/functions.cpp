#include "functions.h"

// ----------------------------------------------------------------------------
// .mshをDMPlexで読み込む 
PetscErrorCode read_gmsh( const std::string& mesh_path, DM &dm )
{
  // メッシュを補完ありで読み込む（第4引数はinterpolate=1に対応）
  PetscCall( DMPlexCreateFromFile( PETSC_COMM_WORLD, mesh_path.c_str(), NULL, PETSC_TRUE, &dm ) );
  // 次元の取得
  PetscInt dim;
  PetscCall( DMGetDimension( dm, &dim ) );
  // 次元が2なら何もしない．そうでなければメッセージを出力して処理を中断 
  PetscCheck( dim==2, PETSC_COMM_WORLD, PETSC_ERR_SUP, "このサンプルは2D専用です");
  //+++
  PetscPrintf( PETSC_COMM_WORLD, "%s[%d] dim = %d\n", __FUNCTION__, __LINE__, (int)dim );
  //---
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// dm を領域分割する
PetscErrorCode partition_mesh( DM &dm, DM &dm_dist )
{
  PetscMPIInt nproc;
  MPI_Comm_size( PETSC_COMM_WORLD, &nproc );
  if( nproc > 1 )
  {
    PetscPartitioner part;
    PetscCall( DMPlexGetPartitioner( dm, &part ) );
    PetscCall( PetscPartitionerSetType( part, PETSCPARTITIONERPARMETIS ) );
    PetscCall( DMPlexDistribute( dm, 0, NULL, &dm_dist ) );
    if( dm_dist )
    {
      PetscCall( DMDestroy( &dm ) );
      dm = dm_dist;
    }
  }
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// DMPlexのセルpointIDとgmshのelementTagの紐づけ
PetscErrorCode get_elemID_map( const DM& dm, const node_vec& nodes, const elem_vec& elems,
  std::map<int,int>& eID2pID, std::map<int,int>& pID2eID, const bool debug )
{
  // rankの取得
  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

  // 次元の取得
  PetscInt dim;
  PetscCall( DMGetDimension( dm, &dim ) );

  // ローカルで座標の配列を取得
  Vec crds_loc = NULL;
  PetscCall( DMGetCoordinatesLocal( dm, &crds_loc ) );
  const PetscScalar *crds_loc_arr;
  if( crds_loc ) PetscCall( VecGetArrayRead( crds_loc, &crds_loc_arr ) );

  // 座標セクションとセルのポイント番号の範囲を取得
  PetscSection csec = NULL;
  PetscInt c_start=0, c_end=0;
  PetscCall( DMGetCoordinateSection( dm, &csec ) );
  PetscCall( DMPlexGetHeightStratum( dm, 0, &c_start, &c_end ) );

  std::vector<bool> visited_elem( elems.size(), false );
  for( PetscInt c=c_start; c<c_end; c++ )
  {
    // セルごとに節点の座標値を取得
    PetscInt dof, off;
    PetscCall( PetscSectionGetDof(    csec, c, &dof ) );
    PetscCall( PetscSectionGetOffset( csec, c, &off ) );
    PetscInt num_node = dof/dim;
    std::vector<std::vector<double>> xyz( num_node, std::vector<double>( dim, 0.0 ) );
    for( int n=0; n<num_node; n++ )
    {
      for( int i=0; i<dim; i++ )
      {
        xyz[n][i] = (double)PetscRealPart( crds_loc_arr[ off + n*dim + i ] );
      }
    }
    if( debug )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d %s[%d] : coordinates\n", rank, __FUNCTION__, __LINE__ );
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d num_node=%5d\n", rank, c, num_node );
      for( int n=0; n<num_node; n++ )
      {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d", rank );
        for( int i=0; i<dim; i++ )
        {
          PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", xyz[n][i] );
        }
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
      }
    }

    //---
    // elemsと突き合わせて，6点の座標値が一致していれば，そのelemsのIDとpのマップを登録する
    double tol = 1.0e-8;

    for( const elem &e : elems )
    {
      if( e.nodeIDs.size() != num_node ) continue;
      int eidx = elems.idx_of_id( e.ID );
      if( visited_elem[eidx] ) continue;
      
      if( 0 )
      {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "check elem ID=%d\n", e.ID );
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "eidx=%d\n", eidx );
      }
      
      std::vector<bool> visited_node( num_node, false );
      bool identical = true;
      for( int nid=0; nid<num_node; nid++ )
      {
        const node& nd = nodes.id_of(e.nodeIDs[nid]);
        
        if( 0 )
        {
          PetscSynchronizedPrintf( PETSC_COMM_WORLD, "  e.nodeIDs[%5d]=%d\n", nid, e.nodeIDs[nid] );
          PetscSynchronizedPrintf( PETSC_COMM_WORLD, "  node ID=%d\n", nd.ID );
          PetscSynchronizedPrintf( PETSC_COMM_WORLD, "  " );
          if( dim >= 1 ) PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", nd.x );
          if( dim >= 2 ) PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", nd.y );
          if( dim >= 3 ) PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", nd.z );
          PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
        }

        bool find = false;
        for( int n=0; n<num_node; n++ )
        {
          if( visited_node[n] ) continue;
          bool same = true;
          if( dim == 2 )
          {
            same = same && close2( nd.x, xyz[n][0], tol ) && close2( nd.y, xyz[n][1], tol );
          }
          if( same )
          {
            visited_node[n] = true;
            find = true;
            break;
          }
        }
        
        if( 0 ) PetscSynchronizedPrintf( PETSC_COMM_WORLD, "  find=%d\n", find );
        
        identical = identical && find;
        if( !find ) break;
      }
      if( identical )
      {
        visited_elem[eidx] = true;
        eID2pID[e.ID] = c;
        pID2eID[c] = e.ID;
        if( 0 )
        {
          PetscSynchronizedPrintf( PETSC_COMM_WORLD, "elems.ID = %d is identical with %d\n", e.ID, c );
          PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
        }
        break;
      }
    }
  }

  if (crds_loc)  PetscCall( VecRestoreArrayRead( crds_loc,  &crds_loc_arr ) );
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );

  if( debug )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d %s[%d] : pID2eID\n", rank, __FUNCTION__, __LINE__ );
    for( const auto& [pID, eID] : pID2eID )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, pID=%7d, eID=%7d\n", rank, pID, eID );
    }
  }
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// FE空間を作成する
PetscErrorCode create_FE( const DM& dm, PetscFE& fe )
{
  // 次元の取得
  PetscInt dim;
  PetscCall( DMGetDimension( dm, &dim ) );

  // 2D三角形要素（P2）で2成分ベクトル場（変位）に対する2次有限要素空間を作る
  // 引数は，コミュニケータ，次元，フィールド成分数，要素形状がsimplexか，P1/P2, 積分次数，出力先の有限要素オブジェクト
  PetscCall( PetscFECreateLagrange( PETSC_COMM_WORLD, dim, dim, PETSC_TRUE, 2, PETSC_DETERMINE, &fe ) );

  // dm にフィールド0を登録
  // 引数は，メッシュオブジェクト，フィールド番号，ラベル，そのフィールドに対応する有限要素オブジェクト
  PetscCall( DMSetField( dm, 0, NULL, (PetscObject)fe ) );

  // Discrete System (PetscDS) を dm から生成する （自由度構造の確定）
  // 内部では，dmに紐づいた PetscSection を構築，PDE用のオブジェクト PetscDS を作成，各フィールドごとに PetscFE の情報を関連づけを行っている
  PetscCall( DMCreateDS( dm ) );

  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// Dマトリクスを計算する（線形弾性）
PetscErrorCode cal_D_matrix( const double E, const double nu, PetscScalar* D )
{
  double lamb = (E*nu)/((1.0+nu)*(1.0-2.0*nu));
  double mu = E/(2.0*(1.0+nu));
  for( int i=0; i<16; i++ ) D[i] = 0.0;
  D[ 0] = lamb + 2.0*mu;  D[ 1] = lamb;           D[ 2] = lamb;           D[ 3] = 0.0;
  D[ 4] = lamb;           D[ 5] = lamb + 2.0*mu;  D[ 6] = lamb;           D[ 7] = 0.0;
  D[ 8] = lamb;           D[ 9] = lamb;           D[10] = lamb + 2.0*mu;  D[11] = 0.0;
  D[12] = 0.0;            D[13] = 0.0;            D[14] = 0.0;            D[15] = mu;
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// Kマトリクスを計算してマージする
PetscErrorCode merge_Kuu_matrix( const DM& dm )
{
  // ローカルセクション，グローバルセクションを取得
  PetscSection loc_sec, glb_sec;
  

  PetscFunctionReturn( PETSC_SUCCESS );
}

// Dirichlet境界条件 の設定
PetscErrorCode set_Dirichlet_zero( const int rank, const DM& dm, const PetscInt phys_id, Mat& A, Vec& b )
{
  // Face Sets ラベル取得
  DMLabel label = NULL;
  PetscCall( DMGetLabel( dm, "Face Sets", &label ) );

  // 対象エッジ集合を取得
  IS faceIS = NULL;
  PetscCall( DMLabelGetStratumIS( label, phys_id, &faceIS ) );

  // セクション取得
  PetscSection loc_sec;
  PetscSection glb_sec;
  PetscCall( DMGetLocalSection( dm, &loc_sec ) );
  PetscCall( DMGetGlobalSection( dm, &glb_sec ) );

  // エッジ上のDOFとそのエッジの両端点のDOFを集める
  std::vector<PetscInt> row_idxs; // グローバル方程式番号

  if( faceIS )
  {
    const PetscInt* faces = NULL;
    PetscInt nfaces = 0;
    PetscCall( ISGetLocalSize( faceIS, &nfaces ) );
    PetscCall( ISGetIndices( faceIS, &faces ) );

    for( PetscInt i=0; i<nfaces; i++ )
    {
      const PetscInt f = faces[i];

      // f の節点出力
      PetscCall( show_coords_boundary( rank, dm, f ) );

      // P2の辺中点DOF
      PetscInt dof=0, off=0;
      PetscCall( PetscSectionGetDof(    glb_sec, f, &dof ) );
      PetscCall( PetscSectionGetOffset( glb_sec, f, &off ) );
      for( PetscInt k=0; k<dof; k++ )
      {
        if( off + k >= 0 ) row_idxs.push_back( off + k );
      }

      // P2の頂点DOF
      const PetscInt* cone = NULL;
      PetscInt ncone = 0;
      PetscCall( DMPlexGetConeSize( dm, f, &ncone ) );
      PetscCall( DMPlexGetCone(     dm, f, &cone  ) );
      for( PetscInt cn=0; cn<ncone; cn++ )
      {
        PetscInt v = cone[cn];
        PetscInt vdof=0, voff=0;
        PetscCall( PetscSectionGetDof(    glb_sec, v, &vdof ) );
        PetscCall( PetscSectionGetOffset( glb_sec, v, &voff ) );
        for( PetscInt k=0; k<vdof; k++ )
        {
          if( voff + k >= 0 ) row_idxs.push_back( voff + k );
        }
      }
    }

    PetscCall( ISRestoreIndices( faceIS, &faces ) );
    PetscCall( ISDestroy( &faceIS ) );
  }

  // 重複除去
  std::sort( row_idxs.begin(), row_idxs.end() );
  row_idxs.erase( std::unique( row_idxs.begin(), row_idxs.end() ), row_idxs.end() );

  // 追加の範囲チェック（デバッグ用）
  PetscInt nrowsA, ncolsA;
  PetscCall(MatGetSize(A, &nrowsA, &ncolsA));
  for (size_t i = 0; i < row_idxs.size(); ++i) {
    if (row_idxs[i] < 0 || row_idxs[i] >= nrowsA) {
      PetscPrintf(PETSC_COMM_SELF,
        "BC index out of range: %d (valid [0,%d))\n",
        (int)row_idxs[i], (int)nrowsA);
    }
  }

  // Local -> Global
  //IS is_loc = NULL;
  IS is_glb = NULL;
  PetscCall( ISCreateGeneral( PETSC_COMM_WORLD, (PetscInt)row_idxs.size(), row_idxs.data(), PETSC_COPY_VALUES, &is_glb ) );
  //PetscCall( ISCreateGeneral( PETSC_COMM_WORLD, (PetscInt)row_idxs.size(), row_idxs.data(), PETSC_COPY_VALUES, &is_loc ) );
  //ISLocalToGlobalMapping l2g;
  //PetscCall( DMGetLocalToGlobalMapping( dm, &l2g ) );
  //PetscCall( ISLocalToGlobalMappingApplyIS( l2g, is_loc, &is_glb ) );
  //PetscCall( ISDestroy(&is_loc) );

  Vec xbc;
  PetscCall( DMCreateGlobalVector( dm, &xbc ) );
  PetscCall( VecZeroEntries( xbc ) );

  // 係数行列と右辺ベクトルへ Diriclet境界を適用 (u,v=0)
  PetscCall( MatZeroRowsColumnsIS( A, is_glb, 1.0, xbc, b ) );

  PetscCall( ISDestroy(&is_glb) );
  PetscCall( VecDestroy(&xbc) );

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// 節点力を設定
PetscErrorCode set_nodal_force( const int rank, const DM& dm, const PetscFE& fe,
  const PetscInt phys_id, const PetscScalar F, const PetscInt dir, Vec& b )
{
  // Face Sets ラベル取得
  DMLabel label = NULL;
  PetscCall( DMGetLabel( dm, "Face Sets", &label ) );

  // 対象エッジ集合を取得
  IS faceIS = NULL;
  PetscCall( DMLabelGetStratumIS( label, phys_id, &faceIS ) );

  // ローカル Vec の作成
  Vec bloc = NULL;
  PetscCall( DMCreateLocalVector( dm, &bloc ) );
  PetscCall( VecZeroEntries( bloc ) );

  // セクション取得
  PetscSection loc_sec = NULL;
  PetscCall( DMGetLocalSection( dm, &loc_sec ) );

  //成分数を取得(Nc=2を想定)
  PetscInt Nc = 0;
  PetscCall( PetscFEGetNumComponents( fe, &Nc ) );

  if( faceIS )
  {

    const PetscInt* faces = NULL;
    PetscInt nfaces = 0;
    PetscCall( ISGetLocalSize( faceIS, &nfaces ) );
    PetscCall( ISGetIndices( faceIS, &faces ) );

    for( PetscInt i=0; i<nfaces; i++ )
    {
      const PetscInt f = faces[i];

      // f の節点出力
      PetscCall( show_coords_boundary( rank, dm, f ) );

      // P2の辺中点DOF
      PetscInt dof=0, off=0;
      PetscCall( PetscSectionGetDof(    loc_sec, f, &dof ) );
      PetscCall( PetscSectionGetOffset( loc_sec, f, &off ) );
      if( dof > 0 )
      {
        PetscInt nbf = dof / Nc;
        for( PetscInt j=0; j<nbf; j++ )
        {
          const PetscInt rloc = off + j*Nc + dir; // ローカル添え字
          PetscCall( VecSetValuesLocal( bloc, 1, &rloc, &F, ADD_VALUES ) );
        }
      }

      // P2の頂点DOF
      const PetscInt* cone = NULL;
      PetscInt ncone = 0;
      PetscCall( DMPlexGetConeSize( dm, f, &ncone ) );
      PetscCall( DMPlexGetCone(     dm, f, &cone  ) );
      for( PetscInt cn=0; cn<ncone; cn++ )
      {
        PetscInt v = cone[cn];
        PetscInt vdof=0, voff=0;
        PetscCall( PetscSectionGetDof(    loc_sec, v, &vdof ) );
        PetscCall( PetscSectionGetOffset( loc_sec, v, &voff ) );
        if( vdof > 0 )
        {
          PetscInt nbf = vdof / Nc;
          for( PetscInt j=0; j<nbf; j++ )
          {
            const PetscInt rloc = voff + j*Nc + dir; // ローカル添え字
            PetscCall( VecSetValuesLocal( bloc, 1, &rloc, &F, ADD_VALUES ) );
          }
        }
      }
    }
    PetscCall( ISRestoreIndices( faceIS, &faces ) );
    PetscCall( ISDestroy(&faceIS) );
  }

  // ローカル -> グローバルへ集約
  PetscCall( VecAssemblyBegin(bloc) );
  PetscCall( VecAssemblyEnd(bloc) );
  PetscCall( DMLocalToGlobalBegin( dm, bloc, ADD_VALUES, b ) );
  PetscCall( DMLocalToGlobalEnd( dm, bloc, ADD_VALUES, b ) );

  PetscCall( VecDestroy( &bloc ) );

  // check
  PetscScalar sum=0;
  PetscCall( VecNorm( b, NORM_1, &sum ) );
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d vec_norm=%15.5e\n", rank, sum );

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// pointID が depth=0 (頂点)のとき，座標を計算する 
PetscErrorCode get_coords_vertex( const int rank, const DM& dm, const PetscInt p, std::vector<double>& xy )
{
  DM cdm = NULL; //座標用DM
  Vec coords_loc = NULL;
  PetscSection csec = NULL; //座標用セクション
  PetscInt dim = 0;

  PetscCall( DMGetCoordinateDM( dm, &cdm ) );
  PetscCall( DMGetCoordinatesLocal( dm, &coords_loc ) );
  PetscCall( DMGetCoordinateSection( dm, &csec ) );
  PetscCall( DMGetCoordinateDim( dm, &dim ) );

  // 頂点 p を持つ辺 f を取得
  const PetscInt *supp = NULL;
  PetscInt nsupp = 0;
  PetscCall( DMPlexGetSupportSize(dm, p, &nsupp ) );
  PetscCall( DMPlexGetSupport( dm, p, &supp ) );
  const PetscInt f = supp[0];

  // 辺 f を持つセル c を取得
  supp = NULL;
  nsupp = 0;
  PetscCall( DMPlexGetSupportSize(dm, f, &nsupp ) );
  PetscCall( DMPlexGetSupport( dm, f, &supp ) );
  const PetscInt c = supp[0];

  // セル座標のクロシージャ
  PetscInt cdof = 0;
  PetscScalar *xc = NULL;
  PetscCall( DMPlexVecGetClosure(cdm, csec, coords_loc, c, &cdof, &xc ) );

  //+++
  //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d p(%5d) c=%5d xc:\n", rank, p, c );
  //for( int i=0; i<cdof; i=i+2 )
  //{
  //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d xc[%5d]=%15.5e%15.5e\n", rank, i, xc[i], xc[i+1] );
  //}
  //---

  // セル内での辺 f のローカル番号 ie を求める
  const PetscInt *cone = NULL;
  PetscInt ncone = 0;
  const PetscInt *cori = NULL;
  PetscCall( DMPlexGetConeSize( dm, c, &ncone ) );
  PetscCall( DMPlexGetCone( dm, c, &cone ) );
  PetscCall( DMPlexGetConeOrientation( dm, c, &cori ) );
  PetscInt ie = -1;
  for( PetscInt j=0; j<ncone; j++ )
  {
    if( cone[j] == f )
    {
      ie = j;
      break;
    }
  }
  PetscInt ori = cori[ie];

  //+++ 各 cone の並び確認
  //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d p(%5d) c=%5d:\n", rank, p, c );
  //for( int i=0; i<ncone; i++ )
  //{
  //  PetscInt fc = cone[i];
  //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "  rank=%3d cone[%5d]=%5d cori[%5d]=%5d\n", rank, i, fc, i, cori[i] );
  //  const PetscInt *vc = NULL;
  //  PetscInt nvc = 0;
  //  PetscCall( DMPlexGetConeSize( dm, fc, &nvc ) );
  //  PetscCall( DMPlexGetCone( dm, fc, &vc ) );
  //  for( int j=0; j<nvc; j++ )
  //  {
  //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "    rank=%3d vcone[%5d]=%5d\n", rank, j, vc[j] );
  //  }
  //}
  //---

  // 辺 f のどちらかの番号 iv を求める
  const PetscInt *vcone = NULL;
  PetscInt nvcone = 0;
  PetscCall( DMPlexGetConeSize( dm, f, &nvcone ) );
  PetscCall( DMPlexGetCone( dm, f, &vcone ) );
  PetscInt iv = -1;
  for( PetscInt j=0; j<nvcone; j++ )
  {
    if( vcone[j] == p )
    {
      if( ori >= 0 ) iv = j;
      if( ori <  0 ) iv = ( nvcone - 1 ) - j;
      break;
    }
  }

  // Tri-P2: closure中のノード -> { V0, E01, V1, E02, E12, V2 }
  static const PetscInt V_ofs[3] = { 0, 2, 5 }; // V0, V1, V2の位置
  static const PetscInt E_ofs[3] = { 1, 4, 3 }; // E01, E12, E20
  static const PetscInt edgeVerts[3][2] = { {0,1}, {1,2}, {2,0} };

  // 頂点 p の座標
  const PetscInt lv = edgeVerts[ie][iv];
  const PetscScalar *x_vtx= xc + dim * V_ofs[lv];

  //+++
  //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d p(%5d) is contained in the face, cell(%5d,%5d) ie=%5d iv=%5d lv=%5d\n", rank, p, f, c, ie, iv, lv );
  //---

  //
  xy.resize( dim );
  xy[0] = x_vtx[0];
  xy[1] = x_vtx[1];

  // 後片付け
  PetscCall( DMPlexVecRestoreClosure( cdm, csec, coords_loc, c, &cdof, &xc ) );

  PetscFunctionReturn( PETSC_SUCCESS );
}

// pointID が depth=1 (辺)のとき，座標を計算する 
PetscErrorCode get_coords_face( const int rank, const DM& dm, const PetscInt p, std::vector<double>& xy )
{
  DM cdm = NULL; //座標用DM
  Vec coords_loc = NULL;
  PetscSection csec = NULL; //座標用セクション
  PetscInt dim = 0;

  PetscCall( DMGetCoordinateDM( dm, &cdm ) );
  PetscCall( DMGetCoordinatesLocal( dm, &coords_loc ) );
  PetscCall( DMGetCoordinateSection( dm, &csec ) );
  PetscCall( DMGetCoordinateDim( dm, &dim ) );

  // 境界辺に隣接するセル c を取得
  const PetscInt *supp = NULL;
  PetscInt nsupp = 0;
  PetscCall( DMPlexGetSupportSize(dm, p, &nsupp ) );
  PetscCall( DMPlexGetSupport( dm, p, &supp ) );
  const PetscInt c = supp[0];

  // セル座標のクロシージャ
  PetscInt cdof = 0;
  PetscScalar *xc = NULL;
  PetscCall( DMPlexVecGetClosure(cdm, csec, coords_loc, c, &cdof, &xc ) );

  // セル内での辺 f のローカル番号 ie を求める
  const PetscInt *cone = NULL;
  PetscInt ncone = 0;
  PetscCall( DMPlexGetConeSize( dm, c, &ncone ) );
  PetscCall( DMPlexGetCone( dm, c, &cone ) );
  PetscInt ie = -1;
  for( PetscInt j=0; j<ncone; j++ )
  {
    if( cone[j] == p )
    {
      ie = j;
      break;
    }
  }
  if (ie < 0) {
    PetscCall( DMPlexVecRestoreClosure( cdm, csec, coords_loc, c, &cdof, &xc ) );
    PetscFunctionReturn( PETSC_SUCCESS );
  }

  // Tri-P2: closure中のノード -> { V0, E01, V1, E02, E12, V2 }
  static const PetscInt E_ofs[3] = { 1, 4, 3 }; // E01, E12, E20

  // 辺 p の中点座標
  const PetscScalar *xmid = xc + dim*E_ofs[ie];

  //
  xy.resize( dim );
  xy[0] = xmid[0];
  xy[1] = xmid[1];

  // 後片付け
  PetscCall( DMPlexVecRestoreClosure( cdm, csec, coords_loc, c, &cdof, &xc ) );

  PetscFunctionReturn( PETSC_SUCCESS );
}

// 変位の出力
PetscErrorCode show_displacement( const int rank, const DM& dm, const Vec& sol )
{
  // セクション取得など
  PetscSection sec;
  Vec coords_loc = NULL;
  PetscInt dim = 0;
  PetscCall( DMGetCoordinatesLocal( dm, &coords_loc ) );
  PetscCall( DMGetLocalSection( dm, &sec ) );
  PetscCall( DMGetCoordinateDim( dm, &dim ) );

  // グローバル解をローカルに
  Vec sol_loc;
  PetscCall( DMGetLocalVector( dm, &sol_loc ) );
  PetscCall( DMGlobalToLocalBegin( dm, sol, INSERT_VALUES, sol_loc ) );
  PetscCall( DMGlobalToLocalEnd  ( dm, sol, INSERT_VALUES, sol_loc ) );

  // 配列ポインタを取得
  const PetscScalar* sol_arr = NULL;
  PetscCall( VecGetArrayRead( sol_loc, &sol_arr ) );

  // 範囲（セル）
  PetscInt c_start=0, c_end=0;
  PetscCall( DMPlexGetHeightStratum( dm, 0, &c_start, &c_end ) );

  // セルでループ
  for( PetscInt c=c_start; c<c_end; c++ )
  {
    PetscInt npts = 0;
    PetscInt* pts = NULL;
    PetscCall( DMPlexGetTransitiveClosure( dm, c, PETSC_TRUE, &npts, &pts ) );

    // このセルのポイントでループ
    for( PetscInt k=0; k<npts; k++ )
    {
      const PetscInt p = pts[2*k];
      PetscInt depth;
      PetscCall( DMPlexGetPointDepth( dm, p, &depth ) );
      if( depth == 2 ) continue; // pがセルなら飛ばす

      PetscInt dof = 0;
      PetscInt off = 0;
      PetscCall( PetscSectionGetDof( sec, p, &dof ) );
      PetscCall( PetscSectionGetOffset( sec, p, &off ) );

      // 解ベクトル
      const PetscScalar ux  = sol_arr[off + 0];
      const PetscScalar uy  = sol_arr[off + 1];

      // 座標
      std::vector<double> xy( dim, 0.0 );
      if( depth == 1 )
      {
        PetscCall( get_coords_face( rank, dm, p, xy ) );
      }
      if( depth == 0 )
      {
        PetscCall( get_coords_vertex( rank, dm, p, xy ) );
      }

      // 出力
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c= %5d k=%5d p=%5d depth=%5d dof=%5d off=%5d (x y)=%15.5e%15.5e (u v)=%15.5e%15.5e\n",
          rank, c, k, p, depth, dof, off, xy[0], xy[1], ux, uy );
    }
    PetscCall( DMPlexRestoreTransitiveClosure( dm, c, PETSC_TRUE, &npts, &pts ) );
  }

  // 後片付け
  PetscCall( VecRestoreArrayRead( sol_loc, &sol_arr ) );
  PetscCall( DMRestoreLocalVector( dm,     &sol_loc ) );


  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// セル c のローカル DOF 対応とke行番号の対応を作る
PetscErrorCode build_cell_dof_map( const int rank, const DM& dm, const PetscSection& sec, const PetscInt c,
  PetscInt& ncelldof, PetscInt* idx, PetscInt* pt, PetscInt* comp )
{
  // セル c のトランジティブ・クロージャ（点の並び）を取得
  // clos は [point, orientation, point, orientation, ...] の 2*nclos 長
  PetscInt *clos = NULL;
  PetscInt nclos = 0;
  PetscCall( DMPlexGetTransitiveClosure( dm, c, PETSC_TRUE, &nclos, &clos ) );

  // 変位の成分数 Nc を取得（単一フィールドのベクトル2成分を想定）
  PetscInt Nc = 1; // 成分数
  PetscInt Nf = 0; // フィールド数
  PetscCall( PetscSectionGetNumFields(sec, &Nf) );
  if( Nf > 0 )
  {
    // 有限要素なら DMGetField で PetscFE から成分数をとるのが確実
    PetscObject fobj = NULL;
    DMGetField( dm, 0, NULL, &fobj );
    if( fobj )
    {
      PetscFE fe = (PetscFE)fobj;
      PetscCall( PetscFEGetNumComponents( fe, &Nc ) );
    }
  }
  else 
  {
    // Nf==0 の場合は「全部ひとつの場」として格納されており、
    // P2弾性なら頂点/辺の dof は 2 になるはず。最初に正の dof を見つけて推定しておく
  }
  //+++
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d Nf=%5d Nc=%5d\n", rank, c, Nf, Nc );
  for( int i=0; i<nclos; i++ )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d clos[%3d]=%5d%5d (point, orientation)\n", rank, c, i, clos[2*i], clos[2*i+1] );
  }
  //---

  // クロージャ順に sec から（offset, dof）を拾ってローカル添字を並べる
  PetscInt pos = 0;
  for( PetscInt i=0; i<nclos; i++ )
  {
    PetscInt p = clos[2*i];
    PetscInt dof=0, off=0;
    PetscCall( PetscSectionGetDof( sec, p, &dof ) );
    if( !dof ) continue; // セル本体など dof=0 はスキップ
    PetscCall( PetscSectionGetOffset( sec, p, &off ) );
    //+++
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d p=%5d dof=%5d off=%5d\n", rank, c, p, dof, off );
    //---
    for( PetscInt k=0; k<dof; k++ )
    {
      idx[pos] = off + k;
      pt[pos] = p;
      comp[pos] = (Nc>0)? (k%Nc) : 0;
      pos++;
    }
  }
  ncelldof = pos;
  //+++
  for( int i=0; i<ncelldof; i++ )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d idx[%3d]=%5d pt[%3d]=%5d comp[%3d]=%5d\n",
      rank, c, i, idx[i], i, pt[i], i, comp[i] );
  }
  //---

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );

  PetscCall( DMPlexRestoreTransitiveClosure( dm, c, PETSC_TRUE, &nclos, &clos ) );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// 頂点のIDの範囲を出力
PetscErrorCode show_vertexID_range( const DM& dm )
{
  // rankの取得
  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

  //最初のIDと最後のIDを取得
  PetscInt start=0, end=0, num=0;
  PetscCall( DMPlexGetDepthStratum( dm, 0, &start, &end ) );
  num = end - start;

  //出力
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d [vertex_start, vertex_end) = [%5d,%5d ) num = %5d\n", rank, start, end, num );
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// 面のIDの範囲を出力
PetscErrorCode show_faceID_range( const DM& dm )
{
  // rankの取得
  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

  // 最初のIDと最後のIDを取得
  PetscInt start=0, end=0, num=0;
  PetscCall( DMPlexGetHeightStratum( dm, 1, &start, &end ) );
  num = end - start;
  
  // 出力
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d [face_start,   face_end  ) = [%5d,%5d ) num = %5d\n", rank, start, end, num );
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// セルのIDの範囲を出力
PetscErrorCode show_cellID_range( const DM& dm )
{
  // rankの取得
  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

  // 最初のIDと最後のIDを取得
  PetscInt start=0, end=0, num=0;
  PetscCall( DMPlexGetHeightStratum( dm, 0, &start, &end ) );
  num = end - start;
  
  // 出力
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d [cell_start,   cell_end  ) = [%5d,%5d ) num = %5d\n", rank, start, end, num );
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// 座標セクションを使って，各セルの節点座標を出力する
PetscErrorCode show_coords_each_cell( const DM& dm )
{
  // rankの取得
  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

  DM cdm = NULL; //座標用DM
  Vec crds_loc = NULL;
  PetscSection csec = NULL; //座標用セクション

  PetscCall( DMGetCoordinateDM( dm, &cdm ) );
  PetscCall( DMGetCoordinatesLocal( dm, &crds_loc ) );
  PetscCall( DMGetCoordinateSection( dm, &csec ) );

  PetscInt c_start=0, c_end=0;
  PetscCall( DMPlexGetHeightStratum( cdm, 0, &c_start, &c_end ) );

  for( PetscInt c=c_start; c<c_end; c++ )
  {
    PetscInt cdof = 0;
    PetscScalar *xc = NULL;
    PetscCall( DMPlexVecGetClosure( cdm, csec, crds_loc, c, &cdof, &xc ) );
    //+++
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d\n", rank, c );
    for( int i=0; i<cdof; i++ )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d xc[%5d]=%15.5e\n", rank, i, xc[i] );
    }
    //---
    PetscCall( DMPlexVecRestoreClosure( cdm, csec, crds_loc, c, &cdof, &xc ) );
  }

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// f が境界のとき，その境界上の節点の座標値を出力する（P2三角形のみ対応）
PetscErrorCode show_coords_boundary( const int rank, const DM& dm, const PetscInt f )
{
  DM cdm = NULL; //座標用DM
  Vec coords_loc = NULL;
  PetscSection csec = NULL; //座標用セクション
  PetscInt dim = 0;

  PetscCall( DMGetCoordinateDM( dm, &cdm ) );
  PetscCall( DMGetCoordinatesLocal( dm, &coords_loc ) );
  PetscCall( DMGetCoordinateSection( dm, &csec ) );
  PetscCall( DMGetCoordinateDim( dm, &dim ) );

  // 境界辺に隣接するセル c を取得
  const PetscInt *supp = NULL;
  PetscInt nsupp = 0;
  PetscCall( DMPlexGetSupportSize(dm, f, &nsupp ) );
  PetscCall( DMPlexGetSupport( dm, f, &supp ) );
  const PetscInt c = supp[0];

  // セル座標のクロシージャ
  PetscInt cdof = 0;
  PetscScalar *xc = NULL;
  PetscCall( DMPlexVecGetClosure(cdm, csec, coords_loc, c, &cdof, &xc ) );

  // セル内での辺 f のローカル番号 ie を求める
  const PetscInt *cone = NULL;
  PetscInt ncone = 0;
  PetscCall( DMPlexGetConeSize( dm, c, &ncone ) );
  PetscCall( DMPlexGetCone( dm, c, &cone ) );
  PetscInt ie = -1;
  for( PetscInt j=0; j<ncone; j++ )
  {
    if( cone[j] == f )
    {
      ie = j;
      break;
    }
  }
  if (ie < 0) {
    PetscCall( DMPlexVecRestoreClosure( cdm, csec, coords_loc, c, &cdof, &xc ) );
    PetscFunctionReturn( PETSC_SUCCESS );
  }

  // Tri-P2: closure中のノード -> { V0, E01, V1, E02, E12, V2 }
  static const PetscInt V_ofs[3] = { 0, 2, 5 }; // V0, V1, V2の位置
  static const PetscInt E_ofs[3] = { 1, 4, 3 }; // E01, E12, E20

  // 辺 f の中点座標
  const PetscScalar *xmid = xc + dim*E_ofs[ie];

  // 辺 f の両端点座標
  static const PetscInt edgeVerts[3][2] = { {0,1}, {1,2}, {2,0} };
  const PetscInt lv0 = edgeVerts[ie][0];
  const PetscInt lv1 = edgeVerts[ie][1];
  const PetscScalar *xEnd0 = xc + dim * V_ofs[lv0];
  const PetscScalar *xEnd1 = xc + dim * V_ofs[lv1];
  //+++
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d coordinates of nodes on the boundary f=%5d\n", rank, f );
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d (x y) =%15.5e%15.5e\n", rank, xmid[0], xmid[1] );
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d (x y) =%15.5e%15.5e\n", rank, xEnd0[0], xEnd0[1] );
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d (x y) =%15.5e%15.5e\n", rank, xEnd1[0], xEnd1[1] );
  //---

  // 後片付け
  PetscCall( DMPlexVecRestoreClosure( cdm, csec, coords_loc, c, &cdof, &xc ) );

  PetscFunctionReturn( PETSC_SUCCESS );
}