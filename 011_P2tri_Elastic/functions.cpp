#include "functions.h"

// .mshをDMPlexで読み込む 
PetscErrorCode read_gmsh( const std::string& mesh_path, DM &dm, PetscInt &dim )
{
  // メッシュを補完ありで読み込む（第4引数はinterpolate=1に対応）
  PetscCall( DMPlexCreateFromFile( PETSC_COMM_WORLD, mesh_path.c_str(), NULL, PETSC_TRUE, &dm ) );
  // 次元の取得
  PetscCall( DMGetDimension( dm, &dim ) );
  // 次元が2なら何もしない．そうでなければメッセージを出力して処理を中断 
  PetscCheck( dim==2, PETSC_COMM_WORLD, PETSC_ERR_SUP, "このサンプルは2D専用です");
  //+++
  {
    PetscPrintf( PETSC_COMM_WORLD, "dim = %d\n", (int)dim );
  }
  //---
  PetscFunctionReturn( PETSC_SUCCESS );
}

// 座標を取得
PetscErrorCode get_coords( const int rank, const DM& dm, const PetscInt dim, const PetscScalar *a_loc, const PetscScalar *a_glob, const bool debug )
{
  Vec coords_loc  = NULL;
  Vec coords_glob = NULL;
  
  // dm が持つ「座標ベクトル（ローカル）」を取り出して coords_loc に入れる
  PetscCall( DMGetCoordinatesLocal( dm, &coords_loc ) );
  // coords_locがあれば，生ポインタに格納
  if (coords_loc) PetscCall( VecGetArrayRead(coords_loc, &a_loc) );
  
  // dm が持つ「座標ベクトル（グローバル）」を取り出して　coords_glob に入れる
  PetscCall( DMGetCoordinates( dm, &coords_glob ) );
  // coords_globがあれば，生ポインタに格納
  if (coords_glob) PetscCall( VecGetArrayRead( coords_glob, &a_glob ) );

  //+++
  if( debug )
  {
    // a_loc と a_glob のアドレスを出力
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d coords_loc  = %p\n", rank, coords_loc  );
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d coords_glob = %p\n", rank, coords_glob );
    // a_loc の内容を出力
    if( coords_loc )
    {
      PetscInt num_loc;
      PetscCall( VecGetLocalSize( coords_loc, &num_loc ) );
      for( PetscInt i=0; i<num_loc; i=i+dim )
      {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d a_loc[%5d]=%15.5e", rank, i, (double)PetscRealPart( a_loc[i] ) );
        if( dim >= 2 ) PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", (double)PetscRealPart( a_loc[i+1] ) );
        if( dim >= 3 ) PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", (double)PetscRealPart( a_loc[i+2] ) );
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
      }
    }
    PetscSection section;
    PetscInt pointID_start, pointID_end;
    PetscCall( DMGetCoordinateSection( dm, &section ) );
    // 全てのdmの ID の範囲を取得
    PetscCall( DMPlexGetChart( dm, &pointID_start, &pointID_end ) );
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d [pointID_start, pointID_end ) = [%5d,%5d )\n", rank, pointID_start, pointID_end );
    for( PetscInt p=pointID_start; p<pointID_end; p++ )
    {
      PetscInt dof, offset;
      PetscCall( PetscSectionGetDof(    section, p, &dof    ) );
      PetscCall( PetscSectionGetOffset( section, p, &offset ) );
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d pointID=%5d offset=%5d dof=%5d\n", rank, p, offset, dof );
    }
  }
  //---

  if (coords_loc)  PetscCall(VecRestoreArrayRead(coords_loc,  &a_loc ) );
  if (coords_glob) PetscCall(VecRestoreArrayRead(coords_glob, &a_glob) );

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// 各セルの節点座標を出力する
PetscErrorCode show_coords_each_cell( const int rank, const DM& dm )
{
  DM cdm = NULL; //座標用DM
  Vec coords_loc = NULL;
  PetscSection csec = NULL; //座標用セクション

  PetscCall( DMGetCoordinateDM( dm, &cdm ) );
  PetscCall( DMGetCoordinatesLocal( dm, &coords_loc ) );
  PetscCall( DMGetCoordinateSection( dm, &csec ) );

  PetscInt c_start=0, c_end=0;
  PetscCall( DMPlexGetHeightStratum( cdm, 0, &c_start, &c_end ) );

  for( PetscInt c=c_start; c<c_end; c++ )
  {
    PetscInt cdof = 0;
    PetscScalar *xc = NULL;
    PetscCall( DMPlexVecGetClosure( cdm, csec, coords_loc, c, &cdof, &xc ) );
    //+++
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d\n", rank, c );
    for( int i=0; i<cdof; i++ )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d xc[%5d]=%15.5e\n", rank, i, xc[i] );
    }
    //---
    PetscCall( DMPlexVecRestoreClosure( cdm, csec, coords_loc, c, &cdof, &xc ) );
  }

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// 頂点のIDの範囲を取得
PetscErrorCode get_vertex_ID_range( const int rank, const DM& dm, PetscInt& ID_start, PetscInt& ID_end, const bool debug )
{
  // DMPlex内部の頂点の最初のIDと最後のIDを取得
  PetscCall( DMPlexGetDepthStratum( dm, 0, &ID_start, &ID_end ) );
  if( debug )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d [vertex_start, vertex_end) = [%5d,%5d )\n", rank, ID_start, ID_end );
  }
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// 面のIDの範囲を取得
PetscErrorCode get_face_ID_range( const int rank, const DM& dm, PetscInt& ID_start, PetscInt& ID_end, const bool debug )
{
  // DMPlex内部の面の最初のIDと最後のIDを取得
  PetscCall( DMPlexGetHeightStratum( dm, 1, &ID_start, &ID_end ) );
  if( debug )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d [face_start,   face_end  ) = [%5d,%5d )\n", rank, ID_start, ID_end );
  }
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// セルのIDの範囲を取得
PetscErrorCode get_cell_ID_range( const int rank, const DM& dm, PetscInt& ID_start, PetscInt& ID_end, const bool debug )
{
  // DMPlex内部のセルの最初のIDと最後のIDを取得
  PetscCall( DMPlexGetHeightStratum( dm, 0, &ID_start, &ID_end ) );
  if( debug )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d [cell_start,   cell_end  ) = [%5d,%5d )\n", rank, ID_start, ID_end );
  }
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// 頂点数を取得
PetscErrorCode get_num_vertex( const int rank, const DM& dm, PetscInt& num, const bool debug )
{
  PetscInt start, end;
  get_vertex_ID_range( rank, dm, start, end );
  num = end - start;
  if( debug )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d num_vertex   = %5d\n", rank, num );
  }
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// 面数を取得
PetscErrorCode get_num_face( const int rank, const DM& dm, PetscInt& num, const bool debug )
{
  PetscInt start, end;
  get_face_ID_range( rank, dm, start, end );
  num = end - start;
  if( debug )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d num_face     = %5d\n", rank, num );
  }
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// セル数を取得
PetscErrorCode get_num_cell( const int rank, const DM& dm, PetscInt& num, const bool debug )
{
  PetscInt start, end;
  get_cell_ID_range( rank, dm, start, end );
  num = end - start;
  if( debug )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d num_cell     = %5d\n", rank, num );
  }
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// 境界数を取得
PetscErrorCode get_num_boundary( const int rank, const DM& dm, PetscInt& num, const bool debug )
{
  PetscInt start, end;
  get_face_ID_range( rank, dm, start, end );
  num = 0;
  for( PetscInt f=start; f<end; f++ )
  {
    PetscInt support_size;
    PetscCall( DMPlexGetSupportSize( dm, f, &support_size) );
    if( support_size == 1 ) num++;
  }
  if( debug )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d num_boundary = %5d\n", rank, num );
  }
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ラベル（GmshのPhysical Group）を取得
PetscErrorCode get_label_name( const int rank, const DM& dm, std::vector<std::string> &label_names, const bool debug )
{
  PetscInt num_labels;
  PetscCall( DMGetNumLabels( dm, &num_labels ) );
  for( PetscInt i=0; i<num_labels; i++ )
  {
    const char* name = NULL;
    PetscCall( DMGetLabelName( dm, i, &name ) );
    std::string sname(name);
    label_names.push_back( sname );
  }
  if( debug )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, num_labels =%5d\n", rank, num_labels );
    for( int i=0; i<label_names.size(); i++ )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, label_name[%5d] = %s\n", rank, i, label_names[i].c_str() );
    }
  }
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// あるラベルについて値（=Physical ID）ごとの要素数を取得
PetscErrorCode get_label_num( const int rank, const DM& dm, const std::string& label_name, PetscInt& num, const bool debug )
{
  // dmが持つラベルから label_name を取得
  DMLabel label;
  PetscCall( DMGetLabel( dm, label_name.c_str(), &label ) );
  if( label )
  {
    // このプロセスが保持している値（物理ID）の集合をvalueISに取得
    IS valueIS;
    PetscCall( DMLabelGetValueIS( label, &valueIS ) );
    if( valueIS )
    {
      // ローカルに取得された物理IDの数を取得
      PetscInt nvals;
      PetscCall( ISGetLocalSize( valueIS, &nvals ) );
      // 物理IDを取得
      const PetscInt *vals;
      PetscCall( ISGetIndices( valueIS, &vals ) );
      if( debug )
      {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, nvals = %d\n", rank, nvals );
      }
      for( PetscInt k=0; k<nvals; k++ )
      {
        PetscInt v = vals[k];
        IS IDs;
        const PetscInt *a_IDs;
        num = 0;
        // ラベル値 v を持つ全てのDMエンティティIDの集合を取得
        PetscCall( DMLabelGetStratumIS( label, v, &IDs ) );
        if( IDs )
        {
          // このランクが保持する該当エンティティ数
          PetscCall( ISGetLocalSize( IDs, &num ) );
          PetscCall( ISGetIndices( IDs, &a_IDs ) );
          if( debug )
          {
            PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, %s value = %d : num = %d\n", rank, label_name.c_str(), v, num );
          }
          PetscCall( ISRestoreIndices( IDs, &a_IDs ) );
          PetscCall( ISDestroy( &IDs ) );
        }
      }
      PetscCall( ISRestoreIndices( valueIS, &vals ) );
      PetscCall( ISDestroy( &valueIS ) );
    }
  }

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// dm/secの辺，頂点pointIDとgmshのnodeTagの紐づけ
PetscErrorCode get_nodeID_map()
{
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// DMPlexのセルpointIDとgmshのelementTagの紐づけ
PetscErrorCode get_elemID_map( const int rank, const DM& dm, const PetscInt dim, const node_vec& nodes, const elem_vec& elems,
  std::map<int,int>& eID2pID, std::map<int,int>& pID2eID, const bool debug )
{
  Vec coords_loc = NULL;
  PetscCall( DMGetCoordinatesLocal( dm, &coords_loc ) );
  const PetscScalar *a_loc;
  if (coords_loc) PetscCall( VecGetArrayRead(coords_loc, &a_loc) );

  PetscSection section;
  PetscInt cellID_start, cellID_end;
  PetscCall( DMGetCoordinateSection( dm, &section ) );
  PetscCall( DMPlexGetHeightStratum( dm, 0, &cellID_start, &cellID_end ) );

  std::vector<bool> visited_elem( elems.size(), false );
  for( PetscInt p=cellID_start; p<cellID_end; p++ )
  {
    // セルごとに節点の座標値を取得
    PetscInt dof, offset;
    PetscCall( PetscSectionGetDof(    section, p, &dof    ) );
    PetscCall( PetscSectionGetOffset( section, p, &offset ) );
    PetscInt num_node = dof/dim;
    std::vector<std::vector<double>> xyz( num_node, std::vector<double>( dim, 0.0 ) );
    for( int n=0; n<num_node; n++ )
    {
      for( int i=0; i<dim; i++ )
      {
        xyz[n][i] = (double)PetscRealPart(a_loc[offset + n*dim + i]);
      }
    }
    if( debug )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d pID=%7d num_node=%5d\n", rank, p, num_node );
      std::cout << std::scientific << std::setprecision(5);
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
    if( debug ) PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
    double tol = 1.0e-8;

    for( const elem &e : elems )
    {
      if( e.nodeIDs.size() != num_node ) continue;
      int eidx = elems.idx_of_id( e.ID );
      if( visited_elem[eidx] ) continue;
      
      if( debug ) PetscSynchronizedPrintf( PETSC_COMM_WORLD, "check elem ID=%d\n", e.ID );
      if( debug ) PetscSynchronizedPrintf( PETSC_COMM_WORLD, "eidx=%d\n", eidx );
      
      std::vector<bool> visited_node( num_node, false );
      bool identical = true;
      for( int nid=0; nid<num_node; nid++ )
      {
        const node& nd = nodes.id_of(e.nodeIDs[nid]);
        
        if( debug )
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
        
        if( debug ) PetscSynchronizedPrintf( PETSC_COMM_WORLD, "  find=%d\n", find );
        
        identical = identical && find;
        if( !find ) break;
      }
      if( identical )
      {
        visited_elem[eidx] = true;
        eID2pID[e.ID] = p;
        pID2eID[p] = e.ID;
        if( debug ) PetscSynchronizedPrintf( PETSC_COMM_WORLD, "elems.ID = %d is identical with %d\n", e.ID, p );
        if( debug ) PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
        break;
      }
    }
  }

  if (coords_loc)  PetscCall(VecRestoreArrayRead(coords_loc,  &a_loc ) );
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// 節点の並び順について，csec/cdm 順から sec/dm 順に変換するマップを計算する
// この関数は未完成：Permutationが設定されていない場合，自分で設定する必要がある
PetscErrorCode get_map_nic2ni( const int rank, const DM& dm, std::vector<PetscInt>& map )
{
  DM cdm = NULL;  // 座標用DM
  PetscSection sec = NULL; // dmのセクション
  PetscSection csec = NULL; // cdmのセクション
  PetscCall( DMGetCoordinateDM( dm, &cdm ) );
  PetscCall( DMGetLocalSection( dm, &sec ) );
  PetscCall( DMGetCoordinateSection( dm, &csec ) );

  PetscInt depth = -1;
  PetscCall( DMPlexGetDepth( dm, &depth ) );

  //+++
  PetscPrintf( PETSC_COMM_WORLD, "depth = %d\n", depth );
  //---

  // 各セクションのクロージャ置換を取得
  IS is_perm_nic = NULL;
  IS is_perm_ni  = NULL;
  PetscCall( PetscSectionGetClosurePermutation( csec, (PetscObject)cdm, depth, 2, &is_perm_nic ) );
  PetscCall( PetscSectionGetClosurePermutation( sec,  (PetscObject)dm,  depth, 2, &is_perm_ni  ) );
  const PetscInt *perm_nic = NULL;
  const PetscInt *perm_ni  = NULL;
  PetscInt num_nic = 0;
  PetscInt num_ni  = 0;
  if( is_perm_nic )
  {
    PetscCall( ISGetLocalSize( is_perm_nic, &num_nic ) );
    PetscCall( ISGetIndices( is_perm_nic, &perm_nic ) );
  }
  if( is_perm_ni )
  {
    PetscCall( ISGetLocalSize( is_perm_ni, &num_ni ) );
    PetscCall( ISGetIndices( is_perm_ni, &perm_ni ) );
  }

  //+++
  PetscPrintf( PETSC_COMM_WORLD, "is_perm_nic = %p\n", is_perm_nic );
  PetscPrintf( PETSC_COMM_WORLD, "is_perm_ni  = %p\n", is_perm_ni  );
  //---

  std::vector<PetscInt> inv_perm_nic( num_nic, -1 );
  std::vector<PetscInt> map_nic2ni( num_nic, -1 );
  for( PetscInt k=0; k<num_nic; k++ ) inv_perm_nic[ perm_nic[k] ] = k;
  for( PetscInt i=0; i<num_nic; i++ ) map_nic2ni[i] = inv_perm_nic[ perm_ni[i] ];

  if( is_perm_nic )
  {
    PetscCall( ISRestoreIndices( is_perm_nic, &perm_nic ) );
    PetscCall( ISDestroy( &is_perm_nic ) );
  }
  if( is_perm_ni )
  {
    PetscCall( ISRestoreIndices( is_perm_ni, &perm_ni ) );
    PetscCall( ISDestroy( &is_perm_ni ) );
  }

  PetscFunctionReturn( PETSC_SUCCESS );
}

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