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