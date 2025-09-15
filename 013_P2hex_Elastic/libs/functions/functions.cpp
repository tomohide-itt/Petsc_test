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
// dm の情報を出力する
PetscErrorCode show_DM_info( const DM& dm )
{
  // rankの取得
  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

  // 次元の取得
  PetscInt dim;
  PetscCall( DMGetDimension( dm, &dim ) );

  // チャート範囲取得
  PetscInt chart_start, chart_end;
  PetscCall( DMPlexGetChart( dm, &chart_start, &chart_end ) );

  // Height順に範囲を取得
  std::vector<int> h_start(dim+1);
  std::vector<int> h_end(dim+1);
  for( int h=0; h<=dim; h++ )
  {
    PetscCall( DMPlexGetHeightStratum( dm, h, &(h_start[h]), &(h_end[h]) ) );
  }

  // Depth順に範囲を取得
  std::vector<int> d_start(dim+1);
  std::vector<int> d_end(dim+1);
  for( int d=0; d<=dim; d++ )
  {
    PetscCall( DMPlexGetDepthStratum( dm, d, &(d_start[d]), &(d_end[d]) ) );
  }

  //+++
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d chart=[%5d,%5d)\n", rank, chart_start, chart_end );
  for( int h=0; h<=dim; h++ )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d h%d=[%5d,%5d)\n", rank, h, h_start[h], h_end[h] );
  }
  for( int d=0; d<=dim; d++ )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d d%d=[%5d,%5d)\n", rank, d, d_start[d], d_end[d] );
  }
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  //----

  // 座標セクションから dof と off を取得 して出力
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-------- rank=%3d dof & off from csec\n", rank );
  PetscSection loc_sec;
  PetscCall( DMGetLocalSection( dm, &loc_sec ) );
  Vec coords_loc = NULL;
  PetscCall( DMGetCoordinatesLocal( dm, &coords_loc) );
  PetscSection csec;
  PetscCall( DMGetCoordinateSection( dm, &csec ) );
  const PetscScalar *coords_loc_arr;
  PetscCall( VecGetArrayRead( coords_loc, &coords_loc_arr ) );
  for( int p=chart_start; p<chart_end; p++ )
  {
    PetscInt dof=0, off=0;
    PetscCall( PetscSectionGetDof( csec, p, &dof ) );
    PetscCall( PetscSectionGetOffset( csec, p, &off ) );
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d p=%5d dof=%5d off=%5d\n", rank, p, dof, off );
    for( int k=0; k<dof; k++ )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "  rank=%3d %15.5e\n", rank, coords_loc_arr[off+k] );
    }
  }
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscCall( VecRestoreArrayRead( coords_loc, &coords_loc_arr ) );

  // セルのトランジティブクロージャの情報
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-------- rank=%3d transitive closure\n", rank );
  PetscInt c_start=0, c_end=0;
  PetscCall( DMPlexGetHeightStratum( dm, 0, &c_start, &c_end ) );
  for( PetscInt c=c_start; c<c_end; c++ )
  {
    PetscInt npts = 0;
    PetscInt* pts = NULL;
    PetscCall( DMPlexGetTransitiveClosure( dm, c, PETSC_TRUE, &npts, &pts ) );
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d npts=%5d\n", rank, c, npts );
    for( PetscInt k=0; k<npts; k++ )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "  rank=%3d pts=%5d%5d\n", rank, pts[k*2+0], pts[k*2+1] );
    }
    PetscCall( DMPlexRestoreTransitiveClosure( dm, c, PETSC_TRUE, &npts, &pts ) );
  }

  //---
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// nodesの設定
PetscErrorCode set_nodes( DM& dm, node_vec& nodes )
{
  // rankの取得
  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

  // 次元の取得
  PetscInt dim;
  PetscCall( DMGetDimension( dm, &dim ) );

  // 範囲（セル）
  PetscInt c_start=0, c_end=0;
  PetscCall( DMPlexGetHeightStratum( dm, 0, &c_start, &c_end ) );

  std::set<int> added_p;

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
      if( added_p.count( p ) ) continue;

      std::vector<double> xy( 3, 0.0 );
      bool bxy = false;
      if( dim==2 && npts== 7 ) bxy = trian6::get_coords( dm, p, xy );
      if( dim==3 && npts==27 ) bxy = hexl27::get_coords( dm, p, xy );

      if( bxy )
      {
        nodes.create_new( p, xy[0], xy[1], xy[2] );
        added_p.insert(p);
      }
    }
    PetscCall( DMPlexRestoreTransitiveClosure( dm, c, PETSC_TRUE, &npts, &pts ) );
  }
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// elemsの設定
PetscErrorCode set_elems( DM& dm, node_vec& nodes, elem_vec& elems )
{
  // rankの取得
  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

  // 次元の取得
  PetscInt dim;
  PetscCall( DMGetDimension( dm, &dim ) );

  // 範囲（セル）
  PetscInt c_start=0, c_end=0;
  PetscCall( DMPlexGetHeightStratum( dm, 0, &c_start, &c_end ) );

  // セルでループ
  for( PetscInt c=c_start; c<c_end; c++ )
  {
    PetscInt npts = 0;
    PetscInt* pts = NULL;
    PetscCall( DMPlexGetTransitiveClosure( dm, c, PETSC_TRUE, &npts, &pts ) );

    std::vector<int> nd_clos_ids;
    // このセルのポイントでループ
    for( PetscInt k=0; k<npts; k++ )
    {
      const PetscInt p = pts[2*k];
      PetscInt depth;
      PetscCall( DMPlexGetPointDepth( dm, p, &depth ) );
      if( dim==2 && npts== 7 && depth == 2 ) continue; // pがセルなら飛ばす( trian6 のとき )

      nd_clos_ids.push_back(p);
    }

    if( dim==2 && npts== 7 ) elems.create_new<trian6>( c, nd_clos_ids, nodes );
    if( dim==3 && npts==27 ) elems.create_new<hexl27>( c, nd_clos_ids, nodes );

    PetscCall( DMPlexRestoreTransitiveClosure( dm, c, PETSC_TRUE, &npts, &pts ) );
  }
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// FE空間を作成する
PetscErrorCode create_FE( const DM& dm, PetscFE& fe )
{
  // 次元の取得
  PetscInt dim;
  PetscCall( DMGetDimension( dm, &dim ) );

  // 2次要素（P2）でdim成分ベクトル場（変位）に対する有限要素空間を作る
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
PetscErrorCode cal_D_matrix( const DM& dm, const double E, const double nu, std::vector<PetscScalar>& D )
{
  // 次元の取得
  PetscInt dim;
  PetscCall( DMGetDimension( dm, &dim ) );

  double lamb = (E*nu)/((1.0+nu)*(1.0-2.0*nu));
  double mu = E/(2.0*(1.0+nu));

  std::vector<PetscScalar> delxdel;
  std::vector<PetscScalar> idn;

  int tens2_size = 4;
  if( dim == 3 ) tens2_size = 6;

  D.resize(       tens2_size * tens2_size );
  delxdel.resize( tens2_size * tens2_size );
  idn.resize(     tens2_size * tens2_size );

  for( int i=0; i<D.size(); i++ ) D[i] = 0.0;
  for( int i=0; i<delxdel.size(); i++ ) delxdel[i] = 0.0;
  for( int i=0; i<idn.size(); i++ ) idn[i] = 0.0;

  for( int i=0; i<3; i++ ){
    for( int j=0; j<3; j++ )
    {
      delxdel[i*tens2_size+j] = 1.0;
    }
  }

  for( int i=0; i<3; i++ ) idn[i*tens2_size+i] = 1.0;
  for( int i=3; i<tens2_size; i++ ) idn[i*tens2_size+i] = 0.5;

  for( int i=0; i<tens2_size; i++ )
  {
    for( int j=0; j<tens2_size; j++ )
    {
      D[i*tens2_size+j] = lamb*delxdel[i*tens2_size+j] + 2.0*mu*idn[i*tens2_size+j];
    }
  }
  PetscFunctionReturn( PETSC_SUCCESS );
}