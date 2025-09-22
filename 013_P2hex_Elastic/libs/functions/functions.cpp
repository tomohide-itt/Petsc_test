#include "functions.h"
#include "timer.h"

// ----------------------------------------------------------------------------
// .mshをDMPlexで読み込む 
PetscErrorCode read_gmsh( const std::string& mesh_path, DM &dm )
{
  tmr.start();
  // メッシュを補完ありで読み込む（第4引数はinterpolate=1に対応）
  PetscCall( DMPlexCreateFromFile( PETSC_COMM_WORLD, mesh_path.c_str(), NULL, PETSC_TRUE, &dm ) );
  //
  DMSetUp(dm);
  // 次元の取得
  PetscInt dim;
  PetscCall( DMGetDimension( dm, &dim ) );
  //+++
  //PetscPrintf( PETSC_COMM_WORLD, "%s[%d] dim = %d\n", __FUNCTION__, __LINE__, (int)dim );
  //---
  tmr.stop( __FUNCTION__ );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// dm を領域分割する
PetscErrorCode partition_mesh( DM &dm, DM &dm_dist )
{
  tmr.start();
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
  tmr.stop( __FUNCTION__ );
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
  PetscSection loc_sec = NULL;
  PetscCall( DMGetLocalSection( dm, &loc_sec ) );
  Vec coords_loc = NULL;
  PetscCall( DMGetCoordinatesLocal( dm, &coords_loc) );
  PetscSection csec = NULL;
  PetscCall( DMGetCoordinateSection( dm, &csec ) );
  const PetscScalar *coords_loc_arr = NULL;
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
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );

  // ラベル情報
  PetscInt num_labels;
  PetscCall( DMGetNumLabels( dm, &num_labels ) );
  std::vector<std::string> label_names;
  for( PetscInt i=0; i<num_labels; i++ )
  {
    const char* name = NULL;
    PetscCall( DMGetLabelName( dm, i, &name ) );
    std::string ss( name );
    label_names.push_back( ss );
  }
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-------- rank=%3d label names : num_labels=%5d\n", rank, num_labels );
  for( int i=0; i<label_names.size(); i++ )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "  rank=%3d, label_name[%5d] = %s\n", rank, i, label_names[i].c_str() );
  }
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );

  //---
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// nodesの設定
PetscErrorCode set_nodes( DM& dm, node_vec& nodes )
{
  tmr.start();
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
  tmr.stop( __FUNCTION__ );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// elemsの設定
PetscErrorCode set_elems( DM& dm, node_vec& nodes, elem_vec& elems )
{
  tmr.start();

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
  tmr.stop( __FUNCTION__ );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// FE空間を作成する
PetscErrorCode create_FE( DM dm, const bool debug )
{
  tmr.start();

  // rankの取得
  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

  // 次元の取得
  PetscInt dim;
  PetscCall( DMGetDimension( dm, &dim ) );

  // メッシュが simplex かどうかを判定（最初のセルの cone サイズで判定）
  // cone_size: 2D tri=3, quad=4 / 3D tet=4, hex=6
  // tri, tet: simplex, qual, hex: non-simplex
  PetscInt c_start, c_end;
  PetscCall( DMPlexGetHeightStratum( dm, 0, &c_start, &c_end ) );
  PetscInt cone_size = 0;
  PetscBool is_simplex = PETSC_TRUE;
  if( c_start < c_end )
  {
    PetscCall( DMPlexGetConeSize( dm, c_start, &cone_size ) );
    if(      dim == 2 ) is_simplex = (cone_size==3)? PETSC_TRUE : PETSC_FALSE;
    else if( dim == 3 ) is_simplex = (cone_size==4)? PETSC_TRUE : PETSC_FALSE;
  }

  // 2次要素（P2）でdim成分ベクトル場（変位）に対する有限要素空間を作る
  // 引数は，コミュニケータ，次元，フィールド成分数，要素形状がsimplexか，P1/P2, 積分次数，出力先の有限要素オブジェクト
  PetscFE fe;
  PetscCall( PetscFECreateLagrange( PETSC_COMM_WORLD, dim, dim, is_simplex, 2, PETSC_DETERMINE, &fe ) );

  // dm にフィールド0を登録
  // 引数は，メッシュオブジェクト，フィールド番号，ラベル，そのフィールドに対応する有限要素オブジェクト
  //PetscCall( DMSetField( dm, 0, NULL, (PetscObject)fe ) );
  PetscCall( DMAddField( dm, NULL, (PetscObject)fe ) );

  // Discrete System (PetscDS) を dm から生成する （自由度構造の確定）
  // 内部では，dmに紐づいた PetscSection を構築，PDE用のオブジェクト PetscDS を作成，各フィールドごとに PetscFE の情報を関連づけを行っている
  PetscCall( DMCreateDS( dm ) );

  // --- ここからがポイント：FE の numDof から Section を明示生成 ---
  PetscDS ds = NULL;
  PetscCall( DMGetDS( dm, &ds ) );
  PetscFE fe0 = NULL;
  PetscCall( PetscDSGetDiscretization( ds, 0, (PetscObject*)&fe0 ) );

  const PetscInt *feNumDof = NULL;     // ← {3,3,3,3} が入っている（Nc込み）
  PetscCall( PetscFEGetNumDof( fe0, &feNumDof ) );

  PetscInt Nc = 0, dimMesh = 0;
  PetscCall( PetscFEGetNumComponents( fe0, &Nc ) );  // ここは 3 のはず
  PetscCall( DMGetDimension( dm, &dimMesh ) );       // 3

  // DMPlexCreateSection へ渡す配列を組む（※Ncで割らない！）
  PetscInt numComp[1]   = { Nc };                // =3
  PetscInt numDofFlat[4];
  for (PetscInt d=0; d<=dimMesh; ++d) numDofFlat[d] = feNumDof[d];  // {3,3,3,3}

  PetscSection sec = NULL;
  // args: dm, labels, numComp, numDof, numBC, bcields, bcPoints, bcComps, perm, sec
  PetscCall( DMPlexCreateSection( dm, NULL, numComp, numDofFlat, 0, NULL, NULL, NULL, NULL, &sec ) );
  PetscCall( DMSetLocalSection( dm, sec ) );
  PetscCall( PetscSectionDestroy( &sec ) );
  PetscCall( PetscFEDestroy( &fe ) );
// -- ここまで

  // --- 検証: 本当に Section が張られたか ---
  if( debug )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d : Local DOFS\n", rank );

    PetscSection loc_sec = NULL;
    PetscCall( DMGetLocalSection( dm, &loc_sec ) );

    std::vector<PetscInt> start( dim+1 );
    std::vector<PetscInt> end( dim+1 );
    for( int i=0; i<=dim; i++ ) PetscCall( DMPlexGetDepthStratum( dm, i, &start[i], &end[i] ) );

    std::vector<PetscInt> dofs( dim+1, 0 );
    PetscInt dof = 0;
    for( int i=0; i<=dim; i++ )
    {
      for( PetscInt p=start[i]; p<end[i]; p++ )
      {
        PetscSectionGetDof( loc_sec, p, &dof );
        dofs[i] += dof;
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "p=%5d depth=%5d dof=%5d\n", p, i, dof );
      }
    }
    PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  }
  tmr.stop( __FUNCTION__ );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// Dマトリクスを計算する（線形弾性）
PetscErrorCode cal_D_matrix( const DM& dm, const double E, const double nu, std::vector<PetscScalar>& D )
{
  tmr.start();

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

  //+++
  //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "----------- dmat\n" );
  //for( int i=0; i<tens2_size; i++ )
  //{
  //  for( int j=0; j<tens2_size; j++ )
  //  {
  //    PetscPrintf( PETSC_COMM_WORLD, "%15.5e", D[i*tens2_size+j] );
  //  }
  //  PetscPrintf( PETSC_COMM_WORLD, "\n" );
  //}
  //---
  tmr.stop( __FUNCTION__ );
  PetscFunctionReturn( PETSC_SUCCESS );
}

PetscErrorCode merge_Kuu_matrix( const DM& dm, const std::vector<PetscScalar>& D, const elem_vec& elems, Mat& A, const bool debug )
{
  tmr.start();

  // rankの取得
  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

  // 次元の取得
  PetscInt dim;
  PetscCall( DMGetDimension( dm, &dim ) );

  // ローカルセクション，グローバルセクションを取得
  PetscSection loc_sec, glb_sec;
  PetscCall( DMGetLocalSection(  dm, &loc_sec ) );
  PetscCall( DMGetGlobalSection( dm, &glb_sec ) );

  // セルの範囲を取得
  PetscInt c_start=0, c_end=0;
  PetscCall( DMPlexGetHeightStratum( dm, 0, &c_start, &c_end ) );

  for( PetscInt c=c_start; c<c_end; c++ )
  {
    // 局所自由度インデックス（プロセス内のローカルVecの位置）を取得
    PetscInt* idx = NULL;
    PetscInt nidx = 0;
    PetscCall( DMPlexGetClosureIndices( dm, loc_sec, glb_sec, c, PETSC_TRUE, &nidx, &idx, NULL, NULL ) );

    //+++
    if( debug )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d nidx=%5d: Local DOF indices\n", rank, c, nidx );
      for( int i=0; i<nidx; i++ )
      {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "idx[%5d]=%5d\n", i, idx[i] );
      }
    }
    //---
    std::vector<double> Kuu;
    elems.pid_is(c)->cal_Kuu_matrix( Kuu, D );

    //int num_nods = elems.pid_is(c)->num_nods;
    //int Kuu_size = num_nods*dim * num_nods*dim;

    //PetscScalar* Ke = new PetscScalar[ Kuu_size ];

    //for( int i=0; i<Kuu_size; i++ ) Ke[i]=Kuu[i];

    // アセンブリ
    PetscCall( DMPlexMatSetClosure( dm, loc_sec, glb_sec, A, c, Kuu.data(), ADD_VALUES ) );

    // 局所自由度インデックスを返す
    PetscCall( DMPlexRestoreClosureIndices( dm, loc_sec, glb_sec, c, PETSC_TRUE, &nidx, &idx, NULL, NULL ) );

    //delete[] Ke;

  } // for( c )

  PetscCall( MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY ) );
  PetscCall( MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY ) );

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  tmr.stop( __FUNCTION__ );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// 節点力を設定
PetscErrorCode set_nodal_force( const DM& dm, const PetscInt phys_id, const PetscScalar F, const PetscInt dir, Vec& b )
{
  tmr.start();

  // rankの取得
  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

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
  PetscSection glb_sec = NULL;
  PetscCall( DMGetLocalSection( dm, &loc_sec ) );
  PetscCall( DMGetGlobalSection( dm, &glb_sec ) );

  // DSの取得
  PetscDS ds = NULL;
  PetscCall( DMGetDS( dm, &ds ) );

  // FEの取得
  PetscFE fe = NULL;
  PetscCall( PetscDSGetDiscretization( ds, 0, (PetscObject*)&fe ) );

  // 成分数を取得
  PetscInt Nc = 0;
  PetscCall( PetscFEGetNumComponents( fe, &Nc ) );

  //+++
  //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
  //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d set_nodal_force phys_id=%5d F=%15.5e dir=%5d\n", rank, phys_id, F, dir );
  //---

  // 境界に載っている節点のrlocをユニークに収集する（所有ランクのみ）
  std::vector<PetscInt> rlocs;

  if( faceIS )
  {
    const PetscInt* faces = NULL;
    PetscInt nfaces = 0;
    PetscCall( ISGetLocalSize( faceIS, &nfaces ) );
    PetscCall( ISGetIndices( faceIS, &faces ) );

    for( PetscInt i=0; i<nfaces; i++ )
    {
      const PetscInt f = faces[i];
      
      PetscInt npts=0;
      PetscInt* pts = NULL;
      PetscCall( DMPlexGetTransitiveClosure( dm, f, PETSC_TRUE, &npts, &pts ) );

      for( PetscInt k=0; k<npts; k++ )
      {
        PetscInt p = pts[2*k];
        PetscInt gdof=0, goff=0;
        PetscCall( PetscSectionGetDof(    glb_sec, p, &gdof ) );
        PetscCall( PetscSectionGetOffset( glb_sec, p, &goff ) );
        if( gdof <= 0 || goff < 0 ) continue;  //所有するランクのみ

        PetscInt ldof=0, loff=0;
        PetscCall( PetscSectionGetDof(    loc_sec, p, &ldof ) );
        PetscCall( PetscSectionGetOffset( loc_sec, p, &loff ) );
        if( ldof <= 0 ) continue;

        const PetscInt nbf = ldof / Nc;
        for( PetscInt j=0; j<nbf; j++ )
        {
          rlocs.push_back( loff + j*Nc + dir );
        }

      }
      PetscCall( DMPlexRestoreTransitiveClosure( dm, f, PETSC_TRUE, &npts, &pts ) );
    }
    PetscCall( ISRestoreIndices( faceIS, &faces ) );
    PetscCall( ISDestroy(&faceIS) );
  }

  // 重複除去
  std::sort( rlocs.begin(), rlocs.end() );
  rlocs.erase( std::unique( rlocs.begin(), rlocs.end()), rlocs.end() );

  for( PetscInt i=0; i<(PetscInt)rlocs.size(); i++ )
  {
    PetscCall( VecSetValuesLocal( bloc, 1, &rlocs[i], &F, INSERT_VALUES ) );
  }

  // ローカル -> グローバルへ集約
  PetscCall( VecAssemblyBegin(bloc) );
  PetscCall( VecAssemblyEnd(bloc) );
  PetscCall( DMLocalToGlobalBegin( dm, bloc, ADD_VALUES, b ) );
  PetscCall( DMLocalToGlobalEnd( dm, bloc, ADD_VALUES, b ) );

  PetscCall( VecDestroy( &bloc ) );

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  tmr.stop( __FUNCTION__ );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// Dirichlet境界条件 の設定
PetscErrorCode set_Dirichlet_zero( const DM& dm, const PetscInt phys_id, Mat& A, Vec& b )
{
  tmr.start();

  // rankの取得
  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

  // Face Sets ラベル取得
  DMLabel label = NULL;
  PetscCall( DMGetLabel( dm, "Face Sets", &label ) );

  // 対象エッジ集合を取得
  IS faceIS = NULL;
  PetscCall( DMLabelGetStratumIS( label, phys_id, &faceIS ) );

  // ローカル Vec の作成
  Vec xbc_loc = NULL;
  PetscCall( DMCreateLocalVector( dm, &xbc_loc ) );
  PetscCall( VecZeroEntries( xbc_loc ) );

  // グローバル Vec の作成
  Vec xbc_glb = NULL;
  PetscCall( DMCreateGlobalVector( dm, &xbc_glb ) );
  PetscCall( VecZeroEntries( xbc_glb ) );

  // セクション取得
  PetscSection loc_sec;
  PetscSection glb_sec;
  PetscCall( DMGetLocalSection( dm, &loc_sec ) );
  PetscCall( DMGetGlobalSection( dm, &glb_sec ) );

  PetscScalar gbc = 0.0;

  // エッジ上のDOFとそのエッジの両端点のDOFを集める
  std::vector<PetscInt> row_idxs; // グローバル方程式番号

  //+++
  //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
  //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d set_Dirichlet_zero phys_id=%5d\n", rank, phys_id );
  //---

  if( faceIS )
  {
    const PetscInt* faces = NULL;
    PetscInt nfaces = 0;
    PetscCall( ISGetLocalSize( faceIS, &nfaces ) );
    PetscCall( ISGetIndices( faceIS, &faces ) );

    for( PetscInt i=0; i<nfaces; i++ )
    {
      const PetscInt f = faces[i];

      PetscInt depth;
      PetscCall( DMPlexGetPointDepth( dm, f, &depth ) );

      // 3D:面の中点DOF / 2D:辺の中点DOF
      PetscInt dof=0, off=0;
      PetscCall( PetscSectionGetDof(    glb_sec, f, &dof ) );
      PetscCall( PetscSectionGetOffset( glb_sec, f, &off ) );
      //+++
      //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "f =%5d dof =%5d\n", f, dof );
      //---
      for( PetscInt k=0; k<dof; k++ )
      {
        const PetscInt rloc = off + k;
        //+++
        //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rloc =%5d\n", rloc );
        //---
        if( rloc >= 0 )
        {
          row_idxs.push_back( rloc );
          //PetscCall( VecSetValuesLocal( xbc_loc, 1, &rloc, &gbc, ADD_VALUES ) );
        }
      }

      // P2の頂点DOF
      const PetscInt* cone1 = NULL;
      PetscInt ncone1 = 0;
      PetscCall( DMPlexGetConeSize( dm, f, &ncone1 ) );
      PetscCall( DMPlexGetCone(     dm, f, &cone1  ) );
      for( PetscInt cn1=0; cn1<ncone1; cn1++ )
      {
        PetscInt p1 = cone1[cn1];
        PetscInt dof1=0, off1=0;
        PetscCall( PetscSectionGetDof(    glb_sec, p1, &dof1 ) );
        PetscCall( PetscSectionGetOffset( glb_sec, p1, &off1 ) );
        //+++
        //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "p1=%5d dof1=%5d\n", p1, dof1 );
        //---
        for( PetscInt k=0; k<dof1; k++ )
        {
          const PetscInt rloc1 = off1 + k;
          //+++
          //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rloc1=%5d\n", rloc1 );
          //---
          if( rloc1 >= 0 )
          {
            row_idxs.push_back( rloc1 );
            //PetscCall( VecSetValuesLocal( xbc_loc, 1, &rloc, &gbc, ADD_VALUES ) );
          }
        }
        if( depth == 2 )
        {
          const PetscInt* cone2 = NULL;
          PetscInt ncone2 = 0;
          PetscCall( DMPlexGetConeSize( dm, p1, &ncone2 ) );
          PetscCall( DMPlexGetCone(     dm, p1, &cone2  ) );
          for( PetscInt cn2=0; cn2<ncone2; cn2++ )
          {
            PetscInt p2 = cone2[cn2];
            PetscInt dof2=0, off2=0;
            PetscCall( PetscSectionGetDof(    glb_sec, p2, &dof2 ) );
            PetscCall( PetscSectionGetOffset( glb_sec, p2, &off2 ) );
            //+++
            //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "p2=%5d dof2=%5d\n", p2, dof2 );
            //---
            for( PetscInt k=0; k<dof2; k++ )
            {
              const PetscInt rloc2 = off2 + k;
              //+++
              //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rloc2=%5d\n", rloc2 );
              //---
              if( rloc2 >= 0 )
              {
                row_idxs.push_back( rloc2 );
                //PetscCall( VecSetValuesLocal( xbc_loc, 1, &rloc, &gbc, ADD_VALUES ) );
              }
            }
          }
        }
      }
    }

    PetscCall( ISRestoreIndices( faceIS, &faces ) );
    PetscCall( ISDestroy( &faceIS ) );
  }

  // 重複除去
  std::sort( row_idxs.begin(), row_idxs.end() );
  row_idxs.erase( std::unique( row_idxs.begin(), row_idxs.end() ), row_idxs.end() );
  
  //+++
  //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "----- rank=%3d row_idxs to set Dirichlet row_idxs.size()=%5ld\n",
  //  rank, row_idxs.size() );
  //for( int i=0; i<row_idxs.size(); i++ )
  //{
  //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%5d", row_idxs[i] );
  //}
  //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
  //---

  // Local -> Global
  IS is_glb = NULL;
  PetscCall( ISCreateGeneral( PETSC_COMM_WORLD, (PetscInt)row_idxs.size(), row_idxs.data(), PETSC_COPY_VALUES, &is_glb ) );

  PetscCall( VecAssemblyBegin( xbc_loc ) );
  PetscCall( VecAssemblyEnd(   xbc_loc ) );
  PetscCall( DMLocalToGlobalBegin( dm, xbc_loc, ADD_VALUES, xbc_glb ) );
  PetscCall( DMLocalToGlobalEnd(   dm, xbc_loc, ADD_VALUES, xbc_glb ) );

  // 係数行列と右辺ベクトルへ Diriclet境界を適用 (u,v=0)
  PetscCall( MatZeroRowsColumnsIS( A, is_glb, 1.0, xbc_glb, b ) );

  PetscCall( ISDestroy( &is_glb ) );
  PetscCall( VecDestroy( &xbc_loc ) );
  PetscCall( VecDestroy( &xbc_glb ) );

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  tmr.stop( __FUNCTION__ );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// 変位のセット
PetscErrorCode set_displacement( const DM& dm, const Vec& sol, node_vec& nodes )
{
  tmr.start();

  // rankの取得
  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

  // 次元の取得
  PetscInt dim;
  PetscCall( DMGetDimension( dm, &dim ) );

  // セクション取得
  PetscSection sec;
  PetscCall( DMGetLocalSection( dm, &sec ) );

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

      PetscInt dof = 0;
      PetscInt off = 0;
      PetscCall( PetscSectionGetDof( sec, p, &dof ) );
      PetscCall( PetscSectionGetOffset( sec, p, &off ) );
      //+++
      //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d p=%5d depth=%5d dof=%5d off=%5d\n", rank, c, p, depth, dof, off );
      //---
      for( int i=0; i<dof; i++ )
      {
        nodes.pid_is(p)->uv[i] = sol_arr[off+i];
      }
    }
    PetscCall( DMPlexRestoreTransitiveClosure( dm, c, PETSC_TRUE, &npts, &pts ) );
  }

  // 後片付け
  PetscCall( VecRestoreArrayRead( sol_loc, &sol_arr ) );
  PetscCall( DMRestoreLocalVector( dm,     &sol_loc ) );

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  tmr.stop( __FUNCTION__ );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// 変位の出力
PetscErrorCode show_displacement( const elem_vec& elems )
{
  // rankの取得
  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

  for( const auto& e : elems )
  {
    // 出力
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d elem.pid=%5d\n", rank, e->pid );
    for( int i=0; i<e->num_nods; i++ )
    {
      const auto& nd = e->nod[i];
      if( e->dim == 2 )
      {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d node.pid=%5d (x y)=%15.5e%15.5e (u v)=%15.5e%15.5e\n",
          rank, nd->pid, nd->xy[0], nd->xy[1], nd->uv[0], nd->uv[1] );
      }
      else if( e->dim == 3 )
      {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d node.pid=%5d (x y z)=%15.5e%15.5e%15.5e (u v w)=%15.5e%15.5e%15.5e\n",
          rank, nd->pid, nd->xy[0], nd->xy[1], nd->xy[2], nd->uv[0], nd->uv[1], nd->uv[2] );
      }
    }
  }

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// vtkファイルの出力
void output_vtk( const std::string& vtk_path, const node_vec& nodes, const elem_vec& elems,
  const std::map<int,int>& lpid2ntag )
{
  tmr.start();

  // rank, nprocの取得
  PetscMPIInt rank, nproc;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
  MPI_Comm_size( PETSC_COMM_WORLD, &nproc );

  std::vector<double> xy_vec;
  std::vector<double> uv_vec;
  std::vector<int> ntag_vec;
  std::map<int,int> ntag2vidx;

  int node_size = nodes.size();

  if( nproc > 1 )
  {
    // 各ランクの xy 座標をベクトルに格納する
    std::vector<double> loc_xy;
    for( int n=0; n<nodes.size(); n++ )
    {
      loc_xy.push_back( nodes[n]->xy[0] );
      loc_xy.push_back( nodes[n]->xy[1] );
      loc_xy.push_back( nodes[n]->xy[2] );
    }
    // 各ランクの uv をベクトルに格納する
    std::vector<double> loc_uv;
    for( int n=0; n<nodes.size(); n++ )
    {
      loc_uv.push_back( nodes[n]->uv[0] );
      loc_uv.push_back( nodes[n]->uv[1] );
      loc_uv.push_back( nodes[n]->uv[2] );
    }
    // 各ランクの 節点タグ をベクトルに格納する
    std::vector<int> loc_ntag;
    for( int n=0; n<nodes.size(); n++ )
    {
      int lpid = nodes[n]->pid;
      int ntag = lpid2ntag.at(lpid);
      loc_ntag.push_back( ntag );
    }
    //+++
    //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "----------- local vectors\n" );
    //for( int i=0; i<nodes.size(); i++ )
    //{
    //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d i=%5d tag=%5d (x y)=%15.5e%15.5e\n",
    //    rank, i, loc_ntag[i], loc_xy[i*3+0], loc_xy[i*3+1] );
    //}
    //PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
    //---

    // 各ランクの nodeのサイズ をベクトル化
    int nsize = nodes.size();
    std::vector<int> nsize_vec( nproc );
    MPI_Allgather( &nsize, 1, MPI_INT, nsize_vec.data(), 1, MPI_INT, PETSC_COMM_WORLD );
    std::vector<int> nidxs( nproc+1 );
    nidxs[0] = 0;
    for( int i=1; i<=nproc; i++ ) nidxs[i] = nidxs[i-1] + nsize_vec[i-1];
    int ntotal = nidxs[nproc];
    node_size = ntotal;

    // rank=0 に localなベクトルを集める
    std::vector<double> glb_xy;
    std::vector<double> glb_uv;
    std::vector<int> glb_ntag;
    if( rank == 0 )
    {
      glb_xy.resize( ntotal * 3 );
      glb_uv.resize( ntotal * 3 );
      glb_ntag.resize( ntotal );
    }
    if( rank == 0 )
    {
      memcpy( glb_xy.data(), loc_xy.data(), nsize*3*sizeof(double) );
      memcpy( glb_uv.data(), loc_uv.data(), nsize*3*sizeof(double) );
      memcpy( glb_ntag.data(), loc_ntag.data(), nsize*sizeof(int) );
    }
    for( int i=1; i<nproc; i++ )
    {
      if( rank == i )
      {
        MPI_Send( loc_xy.data(), nsize*3, MPI_DOUBLE, 0, 0, PETSC_COMM_WORLD );
        MPI_Send( loc_uv.data(), nsize*3, MPI_DOUBLE, 0, 0, PETSC_COMM_WORLD );
        MPI_Send( loc_ntag.data(), nsize, MPI_INT, 0, 0, PETSC_COMM_WORLD );
      }
      if( rank == 0 )
      {
        MPI_Recv( glb_xy.data()+nidxs[i]*3, nsize_vec[i]*3, MPI_DOUBLE, i, 0, PETSC_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Recv( glb_uv.data()+nidxs[i]*3, nsize_vec[i]*3, MPI_DOUBLE, i, 0, PETSC_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Recv( glb_ntag.data()+nidxs[i], nsize_vec[i], MPI_INT, i, 0, PETSC_COMM_WORLD, MPI_STATUS_IGNORE );
      }
    }
    //+++
    //if( rank == 0 )
    //{
    //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "----------- global vectors\n" );
    //  for( int i=0; i<ntotal; i++ )
    //  {
    //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d i=%5d tag=%5d (x y)=%15.5e%15.5e\n",
    //      rank, i, glb_ntag[i], glb_xy[i*3+0], glb_xy[i*3+1] );
    //  }
    //}
    //PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
    //---

    // tag が 重複しないようにベクトルを作る
    if( rank == 0 )
    {
      std::set<int> added_tags;

      for( int i=0; i<ntotal; i++ )
      {
        if( !added_tags.count( glb_ntag[i] ) )
        {
          xy_vec.push_back( glb_xy[i*3+0] );
          xy_vec.push_back( glb_xy[i*3+1] );
          xy_vec.push_back( glb_xy[i*3+2] );
          uv_vec.push_back( glb_uv[i*3+0] );
          uv_vec.push_back( glb_uv[i*3+1] );
          uv_vec.push_back( glb_uv[i*3+2] );
          ntag_vec.push_back( glb_ntag[i] );
          added_tags.insert( glb_ntag[i] );
          ntag2vidx[ glb_ntag[i] ] = ntag_vec.size()-1;
        }
      }
      node_size = ntag_vec.size();
    }
  } // if( nproc > 1 )
  else
  {
    for( int i=0; i<node_size; i++ )
    {
      xy_vec.push_back( nodes[i]->xy[0] );
      xy_vec.push_back( nodes[i]->xy[1] );
      xy_vec.push_back( nodes[i]->xy[2] );
      uv_vec.push_back( nodes[i]->uv[0] );
      uv_vec.push_back( nodes[i]->uv[1] );
      uv_vec.push_back( nodes[i]->uv[2] );
      int lpid = nodes[i]->pid;
      int ntag = lpid2ntag.at(lpid);
      ntag_vec.push_back( ntag );
      ntag2vidx[ ntag ] = i;
    }
  }

  //if( rank == 0 )
  //{
  //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "----------- gathered vectors for nodes\n" );
  //  for( int i=0; i<node_size; i++ )
  //  {
  //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d i=%5d tag=%5d (x y)=%15.5e%15.5e\n",
  //      rank, i, ntag_vec[i], xy_vec[i*3+0], xy_vec[i*3+1] );
  //  }
  //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "----------- ntag2vidx\n" );
  //  for( const auto& [ntag,vidx] : ntag2vidx )
  //  {
  //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d ntag=%5d - vidx=%5d\n", rank, ntag, vidx );
  //  }
  //}
  //PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );

  // 要素に関する変数について
  std::vector<int> num_nods_vec;
  std::vector<int> node_idx_vec;
  int elem_size = elems.size();
  std::vector<int> esize_vec;
  std::vector<int> cell_type_vec;
  std::vector<int> num_vertex_vec;
  int max_node_per_elem = 0;

  if( nproc > 1 )
  {
    // 各ランクの elemのサイズをベクトル化
    int esize = elems.size();
    esize_vec.resize( nproc );
    MPI_Allgather( &esize, 1, MPI_INT, esize_vec.data(), 1, MPI_INT, PETSC_COMM_WORLD );
    std::vector<int> eidxs( nproc+1 );
    eidxs[0] = 0;
    for( int i=0; i<=nproc; i++ ) eidxs[i] = eidxs[i-1] + esize_vec[i-1];
    int etotal = eidxs[nproc];
    elem_size = etotal;

    // 各ランクの num_nods をベクトルに格納する
    std::vector<int> loc_num_nods;
    for( int i=0; i<elems.size(); i++ ) loc_num_nods.push_back( elems[i]->num_nods );
    // 各ランクの cell_type をベクトルに格納する
    std::vector<int> loc_cell_type;
    for( int i=0; i<elems.size(); i++ ) loc_cell_type.push_back( elems[i]->vtk_cell_type() );
    // 各ランクの num_vertex をベクトルに格納する
    std::vector<int> loc_num_vertex;
    for( int i=0; i<elems.size(); i++ ) loc_num_vertex.push_back( elems[i]->vtk_num_vertex() );

    //+++
    //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "----------- local vectors of element\n" );
    //for( int i=0; i<elems.size(); i++ )
    //{
    //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d i=%5d loc_num_nods=%5d loc_cell_type=%5d loc_numv=%5d\n",
    //    rank, i, loc_num_nods[i], loc_cell_type[i], loc_num_vertex[i] );
    //}
    //PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
    //---

    // rank=0 に localなベクトルを集める
    std::vector<int> glb_num_nods;
    std::vector<int> glb_cell_type;
    std::vector<int> glb_num_vertex;
    if( rank == 0 )
    {
      glb_num_nods.resize( etotal );
      glb_cell_type.resize( etotal );
      glb_num_vertex.resize( etotal );
    }
    if( rank == 0 )
    {
      memcpy( glb_num_nods.data(), loc_num_nods.data(), esize*sizeof(int) );
      memcpy( glb_cell_type.data(), loc_cell_type.data(), esize*sizeof(int) );
      memcpy( glb_num_vertex.data(), loc_num_vertex.data(), esize*sizeof(int) );
    }
    for( int i=1; i<nproc; i++ )
    {
      if( rank == i )
      {
        MPI_Send( loc_num_nods.data(), esize, MPI_INT, 0, 0, PETSC_COMM_WORLD );
        MPI_Send( loc_cell_type.data(), esize, MPI_INT, 0, 0, PETSC_COMM_WORLD );
        MPI_Send( loc_num_vertex.data(), esize, MPI_INT, 0, 0, PETSC_COMM_WORLD );
      }
      if( rank == 0 )
      {
        MPI_Recv( glb_num_nods.data()+eidxs[i], esize_vec[i], MPI_INT, i, 0, PETSC_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Recv( glb_cell_type.data()+eidxs[i], esize_vec[i], MPI_INT, i, 0, PETSC_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Recv( glb_num_vertex.data()+eidxs[i], esize_vec[i], MPI_INT, i, 0, PETSC_COMM_WORLD, MPI_STATUS_IGNORE );
      }
    }

    //
    if( rank == 0 )
    {
      for( int i=0; i<etotal; i++ )
      {
        num_nods_vec.push_back( glb_num_nods[i] );
        cell_type_vec.push_back( glb_cell_type[i] );
        num_vertex_vec.push_back( glb_num_vertex[i] );
      }
    }

    if( rank==0 )
    {
      for( int i=0; i<etotal; i++ )
      {
        if( max_node_per_elem < glb_num_nods[i] ) max_node_per_elem = glb_num_nods[i];
      }
    }
    MPI_Bcast( &max_node_per_elem, 1, MPI_INT, 0, PETSC_COMM_WORLD );

    // 各ランクの 各要素に含まれる節点の tag -> index をベクトルに格納する
    std::vector<int> loc_node_tags( elems.size() * max_node_per_elem );
    for( int m=0; m<elems.size(); m++ )
    {
      for( int i=0; i<elems[m]->num_nods; i++ )
      {
        int npid = elems[m]->node_pids[i];
        int ntag = lpid2ntag.at(npid);
        loc_node_tags[m*max_node_per_elem + i] = ntag;
      }
    }

    // rank=0 に localなベクトルを集める
    std::vector<int> glb_node_tags;
    if( rank == 0 )
    {
      glb_node_tags.resize( etotal * max_node_per_elem );
    }
    if( rank == 0 )
    {
      memcpy( glb_node_tags.data(), loc_node_tags.data(), esize*max_node_per_elem*sizeof(int) );
    }
    for( int i=1; i<nproc; i++ )
    {
      if( rank == i )
      {
        MPI_Send( loc_node_tags.data(), esize*max_node_per_elem, MPI_INT, 0, 0, PETSC_COMM_WORLD );
      }
      if( rank == 0 )
      {
        MPI_Recv( glb_node_tags.data()+eidxs[i]*max_node_per_elem, esize_vec[i]*max_node_per_elem, MPI_INT, i, 0, PETSC_COMM_WORLD, MPI_STATUS_IGNORE );
      }
    }
    //+++
    //if( rank == 0 )
    //{
    //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "----------- global vectors for element\n" );
    //  for( int i=0; i<etotal; i++ )
    //  {
    //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d i=%5d ntags:", rank, i );
    //    for( int j=0; j<max_node_per_elem; j++ )
    //    {
    //      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%5d", glb_node_tags[i*max_node_per_elem+j] );
    //    }
    //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
    //  }
    //}
    //PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
    //---

    // glb_node_tags の tag を idxへ変換
    if( rank == 0 )
    {
      node_idx_vec.resize( etotal * max_node_per_elem );
      for( int i=0; i<etotal; i++ )
      {
        for( int j=0; j<max_node_per_elem; j++ )
        {
          int ntag = glb_node_tags[i*max_node_per_elem+j];
          int nidx = ntag2vidx.at(ntag);
          node_idx_vec[i*max_node_per_elem+j] = nidx;
        }
      }
    }
    //+++
    //if( rank == 0 )
    //{
    //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "----------- global vectors for element\n" );
    //  for( int i=0; i<etotal; i++ )
    //  {
    //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d i=%5d idxs:", rank, i );
    //    for( int j=0; j<max_node_per_elem; j++ )
    //    {
    //      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%5d", node_idx_vec[i*max_node_per_elem+j] );
    //    }
    //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
    //  }
    //}
    //PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
    //---
  } // if( nproc > 1 )
  else
  {
    // make vector of num_nods
    num_nods_vec.resize( elem_size );
    for( int i=0; i<elem_size; i++ )
    {
      num_nods_vec[i] = elems[i]->num_nods;
      if( max_node_per_elem < elems[i]->num_nods ) max_node_per_elem = elems[i]->num_nods;
    }
    // make vector of node_idx
    node_idx_vec.resize( elem_size * max_node_per_elem );
    for( int i=0; i<elem_size; i++ )
    {
      for( int j=0; j<max_node_per_elem; j++ )
      {
        int npid = elems[i]->node_pids[j];
        int ntag = lpid2ntag.at(npid);
        int nidx = ntag2vidx.at(ntag);
        node_idx_vec[i*max_node_per_elem+j] = nidx;
      }
    }
    // make cell_type vec
    cell_type_vec.resize( elem_size );
    for( int i=0; i<elem_size; i++ ) cell_type_vec[i] = elems[i]->vtk_cell_type();
    // make num_vertex_vec
    num_vertex_vec.resize( elem_size );
    for( int i=0; i<elem_size; i++ ) num_vertex_vec[i] = elems[i]->vtk_num_vertex();
    // append elem_size to esize_vec that is to identify the rank owning the element
    esize_vec.push_back( elem_size );
  }

  //if( rank == 0 )
  //{
  //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "----------- gathered vectors for elems\n" );
  //  for( int i=0; i<elem_size; i++ )
  //  {
  //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d elem[%5d] nidxs:", rank, i );
  //    for( int j=0; j<6; j++ )
  //    {
  //      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%5d", node_idx_vec[i*6+j] );
  //    }
  //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n");
  //  }
  //}
  //PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );

  // vtk ファイルへの出力
  if( rank == 0 )
  {
    std::ofstream fprv;
    fprv.open( vtk_path, std::ios::out );

    std::ostringstream s_fprv;

    // header of vtk file
    s_fprv << "# vtk DataFile Version 4.1\n";
		s_fprv << "output\n";
		s_fprv << "ASCII\n";
		s_fprv << "DATASET UNSTRUCTURED_GRID\n";
    // 節点データの出力
    s_fprv << "POINTS " << node_size << " float" << std::endl;
    for( int nidx=0; nidx<node_size; nidx++ )
    {
      float x = static_cast<float>( xy_vec[nidx*3+0] );
      float y = static_cast<float>( xy_vec[nidx*3+1] );
      float z = static_cast<float>( xy_vec[nidx*3+2] );
      s_fprv << x << " " << y << " " << z << std::endl;
    }
    // 頂点の数を数える
    int numv = 0;
    for( int eidx=0; eidx<elem_size; eidx++ ) numv += ( num_vertex_vec[eidx] + 1 );
    // 要素データの出力
    s_fprv << "CELLS " << elem_size << " " << numv << std::endl;
    for( int eidx=0; eidx<elem_size; eidx++ )
    {
      s_fprv << num_vertex_vec[eidx];
      for( int i=0; i<num_vertex_vec[eidx]; i++ )
      {
        int nidx = node_idx_vec[eidx*max_node_per_elem+i];
        s_fprv << " " << nidx;
      }
      s_fprv << std::endl;
    }
    // セルタイプの出力
    s_fprv << "CELL_TYPES " << elem_size << std::endl;
    for( int eidx=0; eidx<elem_size; eidx++ )
    {
      s_fprv << cell_type_vec[eidx] << std::endl;
    }
    // 変位の出力
    s_fprv << "POINT_DATA " << node_size << std::endl;
    s_fprv << "VECTORS displacement float" << std::endl;
    for( int nidx=0; nidx<node_size; nidx++ )
    {
      float ux = static_cast<float>( uv_vec[nidx*3+0] );
      float uy = static_cast<float>( uv_vec[nidx*3+1] );
      float uz = static_cast<float>( uv_vec[nidx*3+2] );
      s_fprv << ux << " " << uy << " " << uz << std::endl;
    }
    // ランクの出力
    s_fprv << "CELL_DATA " << elem_size << std::endl;
    s_fprv << "SCALARS rank int" << std::endl;
    s_fprv << "LOOKUP_TABLE default" << std::endl;
    for( int r=0; r<nproc; r++ )
    {
      for( int i=0; i<esize_vec[r]; i++ )
      {
        s_fprv << r << std::endl;
      }
    }
    fprv << s_fprv.str();
    fprv.close();
  }
  tmr.stop( __FUNCTION__ );
}