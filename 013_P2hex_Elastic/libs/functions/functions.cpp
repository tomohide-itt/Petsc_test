#include "functions.h"

// ----------------------------------------------------------------------------
// .mshをDMPlexで読み込む 
PetscErrorCode read_gmsh( const std::string& mesh_path, DM &dm )
{
  // メッシュを補完ありで読み込む（第4引数はinterpolate=1に対応）
  PetscCall( DMPlexCreateFromFile( PETSC_COMM_WORLD, mesh_path.c_str(), NULL, PETSC_TRUE, &dm ) );
  //
  DMSetUp(dm);
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
PetscErrorCode create_FE( DM dm, const bool debug )
{
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
      }
    }

    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d : Local DOFS\n", rank );
    for( int i=0; i<=dim; i++ )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "DOFS(depth=%3d)=%5d\n", i, dofs[i] );
    }
    PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  }
    
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

PetscErrorCode merge_Kuu_matrix( const DM& dm, const std::vector<PetscScalar>& D, const elem_vec& elems, Mat& A, const bool debug )
{
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

    int num_nods = elems.pid_is(c)->num_nods;
    int Kuu_size = num_nods*dim * num_nods*dim;

    PetscScalar* Ke = new PetscScalar[ Kuu_size ];

    for( int i=0; i<Kuu_size; i++ ) Ke[i]=Kuu[i];
    for( int i=0; i<Kuu_size; i++ ) Ke[i] = 0.0;

    // アセンブリ
    PetscCall( DMPlexMatSetClosure( dm, loc_sec, glb_sec, A, c, Ke, ADD_VALUES ) );

    // 局所自由度インデックスを返す
    PetscCall( DMPlexRestoreClosureIndices( dm, loc_sec, glb_sec, c, PETSC_TRUE, &nidx, &idx, NULL, NULL ) );

    delete[] Ke;

  } // for( c )

  PetscCall( MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY ) );
  PetscCall( MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY ) );

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// 節点力を設定
PetscErrorCode set_nodal_force( const DM& dm, const PetscInt phys_id, const PetscScalar F, const PetscInt dir, Vec& b )
{
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
  PetscCall( DMGetLocalSection( dm, &loc_sec ) );

  // DSの取得
  PetscDS ds = NULL;
  PetscCall( DMGetDS( dm, &ds ) );

  // FEの取得
  PetscFE fe = NULL;
  PetscCall( PetscDSGetDiscretization( ds, 0, (PetscObject*)&fe ) );

  //成分数を取得
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

      PetscInt depth;
      PetscCall( DMPlexGetPointDepth( dm, f, &depth ) );

      //+++
      PetscPrintf( PETSC_COMM_WORLD, "f=%5d depth=%5d\n", f, depth );
      //---

      // 3D:面の中点DOF / 2D:辺の中点DOF
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

      // 3D:辺の中点DOF / 2D:頂点のDOF
      const PetscInt* cone1 = NULL;
      PetscInt ncone1 = 0;
      PetscCall( DMPlexGetConeSize( dm, f, &ncone1 ) );
      PetscCall( DMPlexGetCone(     dm, f, &cone1  ) );
      for( PetscInt cn1=0; cn1<ncone1; cn1++ )
      {
        PetscInt p1 = cone1[cn1];
        //+++
        PetscPrintf( PETSC_COMM_WORLD, "f=%5d cone1[%5d]=%5d\n", f, cn1, p1 );
        //---
        PetscInt dof1=0, off1=0;
        PetscCall( PetscSectionGetDof(    loc_sec, p1, &dof1 ) );
        PetscCall( PetscSectionGetOffset( loc_sec, p1, &off1 ) );
        if( dof1 > 0 )
        {
          PetscInt nbf1 = dof1/Nc;
          for( PetscInt j=0; j<nbf1; j++ )
          {
            const PetscInt rloc1 = off1 + j*Nc + dir; //ローカル添え字
            PetscCall( VecSetValuesLocal( bloc, 1, &rloc1, &F, ADD_VALUES ) );
          } 
        }
        // 3D:頂点のDOF
        if( depth == 2 )
        {
          const PetscInt* cone2 = NULL;
          PetscInt ncone2 = 0;
          PetscCall( DMPlexGetConeSize( dm, p1, &ncone2 ) );
          PetscCall( DMPlexGetCone(     dm, p1, &cone2  ) );
          for( PetscInt cn2=0; cn2<ncone2; cn2++ )
          {
            PetscInt p2 = cone2[cn2];
            //+++
            PetscPrintf( PETSC_COMM_WORLD, "f=%5d p1=%5d cone2[%5d]=%5d\n", f, p1, cn2, p2 );
            //---
            PetscInt dof2=0, off2=0;
            PetscCall( PetscSectionGetDof(    loc_sec, p2, &dof2 ) );
            PetscCall( PetscSectionGetOffset( loc_sec, p2, &off2 ) );
            if( dof2 > 0 )
            {
              PetscInt nbf2 = dof2/Nc;
              for( PetscInt j=0; j<nbf2; j++ )
              {
                const PetscInt rloc2 = off2 + j*Nc + dir; //ローカル添え字
                PetscCall( VecSetValuesLocal( bloc, 1, &rloc2, &F, ADD_VALUES ) );
              } 
            }
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

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// Dirichlet境界条件 の設定
PetscErrorCode set_Dirichlet_zero( const DM& dm, const PetscInt phys_id, Mat& A, Vec& b )
{
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

      // f の節点出力
      //PetscCall( show_coords_boundary( rank, dm, f ) );

      // 3D:面の中点DOF / 2D:辺の中点DOF
      PetscInt dof=0, off=0;
      PetscCall( PetscSectionGetDof(    glb_sec, f, &dof ) );
      PetscCall( PetscSectionGetOffset( glb_sec, f, &off ) );
      for( PetscInt k=0; k<dof; k++ )
      {
        const PetscInt rloc = off + k;
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
        for( PetscInt k=0; k<dof1; k++ )
        {
          const PetscInt rloc1 = off1 + k;
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
            PetscCall( PetscSectionGetDof(    loc_sec, p2, &dof2 ) );
            PetscCall( PetscSectionGetOffset( loc_sec, p2, &off2 ) );
            for( PetscInt k=0; k<dof2; k++ )
            {
              const PetscInt rloc2 = off2 + k;
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
  PetscFunctionReturn( PETSC_SUCCESS );
}