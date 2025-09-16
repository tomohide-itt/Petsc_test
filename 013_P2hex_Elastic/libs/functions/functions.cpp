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
PetscErrorCode create_FE( DM dm, PetscFE& fe )
{
  // 次元の取得
  PetscInt dim;
  PetscCall( DMGetDimension( dm, &dim ) );

  // 2次要素（P2）でdim成分ベクトル場（変位）に対する有限要素空間を作る
  // 引数は，コミュニケータ，次元，フィールド成分数，要素形状がsimplexか，P1/P2, 積分次数，出力先の有限要素オブジェクト
  PetscCall( PetscFECreateLagrange( PETSC_COMM_WORLD, dim, dim, PETSC_FALSE, 2, PETSC_DETERMINE, &fe ) );

  // dm にフィールド0を登録
  // 引数は，メッシュオブジェクト，フィールド番号，ラベル，そのフィールドに対応する有限要素オブジェクト
  //PetscCall( DMSetField( dm, 0, NULL, (PetscObject)fe ) );
  PetscCall( DMAddField( dm, NULL, (PetscObject)fe ) );

  // Discrete System (PetscDS) を dm から生成する （自由度構造の確定）
  // 内部では，dmに紐づいた PetscSection を構築，PDE用のオブジェクト PetscDS を作成，各フィールドごとに PetscFE の情報を関連づけを行っている
  PetscCall( DMCreateDS( dm ) );

  // --- ここからがポイント：FE の numDof から Section を明示生成 ---
  PetscDS ds = NULL; PetscCall(DMGetDS(dm, &ds));
PetscFE fe0 = NULL; PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject*)&fe0));

const PetscInt *feNumDof = NULL;     // ← {3,3,3,3} が入っている（Nc込み）
PetscCall(PetscFEGetNumDof(fe0, &feNumDof));

PetscInt Nc = 0, dimMesh = 0;
PetscCall(PetscFEGetNumComponents(fe0, &Nc));  // ここは 3 のはず
PetscCall(DMGetDimension(dm, &dimMesh));       // 3

// DMPlexCreateSection へ渡す配列を組む（※Ncで割らない！）
PetscInt numComp[1]   = { Nc };                // =3
PetscInt numDofFlat[4];
for (PetscInt d=0; d<=dimMesh; ++d) numDofFlat[d] = feNumDof[d];  // {3,3,3,3}

PetscSection sec = NULL;
PetscCall(DMPlexCreateSection(dm,
                              /*labels*/   NULL,
                              /*numComp*/  numComp,     // Nc
                              /*numDof*/   numDofFlat,  // {3,3,3,3}（Nc込み）
                              /*numBC*/    0,
                              /*bcFields*/ NULL,
                              /*bcPoints*/ NULL,
                              /*bcComps*/  NULL,
                              /*perm*/     NULL,
                              &sec));
PetscCall(DMSetLocalSection(dm, sec));
PetscCall(PetscSectionDestroy(&sec));
// -- ここまで

  // --- 検証: 本当に Section が張られたか ---
  /*
  PetscDS ds=NULL; PetscInt nf=0;
PetscCall(DMGetDS(dm, &ds));
PetscCall(PetscDSGetNumFields(ds, &nf));
PetscPrintf(PETSC_COMM_WORLD, "DS num fields = %" PetscInt_FMT "\n", nf);

// depthごとの点数と DoF 合計
PetscSection loc=NULL; PetscCall(DMGetLocalSection(dm, &loc));
PetscInt vS,vE,eS,eE,fS,fE,cS,cE;
PetscCall(DMPlexGetDepthStratum(dm,0,&vS,&vE));
PetscCall(DMPlexGetDepthStratum(dm,1,&eS,&eE));
PetscCall(DMPlexGetDepthStratum(dm,2,&fS,&fE));
PetscCall(DMPlexGetHeightStratum(dm,0,&cS,&cE)); // cells
auto sumdof=[&](PetscInt s,PetscInt e){
  PetscInt ssum=0; for(PetscInt p=s;p<e;++p){PetscInt d=0;PetscSectionGetDof(loc,p,&d); ssum+=d;} return ssum;
};
PetscPrintf(PETSC_COMM_WORLD,
  "counts: V=%" PetscInt_FMT " E=%" PetscInt_FMT " F=%" PetscInt_FMT " C=%" PetscInt_FMT "\n",
  vE-vS, eE-eS, fE-fS, cE-cS);
PetscPrintf(PETSC_COMM_WORLD,
  "dofs:   V=%" PetscInt_FMT " E=%" PetscInt_FMT " F=%" PetscInt_FMT " C=%" PetscInt_FMT "\n",
  sumdof(vS,vE), sumdof(eS,eE), sumdof(fS,fE), sumdof(cS,cE));
  */
 /*
 PetscDS ds=NULL; PetscCall(DMGetDS(dm, &ds));
PetscFE fe0=NULL; PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject*)&fe0));
const PetscInt *numDof=NULL; PetscCall(PetscFEGetNumDof(fe0, &numDof));
PetscPrintf(PETSC_COMM_WORLD,
  "FE numDof by topodim: d0=%" PetscInt_FMT " d1=%" PetscInt_FMT
  " d2=%" PetscInt_FMT " d3=%" PetscInt_FMT "\n",
  numDof[0], numDof[1], numDof[2], numDof[3]);
  */
 PetscSection loc=NULL;
PetscCall(DMGetLocalSection(dm,&loc));

PetscInt vS,vE,eS,eE,fS,fE,cS,cE;
PetscCall(DMPlexGetDepthStratum(dm,0,&vS,&vE));
PetscCall(DMPlexGetDepthStratum(dm,1,&eS,&eE));
PetscCall(DMPlexGetDepthStratum(dm,2,&fS,&fE));
PetscCall(DMPlexGetHeightStratum(dm,0,&cS,&cE));

PetscInt sumV=0,sumE=0,sumF=0,sumC=0,dof=0;
for (PetscInt p=vS;p<vE;++p){PetscSectionGetDof(loc,p,&dof); sumV+=dof;}
for (PetscInt p=eS;p<eE;++p){PetscSectionGetDof(loc,p,&dof); sumE+=dof;}
for (PetscInt p=fS;p<fE;++p){PetscSectionGetDof(loc,p,&dof); sumF+=dof;}
for (PetscInt p=cS;p<cE;++p){PetscSectionGetDof(loc,p,&dof); sumC+=dof;}
PetscPrintf(PETSC_COMM_WORLD,"dofs(Local): V=%" PetscInt_FMT " E=%" PetscInt_FMT
            " F=%" PetscInt_FMT " C=%" PetscInt_FMT "\n",sumV,sumE,sumF,sumC);

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
/*
    std::vector<double> Kuu;
    elems.pid_is(c)->cal_Kuu_matrix( Kuu, D );

    int num_nods = elems.pid_is(c)->num_nods;
    int Kuu_size = num_nods*dim * num_nods*dim;

    PetscScalar* Ke = new PetscScalar[ Kuu_size ];

    for( int i=0; i<Kuu_size; i++ ) Ke[i]=Kuu[i];

    // アセンブリ
    PetscCall( DMPlexMatSetClosure( dm, loc_sec, glb_sec, A, c, Ke, ADD_VALUES ) );
*/
    // 局所自由度インデックスを返す
    PetscCall( DMPlexRestoreClosureIndices( dm, loc_sec, glb_sec, c, PETSC_TRUE, &nidx, &idx, NULL, NULL ) );

//    delete[] Ke;

  } // for( c )

  /*
  PetscCall( MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY ) );
  PetscCall( MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY ) );
*/
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}