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
      PetscInt depth;
      PetscCall( DMPlexGetPointDepth( dm, p, &depth ) );
      if( depth == 2 ) continue; // pがセルなら飛ばす

      // 座標
      std::vector<double> xy( 3, 0.0 );
      PetscCall( get_coords( dm, p, xy ) );

      node nd( p, xy[0], xy[1], xy[2] );
      if( !added_p.count( p ) )
      {
        added_p.insert( p );
        nodes.push_back( nd );
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
      if( depth == 2 ) continue; // pがセルなら飛ばす

      nd_clos_ids.push_back(p);
    }
    elem e( c, nd_clos_ids, nodes );
    elems.push_back( e );
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
PetscErrorCode merge_Kuu_matrix( const DM& dm, const PetscScalar* D, const elem_vec& elems, Mat& A, const bool debug )
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

  std::array<double,144> Kuu;

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
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d : Local DOF indices\n", rank, c );
      for( int i=0; i<nidx; i++ )
      {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "idx[%5d]=%5d\n", i, idx[i] );
      }
    }
    //---

    Kuu = elems.pid_is(c).Kuu_matrix( D );

    //+++
    if( debug )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d : Kuu\n", rank, c );
      for( int i=0; i<12; i++ )
      {
        for( int j=0; j<12; j++ )
        {
          PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%12.3e", Kuu[i*12+j] );
        }
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
      }
    }
    //---

    // Keの並べ替え
    elems.pid_is(c).permutate_Kuu_matrix( Kuu );

    PetscScalar Ke[144];
    for( int i=0; i<144; i++ ) Ke[i]=Kuu[i];

    // アセンブリ
    PetscCall( DMPlexMatSetClosure( dm, loc_sec, glb_sec, A, c, Ke, ADD_VALUES ) );
    
    // 局所自由度インデックスを返す
    PetscCall( DMPlexRestoreClosureIndices( dm, loc_sec, glb_sec, c, PETSC_TRUE, &nidx, &idx, NULL, NULL ) );

  } // for( c )

  PetscCall( MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY ) );
  PetscCall( MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY ) );

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

      // f の節点出力
      //PetscCall( show_coords_boundary( rank, dm, f ) );

      // P2の辺中点DOF
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
          const PetscInt rloc = voff + k;
          if( rloc >= 0 )
          {
            row_idxs.push_back( rloc );
            //PetscCall( VecSetValuesLocal( xbc_loc, 1, &rloc, &gbc, ADD_VALUES ) );
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

// ----------------------------------------------------------------------------
// 節点力を設定
PetscErrorCode set_nodal_force( const DM& dm, const PetscFE& fe,
  const PetscInt phys_id, const PetscScalar F, const PetscInt dir, Vec& b )
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
      //PetscCall( show_coords_boundary( rank, dm, f ) );

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
  //PetscScalar sum=0;
  //PetscCall( VecNorm( b, NORM_1, &sum ) );
  //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d vec_norm=%15.5e\n", rank, sum );

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// 変位のセット
PetscErrorCode set_displacement( const DM& dm, const Vec& sol, node_vec& nodes )
{
  // rankの取得
  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

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

      nodes.pid_is(p).uv[0] = ux;
      nodes.pid_is(p).uv[1] = uy;
    }
    PetscCall( DMPlexRestoreTransitiveClosure( dm, c, PETSC_TRUE, &npts, &pts ) );
  }

  // 後片付け
  PetscCall( VecRestoreArrayRead( sol_loc, &sol_arr ) );
  PetscCall( DMRestoreLocalVector( dm,     &sol_loc ) );

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
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
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d elem.pid=%5d\n", rank, e.pid );
    for( int i=0; i<e.num_nods; i++ )
    {
      node* nd = e.nod[i];
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d node.pid=%5d (x y)=%15.5e%15.5e (u v)=%15.5e%15.5e\n",
        rank, nd->pid, nd->xy[0], nd->xy[1], nd->uv[0], nd->uv[1] );
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
      loc_xy.push_back( nodes[n].xy[0] );
      loc_xy.push_back( nodes[n].xy[1] );
      loc_xy.push_back( nodes[n].xy[2] );
    }
    // 各ランクの uv をベクトルに格納する
    std::vector<double> loc_uv;
    for( int n=0; n<nodes.size(); n++ )
    {
      loc_uv.push_back( nodes[n].uv[0] );
      loc_uv.push_back( nodes[n].uv[1] );
      loc_uv.push_back( nodes[n].uv[2] );
    }
    // 各ランクの 節点タグ をベクトルに格納する
    std::vector<int> loc_ntag;
    for( int n=0; n<nodes.size(); n++ )
    {
      int lpid = nodes[n].pid;
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
      xy_vec.push_back( nodes[i].xy[0] );
      xy_vec.push_back( nodes[i].xy[1] );
      xy_vec.push_back( nodes[i].xy[2] );
      uv_vec.push_back( nodes[i].uv[0] );
      uv_vec.push_back( nodes[i].uv[1] );
      uv_vec.push_back( nodes[i].uv[2] );
      int lpid = nodes[i].pid;
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
  std::vector<int> node_idx_vec;
  int elem_size = elems.size();
  std::vector<int> esize_vec;

  if( nproc > 1 )
  {
    int max_node_per_elem = 6;
    // 各ランクの 各要素に含まれる節点の tag -> index をベクトルに格納する
    std::vector<int> loc_node_tags( elems.size() * max_node_per_elem );
    for( int m=0; m<elems.size(); m++ )
    {
      for( int i=0; i<elems[m].num_nods; i++ )
      {
        int npid = elems[m].node_pids[i];
        int ntag = lpid2ntag.at(npid);
        loc_node_tags[m*max_node_per_elem + i] = ntag;
      }
    }

    // 各ランクの elemのサイズをベクトル化
    int esize = elems.size();
    esize_vec.resize( nproc );
    MPI_Allgather( &esize, 1, MPI_INT, esize_vec.data(), 1, MPI_INT, PETSC_COMM_WORLD );
    std::vector<int> eidxs( nproc+1 );
    eidxs[0] = 0;
    for( int i=0; i<=nproc; i++ ) eidxs[i] = eidxs[i-1] + esize_vec[i-1];
    int etotal = eidxs[nproc];
    elem_size = etotal;

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
  } // if( nproc > 1 )
  else
  {
    int max_node_per_elem = 6;
    node_idx_vec.resize( elem_size * max_node_per_elem );
    for( int i=0; i<elem_size; i++ )
    {
      for( int j=0; j<max_node_per_elem; j++ )
      {
        int npid = elems[i].node_pids[j];
        int ntag = lpid2ntag.at(npid);
        int nidx = ntag2vidx.at(ntag);
        node_idx_vec[i*max_node_per_elem+j] = nidx;
      }
    }
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
    // 頂点の数を数える（3角形のみに対応）
    int numv = elem_size*(3+1);
    // 要素データの出力
    s_fprv << "CELLS " << elem_size << " " << numv << std::endl;
    for( int eidx=0; eidx<elem_size; eidx++ )
    {
      s_fprv << 3;
      for( int i=0; i<3; i++ )
      {
        int nidx = node_idx_vec[eidx*6+i];
        s_fprv << " " << nidx;
      }
      s_fprv << std::endl;
    }
    // セルタイプの出力
    s_fprv << "CELL_TYPES " << elem_size << std::endl;
    for( int eidx=0; eidx<elem_size; eidx++ )
    {
      s_fprv << 5 << std::endl;
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
}

// ----------------------------------------------------------------------------
// pointID が depth=0 (頂点)のとき，座標を計算する 
PetscErrorCode get_coords_vertex( const DM& dm, const PetscInt p, std::vector<double>& xy )
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

// ----------------------------------------------------------------------------
// pointID が depth=1 (辺)のとき，座標を計算する 
PetscErrorCode get_coords_face( const DM& dm, const PetscInt p, std::vector<double>& xy )
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

// ----------------------------------------------------------------------------
// pointID に対応する点の座標を計算する
PetscErrorCode get_coords( const DM& dm, const PetscInt p, std::vector<double>& xy )
{
  // 次元の取得
  PetscInt dim;
  PetscCall( DMGetDimension( dm, &dim ) );

  xy.resize( dim, 0.0 );

  PetscInt depth;
  PetscCall( DMPlexGetPointDepth( dm, p, &depth ) );
  if( depth == 2 ) PetscFunctionReturn( PETSC_SUCCESS );

  if( depth == 1 ) PetscCall( get_coords_face( dm, p, xy ) );
  if( depth == 0 ) PetscCall( get_coords_vertex( dm, p, xy ) );

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