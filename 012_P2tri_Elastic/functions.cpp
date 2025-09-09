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
  //PetscPrintf( PETSC_COMM_WORLD, "%s[%d] dim = %d\n", __FUNCTION__, __LINE__, (int)dim );
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
PetscErrorCode merge_Kuu_matrix( const DM& dm, const PetscScalar* D, Mat& A, const bool debug )
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
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d : Local DOF indices\n", rank, c );
      for( int i=0; i<nidx; i++ )
      {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "idx[%5d]=%5d\n", i, idx[i] );
      }
    }
    //---

    // Transitive Closure 
    PetscInt npts = 0;
    PetscInt* pts = NULL;
    PetscCall( DMPlexGetTransitiveClosure( dm, c, PETSC_TRUE, &npts, &pts ) );

    // クロージャ順 -> Ke計算用順に交換するための対応関係を取得
    std::vector<int> permt;
    PetscCall( get_permutation( permt ) );

    // Ke計算用の順序で座標マトリクスを得る
    PetscScalar xye[12]; // Ke計算用の順序の座標マトリクス (6x2) 
    for( PetscInt k=0; k<npts; k++ )
    {
      const PetscInt p = pts[2*k];
      PetscInt depth;
      PetscCall( DMPlexGetPointDepth( dm, p, &depth ) );
      if( depth == 2 ) continue;

      std::vector<double> xy( dim );
      PetscCall( get_coords( dm, p, xy ) );
      
      int ke = permt[ k - 1 ];
      for( int i=0; i<dim; i++ ) xye[ ke*dim + i ] = xy[i];
    }
    //+++
    if( debug )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d : coordinates in Ke order\n", rank, c );
      for( int i=0; i<6; i++ )
      {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%5d: %15.5e%15.5e\n", i, xye[i*dim+0], xye[i*dim+1] );
      }
    }
    //---

    // 積分点位置
    PetscScalar r[4];
    r[0] = 0.816847572980459; r[1] = 0.091576213509771;
    r[2] = 0.108103018168070; r[3] = 0.445948490915965;

    // 積分点重み
    PetscScalar w[2];
    w[0] = 0.109951743655322; w[1] = 0.223381589678011;

    PetscScalar Ke[144];
    for( int i=0; i<144; i++ ) Ke[i]=0.0;

    PetscInt num_gp = 6; // 積分点数
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
    } // for( gp )

    //+++
    if( debug )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d : Ke\n", rank, c );
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

    // Keの並べ替え
    PetscCall( permutate_Kuue_matrix( 6, dim, Ke ) );

    // アセンブリ
    PetscCall( DMPlexMatSetClosure( dm, loc_sec, glb_sec, A, c, Ke, ADD_VALUES ) );
    
    // 局所自由度インデックスを返す
    PetscCall( DMPlexRestoreClosureIndices( dm, loc_sec, glb_sec, c, PETSC_TRUE, &nidx, &idx, NULL, NULL ) );
    PetscCall( DMPlexRestoreTransitiveClosure( dm, c, PETSC_TRUE, &npts, &pts ) );

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

// 変位の出力
PetscErrorCode show_displacement( const DM& dm, const Vec& sol )
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

      // 座標
      std::vector<double> xy( dim, 0.0 );
      PetscCall( get_coords( dm, p, xy ) );

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
// セルのクロージャ順とKe行番号順の対応を作る（P2三角形）
PetscErrorCode get_permutation( std::vector<int>& permt )
{
  permt = { 5, 3, 4, 0, 1, 2 };
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ----------------------------------------------------------------------------
// Ke をセルのクロージャ順に並べ替える（P2三角形）
PetscErrorCode permutate_Kuue_matrix( const int nnode, const int dim, PetscScalar* Kuue )
{
  int rows = nnode * dim;
  int cols = nnode * dim;
  double *Kuue_tmp = new double[ rows * cols ];

  for( int i=0; i<rows*cols; i++ ) Kuue_tmp[i] = Kuue[i];

  std::vector<int> permt;
  PetscCall( get_permutation( permt ) );

  for( int i=0; i<nnode; i++ ){
    int io = permt[i];
    for( int k=0; k<dim; k++ ){
      int ik  = i *dim+k;
      int iok = io*dim+k;
      for( int j=0; j<nnode; j++ ){
        int jo = permt[j];
        for( int l=0; l<dim; l++ ){
          int jl  = j *dim+l;
          int jol = jo*dim+l;
          Kuue[ik*cols+jl] = Kuue_tmp[iok*cols+jol];
        }
      }
    }
  }

  delete[] Kuue_tmp;
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