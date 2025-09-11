#include "mesh.h"

//==========================================================================================

// 2つのdouble値の差の絶対値がtol以下か？
bool close2( const double a, const double b, const double tol )
{
  return fabs( a - b ) <= tol;
}

//
void read_msh_nodes( const std::string& mesh_path, msh::node_vec &nodes )
{
  // mshファイルを開く
  std::ifstream fmsh;
  fmsh.open( mesh_path, std::ios::in );
  if( !fmsh )
  {
    std::cout << mesh_path << " could not be opened" << std::endl;
    exit(1);
  }
  
  // $Nodesを探す
  std::string ss;
  while( ss != "$Nodes" )
  {
    getline( fmsh, ss );
  }

  // <numNodes>を読む
  int num_nodes = 0;
  getline( fmsh, ss );
  num_nodes = stoi( ss );

  // 節点情報を読む
  msh::node nd;
  for( int n=1; n<=num_nodes; n++ )
  {
    fmsh >> nd.ID >> nd.x >> nd.y >> nd.z;
    nodes.push_back( nd );
  }

  fmsh.close();
}

//
void read_msh_elems( const std::string& mesh_path, msh::elem_vec &elems )
{
  // mshファイルを開く
  std::ifstream fmsh;
  fmsh.open( mesh_path, std::ios::in );
  if( !fmsh )
  {
    std::cout << mesh_path << " could not be opened" << std::endl;
    exit(1);
  }
  
  // $Elementsを探す
  std::string ss;
  while( ss != "$Elements" )
  {
    getline( fmsh, ss );
  }

  // <numElements>を読む
  int num_elems = 0;
  getline( fmsh, ss );
  num_elems = stoi( ss );

  // 要素情報を読む
  for( int m=1; m<=num_elems; m++ )
  {
    int elem_tag, elem_type, num_tags;
    fmsh >> elem_tag >> elem_type >> num_tags;

    for( int i=0; i<num_tags; i++ )
    {
      int tag;
      fmsh >> tag;
    }
    bool find = false;
    std::vector<int> nodeIDs;
    if( elem_type== 9 ) // 6節点3角形要素
    {
      for( int i=0; i<6; i++ )
      {
        int nid=0;
        fmsh >> nid;
        nodeIDs.push_back( nid );
      }
      find = true;
    }

    if( !find )
    {
      getline( fmsh, ss );
    }
    if( find )
    {
      msh::elem elm;
      elm.ID = elem_tag;
      elm.type = elem_type;
      elm.nodeIDs = nodeIDs;
      if( elem_type == 9 )
      {
        elm.nodeIDs[3] = nodeIDs[4];
        elm.nodeIDs[4] = nodeIDs[5];
        elm.nodeIDs[5] = nodeIDs[3];
      }
      elems.push_back( elm );
    }
  }

  fmsh.close();
}

// ----------------------------------------------------------------------------
// DMPlexのセルpointIDとgmshのelementTagの紐づけ
PetscErrorCode get_elemID_map( const DM& dm, const msh::node_vec& nodes, const msh::elem_vec& elems,
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

    for( const msh::elem &e : elems )
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
        const msh::node& nd = nodes.id_of(e.nodeIDs[nid]);
        
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
PetscErrorCode get_elemID_map( const std::string& mesh_path, const DM& dm,
    std::map<int,int>& eID2pID, std::map<int,int>& pID2eID )
{
  // .mshから節点を読込む
  msh::node_vec nodes;
  read_msh_nodes( mesh_path, nodes );

  // .mshから要素を読込む
  msh::elem_vec elems;
  read_msh_elems( mesh_path, elems );

  PetscCall( get_elemID_map( dm, nodes, elems, eID2pID, pID2eID, true ) );

  PetscFunctionReturn( PETSC_SUCCESS );
}

//
void output_vtk( const std::string& vtk_path, const int rank, const int nproc, const std::map<int,int>& pID2eID,
  msh::node_vec &nodes, msh::elem_vec& elems )
{
  int size = pID2eID.size();
  int* rank_of_elem;
  // elemsのindex順で，その要素を所有するrankを特定し，rank_of_elemに格納する
  if( nproc > 1 )
  {
    int* size_vec = new int[nproc];
    //各rankのsizeをベクトル化する
    MPI_Allgather( &size, 1, MPI_INT, size_vec, 1, MPI_INT, PETSC_COMM_WORLD );
    //+++
    //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d size_vec:", rank );
    //for( int i=0; i<nproc; i++ )
    //{
    //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%5d", size_vec[i] );
    //}
    //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
    //---

    int* first_idxs = new int[nproc+1];
    first_idxs[0] = 0;
    for( int i=1; i<=nproc; i++ ) first_idxs[i] = first_idxs[i-1] + size_vec[i-1];
    int total_size = first_idxs[nproc];
    rank_of_elem = new int[total_size];

    // pID2eIDのローカルなeIDをベクトル化
    int* elemIDs_local = new int[size];
    int cnt =0;
    for( const auto& [pID,eID] : pID2eID )
    {
      elemIDs_local[cnt++] = eID;
    }

    // elemIDs_rank_order （rank=0の）にrankの順に eID をコピーする
    int* elemIDs_rank_order = new int[total_size];
    if( rank == 0 )
    {
      memcpy( elemIDs_rank_order, elemIDs_local, size*sizeof(int) );
    }
    for( int i=1; i<nproc; i++ )
    {
      if( rank == i ) MPI_Send( elemIDs_local, size, MPI_INT, 0, 0, PETSC_COMM_WORLD );
      if( rank == 0 ) MPI_Recv( elemIDs_rank_order+first_idxs[i], size_vec[i], MPI_INT, i, 0, PETSC_COMM_WORLD, MPI_STATUS_IGNORE );
    }

    if( rank == 0 )
    {
      for( int r=0; r<nproc; r++ )
      {
        for( int i=first_idxs[r]; i<first_idxs[r+1]; i++ )
        {
          int eID = elemIDs_rank_order[i];
          int idx = elems.idx_of_id( eID );
          rank_of_elem[idx] = r;
        }
      }
    }

    //+++
    //for( int i=0; i<total_size; i++ )
    //{
    //  PetscPrintf( PETSC_COMM_WORLD, "idx=%5d eID=%5d rank=%3d\n", i, elems.id_of_idx(i), rank_of_elem[i] );
    //}
    //---

    delete[] size_vec;
    delete[] first_idxs;
    delete[] elemIDs_local;
    delete[] elemIDs_rank_order;
  }
  else
  {
    rank_of_elem = new int[size];
    for( int i=0; i<size; i++ ) rank_of_elem[i] = 0;
  }

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
    s_fprv << "POINTS " << nodes.size() << " float" << std::endl;
    for( int nidx=0; nidx<nodes.size(); nidx++ )
    {
      float x = static_cast<float>( nodes[nidx].x );
      float y = static_cast<float>( nodes[nidx].y );
      float z = static_cast<float>( nodes[nidx].z );
      s_fprv << x << " " << y << " " << z << std::endl;
    }
    // 頂点の数を数える（3角形のみ対応）
    int numv = elems.size()*(3+1);
    // 要素データの出力
    s_fprv << "CELLS " << elems.size() << " " << numv << std::endl;
    for( int eidx=0; eidx<elems.size(); eidx++ )
    {
      s_fprv << 3;
      for( int i=0; i<3; i++ )
      {
        int nid = elems[eidx].nodeIDs[i];
        int nidx = nodes.idx_of_id( nid );
        s_fprv << " " << nidx;
      }
      s_fprv << std::endl;
    }
    // セルタイプの出力
    s_fprv << "CELL_TYPES " << elems.size() << std::endl;
    for( int eidx=0; eidx<elems.size(); eidx++ )
    {
      s_fprv << 5 << std::endl;
    }
    // ランクの出力
    s_fprv << "CELL_DATA " << elems.size() << std::endl;
    s_fprv << "SCALARS rank int" << std::endl;
    s_fprv << "LOOKUP_TABLE default" << std::endl;
    for( int eidx=0; eidx<elems.size(); eidx++ )
    {
      s_fprv << rank_of_elem[eidx] << std::endl;
    }

    fprv << s_fprv.str();

    fprv.close();
  }

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  delete[] rank_of_elem;
}