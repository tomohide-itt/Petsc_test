#include "gmsh.h"
#include "elem.h"
#include "functions.h"

//==========================================================================================
// mshファイルを読んで，gmsh::node_vec を作る
void gmsh::read_nodes( const std::string& mesh_path, node_vec &nodes )
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
  node nd;
  for( int n=1; n<=num_nodes; n++ )
  {
    fmsh >> nd.tag >> nd.xy[0] >> nd.xy[1] >> nd.xy[2];
    nodes.push_back( nd );
  }

  fmsh.close();
}

//-----------------------------------------------------------------------------------------------
// mshファイルを読んで，gmsh::elem_vec を作る
void gmsh::read_elems( const std::string& mesh_path, elem_vec &elems )
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
    std::vector<int> node_tags;
    if( elem_type== 9 ) // 6節点3角形要素
    {
      for( int i=0; i<6; i++ )
      {
        int ntag=0;
        fmsh >> ntag;
        node_tags.push_back( ntag );
      }
      find = true;
    }

    if( !find )
    {
      getline( fmsh, ss );
    }
    if( find )
    {
      elem elm;
      elm.tag = elem_tag;
      elm.type = elem_type;
      elm.node_tags = node_tags;
      if( elem_type == 9 )
      {
        elm.node_tags[3] = node_tags[4];
        elm.node_tags[4] = node_tags[5];
        elm.node_tags[5] = node_tags[3];
      }
      elems.push_back( elm );
    }
  }

  fmsh.close();
}

//-----------------------------------------------------------------------------------------------
// 要素の ローカルなpid と tag の関係を得る
PetscErrorCode get_elem_tag_local_pid_map( const DM& dm, const gmsh::node_vec& nodes, const gmsh::elem_vec& elems,
    std::map<int,int>& etag2lpid, std::map<int,int>& lpid2etag,
    std::map<int,int>& ntag2lpid, std::map<int,int>& lpid2ntag, const bool debug )
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

  std::vector<bool> visited_elem( elems.size(), false );

  // セルでループ
  for( PetscInt c=c_start; c<c_end; c++ )
  {
    PetscInt npts = 0;
    PetscInt* pts = NULL;
    PetscCall( DMPlexGetTransitiveClosure( dm, c, PETSC_TRUE, &npts, &pts ) );

    std::vector<std::vector<double>> xye;
    std::vector<int> pe;

    // このセルのポイントでループ
    for( PetscInt k=0; k<npts; k++ )
    {
      const PetscInt p = pts[2*k];
      PetscInt depth;
      PetscCall( DMPlexGetPointDepth( dm, p, &depth ) );
      if( depth == 2 ) continue; // pがセルなら飛ばす

      std::vector<double> xy;
      PetscCall( get_coords( dm, p, xy ) );
      xye.push_back( xy );
      pe.push_back( p );
    }

    int num_nods = xye.size();
    //+++
    if( debug )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d c=%5d\n", rank, c );
      for( int i=0; i<xye.size(); i++ )
      {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d p=%5d (x y)=%15.5e%15.5e\n", rank, pe[i], xye[i][0], xye[i][1] );
      }
    }
    //---

    // gelemsと突き合わせて，全ての座標が一致しているかチェックし，そのtagとcのマップを登録する
    double tol = 1.0e-8;

    for( const gmsh::elem &ge : elems )
    {
      if( ge.node_tags.size() != num_nods ) continue;
      int eidx = elems.tag2idx.at( ge.tag );
      if( visited_elem[eidx] ) continue;

      std::vector<bool> visited_node( num_nods, false );
      std::vector<int> ntags( num_nods, -1 );
      bool identical = true;
      for( int gni=0; gni<ge.node_tags.size(); gni++ )
      {
        int ntag = ge.node_tags[gni];
        int nidx = nodes.tag2idx.at( ntag );
        const gmsh::node& nd = nodes[nidx];

        bool find = false;
        for( int i=0; i<num_nods; i++ )
        {
          if( visited_node[i] ) continue;
          bool same = true;
          if( dim == 2 )
          {
            same = same && close2( nd.xy[0], xye[i][0], tol ) && close2( nd.xy[1], xye[i][1], tol );
          }
          if( same )
          {
            visited_node[i] = true;
            ntag2lpid[ntag] = pe[i];
            lpid2ntag[pe[i]] = ntag;
            find = true;
            break;
          }
        }
        identical = identical && find;
        if( !find ) break;
      }
      if( identical )
      {
        visited_elem[eidx] = true;
        etag2lpid[ge.tag] = c;
        lpid2etag[c] = ge.tag;
        break;
      }
    }
    PetscCall( DMPlexRestoreTransitiveClosure( dm, c, PETSC_TRUE, &npts, &pts ) );
  }

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

//-----------------------------------------------------------------------------------------------
// mshファイルを読んで，tag - id - pid の関係を得る
PetscErrorCode get_mesh_info( const std::string& mesh_path, const DM& dm,
    std::map<int,int>& ntag2gnid, std::map<int,int>& gnid2ntag,
    std::map<int,int>& etag2geid, std::map<int,int>& geid2etag,
    std::map<int,int>& ntag2lpid, std::map<int,int>& lpid2ntag,
    std::map<int,int>& etag2lpid, std::map<int,int>& lpid2etag )
{
  // rankの取得
  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

  // 節点読込
  gmsh::node_vec nodes;
  gmsh::read_nodes( mesh_path, nodes );
  ntag2gnid = nodes.tag2idx;
  gnid2ntag = nodes.idx2tag;

  //要素読込
  gmsh::elem_vec elems;
  gmsh::read_elems( mesh_path, elems );
  etag2geid = elems.tag2idx;
  geid2etag = elems.idx2tag;

  //+++
  //if( rank == 0 )
  //{
  //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
  //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "ntag2gnid\n" );
  //  for( const auto& [ntag,gnid] : ntag2gnid )
  //  {
  //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "ntag=%5d gnid=%5d\n", ntag, gnid );
  //  }
  //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
  //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "etag2geid\n" );
  //  for( const auto& [etag,geid] : etag2geid )
  //  {
  //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "etag=%5d geid=%5d\n", etag, geid );
  //  }
  //}
  //---

  //std::map<int,int> etag2lpid, lpid2etag, ntag2lpid, lpid2ntag;
  PetscCall( get_elem_tag_local_pid_map( dm, nodes, elems, etag2lpid, lpid2etag, ntag2lpid, lpid2ntag, false ) );

  //+++
  //{
  //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
  //  for( const auto& [lpid, ntag] : lpid2ntag )
  //  {
  //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d lpid=%5d ntag=%5d\n", rank, lpid, ntag );
  //  }
  //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
  //  for( const auto& [ntag, lpid] : ntag2lpid )
  //  {
  //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d ntag=%5d lpid=%5d\n", rank, ntag, lpid );
  //  }
  //  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
  //  for( const auto& [lpid, etag] : lpid2etag )
  //  {
  //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d lpid=%5d etag=%5d\n", rank, lpid, etag );
  //  }
  //}
  //---

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// 2つのdouble値の差の絶対値がtol以下か？
bool close2( const double a, const double b, const double tol )
{
  return fabs( a - b ) <= tol;
}
