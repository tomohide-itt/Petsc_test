#include "gmsh.h"

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
    std::map<int,int>& etag2lpid, std::map<int,int>& lpid2etag, const bool debug )
{
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

//-----------------------------------------------------------------------------------------------
// mshファイルを読んで，tag - id - pid の関係を得る
PetscErrorCode get_mesh_info( const std::string& mesh_path, const DM& dm,
    std::map<int,int>& ntag2gnid, std::map<int,int>& gnid2ntag,
    std::map<int,int>& etag2geid, std::map<int,int>& geid2etag )
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
  if( rank == 0 )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "ntag2gnid\n" );
    for( const auto& [ntag,gnid] : ntag2gnid )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "ntag=%5d gnid=%5d\n", ntag, gnid );
    }
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "-----------\n" );
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "etag2geid\n" );
    for( const auto& [etag,geid] : etag2geid )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "etag=%5d geid=%5d\n", etag, geid );
    }
  }
  //---

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}
