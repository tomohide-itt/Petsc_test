#include "mesh.h"

//==========================================================================================

// 2つのdouble値の差の絶対値がtol以下か？
bool close2( const double a, const double b, const double tol )
{
  return fabs( a - b ) <= tol;
}

//
void read_msh_nodes( const char* mesh_path, node_vec &nodes )
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
    fmsh >> nd.ID >> nd.x >> nd.y >> nd.z;
    nodes.push_back( nd );
  }

  fmsh.close();
}

//
void read_msh_elems( const char* mesh_path, elem_vec &elems )
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
      elem elm;
      elm.ID = elem_tag;
      elm.type = elem_type;
      elm.nodeIDs = nodeIDs;
      elems.push_back( elm );
    }
  }

  fmsh.close();
}

//
void show_elems( const std::vector<elem>& elems )
{
  std::cout << "number of elems = " << elems.size() << std::endl;
  for( int i=0; i<elems.size(); i++ )
  {
    std::cout << "ID: " << std::setw(7) << elems[i].ID;
    std::cout << " nodeIDs: ";
    for( int j=0; j<elems[i].nodeIDs.size(); j++ )
    {
      std::cout << std::setw(7) << elems[i].nodeIDs[j];
    }
    std::cout << std::endl;
  }
} 