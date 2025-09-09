#ifndef MESH_H
#define MESH_H
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <iomanip>
#include <map>
#include <petscdmplex.h>
#include <petscksp.h>

namespace msh
{

class node
{
public:
    int ID;
    double x;
    double y;
    double z;
};

class node_vec
{
public:
    node_vec() : max_idx(0){}
    const int size()const{ return m_nodes.size(); }
    const node& operator[]( const int idx ) const{ return m_nodes[idx]; }
    node& operator[]( const int idx ){ return m_nodes[idx]; }
    const node& id_of( const int id ) const{ return m_nodes[m_id2idx.at(id)]; }
    node& id_of( const int id ){ return m_nodes[m_id2idx.at(id)]; }
    const int idx_of_id( const int id ) const{ return m_id2idx.at(id); }
    const int id_of_idx( const int idx ) const{ return m_idx2id.at(idx); }
    void push_back( const node& nd )
    {
        m_nodes.push_back(nd);
        m_id2idx[nd.ID] = max_idx;
        m_idx2id[max_idx] = nd.ID;
        max_idx++;
    }
    void show() const
    {
        PetscPrintf( PETSC_COMM_WORLD, "number of nodes = %d\n", this->size() );
        for( int idx=0; idx<this->size(); idx++ )
        {
            PetscPrintf( PETSC_COMM_WORLD, "idx:%7d ID:%7d (x y z):%15.5e%15.5e%15.5e\n",
                idx, m_idx2id.at(idx), (*this)[idx].x, (*this)[idx].y, (*this)[idx].z );
        }
    }
private:
    std::vector<node> m_nodes;
    int max_idx;
    std::map<int,int> m_id2idx;
    std::map<int,int> m_idx2id;
};

class elem
{
public:
    int ID;
    int type;
    std::vector<int> nodeIDs;
};

class elem_vec
{
public:
    elem_vec() : max_idx(0){}
    const int size()const{ return m_elems.size(); }
    const elem& operator[]( const int idx ) const{ return m_elems[idx]; }
    elem& operator[]( const int idx ){ return m_elems[idx]; }
    const elem& id_of( const int id ) const{ return m_elems[m_id2idx.at(id)]; }
    elem& id_of( const int id ){ return m_elems[m_id2idx.at(id)]; }
    const int idx_of_id( const int id ) const{ return m_id2idx.at(id); }
    const int id_of_idx( const int idx ) const{ return m_idx2id.at(idx); }
    std::vector<elem>::iterator begin(){ return m_elems.begin(); }
    std::vector<elem>::const_iterator begin() const{ return m_elems.begin(); }
    std::vector<elem>::iterator end(){ return m_elems.end(); }
    std::vector<elem>::const_iterator end() const{ return m_elems.end(); }
    void push_back( const elem& e )
    {
        m_elems.push_back(e);
        m_id2idx[e.ID] = max_idx;
        m_idx2id[max_idx] = e.ID;
        max_idx++;
    }
    void show() const
    {
        PetscPrintf( PETSC_COMM_WORLD, "number of elems = %d\n", this->size() );
        for( int idx=0; idx<this->size(); idx++ )
        {
            PetscPrintf( PETSC_COMM_WORLD, "idx:%7d ID:%7d nodeIDs:", idx, m_idx2id.at(idx) );
            for( int j=0; j<(*this)[idx].nodeIDs.size(); j++ )
            {
                PetscPrintf( PETSC_COMM_WORLD, "%7d", (*this)[idx].nodeIDs[j] );
            }
            PetscPrintf( PETSC_COMM_WORLD, "\n" );
        }
    }
private:
    std::vector<elem> m_elems;
    int max_idx;
    std::map<int,int> m_id2idx;
    std::map<int,int> m_idx2id;
};

}

PetscErrorCode get_elemID_map( const DM& dm, const msh::node_vec& nodes, const msh::elem_vec& elems,
    std::map<int,int>& eID2pID, std::map<int,int>& pID2eID, const bool debug=false );
PetscErrorCode get_elemID_map( const std::string& mesh_path, const DM& dm,
    std::map<int,int>& eID2pID, std::map<int,int>& pID2eID );

bool close2( const double a, const double b, const double tol );
void read_msh_nodes( const std::string& mesh_path, msh::node_vec &nodes );
void read_msh_elems( const std::string& mesh_path, msh::elem_vec &elems );
void output_vtk( const std::string& vtk_path, const int rank, const int nproc, const std::map<int,int>& pID2eID, msh::node_vec& nodes, msh::elem_vec& elems );

#endif

