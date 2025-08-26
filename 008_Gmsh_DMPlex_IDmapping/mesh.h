#ifndef MESH_H
#define MESH_H
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <iomanip>
#include <map>

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
    void push_back( const node& nd )
    {
        m_nodes.push_back(nd);
        m_id2idx[nd.ID] = max_idx;
        m_idx2id[max_idx] = nd.ID;
        max_idx++;
    }
    void show() const
    {
        std::cout << "number of nodes = " << this->size() << std::endl;
        std::cout << std::scientific << std::setprecision(5);
        for( int idx=0; idx<this->size(); idx++ )
        {
            std::cout << "idx: " << std::setw(7) << idx;
            std::cout << " ID: " << std::setw(7) << m_idx2id.at(idx);
            std::cout << "  (x, y, z): ";
            std::cout << std::setw(15) << (*this)[idx].x;
            std::cout << std::setw(15) << (*this)[idx].y;
            std::cout << std::setw(15) << (*this)[idx].z;
            std::cout << std::endl;
        }
        std::cout << std::defaultfloat;
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
        std::cout << "number of elems = " << this->size() << std::endl;
        for( int idx=0; idx<this->size(); idx++ )
        {
            std::cout << "idx: " << std::setw(7) << idx;
            std::cout << " ID: " << std::setw(7) << m_idx2id.at(idx);
            std::cout << " nodeIDs: ";
            for( int j=0; j<(*this)[idx].nodeIDs.size(); j++ )
            {
                std::cout << std::setw(7) << (*this)[idx].nodeIDs[j];
            }
            std::cout << std::endl;
        }
    }
private:
    std::vector<elem> m_elems;
    int max_idx;
    std::map<int,int> m_id2idx;
    std::map<int,int> m_idx2id;
};

bool close2( const double a, const double b, const double tol );
void read_msh_nodes( const char* mesh_path, node_vec &nodes );
void read_msh_elems( const char* mesh_path, elem_vec &elems );
void show_elems( const std::vector<elem>& elems );

#endif

