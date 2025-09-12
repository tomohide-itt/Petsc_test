#ifndef GMSH_H
#define GMSH_H
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <iomanip>
#include <map>
#include <petscdmplex.h>
#include <petscksp.h>

class elem_vec;

namespace gmsh
{
    class node
    {
    public:
        int tag;
        std::array<double,3> xy;
    };

    class node_vec
    {
    public:
        node_vec() : max_idx(0){}
        const int size()const{ return m_nodes.size(); }
        const node& operator[]( const int idx ) const{ return m_nodes[idx]; }
        node& operator[]( const int idx ){ return m_nodes[idx]; }
        void push_back( const node& nd )
        {
            m_nodes.push_back(nd);
            tag2idx[nd.tag]  = max_idx;
            idx2tag[max_idx] = nd.tag;
            max_idx++;
        }
    public:
        std::map<int,int> tag2idx;
        std::map<int,int> idx2tag;
    private:
        int max_idx;
        std::vector<node> m_nodes;
    };

    class elem
    {
    public:
        int tag;
        int type;
        std::vector<int> node_tags;
    };

    class elem_vec
    {
    public:
        elem_vec() : max_idx(0){}
        const int size()const{ return m_elems.size(); }
        std::vector<elem>::iterator begin(){ return m_elems.begin(); }
        std::vector<elem>::const_iterator begin() const{ return m_elems.begin(); }
        std::vector<elem>::iterator end(){ return m_elems.end(); }
        std::vector<elem>::const_iterator end() const{ return m_elems.end(); }
        void push_back( const elem& e )
        {
            m_elems.push_back(e);
            tag2idx[e.tag] = max_idx;
            idx2tag[max_idx] = e.tag;
            max_idx++;
        }
    public:
        std::map<int,int> tag2idx;
        std::map<int,int> idx2tag;
    private:
        int max_idx;
        std::vector<elem> m_elems;

    };

    void read_nodes( const std::string& mesh_path, node_vec& nodes );
    void read_elems( const std::string& mesh_path, elem_vec& elems );
}

PetscErrorCode get_elem_tag_local_pid_map( const DM& dm, const gmsh::node_vec& nodes, const gmsh::elem_vec& elems,
    std::map<int,int>& etag2lpid, std::map<int,int>& lpid2etag, std::map<int,int>& ntag2lpid, std::map<int,int>& lpid2ntag, const bool debug=false );

PetscErrorCode get_mesh_info( const std::string& mesh_path, const DM& dm,
    std::map<int,int>& ntag2gnid, std::map<int,int>& gnid2ntag,
    std::map<int,int>& etag2geid, std::map<int,int>& geid2etag );

#endif

