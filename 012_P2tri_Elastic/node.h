#ifndef NODE_H
#define NODE_H
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <iomanip>
#include <map>
#include <petscdmplex.h>
#include <petscksp.h>

class node
{
public:
    node( int p, double x, double y, double z ) : pid(p)
    {
        xy[0] = x;  xy[1] = y;  xy[2] = z;
    }
public:
    int id;
    int pid;
    std::array<double,3> xy;
};

class node_vec
{
public:
    node_vec() : max_idx(0){}
    const int size()const{ return m_nodes.size(); }
    const node& operator[]( const int idx ) const{ return m_nodes[idx]; }
    node& operator[]( const int idx ){ return m_nodes[idx]; }
    const node& pid_is( const int pid ) const{ return m_nodes[m_pid2idx.at(pid)]; }
    node& pid_is( const int pid ){ return m_nodes[m_pid2idx.at(pid)]; }
    void push_back( const node& nd )
    {
        m_nodes.push_back(nd);
        m_pid2idx[nd.pid] = max_idx;
        m_idx2pid[max_idx] = nd.pid;
        max_idx++;
    }
private:
    std::vector<node> m_nodes;
    int max_idx;
    std::map<int,int> m_pid2idx;
    std::map<int,int> m_idx2pid;
};

#endif

