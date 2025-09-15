#ifndef NODE_H
#define NODE_H
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <iomanip>
#include <map>
#include <memory>
#include <petscdmplex.h>
#include <petscksp.h>

class node
{
public:
    node( int p, double x, double y, double z ) : pid(p)
    {
        xy[0] = x;  xy[1] = y;  xy[2] = z;
        uv = { 0.0, 0.0, 0.0 };
    }
public:
    int id;
    int pid;
    std::array<double,3> xy;
    std::array<double,3> uv;
};

class node_vec
{
public:
    node_vec();
    ~node_vec();
    void create_new( const int p, const double x, const double y, const double z );
    const int size() const{ return m_nodes.size(); }
    //const node* pid_is( const int pid ) const{ return m_nodes[m_pid2idx.at(pid)]; }
    //node* pid_is( const int pid ){ return m_nodes[m_pid2idx.at(pid)]; }
    const std::shared_ptr<node> pid_is( const int pid ) const{ return m_nodes[m_pid2idx.at(pid)]; }
    void show() const;
private:
    std::vector<std::shared_ptr<node>> m_nodes;
    std::map<int,int> m_pid2idx;
    std::map<int,int> m_idx2pid;
};

#endif

