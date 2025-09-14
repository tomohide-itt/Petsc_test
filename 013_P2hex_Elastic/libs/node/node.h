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
    const int size() const{ return max_idx; }
    void show();
private:
    std::vector<node*> m_nodes;
    int max_idx;
    std::map<int,int> m_pid2idx;
    std::map<int,int> m_idx2pid;
};

#endif

