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
    void show()
    {
        // rankの取得
        PetscMPIInt rank;
        MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d : Nodal points\n", rank );

        for( int i=0; i<this->size(); i++ )
        {
            PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d id=%5d pid=%5d (x y z)=%15.5e%15.5e%15.5e\n",
                rank, i, m_nodes[i].pid, m_nodes[i].xy[0], m_nodes[i].xy[1], m_nodes[i].xy[2] );
        }
        PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
    }
private:
    std::vector<node> m_nodes;
    int max_idx;
    std::map<int,int> m_pid2idx;
    std::map<int,int> m_idx2pid;
};

#endif

