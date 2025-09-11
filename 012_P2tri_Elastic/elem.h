#ifndef ELEM_H
#define ELEM_H
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <iomanip>
#include <map>
#include <petscdmplex.h>
#include <petscksp.h>
#include "node.h"


class elem
{
public:
    elem( int p, std::vector<int> nd_clos_ids, node_vec& nodes );

    std::array<double,12> get_xye() const;
    std::array<double,12> dNdr_at( const int ng ) const;
    std::array<double,4>  J_at( const int ng ) const;
    double detJ_at( const int ng ) const;
    std::array<double,4> J_I_T_at( const int ng ) const;
    std::array<double,12> derivN_at( const int ng ) const;
    double fac_at( const int ng ) const;
    std::array<double,48> B_matrix_at( const int ng ) const;
    std::array<double,48> BVOL_matrix_at( const int ng ) const;
    std::array<double,144> Kuu_matrix( const double* D ) const;
    void permutate_Kuu_matrix( std::array<double,144>& Kuu ) const;
public:
    int id;
    int pid;
    int type;
    int num_nods;
    int dim;
    int num_gp;
    std::vector<int> node_pids;
    std::vector<node*> nod;
    std::vector<int> perm;

    std::array<double,4> gp_pos;
    std::array<double,2> gp_wei;
};

class elem_vec
{
public:
    elem_vec() : max_idx(0){}
    const int size()const{ return m_elems.size(); }
    const elem& operator[]( const int idx ) const{ return m_elems[idx]; }
    elem& operator[]( const int idx ){ return m_elems[idx]; }
    const elem& pid_is( const int pid ) const{ return m_elems[m_pid2idx.at(pid)]; }
    elem& pid_is( const int pid ){ return m_elems[m_pid2idx.at(pid)]; }
    std::vector<elem>::iterator begin(){ return m_elems.begin(); }
    std::vector<elem>::const_iterator begin() const{ return m_elems.begin(); }
    std::vector<elem>::iterator end(){ return m_elems.end(); }
    std::vector<elem>::const_iterator end() const{ return m_elems.end(); }
    void push_back( const elem& e )
    {
        m_elems.push_back(e);
        m_pid2idx[e.pid] = max_idx;
        m_idx2pid[max_idx] = e.pid;
        max_idx++;
    }
    void show()
    {
        // rankの取得
        PetscMPIInt rank;
        MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d : Elements\n", rank );

        for( int i=0; i<this->size(); i++ )
        {
            PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d id=%5d pid=%5d node_pids=",
                rank, i, m_elems[i].pid );
            for( int j=0; j<m_elems[i].node_pids.size(); j++ )
            {
                PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%5d", m_elems[i].node_pids[j] );
            }
            PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
        }
        PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
    }
private:
    std::vector<elem> m_elems;
    int max_idx;
    std::map<int,int> m_pid2idx;
    std::map<int,int> m_idx2pid;
};

#endif

