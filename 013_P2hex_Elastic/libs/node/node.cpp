#include "node.h"

node_vec::node_vec(){}

node_vec::~node_vec()
{
    for( int n=0; n<this->size(); n++ ) delete m_nodes[n];
    //printf( "%s\n", __FUNCTION__ );
}

void node_vec::create_new( const int p, const double x, const double y, const double z )
{
    node* pnd = new node( p, x, y, z );
    int idx = m_nodes.size();
    m_nodes.push_back(pnd);
    m_pid2idx[p] = idx;
    m_idx2pid[idx] = p;
}

void node_vec::show()
{
    // rankの取得
    PetscMPIInt rank;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d : Nodal points\n", rank );

    for( int i=0; i<this->size(); i++ )
    {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d id=%5d pid=%5d (x y z)=%15.5e%15.5e%15.5e\n",
            rank, i, m_nodes[i]->pid, m_nodes[i]->xy[0], m_nodes[i]->xy[1], m_nodes[i]->xy[2] );
    }
    PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
}