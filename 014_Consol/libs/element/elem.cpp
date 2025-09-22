#include "elem.h"

elem_vec::elem_vec(){}

void elem_vec::show() const
{
    // rankの取得
    PetscMPIInt rank;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d : Elements\n", rank );

    for( int i=0; i<this->size(); i++ )
    {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d id=%5d pid=%5d node_pids=",
            rank, i, m_elems[i]->pid );
        for( int j=0; j<m_elems[i]->node_pids.size(); j++ )
        {
            PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%5d", m_elems[i]->node_pids[j] );
        }
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
    }
    PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
}