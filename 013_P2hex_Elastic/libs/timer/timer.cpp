#include "timer.h"

timer tmr;

void timer::start()
{
    if( !flag_timer ) return;
    MPI_Barrier( PETSC_COMM_WORLD );
    this->m_start = MPI_Wtime();
}

void timer::stop( const char* ss )
{
    if( !flag_timer ) return;
    MPI_Barrier( PETSC_COMM_WORLD );
    double end = MPI_Wtime();
    double t0 = end - this->m_start;
    double tw;
    MPI_Reduce( &t0, &tw, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD );
    PetscPrintf( PETSC_COMM_WORLD, "%-35s:%15.5e[sec]\n", ss, tw );
}