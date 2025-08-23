#include <petscksp.h>

int main(int argc,char**argv)
{
  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));
  PetscPrintf(PETSC_COMM_WORLD,"PETSc OK\n");
  PetscCall(PetscFinalize());
  return 0;
}