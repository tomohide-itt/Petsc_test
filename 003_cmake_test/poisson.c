#include <petscksp.h>

int main(int argc,char **argv){
  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));

  PetscInt n = 50, rs, re;
  Mat A;
  Vec x, b;
  KSP ksp;

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&rs,&re));
  for (PetscInt i=rs;i<re;i++){
    if(i>0)   PetscCall(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(A,i,i, 2.0,INSERT_VALUES));
    if(i<n-1) PetscCall(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(VecCreate(PETSC_COMM_WORLD,&b));
  PetscCall(VecSetSizes(b,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(b));
  PetscCall(VecDuplicate(b,&x));
  PetscCall(VecSet(b,1.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetType(ksp,KSPCG));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp,b,x));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}