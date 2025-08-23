#include <petscdmplex.h>
#include <petscksp.h>

int main(int argc,char **argv)
{
  DM dm = NULL;
  DM dmDist = NULL;
  PetscMPIInt rank, size;

  PetscCall(PetscInitialize(&argc,&argv,NULL,
    "Load a Gmsh .msh into DMPlex and distribute in parallel.\n" ) );
  
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
  MPI_Comm_size( PETSC_COMM_WORLD, &size );

  // ファイルから読込
  // -dm_plex_filename ****.msh のようにコマンドで渡す形式
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));      // <- ここで -dm_plex_* オプションが効く
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  // 並列分割（コードから明示的に行う）
  if (size > 1) {
    PetscPartitioner part;
    PetscCall(DMPlexGetPartitioner(dm, &part));
    // 既定は "simple"。ParMETIS/PT-Scotchを使うなら実行時オプションで変更推奨。
    // 例: -petscpartitioner_type parmetis
    PetscCall(PetscPartitionerSetFromOptions(part)); 
    PetscCall(DMPlexDistribute(dm, 0, NULL, &dmDist));
    if (dmDist) { PetscCall(DMDestroy(&dm)); dm = dmDist; }
  }

  //  結果表示（rankごとに自領域の要素数など）
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view")); // 例: -dm_view ::ascii_info_detail

  //  VTK出力
  PetscViewer viewer;
  PetscViewerVTKOpen(PETSC_COMM_WORLD, "partition.vtu", FILE_MODE_WRITE, &viewer);
  DMView(dm, viewer);

  PetscCall(DMDestroy(&dm));
  PetscViewerDestroy(&viewer);
  PetscCall(PetscFinalize());
  return 0;
}