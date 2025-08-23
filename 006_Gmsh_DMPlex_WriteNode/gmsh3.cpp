#include <iostream>
#include <string>
#include <petscdmplex.h>
#include <petscksp.h>

static PetscErrorCode WriteNode( DM dm );

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
  // ファイル名を指定して読込
  std::string fmsh_name = "../gmsh_input/test2D.msh";
  PetscCall( DMPlexCreateGmshFromFile( PETSC_COMM_WORLD, fmsh_name.c_str(), PETSC_TRUE, &dm ) );

  // 並列分割（コードから明示的に行う）
  if (size > 1) {
    PetscPartitioner part;
    PetscCall(DMPlexGetPartitioner(dm, &part));
    PetscCall(PetscPartitionerSetType(part, PETSCPARTITIONERPARMETIS));
    PetscCall(DMPlexDistribute(dm, 0, NULL, &dmDist));
    if (dmDist) { PetscCall(DMDestroy(&dm)); dm = dmDist; }
  }

  //  結果表示（rankごとに自領域の要素数など）
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view")); // 例: -dm_view ::ascii_info_detail

  //  VTK出力
  PetscViewer viewer;
  PetscViewerVTKOpen(PETSC_COMM_WORLD, "partition.vtu", FILE_MODE_WRITE, &viewer);
  DMView(dm, viewer);

  // 節点情報を出力
  {
    PetscCall( WriteNode(dm) );
  }

  PetscCall(DMDestroy(&dm));
  PetscViewerDestroy(&viewer);
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode WriteNode( DM dm )
{
  PetscFunctionBeginUser;

  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

  // DMPlex内部の節点IDの範囲とグローバル節点番号ISを取得
  PetscInt vStart, vEnd;
  IS vIS;
  const PetscInt *vnum;
  PetscCall( DMPlexGetDepthStratum( dm, 0, &vStart, &vEnd ) );
  PetscCall( DMPlexGetVertexNumbering( dm, &vIS ) );
  PetscCall( ISGetIndices( vIS, &vnum ) );
  //+++
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank   = %3d\n", (int)rank );
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "vStart = %d\n", (int)vStart );
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "vEnd   = %d\n", (int)vEnd );
  //---

  // 座標アクセスの準備
  DM cdm;
  Vec coords;
  const PetscScalar *ca;
  PetscInt dim;
  PetscCall( DMGetCoordinateDM( dm, &cdm ) );
  PetscCall( DMGetCoordinatesLocal( dm, &coords ) );
  PetscCall( VecGetArrayRead( coords, &ca ) );
  PetscCall( DMGetCoordinateDim( dm, &dim ) );

  for( PetscInt p=vStart; p<vEnd; ++p )
  {
    // グローバル頂点番号（負値は-(id+1）が真のID）
    PetscInt g = vnum[p-vStart];
    PetscBool own = PETSC_TRUE;
    if( g < 0 )
    {
      g = -(g+1);
      own = PETSC_FALSE;
    }

    // 座標を安全に読み出し
    const PetscScalar *xp = NULL;
    PetscCall( DMPlexPointLocalRead( cdm, p, ca, &xp ) );
    if( !xp ) continue;

    // 出力
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%3d, %3d, %3d, %3d, ", (int)rank, (int)p, (int)g, (int)own );
    if( dim >= 1 ) PetscSynchronizedPrintf( PETSC_COMM_WORLD,"%15.5e", (double)xp[0] );
    if( dim >= 2 ) PetscSynchronizedPrintf( PETSC_COMM_WORLD,"%15.5e", (double)xp[1] );
    if( dim >= 3 ) PetscSynchronizedPrintf( PETSC_COMM_WORLD,"%15.5e", (double)xp[2] );
    PetscSynchronizedPrintf( PETSC_COMM_WORLD,"\n" );
  }
  PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
 
  PetscCall( ISRestoreIndices( vIS, &vnum ) );
  PetscCall( VecRestoreArrayRead( coords, &ca ) );
  PetscFunctionReturn( PETSC_SUCCESS );
}