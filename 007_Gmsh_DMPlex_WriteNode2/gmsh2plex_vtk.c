#include <petscdmplex.h>
#include <stdio.h>

static const char helpmsg[] =
  "Gmsh tri6 (order-2) -> rebuild P1 triangles in PETSc -> legacy VTK (.vtk ASCII)\n"
  "Options:\n"
  "  -mesh <mesh.msh>    (required)\n"
  "  -o    <out.vtk>     (default: mesh_linear.vtk)\n";

static inline PetscBool SharesVertex(const PetscInt ev[2], PetscInt x)
{ return (ev[0]==x || ev[1]==x) ? PETSC_TRUE : PETSC_FALSE; }

/* 三角形セルの角点3つを (A,B,C) で返す（辺→両端頂点から抽出） */
static PetscErrorCode GetTriangleCorners(DM dm, PetscInt cell, PetscInt tri[3])
{
  const PetscInt *edges=NULL, *e0v=NULL, *e1v=NULL, *e2v=NULL;
  PetscInt coneSizeC=0, coneSizeE=0;
  PetscInt A,B,C;

  PetscFunctionBegin;
  PetscCall(DMPlexGetConeSize(dm, cell, &coneSizeC));
  PetscCheck(coneSizeC==3, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
             "Cell %" PetscInt_FMT " is not triangle (coneSize=%d)", cell, (int)coneSizeC);
  PetscCall(DMPlexGetCone(dm, cell, &edges)); /* 3 edges */

  PetscCall(DMPlexGetConeSize(dm, edges[0], &coneSizeE)); PetscCheck(coneSizeE==2,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"edge0 not 2");
  PetscCall(DMPlexGetCone(dm, edges[0], &e0v));
  PetscCall(DMPlexGetConeSize(dm, edges[1], &coneSizeE)); PetscCheck(coneSizeE==2,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"edge1 not 2");
  PetscCall(DMPlexGetCone(dm, edges[1], &e1v));
  PetscCall(DMPlexGetConeSize(dm, edges[2], &coneSizeE)); PetscCheck(coneSizeE==2,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"edge2 not 2");
  PetscCall(DMPlexGetCone(dm, edges[2], &e2v));

  A = e0v[0]; B = e0v[1];
  if (!SharesVertex(e1v,B) && !SharesVertex(e2v,B)) { PetscInt t=A; A=B; B=t; }

  if (SharesVertex(e1v,B)) C = (e1v[0]==B) ? e1v[1] : e1v[0];
  else if (SharesVertex(e2v,B)) C = (e2v[0]==B) ? e2v[1] : e2v[0];
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Triangle edge connectivity inconsistent");

  tri[0]=A; tri[1]=B; tri[2]=C;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* legacy VTK (UNSTRUCTURED_GRID, TRIANGLE=5) を ASCII で出力 */
static PetscErrorCode WriteLegacyVTK(const char *path, PetscInt Nv, PetscInt Nc,
                                     PetscInt cdim, const PetscReal *coords,
                                     const PetscInt *cells)
{
  FILE *fp = fopen(path, "w");
  if (!fp) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open %s", path);

  /* ヘッダ */
  fprintf(fp, "# vtk DataFile Version 3.0\n");
  fprintf(fp, "P1 mesh from PETSc\n");
  fprintf(fp, "ASCII\n");
  fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");

  /* 点座標 */
  fprintf(fp, "POINTS %" PetscInt_FMT " double\n", Nv);
  for (PetscInt i=0;i<Nv;i++) {
    double x=0,y=0,z=0;
    if (cdim>=1) x = (double)coords[cdim*i + 0];
    if (cdim>=2) y = (double)coords[cdim*i + 1];
    if (cdim>=3) z = (double)coords[cdim*i + 2];
    fprintf(fp, "%.17g %.17g %.17g\n", x,y,z);
  }

  /* セル（各行: 3 idx0 idx1 idx2） */
  fprintf(fp, "CELLS %" PetscInt_FMT " %" PetscInt_FMT "\n", Nc, Nc*(3+1));
  for (PetscInt c=0;c<Nc;c++) {
    fprintf(fp, "3 %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",
            cells[3*c+0], cells[3*c+1], cells[3*c+2]);
  }

  /* セルタイプ（VTK_TRIANGLE=5） */
  fprintf(fp, "CELL_TYPES %" PetscInt_FMT "\n", Nc);
  for (PetscInt c=0;c<Nc;c++) fprintf(fp, "5\n");

  fclose(fp);
  return 0;
}

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc,&argv,NULL,helpmsg));

  PetscMPIInt rank;
  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

  // ======= 実行時のオプションで，mshのパス，vtkのパスを指定する =============================================
  // 実行時に -mesh <パス> または -mesh=<パス> が渡されていれば，その<パス>を mesh_path バッファにコピーして，hasMesh = PETSC_TRUE にする
  char mesh_path[PETSC_MAX_PATH_LEN] = "";
  PetscBool hasMesh = PETSC_FALSE;
  PetscCall( PetscOptionsGetString( NULL, NULL, "-mesh", mesh_path, sizeof(mesh_path), &hasMesh ) );
  //+++ check
  {
    PetscPrintf( PETSC_COMM_WORLD, "--- check ---\n" );
    PetscPrintf( PETSC_COMM_WORLD, "mesh_path = %s\n", mesh_path );
    PetscPrintf( PETSC_COMM_WORLD, "hasMesh = %d\n", (int)hasMesh );
  }
  //---
  
  // hasMesh が PETSC_FALSE のとき，PETSC_ERR_USER のエラーを発生させ，指定したメッセージを出力して処理を中断する
  PetscCheck( hasMesh, PETSC_COMM_WORLD, PETSC_ERR_USER, "-mesh <mesh.msh> を指定してください" );

  // 任意オプション -o の文字列値を取得して vtk_path バッファにコピーする．指定がなければ何もしない
  char vtk_path[PETSC_MAX_PATH_LEN] = "mesh_linear.vtk";
  PetscOptionsGetString( NULL, NULL, "-o", vtk_path, sizeof(vtk_path), NULL );
  //+++ check
  {
    PetscPrintf( PETSC_COMM_WORLD, "--- check ---\n" );
    PetscPrintf( PETSC_COMM_WORLD, "vtk_path = %s\n", vtk_path );
  }
  //---

  //=== .mshをDMPlexで読み込む ==============================================================================
  DM dm = NULL; // 解析用DM
  PetscInt dim; // 次元
  // メッシュを補完ありで読み込む（第4引数はinterpolate=1に対応）
  PetscCall( DMPlexCreateFromFile( PETSC_COMM_WORLD, mesh_path, NULL, PETSC_TRUE, &dm ) );
  printf( "%s[%d]\n", __FUNCTION__, __LINE__ );
  // コマンドラインオプションから設定
  PetscCall( DMSetFromOptions(dm) );
  // 次元の取得
  PetscCall( DMGetDimension( dm, &dim ) );
  // 次元が2なら何もしない．そうでなければメッセージを出力して処理を中断 
  PetscCheck( dim==2, PETSC_COMM_WORLD, PETSC_ERR_SUP, "このサンプルは2D専用です");
  //+++
  {
    PetscPrintf( PETSC_COMM_WORLD, "--- check ---\n" );
    PetscPrintf( PETSC_COMM_WORLD, "dim = %d\n", (int)dim );
  }
  //---

  //=== 座標（ローカル/グローバル）を取得 ==============================================================================
  DM cdm = NULL; // 座標用DM
  Vec coordsLocal = NULL;  //座標ベクトル（ローカル）
  Vec coordsGlobal = NULL; //座標ベクトル（グローバル）
  const PetscScalar *aLoc  = NULL;
  const PetscScalar *aGlob = NULL;
  // 座標用DMを取得
  PetscCall( DMGetCoordinateDM( dm, &cdm ) );
  // 座標用DMの取得に失敗したら（cdm==NULL）メッセージを出力して処理を中断
  PetscCheck( cdm!=NULL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Coordinate DM (cdm) is NULL");
  // dm が持つ「座標ベクトル（ローカル）」を取り出して coordsLocal に入れる
  PetscCall( DMGetCoordinatesLocal( dm, &coordsLocal ) );
  if (!coordsLocal) {
    PetscCall(DMLocalizeCoordinates(dm));
    PetscCall(DMGetCoordinatesLocal(dm, &coordsLocal));
  }
  // coordsLocalがあれば，生ポインタに格納
  if (coordsLocal) PetscCall( VecGetArrayRead(coordsLocal, &aLoc) );
  // dm が持つ「座標ベクトル（グローバル）」を取り出して　coordsGlobal に入れる
  PetscCall( DMGetCoordinates( dm, &coordsGlobal ) );
  // coordsGlobalがあれば，生ポインタに格納
  if (coordsGlobal) PetscCall( VecGetArrayRead( coordsGlobal, &aGlob ) );
  //+++
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, coordsLocal=%p\n", rank, coordsLocal );
  PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, coordsGlobal=%p\n", rank, coordsGlobal );
  if( coordsLocal )
  {
    PetscInt numLoc;
    PetscCall( VecGetLocalSize( coordsLocal, &numLoc ) );
    for( PetscInt i=0; i<numLoc; i=i+2 )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "aLoc[%d]=%15.5e,%15.5e\n", i, (double)PetscRealPart(aLoc[i]), (double)PetscRealPart(aLoc[i+1]) );
    }
  }
  //---
  //test
  {
    PetscSection section;
    PetscInt pStart, pEnd;
    PetscCall( DMGetCoordinateSection( dm, &section) );
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, [pStart,  pEnd ) = [%5d,%5d )\n", rank, pStart, pEnd );
    for( PetscInt p=pStart; p<pEnd; p++ )
    {
      PetscInt dof;
      PetscCall( PetscSectionGetDof(section, p, &dof) );
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, p=%5d dof=%5d\n", rank, p, dof );
    }
  }

  /*
  //===
  PetscInt cellStart, cellEnd, vtxStart, vtxEnd; // DMPlex内部のセルと頂点の最初のIDと最後のID
  // DMPlex内部のセルの最初のIDと最後のIDを取得
  PetscCall( DMPlexGetHeightStratum( dm, 0, &cellStart, &cellEnd ) );
  // DMPlex内部の頂点の最初のIDと最後のIDを取得
  PetscCall( DMPlexGetDepthStratum( dm, 0, &vtxStart, &vtxEnd ) );
  //+++
  {
    // 頂点の情報を出力
    IS vtxIS;
    const PetscInt *gvtxIDs;
    PetscCall( DMPlexGetVertexNumbering( dm, &vtxIS ) );
    PetscCall( ISGetIndices( vtxIS, &gvtxIDs ) );
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, [vtxStart,  vtxEnd ) = [%5d,%5d )\n", rank, vtxStart,  vtxEnd );
    for( PetscInt pID=vtxStart; pID<vtxEnd; pID++ )
    {
      PetscInt gvtxID = gvtxIDs[pID - vtxStart];
      PetscBool own = PETSC_TRUE;
      if( gvtxID < 0 )
      {
        gvtxID = -( gvtxID + 1 );
        own = PETSC_FALSE;
      }
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%3d, %3d, %3d, %3d,", rank, pID, gvtxID, (int)own );

      const PetscScalar *xyz = NULL; //座標
      PetscCall( DMPlexPointLocalRead( cdm, pID, aLoc, &xyz ) );
      if( !xyz ) PetscCall( DMPlexPointGlobalRead( cdm, pID, aGlob, &xyz ) );
      if( !xyz ) continue;
      if( dim >= 1 ) PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", (double)xyz[0] );
      if( dim >= 2 ) PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", (double)xyz[1] );
      if( dim >= 3 ) PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", (double)xyz[2] );
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
    }
  }
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, [cellStart, cellEnd) = [%5d,%5d )\n", rank, cellStart, cellEnd );
  }
  //---
  */


  //+++
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  //---

  if (coordsLocal) PetscCall(VecRestoreArrayRead(coordsLocal, &aLoc));
  if (coordsGlobal) PetscCall(VecRestoreArrayRead(coordsGlobal, &aGlob));

  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*
  PetscInt cStart, cEnd, vStart, vEnd;

  // セル・頂点の範囲（トポロジ） 
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));  // 2D: cells (tri)
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));   // vertices

  // 三角形セル数カウント（coneSize==3 を tri とみなす） 
  PetscInt Nc=0;
  for (PetscInt c=cStart;c<cEnd;++c) {
    PetscInt coneSize=0; PetscCall(DMPlexGetConeSize(dm,c,&coneSize));
    if (coneSize==3) ++Nc;
  }
  PetscCheck(Nc>0,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"三角形セルが見つかりません");

  // 使う頂点にだけ連番を振る 
  PetscInt pStart,pEnd;
  PetscCall(DMPlexGetChart(dm,&pStart,&pEnd));
  PetscInt *vmap; PetscCall(PetscMalloc1(pEnd-pStart,&vmap));
  for (PetscInt p=0;p<pEnd-pStart;++p) vmap[p] = -1;

  PetscInt *cellsP1; PetscCall(PetscMalloc1(3*Nc,&cellsP1));
  PetscInt Nv=0, cntCell=0;
  for (PetscInt c=cStart;c<cEnd;++c) {
    PetscInt coneSize=0; PetscCall(DMPlexGetConeSize(dm,c,&coneSize));
    if (coneSize!=3) continue;

    PetscInt tri[3]; PetscCall(GetTriangleCorners(dm,c,tri));
    for (int k=0;k<3;++k) {
      PetscInt vp=tri[k];
      PetscCheck(vp>=vStart && vp<vEnd, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                 "Cell %" PetscInt_FMT " corner %" PetscInt_FMT " not a vertex", c, vp);
      PetscInt idx=vmap[vp-pStart];
      if (idx<0) { vmap[vp-pStart]=idx=Nv++; }
      cellsP1[3*cntCell + k] = idx;
    }
    ++cntCell;
  }
  PetscCheck(cntCell==Nc,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Cell count mismatch");

  // 座標配列作成（Coordinate DM に対して Local→Global フォールバックで読む） 
  PetscInt cdim=0; PetscCall(DMGetCoordinateDim(dm,&cdim));
  PetscReal *coords; PetscCall(PetscMalloc1((size_t)cdim*(size_t)Nv,&coords));

  for (PetscInt vp=vStart; vp<vEnd; ++vp) {
    PetscInt idx=vmap[vp-pStart];
    if (idx<0) continue; // 未使用頂点は飛ばす

    // まず Local を試す 
    const PetscScalar *xyz = NULL;
    if (aLoc) PetscCall(DMPlexPointLocalRead(cdm, vp, aLoc, &xyz));

    // 無ければ Global を試す 
    if (!xyz && aGlob) PetscCall(DMPlexPointGlobalRead(cdm, vp, aGlob, &xyz));

    PetscCheck(xyz!=NULL, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
               "No coordinates for vertex %" PetscInt_FMT, vp);

    for (PetscInt d=0; d<cdim; ++d) coords[cdim*idx + d] = (PetscReal)PetscRealPart(xyz[d]);
  }

  if (coordsLocal) PetscCall(VecRestoreArrayRead(coordsLocal, &aLoc));
  if (coordsGlobal) PetscCall(VecRestoreArrayRead(coordsGlobal, &aGlob));

  // 旧VTKで出力（PETScの Viewer は使わない） 
  PetscCall(WriteLegacyVTK(outvtk, Nv, Nc, cdim, coords, cellsP1));

  // 後片付け
  PetscCall(PetscFree(coords));
  PetscCall(PetscFree(cellsP1));
  PetscCall(PetscFree(vmap));
  PetscCall(DMDestroy(&dm));
*/
