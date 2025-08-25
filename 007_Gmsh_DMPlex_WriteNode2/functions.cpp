#include "functions.h"

// .mshをDMPlexで読み込む 
PetscErrorCode read_gmsh( const char* mesh_path, DM &dm, PetscInt &dim )
{
  // メッシュを補完ありで読み込む（第4引数はinterpolate=1に対応）
  PetscCall( DMPlexCreateFromFile( PETSC_COMM_WORLD, mesh_path, NULL, PETSC_TRUE, &dm ) );
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
  PetscFunctionReturn( PETSC_SUCCESS );
}

// 座標を取得
PetscErrorCode get_coords( const DM& dm, const PetscInt dim, const PetscScalar *a_loc, const PetscScalar *a_glob, const bool debug )
{
  Vec coords_loc  = NULL;
  Vec coords_glob = NULL;
  
  // dm が持つ「座標ベクトル（ローカル）」を取り出して coords_loc に入れる
  PetscCall( DMGetCoordinatesLocal( dm, &coords_loc ) );
  // coords_locがあれば，生ポインタに格納
  if (coords_loc) PetscCall( VecGetArrayRead(coords_loc, &a_loc) );
  
  // dm が持つ「座標ベクトル（グローバル）」を取り出して　coords_glob に入れる
  PetscCall( DMGetCoordinates( dm, &coords_glob ) );
  // coords_globがあれば，生ポインタに格納
  if (coords_glob) PetscCall( VecGetArrayRead( coords_glob, &a_glob ) );

  //+++
  if( debug )
  {
    // a_loc と a_glob のアドレスを出力
    PetscPrintf( PETSC_COMM_WORLD, "coords_loc  = %p\n", coords_loc  );
    PetscPrintf( PETSC_COMM_WORLD, "coords_glob = %p\n", coords_glob );
    // a_loc の内容を出力
    if( coords_loc )
    {
      PetscInt num_loc;
      PetscCall( VecGetLocalSize( coords_loc, &num_loc ) );
      for( PetscInt i=0; i<num_loc; i=i+dim )
      {
        PetscPrintf( PETSC_COMM_WORLD, "a_loc[%5d]=%15.5e", i, (double)PetscRealPart( a_loc[i] ) );
        if( dim >= 2 ) PetscPrintf( PETSC_COMM_WORLD, "%15.5e", (double)PetscRealPart( a_loc[i+1] ) );
        if( dim >= 3 ) PetscPrintf( PETSC_COMM_WORLD, "%15.5e", (double)PetscRealPart( a_loc[i+2] ) );
        PetscPrintf( PETSC_COMM_WORLD, "\n" );
      }
    }
    PetscSection section;
    PetscInt ID_start, ID_end;
    PetscCall( DMGetCoordinateSection( dm, &section ) );
    // 全てのdmの ID の範囲を取得
    PetscCall( DMPlexGetChart( dm, &ID_start, &ID_end ) );
    PetscPrintf( PETSC_COMM_WORLD, "[ID_start, ID_end ) = [%5d,%5d )\n", ID_start, ID_end );
    for( PetscInt ID=ID_start; ID<ID_end; ID++ )
    {
      PetscInt dof, offset;
      PetscCall( PetscSectionGetDof(    section, ID, &dof    ) );
      PetscCall( PetscSectionGetOffset( section, ID, &offset ) );
      PetscPrintf( PETSC_COMM_WORLD, "ID=%5d offset=%5d dof=%5d\n", ID, offset, dof );
    }
  }
  //---

  if (coords_loc)  PetscCall(VecRestoreArrayRead(coords_loc,  &a_loc ) );
  if (coords_glob) PetscCall(VecRestoreArrayRead(coords_glob, &a_glob) );

  PetscFunctionReturn( PETSC_SUCCESS );
}

// 頂点のIDの範囲を取得
PetscErrorCode get_vertex_ID_range( const DM& dm, PetscInt& ID_start, PetscInt& ID_end, const bool debug )
{
  // DMPlex内部の頂点の最初のIDと最後のIDを取得
  PetscCall( DMPlexGetDepthStratum( dm, 0, &ID_start, &ID_end ) );
  if( debug )
  {
    PetscPrintf( PETSC_COMM_WORLD, "[vertex_start, vertex_end) = [%5d,%5d )\n", ID_start, ID_end );
  }
  PetscFunctionReturn( PETSC_SUCCESS );
}

// 面のIDの範囲を取得
PetscErrorCode get_face_ID_range( const DM& dm, PetscInt& ID_start, PetscInt& ID_end, const bool debug )
{
  // DMPlex内部の面の最初のIDと最後のIDを取得
  PetscCall( DMPlexGetHeightStratum( dm, 1, &ID_start, &ID_end ) );
  if( debug )
  {
    PetscPrintf( PETSC_COMM_WORLD, "[face_start,   face_end  ) = [%5d,%5d )\n", ID_start, ID_end );
  }
  PetscFunctionReturn( PETSC_SUCCESS );
}

// セルのIDの範囲を取得
PetscErrorCode get_cell_ID_range( const DM& dm, PetscInt& ID_start, PetscInt& ID_end, const bool debug )
{
  // DMPlex内部のセルの最初のIDと最後のIDを取得
  PetscCall( DMPlexGetHeightStratum( dm, 0, &ID_start, &ID_end ) );
  if( debug )
  {
    PetscPrintf( PETSC_COMM_WORLD, "[cell_start,   cell_end  ) = [%5d,%5d )\n", ID_start, ID_end );
  }
  PetscFunctionReturn( PETSC_SUCCESS );
}

// 頂点数を取得
PetscErrorCode get_num_vertex( const DM& dm, PetscInt& num, const bool debug )
{
  PetscInt start, end;
  get_vertex_ID_range( dm, start, end );
  num = end - start;
  if( debug )
  {
    PetscPrintf( PETSC_COMM_WORLD, "num_vertex   = %5d\n", num );
  }
  PetscFunctionReturn( PETSC_SUCCESS );
}

// 面数を取得
PetscErrorCode get_num_face( const DM& dm, PetscInt& num, const bool debug )
{
  PetscInt start, end;
  get_face_ID_range( dm, start, end );
  num = end - start;
  if( debug )
  {
    PetscPrintf( PETSC_COMM_WORLD, "num_face     = %5d\n", num );
  }
  PetscFunctionReturn( PETSC_SUCCESS );
}

// セル数を取得
PetscErrorCode get_num_cell( const DM& dm, PetscInt& num, const bool debug )
{
  PetscInt start, end;
  get_cell_ID_range( dm, start, end );
  num = end - start;
  if( debug )
  {
    PetscPrintf( PETSC_COMM_WORLD, "num_cell     = %5d\n", num );
  }
  PetscFunctionReturn( PETSC_SUCCESS );
}

// 境界数を取得
PetscErrorCode get_num_boundary( const DM& dm, PetscInt& num, const bool debug )
{
  PetscInt start, end;
  get_face_ID_range( dm, start, end );
  num = 0;
  for( PetscInt f=start; f<end; f++ )
  {
    PetscInt support_size;
    PetscCall( DMPlexGetSupportSize( dm, f, &support_size) );
    if( support_size == 1 ) num++;
  }
  if( debug )
  {
    PetscPrintf( PETSC_COMM_WORLD, "num_boundary = %5d\n", num );
  }
  PetscFunctionReturn( PETSC_SUCCESS );
}

// ラベル（GmshのPhysical Group）を取得
PetscErrorCode get_label_name( const int rank, const DM& dm, std::vector<std::string> &label_names, const bool debug )
{
  PetscInt num_labels;
  PetscCall( DMGetNumLabels( dm, &num_labels ) );
  for( PetscInt i=0; i<num_labels; i++ )
  {
    const char* name = NULL;
    PetscCall( DMGetLabelName( dm, i, &name ) );
    std::string sname(name);
    label_names.push_back( sname );
  }
  if( debug )
  {
    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, num_labels =%5d\n", rank, num_labels );
    for( int i=0; i<label_names.size(); i++ )
    {
      PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, label_name[%5d] = %s\n", rank, i, label_names[i].c_str() );
    }
  }
  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}

// あるラベルについて値（=Physical ID）ごとの要素数を取得
PetscErrorCode get_label_num( const int rank, const DM& dm, const std::string& label_name, PetscInt& num, const bool debug )
{
  // dmが持つラベルから label_name を取得
  DMLabel label;
  PetscCall( DMGetLabel( dm, label_name.c_str(), &label ) );
  if( label )
  {
    // このプロセスが保持している値（物理ID）の集合をvalueISに取得
    IS valueIS;
    PetscCall( DMLabelGetValueIS( label, &valueIS ) );
    if( valueIS )
    {
      // ローカルに取得された物理IDの数を取得
      PetscInt nvals;
      PetscCall( ISGetLocalSize( valueIS, &nvals ) );
      // 物理IDを取得
      const PetscInt *vals;
      PetscCall( ISGetIndices( valueIS, &vals ) );
      if( debug )
      {
        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, nvals = %d\n", rank, nvals );
      }
      for( PetscInt k=0; k<nvals; k++ )
      {
        PetscInt v = vals[k];
        IS IDs;
        const PetscInt *a_IDs;
        num = 0;
        // ラベル値 v を持つ全てのDMエンティティIDの集合を取得
        PetscCall( DMLabelGetStratumIS( label, v, &IDs ) );
        if( IDs )
        {
          // このランクが保持する該当エンティティ数
          PetscCall( ISGetLocalSize( IDs, &num ) );
          PetscCall( ISGetIndices( IDs, &a_IDs ) );
          if( debug )
          {
            PetscSynchronizedPrintf( PETSC_COMM_WORLD, "rank=%3d, %s value = %d : num = %d\n", rank, label_name.c_str(), v, num );
          }
          PetscCall( ISRestoreIndices( IDs, &a_IDs ) );
          PetscCall( ISDestroy( &IDs ) );
        }
      }
      PetscCall( ISRestoreIndices( valueIS, &vals ) );
      PetscCall( ISDestroy( &valueIS ) );
    }
  }

  PetscSynchronizedFlush( PETSC_COMM_WORLD, PETSC_STDOUT );
  PetscFunctionReturn( PETSC_SUCCESS );
}