#include "trian6.h"

// 初期化
void trian6::initialize( const int p, const std::vector<int>& nd_clos_ids, node_vec& nodes )
{
    this->pid = p;
    this->dim = 2;
    this->num_gp = 6;
    // クロージャ順 -> マトリクス計算用順に交換するための対応関係
    this->perm = { 5, 3, 4, 0, 1, 2 };
    // 節点数
    this->num_nods = nd_clos_ids.size();
    // この要素が持つ節点の pid
    node_pids.resize( num_nods );
    for( int i=0; i<num_nods; i++ )
    {
        node_pids[perm[i]] = nd_clos_ids[i];
    }
    // この要素が持つ節点へのポインタ 
    nod.resize( num_nods );
    for( int i=0; i<num_nods; i++ )
    {
        nod[i] = nodes.pid_is(node_pids[i]);
    }
    // 積分点位置
    gp_pos.resize(4);
    gp_pos[0] = 0.816847572980459;
    gp_pos[1] = 0.091576213509771;
    gp_pos[2] = 0.108103018168070;
    gp_pos[3] = 0.445948490915965;
    // 積分点重み
    gp_wei.resize(2);
    gp_wei[0] = 0.109951743655322;
    gp_wei[1] = 0.223381589678011;
}

// pointID に対応する点の座標を計算する
bool trian6::get_coords( const DM& dm, const int p, std::vector<double>& xy )
{
    xy.resize( 3, 0.0 );

    PetscInt depth;
    DMPlexGetPointDepth( dm, p, &depth );
    if( depth == 2 ) return false;

    if( depth == 1 ) get_coords_face( dm, p, xy );
    if( depth == 0 ) get_coords_vertex( dm, p, xy );
    
    return true;
}

// pointID が depth=0 (頂点)のとき，座標を計算する 
void trian6::get_coords_vertex( const DM& dm, const int p, std::vector<double>& xy )
{
    PetscInt dim;
    DMGetCoordinateDim( dm, &dim );

    DM cdm = NULL;
    DMGetCoordinateDM( dm, &cdm );

    Vec coords_loc = NULL;
    DMGetCoordinatesLocal( dm, &coords_loc);
    
    PetscSection csec = NULL;
    DMGetCoordinateSection( dm, &csec );

    // 頂点 p を持つ辺 f を取得
    const PetscInt *supp = NULL;
    PetscInt nsupp = 0;
    DMPlexGetSupportSize(dm, p, &nsupp );
    DMPlexGetSupport( dm, p, &supp );
    const PetscInt f = supp[0];

    // 辺 f を持つセル c を取得
    supp = NULL;
    nsupp = 0;
    DMPlexGetSupportSize(dm, f, &nsupp );
    DMPlexGetSupport( dm, f, &supp );
    const PetscInt c = supp[0];

    // セル座標のクロシージャ
    PetscInt cdof = 0;
    PetscScalar *xc = NULL;
    DMPlexVecGetClosure(cdm, csec, coords_loc, c, &cdof, &xc );

    // セル内での辺 f のローカル番号 ie を求める
    const PetscInt *cone = NULL;
    PetscInt ncone = 0;
    const PetscInt *cori = NULL;
    DMPlexGetConeSize( dm, c, &ncone );
    DMPlexGetCone( dm, c, &cone );
    DMPlexGetConeOrientation( dm, c, &cori );
    PetscInt ie = -1;
    for( PetscInt j=0; j<ncone; j++ )
    {
      if( cone[j] == f )
      {
        ie = j;
        break;
      }
    }
    PetscInt ori = cori[ie];

    // 辺 f のどちらかの番号 iv を求める
    const PetscInt *vcone = NULL;
    PetscInt nvcone = 0;
    DMPlexGetConeSize( dm, f, &nvcone );
    DMPlexGetCone( dm, f, &vcone );
    PetscInt iv = -1;
    for( PetscInt j=0; j<nvcone; j++ )
    {
      if( vcone[j] == p )
      {
        if( ori >= 0 ) iv = j;
        if( ori <  0 ) iv = ( nvcone - 1 ) - j;
        break;
      }
    }

    // Tri-P2: closure中のノード -> { V0, E01, V1, E02, E12, V2 }
    static const PetscInt V_ofs[3] = { 0, 2, 5 }; // V0, V1, V2の位置
    static const PetscInt E_ofs[3] = { 1, 4, 3 }; // E01, E12, E20
    static const PetscInt edgeVerts[3][2] = { {0,1}, {1,2}, {2,0} };

    // 頂点 p の座標
    const PetscInt lv = edgeVerts[ie][iv];
    const PetscScalar *x_vtx = xc + dim * V_ofs[lv];

    xy[0] = x_vtx[0];
    xy[1] = x_vtx[1];
    xy[2] = 0.0;

    // 後片付け
    DMPlexVecRestoreClosure( cdm, csec, coords_loc, c, &cdof, &xc );
}

// pointID が depth=1 (辺)のとき，座標を計算する 
void trian6::get_coords_face( const DM& dm, const int p, std::vector<double>& xy )
{
    PetscInt dim;
    DMGetCoordinateDim( dm, &dim );

    DM cdm = NULL;
    DMGetCoordinateDM( dm, &cdm );

    Vec coords_loc = NULL;
    DMGetCoordinatesLocal( dm, &coords_loc);
    
    PetscSection csec = NULL;
    DMGetCoordinateSection( dm, &csec );

    // 辺 p を持つセル c を取得
    const PetscInt *supp = NULL;
    PetscInt nsupp = 0;
    DMPlexGetSupportSize( dm, p, &nsupp );
    DMPlexGetSupport( dm, p, &supp );
    const PetscInt c = supp[0];

    // セル座標のクロシージャ
    PetscInt cdof = 0;
    PetscScalar *xc = NULL;
    DMPlexVecGetClosure(cdm, csec, coords_loc, c, &cdof, &xc );

    // セル内での辺 f のローカル番号 ie を求める
    const PetscInt *cone = NULL;
    PetscInt ncone = 0;
    DMPlexGetConeSize( dm, c, &ncone );
    DMPlexGetCone( dm, c, &cone );
    PetscInt ie = -1;
    for( PetscInt j=0; j<ncone; j++ )
    {
      if( cone[j] == p )
      {
        ie = j;
        break;
      }
    }

    // Tri-P2: closure中のノード -> { V0, E01, V1, E02, E12, V2 }
    static const PetscInt E_ofs[3] = { 1, 4, 3 }; // E01, E12, E20

    // 辺 p の中点座標
    const PetscScalar *xmid = xc + dim*E_ofs[ie];

    xy[0] = xmid[0];
    xy[1] = xmid[1];
    xy[2] = 0.0;

    // 後片付け
    DMPlexVecRestoreClosure( cdm, csec, coords_loc, c, &cdof, &xc );
}


