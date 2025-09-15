#include "hexl27.h"

// 初期化
void hexl27::initialize( const int p, const std::vector<int>& nd_clos_ids, node_vec& nodes )
{
    this->pid = p;
    this->dim = 3;
    this->num_gp = 27;
    // クロージャ順 -> マトリクス計算用順に交換するための対応関係
    this->perm = { 26, 23, 25, 22, 24, 21, 16, 13, 10, 18,
                    9,  8, 20, 11, 15, 17, 12, 14, 19,  1,
                    2,  6,  5,  0,  4,  7,  3 };
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
    gp_pos[0] = -0.861136311594053;
    gp_pos[1] = -0.339981043584856;
    gp_pos[2] = -gp_pos[1];
    gp_pos[3] = -gp_pos[0];
    // 積分点重み
    gp_wei.resize(4);
    gp_wei[0] = 0.347854845137454;
    gp_wei[1] = 0.652145154862546;
    gp_wei[2] = gp_wei[1];
    gp_wei[3] = gp_wei[0];
}

// pointID に対応する点の座標を計算する
bool hexl27::get_coords( const DM& dm, const int p, std::vector<double>& xy )
{
    xy.resize( 3, 0.0 );
    
    PetscSection loc_sec;
    DMGetLocalSection( dm, &loc_sec );
    
    Vec coords_loc = NULL;
    DMGetCoordinatesLocal( dm, &coords_loc);
    
    PetscSection csec;
    DMGetCoordinateSection( dm, &csec );
    
    const PetscScalar *coords_loc_arr;
    VecGetArrayRead( coords_loc, &coords_loc_arr );

    PetscInt dof=0, off=0;
    PetscSectionGetDof( csec, p, &dof );
    PetscSectionGetOffset( csec, p, &off );

    for( int k=0; k<dof; k++ )
    {
        xy[k] = coords_loc_arr[off+k];
    }

    VecRestoreArrayRead( coords_loc, &coords_loc_arr );
    
    return true;
}

