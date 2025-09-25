#include "hexl27.h"

// 初期化
void hexl27::initialize( const int p, const std::vector<int>& nd_clos_ids, node_vec& nodes )
{
    this->pid = p;
    this->dim = 3;
    this->num_gp = 27;
    // クロージャ順 -> マトリクス計算用順に交換するための対応関係
    this->perms = { 26, 23, 25, 22, 24, 21, 16, 13, 10, 18,
                    9,  8, 20, 11, 15, 17, 12, 14, 19,  1,
                    2,  6,  5,  0,  4,  7,  3 };
    this->permw = { 1, 2, 6, 5, 0, 4, 7, 3 };
    // 節点数
    this->num_nods = nd_clos_ids.size();
    this->num_nodw = 8;
    // この要素が持つ節点の pid
    node_pids.resize( num_nods );
    for( int i=0; i<num_nods; i++ )
    {
        node_pids[perms[i]] = nd_clos_ids[i];
    }
    // この要素が持つ節点へのポインタ 
    nod.resize( num_nods );
    for( int i=0; i<num_nods; i++ )
    {
        nod[i] = nodes.pid_is(node_pids[i]);
    }
    // 積分点位置
    gp_pos.resize(4);
    gp_pos[0] = -0.774596669241483;
    gp_pos[1] = 0.0;
    gp_pos[2] = -gp_pos[0];
    // 積分点重み
    gp_wei.resize(4);
    gp_wei[0] = 0.555555555555556;
    gp_wei[1] = 0.888888888888889;
    gp_wei[2] = gp_wei[0];
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

// xyeを取得 (27x3)
std::array<double,81> hexl27::get_xye() const
{
    std::array<double,81> xye;
    for( int i=0; i<num_nods; i++ )
    {
        for( int j=0; j<dim; j++ ) xye[i*dim+j] = this->nod[i]->xy[j];
    }
    return xye;
}

// xye_wを取得 (8x3)
std::array<double,24> hexl27::get_xye_w() const
{
    std::array<double,24> xye_w;
    for( int i=0; i<num_nodw; i++ )
    {
        for( int j=0; j<dim; j++ ) xye_w[i*dim+j] = this->nod[i]->xy[j];
    }
    return xye_w;
}

// dNdrを計算 (27x3)
std::array<double,81> hexl27::dNdr_at( const int ng ) const
{
    std::array<double,81> dNdr;
    int ig = static_cast<int>( ng/3/3 );
    int jg = static_cast<int>( (ng-ig*3*3)/3 );
    int kg = ng - ig*3*3 - jg*3;
    double r1 = gp_pos[ig];
    double r2 = gp_pos[jg];
    double r3 = gp_pos[kg];
    std::array<double,3> N3r1, N3r2, N3r3;
    N3r1[0] = -0.5*r1*(1.0-r1);
    N3r1[1] =  1.0 - r1*r1;
    N3r1[2] =  0.5*r1*(1.0+r1);
    N3r2[0] = -0.5*r2*(1.0-r2);
    N3r2[1] =  1.0 - r2*r2;
    N3r2[2] =  0.5*r2*(1.0+r2);
    N3r3[0] = -0.5*r3*(1.0-r3);
    N3r3[1] =  1.0 - r3*r3;
    N3r3[2] =  0.5*r3*(1.0+r3);
    std::array<double,3> dN3dr1, dN3dr2, dN3dr3;
    dN3dr1[0] = r1 - 0.5;
    dN3dr1[1] = -2.0 * r1;
    dN3dr1[2] = r1 + 0.5;
    dN3dr2[0] = r2 - 0.5;
    dN3dr2[1] = -2.0 * r2;
    dN3dr2[2] = r2 + 0.5;
    dN3dr3[0] = r3 - 0.5;
    dN3dr3[1] = -2.0 * r3;
    dN3dr3[2] = r3 + 0.5;

    dNdr[ 0*3 + 0 ] = dN3dr1[0] * N3r2[0] * N3r3[0];
	dNdr[ 1*3 + 0 ] = dN3dr1[2] * N3r2[0] * N3r3[0];
	dNdr[ 2*3 + 0 ] = dN3dr1[2] * N3r2[2] * N3r3[0];
	dNdr[ 3*3 + 0 ] = dN3dr1[0] * N3r2[2] * N3r3[0];
	dNdr[ 4*3 + 0 ] = dN3dr1[0] * N3r2[0] * N3r3[2];
	dNdr[ 5*3 + 0 ] = dN3dr1[2] * N3r2[0] * N3r3[2];
	dNdr[ 6*3 + 0 ] = dN3dr1[2] * N3r2[2] * N3r3[2];
	dNdr[ 7*3 + 0 ] = dN3dr1[0] * N3r2[2] * N3r3[2];
	dNdr[ 8*3 + 0 ] = dN3dr1[0] * N3r2[0] * N3r3[1];
	dNdr[ 9*3 + 0 ] = dN3dr1[2] * N3r2[0] * N3r3[1];
	dNdr[10*3 + 0 ] = dN3dr1[2] * N3r2[2] * N3r3[1];
	dNdr[11*3 + 0 ] = dN3dr1[0] * N3r2[2] * N3r3[1];
	dNdr[12*3 + 0 ] = dN3dr1[1] * N3r2[0] * N3r3[0];
	dNdr[13*3 + 0 ] = dN3dr1[2] * N3r2[1] * N3r3[0];
	dNdr[14*3 + 0 ] = dN3dr1[1] * N3r2[2] * N3r3[0];
	dNdr[15*3 + 0 ] = dN3dr1[0] * N3r2[1] * N3r3[0];
	dNdr[16*3 + 0 ] = dN3dr1[1] * N3r2[1] * N3r3[0];
	dNdr[17*3 + 0 ] = dN3dr1[1] * N3r2[0] * N3r3[2];
	dNdr[18*3 + 0 ] = dN3dr1[2] * N3r2[1] * N3r3[2];
	dNdr[19*3 + 0 ] = dN3dr1[1] * N3r2[2] * N3r3[2];
	dNdr[20*3 + 0 ] = dN3dr1[0] * N3r2[1] * N3r3[2];
	dNdr[21*3 + 0 ] = dN3dr1[1] * N3r2[1] * N3r3[2];
	dNdr[22*3 + 0 ] = dN3dr1[1] * N3r2[0] * N3r3[1];
	dNdr[23*3 + 0 ] = dN3dr1[2] * N3r2[1] * N3r3[1];
	dNdr[24*3 + 0 ] = dN3dr1[1] * N3r2[2] * N3r3[1];
	dNdr[25*3 + 0 ] = dN3dr1[0] * N3r2[1] * N3r3[1];
	dNdr[26*3 + 0 ] = dN3dr1[1] * N3r2[1] * N3r3[1];

    dNdr[ 0*3 + 1 ] = N3r1[0] * dN3dr2[0] * N3r3[0];
	dNdr[ 1*3 + 1 ] = N3r1[2] * dN3dr2[0] * N3r3[0];
	dNdr[ 2*3 + 1 ] = N3r1[2] * dN3dr2[2] * N3r3[0];
	dNdr[ 3*3 + 1 ] = N3r1[0] * dN3dr2[2] * N3r3[0];
	dNdr[ 4*3 + 1 ] = N3r1[0] * dN3dr2[0] * N3r3[2];
	dNdr[ 5*3 + 1 ] = N3r1[2] * dN3dr2[0] * N3r3[2];
	dNdr[ 6*3 + 1 ] = N3r1[2] * dN3dr2[2] * N3r3[2];
	dNdr[ 7*3 + 1 ] = N3r1[0] * dN3dr2[2] * N3r3[2];
	dNdr[ 8*3 + 1 ] = N3r1[0] * dN3dr2[0] * N3r3[1];
	dNdr[ 9*3 + 1 ] = N3r1[2] * dN3dr2[0] * N3r3[1];
	dNdr[10*3 + 1 ] = N3r1[2] * dN3dr2[2] * N3r3[1];
	dNdr[11*3 + 1 ] = N3r1[0] * dN3dr2[2] * N3r3[1];
	dNdr[12*3 + 1 ] = N3r1[1] * dN3dr2[0] * N3r3[0];
	dNdr[13*3 + 1 ] = N3r1[2] * dN3dr2[1] * N3r3[0];
	dNdr[14*3 + 1 ] = N3r1[1] * dN3dr2[2] * N3r3[0];
	dNdr[15*3 + 1 ] = N3r1[0] * dN3dr2[1] * N3r3[0];
	dNdr[16*3 + 1 ] = N3r1[1] * dN3dr2[1] * N3r3[0];
	dNdr[17*3 + 1 ] = N3r1[1] * dN3dr2[0] * N3r3[2];
	dNdr[18*3 + 1 ] = N3r1[2] * dN3dr2[1] * N3r3[2];
	dNdr[19*3 + 1 ] = N3r1[1] * dN3dr2[2] * N3r3[2];
	dNdr[20*3 + 1 ] = N3r1[0] * dN3dr2[1] * N3r3[2];
	dNdr[21*3 + 1 ] = N3r1[1] * dN3dr2[1] * N3r3[2];
	dNdr[22*3 + 1 ] = N3r1[1] * dN3dr2[0] * N3r3[1];
	dNdr[23*3 + 1 ] = N3r1[2] * dN3dr2[1] * N3r3[1];
	dNdr[24*3 + 1 ] = N3r1[1] * dN3dr2[2] * N3r3[1];
	dNdr[25*3 + 1 ] = N3r1[0] * dN3dr2[1] * N3r3[1];
	dNdr[26*3 + 1 ] = N3r1[1] * dN3dr2[1] * N3r3[1];

    dNdr[ 0*3 + 2 ] = N3r1[0] * N3r2[0] * dN3dr3[0];
	dNdr[ 1*3 + 2 ] = N3r1[2] * N3r2[0] * dN3dr3[0];
	dNdr[ 2*3 + 2 ] = N3r1[2] * N3r2[2] * dN3dr3[0];
	dNdr[ 3*3 + 2 ] = N3r1[0] * N3r2[2] * dN3dr3[0];
	dNdr[ 4*3 + 2 ] = N3r1[0] * N3r2[0] * dN3dr3[2];
	dNdr[ 5*3 + 2 ] = N3r1[2] * N3r2[0] * dN3dr3[2];
	dNdr[ 6*3 + 2 ] = N3r1[2] * N3r2[2] * dN3dr3[2];
	dNdr[ 7*3 + 2 ] = N3r1[0] * N3r2[2] * dN3dr3[2];
	dNdr[ 8*3 + 2 ] = N3r1[0] * N3r2[0] * dN3dr3[1];
	dNdr[ 9*3 + 2 ] = N3r1[2] * N3r2[0] * dN3dr3[1];
	dNdr[10*3 + 2 ] = N3r1[2] * N3r2[2] * dN3dr3[1];
	dNdr[11*3 + 2 ] = N3r1[0] * N3r2[2] * dN3dr3[1];
	dNdr[12*3 + 2 ] = N3r1[1] * N3r2[0] * dN3dr3[0];
	dNdr[13*3 + 2 ] = N3r1[2] * N3r2[1] * dN3dr3[0];
	dNdr[14*3 + 2 ] = N3r1[1] * N3r2[2] * dN3dr3[0];
	dNdr[15*3 + 2 ] = N3r1[0] * N3r2[1] * dN3dr3[0];
	dNdr[16*3 + 2 ] = N3r1[1] * N3r2[1] * dN3dr3[0];
	dNdr[17*3 + 2 ] = N3r1[1] * N3r2[0] * dN3dr3[2];
	dNdr[18*3 + 2 ] = N3r1[2] * N3r2[1] * dN3dr3[2];
	dNdr[19*3 + 2 ] = N3r1[1] * N3r2[2] * dN3dr3[2];
	dNdr[20*3 + 2 ] = N3r1[0] * N3r2[1] * dN3dr3[2];
	dNdr[21*3 + 2 ] = N3r1[1] * N3r2[1] * dN3dr3[2];
	dNdr[22*3 + 2 ] = N3r1[1] * N3r2[0] * dN3dr3[1];
	dNdr[23*3 + 2 ] = N3r1[2] * N3r2[1] * dN3dr3[1];
	dNdr[24*3 + 2 ] = N3r1[1] * N3r2[2] * dN3dr3[1];
	dNdr[25*3 + 2 ] = N3r1[0] * N3r2[1] * dN3dr3[1];
	dNdr[26*3 + 2 ] = N3r1[1] * N3r2[1] * dN3dr3[1];

    return dNdr;
}

// dNdr_w を計算 (8x3)
std::array<double,24> hexl27::dNdr_w_at( const int ng ) const
{
    std::array<double,24> dNdr_w;
    int ig = static_cast<int>( ng/3/3 );
    int jg = static_cast<int>( (ng-ig*3*3)/3 );
    int kg = ng - ig*3*3 - jg*3;
    double r1 = gp_pos[ig];
    double r2 = gp_pos[jg];
    double r3 = gp_pos[kg];
    double r1m = 1.0 - r1;
    double r2m = 1.0 - r2;
    double r3m = 1.0 - r3;
    double r1p = 1.0 + r1;
    double r2p = 1.0 + r2;
    double r3p = 1.0 + r3;
	////
	dNdr_w[0*3 + 0] = -0.125*r2m*r3m;
    dNdr_w[1*3 + 0] =  0.125*r2m*r3m;
    dNdr_w[2*3 + 0] =  0.125*r2p*r3m;
    dNdr_w[3*3 + 0] = -0.125*r2p*r3m;
    dNdr_w[4*3 + 0] = -0.125*r2m*r3p;
    dNdr_w[5*3 + 0] =  0.125*r2m*r3p;
    dNdr_w[6*3 + 0] =  0.125*r2p*r3p;
    dNdr_w[7*3 + 0] = -0.125*r2p*r3p;
    //
    dNdr_w[0*3 + 1] = -0.125*r1m*r3m;
    dNdr_w[1*3 + 1] = -0.125*r1p*r3m;
    dNdr_w[2*3 + 1] =  0.125*r1p*r3m;
    dNdr_w[3*3 + 1] =  0.125*r1m*r3m;
    dNdr_w[4*3 + 1] = -0.125*r1m*r3p;
    dNdr_w[5*3 + 1] = -0.125*r1p*r3p;
    dNdr_w[6*3 + 1] =  0.125*r1p*r3p;
    dNdr_w[7*3 + 1] =  0.125*r1m*r3p;
    //
    dNdr_w[0*3 + 2] = -0.125*r1m*r2m;
    dNdr_w[1*3 + 2] = -0.125*r1p*r2m;
    dNdr_w[2*3 + 2] = -0.125*r1p*r2p;
    dNdr_w[3*3 + 2] = -0.125*r1m*r2p;
    dNdr_w[4*3 + 2] =  0.125*r1m*r2m;
    dNdr_w[5*3 + 2] =  0.125*r1p*r2m;
    dNdr_w[6*3 + 2] =  0.125*r1p*r2p;
    dNdr_w[7*3 + 2] =  0.125*r1m*r2p;
    return dNdr_w;
}

// Jを計算 (3x3)
std::array<double,9> hexl27::J_at( const int ng ) const
{
    std::array<double,81> xye = get_xye();
    std::array<double,81> dNdr = dNdr_at( ng );
    std::array<double,9> J;
    for( int i=0; i<dim; i++ )
    {
        for( int j=0; j<dim; j++ )
        {
            J[i*dim+j] = 0.0;
            for( int k=0; k<num_nods; k++ )
            {
                J[i*dim+j] += dNdr[k*dim+i]*xye[k*dim+j];
            }
        }
    }
    return J;
}

// J_w を計算 (3x3)
std::array<double,9> hexl27::J_w_at( const int ng ) const
{
    std::array<double,24> xye_w = get_xye_w();
    std::array<double,24> dNdr_w = dNdr_w_at( ng );
    std::array<double,9> J_w;
    for( int i=0; i<dim; i++ )
    {
        for( int j=0; j<dim; j++ )
        {
            J_w[i*dim+j] = 0.0;
            for( int k=0; k<num_nodw; k++ )
            {
                J_w[i*dim+j] += dNdr_w[k*dim+i]*xye_w[k*dim+j];
            }
        }
    }
    return J_w;
}

// Jを計算 (3x3) ：dNdr, xyeから
std::array<double,9> hexl27::J( const std::array<double,81>& dNdr, const std::array<double,81>& xye ) const
{
    std::array<double,9> J;
    for( int i=0; i<dim; i++ )
    {
        for( int j=0; j<dim; j++ )
        {
            J[i*dim+j] = 0.0;
            for( int k=0; k<num_nods; k++ )
            {
                J[i*dim+j] += dNdr[k*dim+i]*xye[k*dim+j];
            }
        }
    }
    return J;
}

// J_w を計算 (3x3) ：dNdr, xyeから
std::array<double,9> hexl27::J_w( const std::array<double,24>& dNdr_w, const std::array<double,24>& xye_w ) const
{
    std::array<double,9> J_w;
    for( int i=0; i<dim; i++ )
    {
        for( int j=0; j<dim; j++ )
        {
            J_w[i*dim+j] = 0.0;
            for( int k=0; k<num_nodw; k++ )
            {
                J_w[i*dim+j] += dNdr_w[k*dim+i]*xye_w[k*dim+j];
            }
        }
    }
    return J_w;
}

// detJを計算
double hexl27::detJ_at( const int ng ) const
{
    std::array<double,9> J = J_at( ng );
    return J[0*dim+0]*J[1*dim+1]*J[2*dim+2] + J[0*dim+1]*J[1*dim+2]*J[2*dim+0] + J[0*dim+2]*J[1*dim+0]*J[2*dim+1]
          -J[0*dim+2]*J[1*dim+1]*J[2*dim+0] - J[0*dim+1]*J[1*dim+0]*J[2*dim+2] - J[0*dim+0]*J[1*dim+2]*J[2*dim+1];
}

// detJ_w を計算
double hexl27::detJ_w_at( const int ng ) const
{
    std::array<double,9> J_w = J_w_at( ng );
    return J_w[0*dim+0]*J_w[1*dim+1]*J_w[2*dim+2] + J_w[0*dim+1]*J_w[1*dim+2]*J_w[2*dim+0] + J_w[0*dim+2]*J_w[1*dim+0]*J_w[2*dim+1]
          -J_w[0*dim+2]*J_w[1*dim+1]*J_w[2*dim+0] - J_w[0*dim+1]*J_w[1*dim+0]*J_w[2*dim+2] - J_w[0*dim+0]*J_w[1*dim+2]*J_w[2*dim+1];
}

// detJを計算 : Jから
double hexl27::detJ( const std::array<double,9>& J ) const
{
    return J[0*dim+0]*J[1*dim+1]*J[2*dim+2] + J[0*dim+1]*J[1*dim+2]*J[2*dim+0] + J[0*dim+2]*J[1*dim+0]*J[2*dim+1]
          -J[0*dim+2]*J[1*dim+1]*J[2*dim+0] - J[0*dim+1]*J[1*dim+0]*J[2*dim+2] - J[0*dim+0]*J[1*dim+2]*J[2*dim+1];
}

// J-T を計算 (3x3)
std::array<double,9> hexl27::J_I_T_at( const int ng ) const
{
    std::array<double,9> J = J_at( ng );
    double detJ = detJ_at( ng );
    std::array<double,9> J_I_T;
    J_I_T[0*dim+0] = (J[1*dim+1]*J[2*dim+2] - J[1*dim+2]*J[2*dim+1])/detJ;
	J_I_T[1*dim+0] = (J[0*dim+2]*J[2*dim+1] - J[0*dim+1]*J[2*dim+2])/detJ;
	J_I_T[2*dim+0] = (J[0*dim+1]*J[1*dim+2] - J[0*dim+2]*J[1*dim+1])/detJ;
	J_I_T[0*dim+1] = (J[1*dim+2]*J[2*dim+0] - J[1*dim+0]*J[2*dim+2])/detJ;
	J_I_T[1*dim+1] = (J[0*dim+0]*J[2*dim+2] - J[0*dim+2]*J[2*dim+0])/detJ;
	J_I_T[2*dim+1] = (J[0*dim+2]*J[1*dim+0] - J[0*dim+0]*J[1*dim+2])/detJ;
	J_I_T[0*dim+2] = (J[1*dim+0]*J[2*dim+1] - J[1*dim+1]*J[2*dim+0])/detJ;
	J_I_T[1*dim+2] = (J[0*dim+1]*J[2*dim+0] - J[0*dim+0]*J[2*dim+1])/detJ;
	J_I_T[2*dim+2] = (J[0*dim+0]*J[1*dim+1] - J[0*dim+1]*J[1*dim+0])/detJ;
    return J_I_T;
}

// J_w-T を計算 (3x3)
std::array<double,9> hexl27::J_I_T_w_at( const int ng ) const
{
    std::array<double,9> J_w = J_w_at( ng );
    double detJ_w = detJ_w_at( ng );
    std::array<double,9> J_I_T_w;
    J_I_T_w[0*dim+0] = (J_w[1*dim+1]*J_w[2*dim+2] - J_w[1*dim+2]*J_w[2*dim+1])/detJ_w;
	J_I_T_w[1*dim+0] = (J_w[0*dim+2]*J_w[2*dim+1] - J_w[0*dim+1]*J_w[2*dim+2])/detJ_w;
	J_I_T_w[2*dim+0] = (J_w[0*dim+1]*J_w[1*dim+2] - J_w[0*dim+2]*J_w[1*dim+1])/detJ_w;
	J_I_T_w[0*dim+1] = (J_w[1*dim+2]*J_w[2*dim+0] - J_w[1*dim+0]*J_w[2*dim+2])/detJ_w;
	J_I_T_w[1*dim+1] = (J_w[0*dim+0]*J_w[2*dim+2] - J_w[0*dim+2]*J_w[2*dim+0])/detJ_w;
	J_I_T_w[2*dim+1] = (J_w[0*dim+2]*J_w[1*dim+0] - J_w[0*dim+0]*J_w[1*dim+2])/detJ_w;
	J_I_T_w[0*dim+2] = (J_w[1*dim+0]*J_w[2*dim+1] - J_w[1*dim+1]*J_w[2*dim+0])/detJ_w;
	J_I_T_w[1*dim+2] = (J_w[0*dim+1]*J_w[2*dim+0] - J_w[0*dim+0]*J_w[2*dim+1])/detJ_w;
	J_I_T_w[2*dim+2] = (J_w[0*dim+0]*J_w[1*dim+1] - J_w[0*dim+1]*J_w[1*dim+0])/detJ_w;
    return J_I_T_w;
}

// J-T を計算 (3x3) : J, detJから
std::array<double,9> hexl27::J_I_T( const std::array<double,9>& J, const double detJ ) const
{
    std::array<double,9> J_I_T;
    J_I_T[0*dim+0] = (J[1*dim+1]*J[2*dim+2] - J[1*dim+2]*J[2*dim+1])/detJ;
	J_I_T[1*dim+0] = (J[0*dim+2]*J[2*dim+1] - J[0*dim+1]*J[2*dim+2])/detJ;
	J_I_T[2*dim+0] = (J[0*dim+1]*J[1*dim+2] - J[0*dim+2]*J[1*dim+1])/detJ;
	J_I_T[0*dim+1] = (J[1*dim+2]*J[2*dim+0] - J[1*dim+0]*J[2*dim+2])/detJ;
	J_I_T[1*dim+1] = (J[0*dim+0]*J[2*dim+2] - J[0*dim+2]*J[2*dim+0])/detJ;
	J_I_T[2*dim+1] = (J[0*dim+2]*J[1*dim+0] - J[0*dim+0]*J[1*dim+2])/detJ;
	J_I_T[0*dim+2] = (J[1*dim+0]*J[2*dim+1] - J[1*dim+1]*J[2*dim+0])/detJ;
	J_I_T[1*dim+2] = (J[0*dim+1]*J[2*dim+0] - J[0*dim+0]*J[2*dim+1])/detJ;
	J_I_T[2*dim+2] = (J[0*dim+0]*J[1*dim+1] - J[0*dim+1]*J[1*dim+0])/detJ;
    return J_I_T;
}

// derivNを計算 (27x3)
std::array<double,81> hexl27::derivN_at( const int ng ) const
{
    std::array<double,81> dNdr = dNdr_at( ng );
    std::array<double,9> J_I_T = J_I_T_at( ng );
    std::array<double,81> derivN;
    for( int i=0; i<num_nods; i++ )
    {
        for( int j=0; j<dim; j++ )
        {
            derivN[i*dim+j] = 0.0;
            for( int k=0; k<dim; k++ )
            {
                derivN[i*dim+j] += dNdr[i*dim+k] * J_I_T[k*dim+j];
            }
        }
    }
    return derivN;
}

// derivN_w を計算 (8x3)
std::array<double,24> hexl27::derivN_w_at( const int ng ) const
{
    std::array<double,24> dNdr_w = dNdr_w_at( ng );
    std::array<double,9> J_I_T_w = J_I_T_w_at( ng );
    std::array<double,24> derivN_w;
    for( int i=0; i<num_nodw; i++ )
    {
        for( int j=0; j<dim; j++ )
        {
            derivN_w[i*dim+j] = 0.0;
            for( int k=0; k<dim; k++ )
            {
                derivN_w[i*dim+j] += dNdr_w[i*dim+k] * J_I_T_w[k*dim+j];
            }
        }
    }
    return derivN_w;
}

// derivNを計算 (27x3) : dNdr, J_I_Tから
std::array<double,81> hexl27::derivN( const std::array<double,81>& dNdr, const std::array<double,9>& J_I_T ) const
{
    std::array<double,81> derivN;
    for( int i=0; i<num_nods; i++ )
    {
        for( int j=0; j<dim; j++ )
        {
            derivN[i*dim+j] = 0.0;
            for( int k=0; k<dim; k++ )
            {
                derivN[i*dim+j] += dNdr[i*dim+k] * J_I_T[k*dim+j];
            }
        }
    }
    return derivN;
}

// derivN_w を計算 (27x3) : dNdr_w, J_I_T_w から
std::array<double,24> hexl27::derivN_w( const std::array<double,24>& dNdr_w, const std::array<double,9>& J_I_T_w ) const
{
    std::array<double,24> derivN_w;
    for( int i=0; i<num_nodw; i++ )
    {
        for( int j=0; j<dim; j++ )
        {
            derivN_w[i*dim+j] = 0.0;
            for( int k=0; k<dim; k++ )
            {
                derivN_w[i*dim+j] += dNdr_w[i*dim+k] * J_I_T_w[k*dim+j];
            }
        }
    }
    return derivN_w;
}

// 体積積分時に乗じる係数を計算
double hexl27::fac_at( const int ng ) const
{
    double detJ = detJ_at( ng );
    int ig = static_cast<int>( ng/3/3 );
    int jg = static_cast<int>( (ng-ig*3*3)/3 );
    int kg = ng - ig*3*3 - jg*3;
    return gp_wei[ig] * gp_wei[jg] * gp_wei[kg] * detJ;
}

// 体積積分時に乗じる係数を計算　： detJから
double hexl27::fac( const int ng, const double detJ ) const
{
    int ig = static_cast<int>( ng/3/3 );
    int jg = static_cast<int>( (ng-ig*3*3)/3 );
    int kg = ng - ig*3*3 - jg*3;
    return gp_wei[ig] * gp_wei[jg] * gp_wei[kg] * detJ;
}

// Bマトリクスの計算 (6x81)
std::array<double,486> hexl27::B_matrix_at( const int ng ) const
{
    std::array<double,81> derivN = derivN_at( ng );
    std::array<double,486> B;
    for( int i=0; i<486; i++ ) B[i] = 0.0;
    for( int i=0; i<num_nods; i++ )
    {
        B[0*81 + (3*i+0)] = -derivN[i*3+0];
        B[1*81 + (3*i+1)] = -derivN[i*3+1];
        B[2*81 + (3*i+2)] = -derivN[i*3+2];
        B[3*81 + (3*i+0)] = -derivN[i*3+1]*0.5;
        B[3*81 + (3*i+1)] = -derivN[i*3+0]*0.5;
        B[4*81 + (3*i+1)] = -derivN[i*3+2]*0.5;
        B[4*81 + (3*i+2)] = -derivN[i*3+1]*0.5;
        B[5*81 + (3*i+0)] = -derivN[i*3+2]*0.5;
        B[5*81 + (3*i+2)] = -derivN[i*3+0]*0.5;
    }
    return B;
}

// Bマトリクスの計算 (6x81)
std::array<double,486> hexl27::B_matrix( const std::array<double,81>& derivN ) const
{
    std::array<double,486> B;
    for( int i=0; i<486; i++ ) B[i] = 0.0;
    for( int i=0; i<num_nods; i++ )
    {
        B[0*81 + (3*i+0)] = -derivN[i*3+0];
        B[1*81 + (3*i+1)] = -derivN[i*3+1];
        B[2*81 + (3*i+2)] = -derivN[i*3+2];
        B[3*81 + (3*i+0)] = -derivN[i*3+1]*0.5;
        B[3*81 + (3*i+1)] = -derivN[i*3+0]*0.5;
        B[4*81 + (3*i+1)] = -derivN[i*3+2]*0.5;
        B[4*81 + (3*i+2)] = -derivN[i*3+1]*0.5;
        B[5*81 + (3*i+0)] = -derivN[i*3+2]*0.5;
        B[5*81 + (3*i+2)] = -derivN[i*3+0]*0.5;
    }
    return B;
}

// BVOLマトリクスの計算 (6x81)
std::array<double,486> hexl27::BVOL_matrix_at( const int ng ) const
{
    std::array<double,486> BVOL = B_matrix_at( ng );
    double fac = fac_at( ng );
    for( int i=0; i<486; i++ ) BVOL[i] *= fac;
    return BVOL;
}

// BVOLマトリクスの計算 (6x81)
std::array<double,486> hexl27::BVOL_matrix( const std::array<double,486>& B, const double fac ) const
{
    std::array<double,486> BVOL;
    for( int i=0; i<486; i++ ) BVOL[i] = B[i]*fac;
    return BVOL;
}

// Bvマトリクスの計算 (1x81)
std::array<double,81> hexl27::Bv_matrix_at( const int ng ) const
{
    std::array<double,81> derivN = derivN_at( ng );
    std::array<double,81> Bv;
    for( int i=0; i<81; i++ ) Bv[i] = 0.0;
    for( int i=0; i<num_nods; i++ )
    {
        Bv[i*dim+0] = -derivN[i*dim+0];
        Bv[i*dim+1] = -derivN[i*dim+1];
        Bv[i*dim+2] = -derivN[i*dim+2];
    }
    return Bv;
}

// Bvマトリクスの計算 (1x81)
std::array<double,81> hexl27::Bv_matrix( const std::array<double,81>& derivN ) const
{
    std::array<double,81> Bv;
    for( int i=0; i<81; i++ ) Bv[i] = 0.0;
    for( int i=0; i<num_nods; i++ )
    {
        Bv[i*dim+0] = -derivN[i*dim+0];
        Bv[i*dim+1] = -derivN[i*dim+1];
        Bv[i*dim+2] = -derivN[i*dim+2];
    }
    return Bv;
}

// Nhマトリクスの計算 (1x8)
std::array<double,8> hexl27::Nh_matrix_at( const int ng ) const
{
    std::array<double,8> Nh;
    int ig = static_cast<int>( ng/3/3 );
    int jg = static_cast<int>( (ng-ig*3*3)/3 );
    int kg = ng - ig*3*3 - jg*3;
    double r1 = gp_pos[ig];
    double r2 = gp_pos[jg];
    double r3 = gp_pos[kg];
    double r1m = 1.0 - r1;
    double r2m = 1.0 - r2;
	double r3m = 1.0 - r3;
	double r1p = 1.0 + r1;
    double r2p = 1.0 + r2;
	double r3p = 1.0 + r3;
	Nh[0] = 0.125*r1m*r2m*r3m;
	Nh[1] = 0.125*r1p*r2m*r3m;
	Nh[2] = 0.125*r1p*r2p*r3m;
    Nh[3] = 0.125*r1m*r2p*r3m;
    Nh[4] = 0.125*r1m*r2m*r3p;
    Nh[5] = 0.125*r1p*r2m*r3p;
    Nh[6] = 0.125*r1p*r2p*r3p;
	Nh[7] = 0.125*r1m*r2p*r3p;
    return Nh;
}

// NhVOLマトリクスの計算 (1x8)
std::array<double,8> hexl27::NhVOL_matrix_at( const int ng ) const
{
    std::array<double,8> NhVOL = this->Nh_matrix_at( ng );
    double fac = this->fac_at( ng );
    for( int i=0; i<8; i++ ) NhVOL[i] *= fac;
    return NhVOL;
}

// Bhマトリクスの計算 (3x8)
std::array<double,24> hexl27::Bh_matrix_at( const int ng ) const
{
    std::array<double,24> derivN_w = this->derivN_w_at( ng );
    std::array<double,24> Bh;
    for( int i=0; i<24; i++ ) Bh[i] = 0.0;
    for( int i=0; i<num_nodw; i++ )
    {
        for( int j=0; j<dim; j++ )
        {
            Bh[j*8+i] = derivN_w[i*dim+j];
        }
    }
    return Bh;
}

// Bhマトリクスの計算 (3x8)
std::array<double,24> hexl27::Bh_matrix( const std::array<double,24>& derivN_w ) const
{
    std::array<double,24> Bh;
    for( int i=0; i<24; i++ ) Bh[i] = 0.0;
    for( int i=0; i<num_nodw; i++ )
    {
        for( int j=0; j<dim; j++ )
        {
            Bh[j*8+i] = derivN_w[i*dim+j];
        }
    }
    return Bh;
}

// BhVOLマトリクスの計算 (3x8)
std::array<double,24> hexl27::BhVOL_matrix_at( const int ng ) const
{
    std::array<double,24> BhVOL = this->Bh_matrix_at( ng );
    double fac = this->fac_at( ng );
    for( int i=0; i<24; i++ ) BhVOL[i] *= fac;
    return BhVOL;
}

// BhVOLマトリクスの計算 (3x8)
std::array<double,24> hexl27::BhVOL_matrix( const std::array<double,24>& Bh, const double fac ) const
{
    std::array<double,24> BhVOL;
    for( int i=0; i<24; i++ ) BhVOL[i] = Bh[i]*fac;
    return BhVOL;
}

// Kuuマトリクスの計算 (81x81)
std::array<double,6561> hexl27::Kuu_matrix( const std::vector<double>& D ) const
{
    std::array<double,6561> Kuu;
    for( int i=0; i<6561; i++ ) Kuu[i] = 0.0;
    for( int ng=0; ng<num_gp; ng++ )
    {
        //std::array<double,486> B = B_matrix_at( ng );
        //std::array<double,486> BVOL = BVOL_matrix_at( ng );
        std::array<double,81> xye = this->get_xye();
        std::array<double,81> dNdr = this->dNdr_at( ng );
        std::array<double,9>  J    = this->J( dNdr, xye );
        double detJ = this->detJ( J );
        std::array<double,9> J_I_T = this->J_I_T( J, detJ );
        std::array<double,81> derivN = this->derivN( dNdr, J_I_T );
        double fac = this->fac( ng, detJ );
        std::array<double,486> B = this->B_matrix( derivN );
        std::array<double,486> BVOL = this->BVOL_matrix( B, fac );

        std::array<double,486> BTD;
        for( int i=0; i<81; i++ )
        {
            for( int j=0; j<6; j++ )
            {
                BTD[i*6+j] = 0.0;
                for( int k=0; k<6; k++ )
                {
                    if( k < 3  ) BTD[i*6+j] += B[k*81+i]*D[k*6+j];
                    else         BTD[i*6+j] += 2.0*B[k*81+i]*D[k*6+j];
                }
            }
        }

        for( int i=0; i<81; i++ )
        {
            for( int j=0; j<81; j++ )
            {
                for( int k=0; k<6; k++ )
                {
                    if( k <  3 ) Kuu[i*81+j] += BTD[i*6+k]*BVOL[k*81+j];
                    else         Kuu[i*81+j] += 2.0*BTD[i*6+k]*BVOL[k*81+j];
                }
            }
        }
    }
    return Kuu;
}

void hexl27::permutate_Kuu_matrix( std::array<double,6561>& Kuu ) const
{
    std::array<double,6561> Kuu_tmp;
    for( int i=0; i<6561; i++ ) Kuu_tmp[i] = Kuu[i];

    for( int i=0; i<num_nods; i++ )
    {
        int io = perms[i];
        for( int k=0; k<dim; k++ )
        {
            int ik = i * dim + k;
            int iok = io * dim + k;
            for( int j=0; j<num_nods; j++ )
            {
                int jo = perms[j];
                for( int l=0; l<dim; l++ )
                {
                    int jl = j * dim + l;
                    int jol = jo * dim + l;
                    Kuu[ik*81+jl] = Kuu_tmp[iok*81+jol];
                }
            }
        }
    }
}

void hexl27::Kuu_matrix( const std::vector<double>& D, std::vector<double>& Kuu ) const
{
    for( int i=0; i<6561; i++ ) Kuu[i] = 0.0;
    for( int ng=0; ng<num_gp; ng++ )
    {
        std::array<double,81> xye = this->get_xye();
        std::array<double,81> dNdr = this->dNdr_at( ng );
        std::array<double,9>  J    = this->J( dNdr, xye );
        double detJ = this->detJ( J );
        std::array<double,9> J_I_T = this->J_I_T( J, detJ );
        std::array<double,81> derivN = this->derivN( dNdr, J_I_T );
        double fac = this->fac( ng, detJ );
        std::array<double,486> B = this->B_matrix( derivN );
        std::array<double,486> BVOL = this->BVOL_matrix( B, fac );

        std::array<double,486> BTD;
        for( int i=0; i<81; i++ )
        {
            for( int j=0; j<6; j++ )
            {
                BTD[i*6+j] = 0.0;
                for( int k=0; k<6; k++ )
                {
                    if( k < 3  ) BTD[i*6+j] += B[k*81+i]*D[k*6+j];
                    else         BTD[i*6+j] += 2.0*B[k*81+i]*D[k*6+j];
                }
            }
        }

        for( int i=0; i<81; i++ )
        {
            for( int j=0; j<81; j++ )
            {
                for( int k=0; k<6; k++ )
                {
                    if( k <  3 ) Kuu[i*81+j] += BTD[i*6+k]*BVOL[k*81+j];
                    else         Kuu[i*81+j] += 2.0*BTD[i*6+k]*BVOL[k*81+j];
                }
            }
        }
    }
}

// Kuhマトリクスの計算 (81x8)
void hexl27::Kuh_matrix( std::vector<double>& Kuh ) const
{
    for( int i=0; i<81*8; i++ ) Kuh[i] = 0.0;
    for( int ng=0; ng<num_gp; ng++ )
    {
        std::array<double,81> xye = this->get_xye();
        std::array<double,81> dNdr = this->dNdr_at( ng );
        std::array<double,9>  J    = this->J( dNdr, xye );
        double detJ = this->detJ( J );
        std::array<double,9> J_I_T = this->J_I_T( J, detJ );
        std::array<double,81> derivN = this->derivN( dNdr, J_I_T );
        std::array<double,81> Bv = this->Bv_matrix( derivN ); // 1x81
        std::array<double,8> NhVOL = this->NhVOL_matrix_at( ng ); // 1x8
        for( int i=0; i<81; i++ )
        {
            for( int j=0; j<8; j++ )
            {
                Kuh[i*8+j] += Bv[i]*NhVOL[j];
            }
        }
    }
}

// Khuマトリクスの計算 (8x81)
void hexl27::Khu_matrix( std::vector<double>& Khu ) const
{
    for( int i=0; i<81*8; i++ ) Khu[i] = 0.0;
    for( int ng=0; ng<num_gp; ng++ )
    {
        std::array<double,81> xye = this->get_xye();
        std::array<double,81> dNdr = this->dNdr_at( ng );
        std::array<double,9>  J    = this->J( dNdr, xye );
        double detJ = this->detJ( J );
        std::array<double,9> J_I_T = this->J_I_T( J, detJ );
        std::array<double,81> derivN = this->derivN( dNdr, J_I_T );
        std::array<double,81> Bv = this->Bv_matrix( derivN ); // 1x81
        std::array<double,8> NhVOL = this->NhVOL_matrix_at( ng ); // 1x8
        for( int i=0; i<8; i++ )
        {
            for( int j=0; j<81; j++ )
            {
                Khu[i*81+j] += NhVOL[i]*Bv[j];
            }
        }
    }
}

// Khhマトリクスの計算 (8x8)
void hexl27::Khh_matrix( std::vector<double>& Khh, const double k, const double gmw ) const
{
    for( int i=0; i<8*8; i++ ) Khh[i] = 0.0;
    for( int ng=0; ng<num_gp; ng++ )
    {
        std::array<double,24> xye_w = this->get_xye_w();
        std::array<double,24> dNdr_w = this->dNdr_w_at( ng );
        std::array<double,9> J_w = this->J_w( dNdr_w, xye_w );
        double detJ_w = this->detJ( J_w );
        std::array<double,9> J_I_T_w = this->J_I_T( J_w, detJ_w );
        std::array<double,24> derivN_w = this->derivN_w( dNdr_w, J_I_T_w );
        double fac = this->fac( ng, detJ_w );
        std::array<double,24> Bh = this->Bh_matrix( derivN_w ); // 3x8
        std::array<double,24> BhVOL = this->BhVOL_matrix( Bh, fac ); // 3x8
        std::array<double,9> kmat = { k/gmw, 0.0, 0.0, 0.0, k/gmw, 0.0, 0.0, 0.0, k/gmw };
        std::array<double,34> BhTk; // 8x3
        //+++
//        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "Bh[%5d]\n", ng );
//        for( int i=0; i<dim; i++ )
//        {
//            for( int j=0; j<num_nodw; j++ )
//            {
//                PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", Bh[i*num_nodw+j] );
//            }
//            PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
//        }
//        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "BhVOL[%5d]\n", ng );
//        for( int i=0; i<dim; i++ )
//        {
//            for( int j=0; j<num_nodw; j++ )
//            {
//                PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", BhVOL[i*num_nodw+j] );
//            }
//            PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
//        }
//        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "k[%5d]\n", ng );
//        for( int i=0; i<dim; i++ )
//        {
//            for( int j=0; j<dim; j++ )
//            {
//                PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", kmat[i*dim+j] );
//            }
//            PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
//        }
        //---
        for( int i=0; i<num_nodw; i++ )
        {
            for( int j=0; j<dim; j++ )
            {
                BhTk[i*dim+j] = 0.0;
                for( int k=0; k<dim; k++ )
                {
                    BhTk[i*dim+j] += Bh[k*num_nodw+i]*kmat[k*dim+j];
                }
            }
        }
        for( int i=0; i<num_nodw; i++ )
        {
            for( int j=0; j<num_nodw; j++ )
            {
                for( int k=0; k<dim; k++ )
                {
                    Khh[i*num_nodw+j] += BhTk[i*dim+k]*BhVOL[k*num_nodw+j];
                }
            }
        }
    }
}

void hexl27::permutate_Kuu_matrix( std::vector<double>& Kuu ) const
{
    std::array<double,6561> Kuu_tmp;
    for( int i=0; i<6561; i++ ) Kuu_tmp[i] = Kuu[i];

    for( int i=0; i<num_nods; i++ )
    {
        int io = perms[i];
        for( int k=0; k<dim; k++ )
        {
            int ik = i * dim + k;
            int iok = io * dim + k;
            for( int j=0; j<num_nods; j++ )
            {
                int jo = perms[j];
                for( int l=0; l<dim; l++ )
                {
                    int jl = j * dim + l;
                    int jol = jo * dim + l;
                    Kuu[ik*81+jl] = Kuu_tmp[iok*81+jol];
                }
            }
        }
    }
}

void hexl27::permutate_Kuh_matrix( std::vector<double>& Kuh ) const
{
    std::array<double,648> Kuh_tmp;
    for( int i=0; i<(num_nods*dim)*num_nodw; i++ ) Kuh_tmp[i] = Kuh[i];
    for( int i=0; i<num_nods; i++ )
    {
        int io = perms[i];
        for( int k=0; k<dim; k++ )
        {
            int ik = i * dim + k;
            int iok = io * dim + k;
            for( int j=0; j<num_nodw; j++ )
            {
                int jo = permw[j];
                Kuh[ik*num_nodw+j] = Kuh_tmp[iok*num_nodw+jo];
            }
        }
    }
}

void hexl27::permutate_Khu_matrix( std::vector<double>& Khu ) const
{
    std::array<double,648> Khu_tmp;
    for( int i=0; i<num_nodw*(num_nods*dim); i++ ) Khu_tmp[i] = Khu[i];
    for( int i=0; i<num_nodw; i++ )
    {
        int io = permw[i];
        for( int j=0; j<num_nods; j++ )
        {
            int jo = perms[j];
            for( int l=0; l<dim; l++ )
            {
                int jl = j * dim + l;
                int jol = jo * dim + l;
                Khu[i*(num_nods*dim)+jl] = Khu_tmp[io*(num_nods*dim)+jol];
            }
        }
    }
}

void hexl27::permutate_Khh_matrix( std::vector<double>& Khh ) const
{
    std::array<double,64> Khh_tmp;
    for( int i=0; i<num_nodw*num_nodw; i++ ) Khh_tmp[i] = Khh[i];

    for( int i=0; i<num_nodw; i++ )
    {
        int io = permw[i];
        for( int j=0; j<num_nodw; j++ )
        {
            int jo = permw[j];
            Khh[i*num_nodw+j] = Khh_tmp[io*num_nodw+jo];
        }
    }
}

// 
void hexl27::cal_Kuu_matrix( std::vector<double>& Kuu, const std::vector<double>& D ) const
{
    Kuu.resize(6561); //81x81
    //std::array<double,6561> Kuu_arr = Kuu_matrix( D );
    //permutate_Kuu_matrix( Kuu_arr );
    //for( int i=0; i<6561; i++ ) Kuu[i] = Kuu_arr[i];
    this->Kuu_matrix( D, Kuu );
    this->permutate_Kuu_matrix( Kuu );
}

//
void hexl27::cal_Kuh_matrix( std::vector<double>& Kuh ) const
{
    Kuh.resize((num_nods*dim)*num_nodw); //81x8
    this->Kuh_matrix( Kuh );
    //+++
    //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "Kuh\n" );
    //for( int i=0; i<num_nods*dim; i++ )
    //{
    //    for( int j=0; j<num_nodw; j++ )
    //    {
    //        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", Kuh[i*num_nodw+j] );
    //    }
    //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
    //}
    //---
    this->permutate_Kuh_matrix( Kuh );
}

//
void hexl27::cal_Khu_matrix( std::vector<double>& Khu ) const
{
    Khu.resize(num_nodw*(num_nods*dim)); // 8x81
    this->Khu_matrix( Khu );
    //+++
    //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "Khu\n" );
    //for( int i=0; i<num_nodw; i++ )
    //{
    //    for( int j=0; j<num_nods*dim; j++ )
    //    {
    //        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", Khu[i*num_nods*dim+j] );
    //    }
    //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
    //}
    //---
    this->permutate_Khu_matrix( Khu );
}

//
void hexl27::cal_Khh_matrix( std::vector<double>& Khh, const double k, const double gmw, const double fac ) const
{
    Khh.resize(num_nodw*num_nodw);
    this->Khh_matrix( Khh, k, gmw );
    for( int i=0; i<num_nodw*num_nodw; i++ ) Khh[i] *= fac;
    //+++
    //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "Khh\n" );
    //for( int i=0; i<num_nodw; i++ )
    //{
    //    for( int j=0; j<num_nodw; j++ )
    //    {
    //        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", Khh[i*num_nodw+j] );
    //    }
    //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
    //}
    //---
    this->permutate_Khh_matrix( Khh );
}