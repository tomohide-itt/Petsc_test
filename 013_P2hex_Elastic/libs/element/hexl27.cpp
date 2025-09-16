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

// detJを計算
double hexl27::detJ_at( const int ng ) const
{
    std::array<double,9> J = J_at( ng );
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

// 体積積分時に乗じる係数を計算
double hexl27::fac_at( const int ng ) const
{
    double detJ = detJ_at( ng );
    int ig = static_cast<int>( ng/3/3 );
    int jg = static_cast<int>( (ng-ig*3*3)/3 );
    int kg = ng - ig*3*3 - jg*3;
    return gp_wei[ig] * gp_wei[jg] * gp_wei[kg] * detJ;
}

// Bマトリクスの計算 (6x27)
std::array<double,486> hexl27::B_matrix_at( const int ng ) const
{
    std::array<double,81> derivN = derivN_at( ng );
    std::array<double,486> B;
    for( int i=0; i<486; i++ ) B[i] = 0.0;
    for( int i=0; i<num_nods; i++ )
    {
        B[0*27 + (3*i+0)] = -derivN[i*3+0];
        B[1*27 + (3*i+1)] = -derivN[i*3+1];
        B[2*27 + (3*i+2)] = -derivN[i*3+2];
        B[3*27 + (3*i+0)] = -derivN[i*3+1]*0.5;
        B[3*27 + (3*i+1)] = -derivN[i*3+0]*0.5;
        B[4*27 + (3*i+1)] = -derivN[i*3+2]*0.5;
        B[4*27 + (3*i+2)] = -derivN[i*3+1]*0.5;
        B[5*27 + (3*i+0)] = -derivN[i*3+2]*0.5;
        B[5*27 + (3*i+2)] = -derivN[i*3+0]*0.5;
    }
    return B;
}

// BVOLマトリクスの計算 (6x27)
std::array<double,486> hexl27::BVOL_matrix_at( const int ng ) const
{
    std::array<double,486> BVOL = B_matrix_at( ng );
    double fac = fac_at( ng );
    for( int i=0; i<486; i++ ) BVOL[i] *= fac;
    return BVOL;
}

// Kuuマトリクスの計算 (27x27)
std::array<double,729> hexl27::Kuu_matrix( const std::vector<double>& D ) const
{
    std::array<double,729> Kuu;
    for( int i=0; i<729; i++ ) Kuu[i] = 0.0;
    for( int ng=0; ng<num_gp; ng++ )
    {
        std::array<double,486> B = B_matrix_at( ng );
        std::array<double,486> BVOL = BVOL_matrix_at( ng );

        std::array<double,486> BTD;
        for( int i=0; i<27; i++ )
        {
            for( int j=0; j<6; j++ )
            {
                BTD[i*6+j] = 0.0;
                for( int k=0; k<6; k++ )
                {
                    if( k < 3  ) BTD[i*6+j] += B[k*27+i]*D[k*6+j];
                    else         BTD[i*6+j] += 2.0*B[k*27+i]*D[k*6+j];
                }
            }
        }

        for( int i=0; i<27; i++ )
        {
            for( int j=0; j<27; j++ )
            {
                for( int k=0; k<6; k++ )
                {
                    if( k <  3 ) Kuu[i*27+j] += BTD[i*6+k]*BVOL[k*27+j];
                    else         Kuu[i*27+j] += 2.0*BTD[i*6+k]*BVOL[k*27+j];
                }
            }
        }
    }
    return Kuu;
}

void hexl27::permutate_Kuu_matrix( std::array<double,729>& Kuu ) const
{
    std::array<double,729> Kuu_tmp;
    for( int i=0; i<729; i++ ) Kuu_tmp[i] = Kuu[i];

    for( int i=0; i<num_nods; i++ )
    {
        int io = perm[i];
        for( int k=0; k<dim; k++ )
        {
            int ik = i * dim + k;
            int iok = io * dim + k;
            for( int j=0; j<num_nods; j++ )
            {
                int jo = perm[j];
                for( int l=0; l<dim; l++ )
                {
                    int jl = j * dim + l;
                    int jol = jo * dim + l;
                    Kuu[ik*27+jl] = Kuu_tmp[iok*27+jol];
                }
            }
        }
    }
}

// 
void hexl27::cal_Kuu_matrix( std::vector<double>& Kuu, const std::vector<double>& D ) const
{
    Kuu.resize(729); //27x27
    std::array<double,729> Kuu_arr = Kuu_matrix( D );
    permutate_Kuu_matrix( Kuu_arr );
    for( int i=0; i<729; i++ ) Kuu[i] = Kuu_arr[i];
}