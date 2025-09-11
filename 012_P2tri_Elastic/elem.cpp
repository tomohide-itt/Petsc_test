#include "elem.h"

// コンストラクタ
elem::elem( int p, std::vector<int> nd_clos_ids, node_vec& nodes ) : pid(p), dim(2), num_gp(6)
{
    // クロージャ順 -> マトリクス計算用順に交換するための対応関係
    perm = { 5, 3, 4, 0, 1, 2 };
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
        nod[i] = &( nodes.pid_is(node_pids[i]) );
    }
    // 積分点位置
    gp_pos[0] = 0.816847572980459;
    gp_pos[1] = 0.091576213509771;
    gp_pos[2] = 0.108103018168070;
    gp_pos[3] = 0.445948490915965;
    // 積分点重み
    gp_wei[0] = 0.109951743655322;
    gp_wei[1] = 0.223381589678011;
}

// xyeを取得 (6x2)
std::array<double,12> elem::get_xye() const
{
    std::array<double,12> xye;
    for( int i=0; i<num_nods; i++ )
    {
        xye[i*dim+0] = this->nod[i]->xy[0];
        xye[i*dim+1] = this->nod[i]->xy[1];
    }
    return xye;
}

// dNdrを計算 (6x2)
std::array<double,12> elem::dNdr_at( const int ng ) const
{
    double L1, L2, L3;
    if( ng == 0 ){ L1 = gp_pos[0]; L2 = gp_pos[1]; }
    if( ng == 1 ){ L1 = gp_pos[1]; L2 = gp_pos[0]; }
    if( ng == 2 ){ L1 = gp_pos[1]; L2 = gp_pos[1]; }
    if( ng == 3 ){ L1 = gp_pos[2]; L2 = gp_pos[3]; }
    if( ng == 4 ){ L1 = gp_pos[3]; L2 = gp_pos[2]; }
    if( ng == 5 ){ L1 = gp_pos[3]; L2 = gp_pos[3]; }
    L3 = 1.0 - L1 - L2;

    std::array<double,12> dNdr;
    dNdr[0*2+0] = 4.0*L1-1.0;
    dNdr[1*2+0] = 0.0;
    dNdr[2*2+0] =-4.0*L3+1.0;
    dNdr[3*2+0] =-4.0*L2;
    dNdr[4*2+0] = 4.0*(L3-L1);
    dNdr[5*2+0] = 4.0*L2;
    dNdr[0*2+1] = 0.0;
    dNdr[1*2+1] = 4.0*L2-1.0;
    dNdr[2*2+1] =-4.0*L3+1.0;
    dNdr[3*2+1] = 4.0*(L3-L2);
    dNdr[4*2+1] =-4.0*L1;
    dNdr[5*2+1] = 4.0*L1;

    return dNdr;
}

// Jを計算 (2x2)
std::array<double,4> elem::J_at( const int ng ) const
{
    std::array<double,12> xye = get_xye();
    std::array<double,12> dNdr = dNdr_at( ng );
    std::array<double,4> J;
    for( int i=0; i<dim; i++ ){
        for( int j=0; j<dim; j++ ){
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
double elem::detJ_at( const int ng ) const
{
    std::array<double,4> J = J_at( ng );
    return J[0]*J[3] - J[1]*J[2];
}

// J-T を計算 (2x2)
std::array<double,4> elem::J_I_T_at( const int ng ) const
{
    std::array<double,4> J = J_at( ng );
    double detJ = detJ_at( ng );
    std::array<double,4> J_I_T;
    J_I_T[0*2+0] = J[1*2+1]/detJ;
    J_I_T[1*2+1] = J[0*2+0]/detJ;
    J_I_T[0*2+1] =-J[1*2+0]/detJ;
    J_I_T[1*2+0] =-J[0*2+1]/detJ;
    return J_I_T;
}

// derivNを計算 (6x2)
std::array<double,12> elem::derivN_at( const int ng ) const
{
    std::array<double,12> dNdr = dNdr_at( ng );
    std::array<double,4> J_I_T = J_I_T_at( ng );
    std::array<double,12> derivN;
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
double elem::fac_at( const int ng ) const
{
    double detJ = detJ_at( ng );
    double wei;
    if( ng < 3 ) wei = gp_wei[0];
    else wei = gp_wei[1];
    return 0.5*wei*detJ;
}

// Bマトリクスの計算 (4x12)
std::array<double,48> elem::B_matrix_at( const int ng ) const
{
    std::array<double,12> derivN = derivN_at( ng );
    std::array<double,48> B;
    for( int i=0; i<48; i++ ) B[i] = 0.0;
    for( int i=0; i<num_nods; i++ )
    {
        B[0*12 + (2*i+0)] = -derivN[i*2+0];
        B[1*12 + (2*i+1)] = -derivN[i*2+1];
        B[3*12 + (2*i+0)] = -derivN[i*2+1]*0.5;
        B[3*12 + (2*i+1)] = -derivN[i*2+0]*0.5;
    }
    return B;
}

// BVOLマトリクスの計算 (4x12)
std::array<double,48> elem::BVOL_matrix_at( const int ng ) const
{
    std::array<double,48> BVOL = B_matrix_at( ng );
    double fac = fac_at( ng );
    for( int i=0; i<48; i++ ) BVOL[i] *= fac;
    return BVOL;
}

// Kuuマトリクスの計算 (12x12)
std::array<double,144> elem::Kuu_matrix( const double* D ) const
{
    std::array<double,144> Kuu;
    for( int i=0; i<144; i++ ) Kuu[i] = 0.0;
    for( int ng=0; ng<num_gp; ng++ )
    {
        std::array<double,48> B = B_matrix_at( ng );
        std::array<double,48> BVOL = BVOL_matrix_at( ng );

        std::array<double,48> BTD;
        for( int i=0; i<12; i++ )
        {
            for( int j=0; j<4; j++ )
            {
                BTD[i*4+j] = 0.0;
                for( int k=0; k<4; k++ )
                {
                    if( k != 3 ) BTD[i*4+j] += B[k*12+i]*D[k*4+j];
                    else         BTD[i*4+j] += 2.0*B[k*12+i]*D[k*4+j];
                }
            }
        }

        for( int i=0; i<12; i++ )
        {
            for( int j=0; j<12; j++ )
            {
                for( int k=0; k<4; k++ )
                {
                    if( k != 3 ) Kuu[i*12+j] += BTD[i*4+k]*BVOL[k*12+j];
                    else         Kuu[i*12+j] += 2.0*BTD[i*4+k]*BVOL[k*12+j];
                }
            }
        }
    }
    return Kuu;
}

void elem::permutate_Kuu_matrix( std::array<double,144>& Kuu ) const
{
    std::array<double,144> Kuu_tmp;
    for( int i=0; i<144; i++ ) Kuu_tmp[i] = Kuu[i];

    for( int i=0; i<num_nods; i++ ){
    int io = perm[i];
    for( int k=0; k<dim; k++ ){
      int ik  = i *dim+k;
      int iok = io*dim+k;
      for( int j=0; j<num_nods; j++ ){
        int jo = perm[j];
        for( int l=0; l<dim; l++ ){
          int jl  = j *dim+l;
          int jol = jo*dim+l;
          Kuu[ik*12+jl] = Kuu_tmp[iok*12+jol];
        }
      }
    }
  }
}