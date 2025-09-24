#include "trian6.h"

// 初期化
void trian6::initialize( const int p, const std::vector<int>& nd_clos_ids, node_vec& nodes )
{
    this->pid = p;
    this->dim = 2;
    this->num_gp = 6;
    // クロージャ順 -> マトリクス計算用順に交換するための対応関係
    this->perms = { 5, 3, 4, 0, 1, 2 };
    this->permw = { 0, 1, 2 };
    // 節点数
    this->num_nods = nd_clos_ids.size();
    this->num_nodw = 3;
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

// xyeを取得 (6x2)
std::array<double,12> trian6::get_xye() const
{
    std::array<double,12> xye;
    for( int i=0; i<num_nods; i++ )
    {
        for( int j=0; j<dim; j++ ) xye[i*dim+j] = this->nod[i]->xy[j];
    }
    return xye;
}

// xyewを取得 (3x2)
std::array<double,6> trian6::get_xyew() const
{
    std::array<double,6> xyew;
    for( int i=0; i<num_nodw; i++ )
    {
        for( int j=0; j<dim; j++ ) xyew[i*dim+j] = this->nod[i]->xy[j];
    }
    return xyew;
}

// dNdrを計算 (6x2)
std::array<double,12> trian6::dNdr_at( const int ng ) const
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

// dNdr_wを計算 (3x2)
std::array<double,6> trian6::dNdr_w_at( const int ng ) const
{
    std::array<double,6> dNdr_w;
    dNdr_w[0*2+0] = 1.0;
    dNdr_w[1*2+0] = 0.0;
    dNdr_w[2*2+0] =-1.0;
    dNdr_w[0*2+1] = 0.0;
    dNdr_w[1*2+1] = 1.0;
    dNdr_w[2*2+1] =-1.0;
    return dNdr_w;
}

// Jを計算 (2x2)
std::array<double,4> trian6::J_at( const int ng ) const
{
    std::array<double,12> xye = get_xye();
    std::array<double,12> dNdr = dNdr_at( ng );
    std::array<double,4> J;
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

// J_wを計算 (2x2)
std::array<double,4> trian6::J_w_at( const int ng ) const
{
    std::array<double,6> xye_w = get_xyew();
    std::array<double,6> dNdr_w = dNdr_w_at( ng );
    std::array<double,4> J_w;
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
double trian6::detJ_at( const int ng ) const
{
    std::array<double,4> J = J_at( ng );
    return J[0]*J[3] - J[1]*J[2];
}

// detJ_wを計算
double trian6::detJ_w_at( const int ng ) const
{
    std::array<double,4> J_w = J_w_at( ng );
    return J_w[0]*J_w[3] - J_w[1]*J_w[2];
}

// J-T を計算 (2x2)
std::array<double,4> trian6::J_I_T_at( const int ng ) const
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

std::array<double,4> trian6::J_I_T_w_at( const int ng ) const
{
    std::array<double,4> J_w = J_w_at( ng );
    double detJ_w = detJ_w_at( ng );
    std::array<double,4> J_I_T_w;
    J_I_T_w[0*2+0] = J_w[1*2+1]/detJ_w;
    J_I_T_w[1*2+1] = J_w[0*2+0]/detJ_w;
    J_I_T_w[0*2+1] =-J_w[1*2+0]/detJ_w;
    J_I_T_w[1*2+0] =-J_w[0*2+1]/detJ_w;
    return J_I_T_w;
}

// derivNを計算 (6x2)
std::array<double,12> trian6::derivN_at( const int ng ) const
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

// derivN_wを計算 (3x2)
std::array<double,6> trian6::derivN_w_at( const int ng ) const
{
    std::array<double,6> dNdr_w = dNdr_w_at( ng );
    std::array<double,4> J_I_T_w = J_I_T_w_at( ng );
    std::array<double,6> derivN_w;
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
double trian6::fac_at( const int ng ) const
{
    double detJ = detJ_at( ng );
    double wei;
    if( ng < 3 ) wei = gp_wei[0];
    else wei = gp_wei[1];
    return 0.5*wei*detJ;
}

// Bマトリクスの計算 (4x12)
std::array<double,48> trian6::B_matrix_at( const int ng ) const
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
std::array<double,48> trian6::BVOL_matrix_at( const int ng ) const
{
    std::array<double,48> BVOL = B_matrix_at( ng );
    double fac = fac_at( ng );
    for( int i=0; i<48; i++ ) BVOL[i] *= fac;
    return BVOL;
}

// Bvマトリクスの計算 (1x12)
std::array<double,12> trian6::Bv_matrix_at( const int ng ) const
{
    std::array<double,12> derivN = derivN_at( ng );
    std::array<double,12> Bv;
    for( int i=0; i<12; i++ ) Bv[i] = 0.0;
    for( int i=0; i<num_nods; i++ )
    {
        Bv[i*2+0] = -derivN[i*2+0];
        Bv[i*2+1] = -derivN[i*2+1];
    }
    return Bv;
}

// Nhマトリクスの計算 (1x3)
std::array<double,3> trian6::Nh_matrix_at( const int ng ) const
{
    double L1, L2, L3;
    if( ng == 0 ){ L1 = gp_pos[0]; L2 = gp_pos[1]; }
    if( ng == 1 ){ L1 = gp_pos[1]; L2 = gp_pos[0]; }
    if( ng == 2 ){ L1 = gp_pos[1]; L2 = gp_pos[1]; }
    if( ng == 3 ){ L1 = gp_pos[2]; L2 = gp_pos[3]; }
    if( ng == 4 ){ L1 = gp_pos[3]; L2 = gp_pos[2]; }
    if( ng == 5 ){ L1 = gp_pos[3]; L2 = gp_pos[3]; }
    L3 = 1.0 - L1 - L2;

    std::array<double,3> Nh;
    Nh[0] = L1;
    Nh[1] = L2;
    Nh[2] = L3;
    return Nh;
}

// NhVOLマトリクスの計算 (1x3)
std::array<double,3> trian6:: NhVOL_matrix_at( const int ng ) const
{
    std::array<double,3> NhVOL = this->Nh_matrix_at( ng );
    double fac = this->fac_at( ng );
    for( int i=0; i<3; i++ ) NhVOL[i] *= fac;
    return NhVOL;
}

// Bhマトリクスの計算 (2x3)
std::array<double,6> trian6::Bh_matrix_at( const int ng ) const
{
    std::array<double,6> derivN_w = derivN_w_at( ng );
    std::array<double,6> Bh;
    for( int i=0; i<6; i++ ) Bh[i] = 0.0;
    for( int i=0; i<num_nodw; i++ )
    {
        for( int j=0; j<dim; j++ )
        {
            Bh[j*3+i] = derivN_w[i*dim+j];
        }
    }
    return Bh;
}

// BhVOLマトリクスの計算 (2x3)
std::array<double,6> trian6::BhVOL_matrix_at( const int ng ) const
{
    std::array<double,6> BhVOL = this->Bh_matrix_at( ng );
    double fac = this->fac_at( ng );
    for( int i=0; i<6; i++ ) BhVOL[i] *= fac;
    return BhVOL;
}

// Kuuマトリクスの計算 (12x12)
std::array<double,144> trian6::Kuu_matrix( const std::vector<double>& D ) const
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
                    if( k < 3  ) BTD[i*4+j] += B[k*12+i]*D[k*4+j];
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
                    if( k <  3 ) Kuu[i*12+j] += BTD[i*4+k]*BVOL[k*12+j];
                    else         Kuu[i*12+j] += 2.0*BTD[i*4+k]*BVOL[k*12+j];
                }
            }
        }
    }
    return Kuu;
}

// Kuhマトリクスの計算 (12x3)
void trian6::Kuh_matrix( std::vector<double>& Kuh ) const
{
    for( int i=0; i<36; i++ ) Kuh[i] = 0.0;
    for( int ng=0; ng<num_gp; ng++ )
    {
        std::array<double,12> Bv = Bv_matrix_at( ng ); //1x12
        std::array<double,3>  NhVOL = NhVOL_matrix_at( ng ); //1x3
        for( int i=0; i<12; i++ )
        {
            for( int j=0; j<3; j++ )
            {
                Kuh[i*3+j] += Bv[i]*NhVOL[j];
            }
        }
    }
}

// Khuマトリクスの計算 (3x12)
void trian6::Khu_matrix( std::vector<double>& Khu ) const
{
    for( int i=0; i<36; i++ ) Khu[i] = 0.0;
    for( int ng=0; ng<num_gp; ng++ )
    {
        std::array<double,12> Bv = Bv_matrix_at( ng ); //1x12
        std::array<double,3>  NhVOL = NhVOL_matrix_at( ng ); //1x3
        for( int i=0; i<3; i++ )
        {
            for( int j=0; j<12; j++ )
            {
                Khu[i*12+j] += NhVOL[i]*Bv[j];
            }
        }
    }
}

// Khhマトリクスの計算 (3x3)
void trian6::Khh_matrix( std::vector<double>& Khh, const double k, const double gmw ) const
{
    for( int i=0; i<9; i++ ) Khh[i] = 0.0;
    for( int ng=0; ng<num_gp; ng++ )
    {
        std::array<double,6> Bh = Bh_matrix_at( ng ); //2x3
        std::array<double,6> BhVOL = BhVOL_matrix_at( ng ); //2x3
        std::array<double,4> kmat = { k/gmw, 0.0, 0.0, k/gmw }; //2x2
        std::array<double,6> BhTk; // 3x2
        for( int i=0; i<num_nodw; i++ )
        {
            for( int j=0; j<dim; j++ )
            {
                BhTk[i*dim+j] = 0.0;
                for( int k=0; k<dim; k++ )
                {
                    BhTk[i*dim+j] += Bh[k*3+i]*kmat[k*2+j];
                }
            }
        }

        for( int i=0; i<num_nodw; i++ )
        {
            for( int j=0; j<num_nodw; j++ )
            {
                for( int k=0; k<dim; k++ )
                {
                    Khh[i*3+j] += BhTk[i*dim+k]*BhVOL[k*3+j];
                }
            }
        }
    }
}

void trian6::permutate_Kuu_matrix( std::array<double,144>& Kuu ) const
{
    std::array<double,144> Kuu_tmp;
    for( int i=0; i<144; i++ ) Kuu_tmp[i] = Kuu[i];

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
                    Kuu[ik*12+jl] = Kuu_tmp[iok*12+jol];
                }
            }
        }
    }
}

void trian6::permutate_Kuh_matrix( std::vector<double>& Kuh ) const
{
    std::array<double,36> Kuh_tmp;
    for( int i=0; i<36; i++ ) Kuh_tmp[i] = Kuh[i];

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
                Kuh[ik*3+j] = Kuh_tmp[iok*3+jo];
            }
        }
    }
}

void trian6::permutate_Khu_matrix( std::vector<double>& Khu ) const
{
    std::array<double,36> Khu_tmp;
    for( int i=0; i<36; i++ ) Khu_tmp[i] = Khu[i];

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
                Khu[i*12+jl] = Khu_tmp[io*12+jol];
            }
        }
    }
}

void trian6::permutate_Khh_matrix( std::vector<double>& Khh ) const
{
    std::array<double,9> Khh_tmp;
    for( int i=0; i<9; i++ ) Khh_tmp[i] = Khh[i];

    for( int i=0; i<num_nodw; i++ )
    {
        int io = permw[i];
        for( int j=0; j<num_nodw; j++ )
        {
            int jo = permw[j];
            Khh[i*3+j] = Khh_tmp[io*3+jo];
        }
    }
}

// 
void trian6::cal_Kuu_matrix( std::vector<double>& Kuu, const std::vector<double>& D ) const
{
    Kuu.resize(144); //12x12
    std::array<double,144> Kuu_arr = Kuu_matrix( D );
    permutate_Kuu_matrix( Kuu_arr );
    for( int i=0; i<144; i++ ) Kuu[i] = Kuu_arr[i];
}

//
void trian6::cal_Kuh_matrix( std::vector<double>& Kuh ) const
{
    Kuh.resize(36); //12x3
    this->Kuh_matrix( Kuh );
    //+++
    //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "Kuh\n" );
    //for( int i=0; i<12; i++ )
    //{
    //    for( int j=0; j<3; j++ )
    //    {
    //        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", Kuh[i*3+j] );
    //    }
    //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
    //}
    //---
    this->permutate_Kuh_matrix( Kuh );
}

//
void trian6::cal_Khu_matrix( std::vector<double>& Khu ) const
{
    Khu.resize(36);
    this->Khu_matrix( Khu );
    //+++
    //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "Khu\n" );
    //for( int i=0; i<3; i++ )
    //{
    //    for( int j=0; j<12; j++ )
    //    {
    //        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", Khu[i*12+j] );
    //    }
    //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
    //}
    //---
    this->permutate_Khu_matrix( Khu );
}

//
void trian6::cal_Khh_matrix( std::vector<double>& Khh, const double k, const double gmw, const double fac ) const
{
    Khh.resize(9);
    this->Khh_matrix( Khh, k, gmw );
    for( int i=0; i<9; i++ ) Khh[i] *= fac;
    //+++
    //PetscSynchronizedPrintf( PETSC_COMM_WORLD, "Khh\n" );
    //for( int i=0; i<3; i++ )
    //{
    //    for( int j=0; j<3; j++ )
    //    {
    //        PetscSynchronizedPrintf( PETSC_COMM_WORLD, "%15.5e", Khh[i*3+j] );
    //    }
    //    PetscSynchronizedPrintf( PETSC_COMM_WORLD, "\n" );
    //}
    //---
    this->permutate_Khh_matrix( Khh );
}


