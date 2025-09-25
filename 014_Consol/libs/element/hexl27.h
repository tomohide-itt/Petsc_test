#ifndef HEXL27_H
#define HEXL27_H
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <iomanip>
#include <map>
#include <petscdmplex.h>
#include <petscksp.h>
#include "elem.h"

class hexl27 : public elem
{
public:
    hexl27() : elem(){}
    void initialize( const int p, const std::vector<int>& nd_clos_ids, node_vec& nodes ) override;
    static bool get_coords( const DM& dm, const int p, std::vector<double>& xy );
    void cal_Kuu_matrix( std::vector<double>& Kuu, const std::vector<double>& D ) const override;
    void cal_Kuh_matrix( std::vector<double>& Kuh ) const override;
    void cal_Khu_matrix( std::vector<double>& Khu ) const override;
    void cal_Khh_matrix( std::vector<double>& Khh, const double k, const double gmw, const double fac ) const override;
    int vtk_num_vertex() const override { return 8; }
    int vtk_cell_type() const override { return 12; }
private:
    std::array<double,81> get_xye() const;
    std::array<double,24> get_xye_w() const;
    std::array<double,81> dNdr_at( const int ng ) const;
    std::array<double,24> dNdr_w_at( const int ng ) const;
    std::array<double,9>  J_at( const int ng ) const;
    std::array<double,9>  J_w_at( const int ng ) const;
    std::array<double,9>  J( const std::array<double,81>& dNdr, const std::array<double,81>& xye ) const;
    std::array<double,9>  J_w( const std::array<double,24>& dNdr_w, const std::array<double,24>& xye_w ) const;
    double detJ_at( const int ng ) const;
    double detJ_w_at( const int ng ) const;
    double detJ( const std::array<double,9>& J ) const;
    std::array<double,9> J_I_T_at( const int ng ) const;
    std::array<double,9> J_I_T_w_at( const int ng ) const;
    std::array<double,9> J_I_T( const std::array<double,9>& J, const double detJ ) const;
    std::array<double,81> derivN_at( const int ng ) const;
    std::array<double,24> derivN_w_at( const int ng ) const;
    std::array<double,81> derivN( const std::array<double,81>& dNdr, const std::array<double,9>& J_I_T ) const;
    std::array<double,24> derivN_w( const std::array<double,24>& dNdr_w, const std::array<double,9>& J_I_T_w ) const;
    double fac_at( const int ng ) const;
    double fac( const int ng, const double detJ ) const;
    std::array<double,486> B_matrix_at( const int ng ) const;
    std::array<double,486> B_matrix( const std::array<double,81>& derivN ) const;
    std::array<double,486> BVOL_matrix_at( const int ng ) const;
    std::array<double,486> BVOL_matrix( const std::array<double,486>& B, const double fac ) const;
    std::array<double,81>  Bv_matrix_at( const int ng ) const;
    std::array<double,81>  Bv_matrix( const std::array<double,81>& derivN ) const;
    std::array<double,8>   Nh_matrix_at( const int ng ) const;
    std::array<double,8>   NhVOL_matrix_at( const int ng ) const;
    std::array<double,24>  Bh_matrix_at( const int ng ) const;
    std::array<double,24>  Bh_matrix( const std::array<double,24>& derivN_w ) const;
    std::array<double,24>  BhVOL_matrix_at( const int ng ) const;
    std::array<double,24>  BhVOL_matrix( const std::array<double,24>& Bh, const double fac ) const;
    std::array<double,6561> Kuu_matrix( const std::vector<double>& D ) const;
    void Kuu_matrix( const std::vector<double>& D, std::vector<double>& Kuu ) const;
    void Kuh_matrix( std::vector<double>& Kuh ) const;
    void Khu_matrix( std::vector<double>& Khu ) const;
    void Khh_matrix( std::vector<double>& Khh, const double k, const double gmw ) const;
    void permutate_Kuu_matrix( std::array<double,6561>& Kuu ) const;
    void permutate_Kuu_matrix( std::vector<double>& Kuu ) const;
    void permutate_Kuh_matrix( std::vector<double>& Kuh ) const;
    void permutate_Khu_matrix( std::vector<double>& Khu ) const;
    void permutate_Khh_matrix( std::vector<double>& Khh ) const;
};

#endif

