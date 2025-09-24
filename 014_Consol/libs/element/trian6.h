#ifndef TRIAN6_H
#define TRIAN6_H
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

class trian6 : public elem
{
public:
    trian6() : elem(){}
    ~trian6() override = default;
    void initialize( const int p, const std::vector<int>& nd_clos_ids, node_vec& nodes ) override;
    static bool get_coords( const DM& dm, const int p, std::vector<double>& xy );
    void cal_Kuu_matrix( std::vector<double>& Kuu, const std::vector<double>& D ) const override;
    void cal_Kuh_matrix( std::vector<double>& Kuh ) const override;
    void cal_Khu_matrix( std::vector<double>& Khu ) const override;
    void cal_Khh_matrix( std::vector<double>& Khh, const double k, const double gmw, const double fac ) const override;
    int vtk_num_vertex() const override { return 3; }
    int vtk_cell_type() const override { return 5; }
private:
    static void get_coords_vertex( const DM& dm, const int p, std::vector<double>& xy );
    static void get_coords_face( const DM& dm, const int p, std::vector<double>& xy );
private:
    std::array<double,12> get_xye() const;
    std::array<double,6>  get_xyew() const;
    std::array<double,12> dNdr_at( const int ng ) const;
    std::array<double,6>  dNdr_w_at( const int ng ) const;
    std::array<double,4>  J_at( const int ng ) const;
    std::array<double,4>  J_w_at( const int ng ) const;
    double detJ_at( const int ng ) const;
    double detJ_w_at( const int ng ) const;
    std::array<double,4> J_I_T_at( const int ng ) const;
    std::array<double,4> J_I_T_w_at( const int ng ) const;
    std::array<double,12> derivN_at( const int ng ) const;
    std::array<double,6>  derivN_w_at( const int ng ) const;
    double fac_at( const int ng ) const;
    std::array<double,48> B_matrix_at( const int ng ) const;
    std::array<double,48> BVOL_matrix_at( const int ng ) const;
    std::array<double,12> Bv_matrix_at( const int ng ) const;
    std::array<double,3>  Nh_matrix_at( const int ng ) const;
    std::array<double,3>  NhVOL_matrix_at( const int ng ) const;
    std::array<double,6>  Bh_matrix_at( const int ng ) const;
    std::array<double,6>  BhVOL_matrix_at( const int ng ) const;
    std::array<double,144> Kuu_matrix( const std::vector<double>& D ) const;
    void Kuh_matrix( std::vector<double>& Kuh ) const;
    void Khu_matrix( std::vector<double>& Khu ) const;
    void Khh_matrix( std::vector<double>& Khh, const double k, const double gmw ) const;
    void permutate_Kuu_matrix( std::array<double,144>& Kuu ) const;
    void permutate_Kuh_matrix( std::vector<double>& Kuh ) const;
    void permutate_Khu_matrix( std::vector<double>& Khu ) const;
    void permutate_Khh_matrix( std::vector<double>& Khh ) const;
};

#endif

