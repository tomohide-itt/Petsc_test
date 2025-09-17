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
private:
    static void get_coords_vertex( const DM& dm, const int p, std::vector<double>& xy );
    static void get_coords_face( const DM& dm, const int p, std::vector<double>& xy );
private:
    std::array<double,12> get_xye() const;
    std::array<double,12> dNdr_at( const int ng ) const;
    std::array<double,4>  J_at( const int ng ) const;
    double detJ_at( const int ng ) const;
    std::array<double,4> J_I_T_at( const int ng ) const;
    std::array<double,12> derivN_at( const int ng ) const;
    double fac_at( const int ng ) const;
    std::array<double,48> B_matrix_at( const int ng ) const;
    std::array<double,48> BVOL_matrix_at( const int ng ) const;
    std::array<double,144> Kuu_matrix( const std::vector<double>& D ) const;
    void permutate_Kuu_matrix( std::array<double,144>& Kuu ) const;
};

#endif

