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
private:
    std::array<double,81> get_xye() const;
    std::array<double,81> dNdr_at( const int ng ) const;
    std::array<double,9>  J_at( const int ng ) const;
    double detJ_at( const int ng ) const;
    std::array<double,9> J_I_T_at( const int ng ) const;
    std::array<double,81> derivN_at( const int ng ) const;
    double fac_at( const int ng ) const;
    std::array<double,486> B_matrix_at( const int ng ) const;
    std::array<double,486> BVOL_matrix_at( const int ng ) const;
    std::array<double,729> Kuu_matrix( const std::vector<double>& D ) const;
    void permutate_Kuu_matrix( std::array<double,729>& Kuu ) const;
};

#endif

