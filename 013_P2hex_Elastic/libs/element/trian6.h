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
    void initialize( const int p, const std::vector<int>& nd_clos_ids, const node_vec& nodes ) override;
    static bool get_coords( const DM& dm, const int p, std::vector<double>& xy );
private:
    static void get_coords_vertex( const DM& dm, const int p, std::vector<double>& xy );
    static void get_coords_face( const DM& dm, const int p, std::vector<double>& xy );
};

#endif

