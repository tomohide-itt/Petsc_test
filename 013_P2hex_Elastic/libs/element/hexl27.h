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
};

#endif

