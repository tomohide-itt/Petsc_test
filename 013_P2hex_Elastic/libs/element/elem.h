#ifndef ELEM_H
#define ELEM_H
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <iomanip>
#include <map>
#include <petscdmplex.h>
#include <petscksp.h>
#include "node.h"

enum class element_type : int
{
    trian6 = 5,
    hexl27 = 7,
};

class elem
{
public:
    elem(){}
public:
    int id;
    int pid;
    int type;
    int num_nods;
    int dim;
    int num_gp;
    std::vector<int> node_pids;
    std::vector<node*> nod;
    std::vector<int> perm;

    std::array<double,4> gp_pos;
    std::array<double,2> gp_wei;
};

class elem_vec
{
public:
    elem_vec();
private:
    std::vector<elem*> m_elems;
    int max_idx;
    std::map<int,int> m_pid2idx;
    std::map<int,int> m_idx2pid;
};

#endif

