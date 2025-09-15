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
    virtual void initialize( const int p, const std::vector<int>& nd_clos_ids, const node_vec& nodes ){}
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
    ~elem_vec();
    template< class ETYPE > void create_new( const int p, const std::vector<int>& nd_clos_ids, const node_vec& nodes );
    const int size() const{ return m_elems.size(); }
private:
    std::vector<elem*> m_elems;
    std::map<int,int> m_pid2idx;
    std::map<int,int> m_idx2pid;
};

#include "elem.hpp"

#endif

