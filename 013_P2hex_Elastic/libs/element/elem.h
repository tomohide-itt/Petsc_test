#ifndef ELEM_H
#define ELEM_H
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <iomanip>
#include <map>
#include <memory>
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
    virtual ~elem() = default;
    virtual void initialize( const int p, const std::vector<int>& nd_clos_ids, node_vec& nodes ){}
    virtual void cal_Kuu_matrix( std::vector<double>& Kuu, const std::vector<double>& D ) const{}
public:
    int id;
    int pid;
    int type;
    int num_nods;
    int dim;
    int num_gp;
    std::vector<int> node_pids;
    std::vector<std::shared_ptr<node>> nod;
    std::vector<int> perm;

    std::vector<double> gp_pos;
    std::vector<double> gp_wei;
};

class elem_vec
{
public:
    elem_vec();
    template< class ETYPE > void create_new( const int p, const std::vector<int>& nd_clos_ids, node_vec& nodes );
    const int size() const{ return m_elems.size(); }
    const std::shared_ptr<elem> pid_is( const int pid ) const{ return m_elems[m_pid2idx.at(pid)]; }
    void show() const;
private:
    std::vector<std::shared_ptr<elem>> m_elems;
    std::map<int,int> m_pid2idx;
    std::map<int,int> m_idx2pid;
};

#include "elem.hpp"

#endif

