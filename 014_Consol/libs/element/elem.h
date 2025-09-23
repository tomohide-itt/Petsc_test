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
    virtual void cal_Kuh_matrix( std::vector<double>& Kuh ) const{}
    virtual void cal_Khu_matrix( std::vector<double>& Khu ) const{}
    virtual void cal_Khh_matrix( std::vector<double>& Khh, const double k, const double gmw ) const{}
    virtual int vtk_num_vertex() const { return -1; }
    virtual int vtk_cell_type() const { return -1; }
public:
    int id;
    int pid;
    int type;
    int num_nods;
    int num_nodw;
    int dim;
    int num_gp;
    std::vector<int> node_pids;
    std::vector<std::shared_ptr<node>> nod;
    std::vector<int> perms;
    std::vector<int> permw;

    std::vector<double> gp_pos;
    std::vector<double> gp_wei;
};

class elem_vec
{
public:
    elem_vec();
    template< class ETYPE > void create_new( const int p, const std::vector<int>& nd_clos_ids, node_vec& nodes );
    const int size() const{ return m_elems.size(); }
    std::shared_ptr<elem> operator[]( const int idx ){ return m_elems[idx]; }
    const std::shared_ptr<elem> operator[]( const int idx ) const { return m_elems[idx]; }
    const std::shared_ptr<elem> pid_is( const int pid ) const{ return m_elems[m_pid2idx.at(pid)]; }
    std::vector<std::shared_ptr<elem>>::iterator begin(){ return m_elems.begin(); }
    std::vector<std::shared_ptr<elem>>::iterator end(){ return m_elems.end(); }
    std::vector<std::shared_ptr<elem>>::const_iterator begin() const{ return m_elems.begin(); }
    std::vector<std::shared_ptr<elem>>::const_iterator end() const{ return m_elems.end(); }
    void show() const;
private:
    std::vector<std::shared_ptr<elem>> m_elems;
    std::map<int,int> m_pid2idx;
    std::map<int,int> m_idx2pid;
};

#include "elem.hpp"

#endif

