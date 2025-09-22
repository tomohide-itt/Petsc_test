#ifndef ELEM_HPP
#define ELEM_HPP

#include "elem.h"

template< class ETYPE >
void elem_vec::create_new( const int p, const std::vector<int>& nd_clos_ids, node_vec& nodes )
{
    std::shared_ptr<elem> pel = std::make_shared<ETYPE>();
    pel->initialize( p, nd_clos_ids, nodes );
    int idx = m_elems.size();
    m_elems.push_back(pel);
    m_pid2idx[p] = idx;
    m_idx2pid[idx] = p;
}


#endif

