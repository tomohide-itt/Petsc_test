#include "elem.h"

elem_vec::elem_vec(){}

elem_vec::~elem_vec()
{
    for( int m=0; m<this->size(); m++ ) delete m_elems[m];
    //printf( "%s\n", __FUNCTION__ );
}