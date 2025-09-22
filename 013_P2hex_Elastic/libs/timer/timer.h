#ifndef TIMER_H
#define TIMER_H
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

class timer
{
public:
    void start();
    void stop( const char* ss );
private:
    static const bool flag_timer = true;
    double m_start;
};

extern timer tmr;

#endif

