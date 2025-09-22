set( LIB_NAME "timer" )

set( HDRS timer.h )
set( SRCS timer.cpp )
set( INCLS ${CMAKE_CURRENT_SOURCE_DIR} )

add_library( ${LIB_NAME} STATIC ${HDRS} ${SRCS} )

target_link_libraries( ${LIB_NAME} PRIVATE MPI::MPI_CXX PkgConfig::PETSC )

target_include_directories( ${LIB_NAME} PUBLIC ${INCLS} )