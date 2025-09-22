set( LIB_NAME "functions" )

set( HDRS functions.h )
set( SRCS functions.cpp )
set( INCLS ${CMAKE_CURRENT_SOURCE_DIR} )

add_library( ${LIB_NAME} STATIC ${HDRS} ${SRCS} )

target_link_libraries( ${LIB_NAME} PRIVATE MPI::MPI_CXX PkgConfig::PETSC node element timer )

target_include_directories( ${LIB_NAME} PUBLIC ${INCLS} )