set( LIB_NAME "element" )

set( HDRS elem.h trian6.h hexl27.h )
set( SRCS elem.cpp trian6.cpp hexl27.cpp )
set( INCLS ${CMAKE_CURRENT_SOURCE_DIR} )

add_library( ${LIB_NAME} STATIC ${HDRS} ${SRCS} )

target_link_libraries( ${LIB_NAME} PRIVATE MPI::MPI_CXX PkgConfig::PETSC node )

target_include_directories( ${LIB_NAME} PUBLIC ${INCLS} )