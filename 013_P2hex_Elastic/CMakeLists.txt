cmake_minimum_required(VERSION 3.18)
set( PRG_NAME "gmsh2plex" )
project( ${PRG_NAME} CXX)
find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET petsc)

add_subdirectory( libs/functions functions )
add_subdirectory( libs/node node )
add_subdirectory( libs/element element )

add_executable(${PRG_NAME} ${PRG_NAME}.cpp )

target_link_libraries(${PRG_NAME} PRIVATE MPI::MPI_CXX PkgConfig::PETSC
    functions node element
)