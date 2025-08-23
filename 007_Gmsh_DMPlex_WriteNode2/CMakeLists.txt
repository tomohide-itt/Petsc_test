cmake_minimum_required(VERSION 3.16)
project(gmsh2plex_vtk C)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET petsc)

find_package(MPI REQUIRED)

add_executable(gmsh2plex_vtk gmsh2plex_vtk.c)
target_link_libraries(gmsh2plex_vtk PRIVATE PkgConfig::PETSC MPI::MPI_C)