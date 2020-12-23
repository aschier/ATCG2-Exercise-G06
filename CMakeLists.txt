cmake_minimum_required(VERSION 2.8.10)
project(ATCG2-Mesh-Exercise)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(OpenMesh)

add_executable(diffusion
	diffusion.cpp
)
target_compile_definitions(diffusion PUBLIC -D_USE_MATH_DEFINES -DOM_STATIC_BUILD)
target_link_libraries(diffusion PUBLIC
	OpenMeshCore
)
target_include_directories(diffusion PUBLIC
	eigen3
)

add_executable(poisson
	poisson.cpp
)
target_compile_definitions(poisson PUBLIC -D_USE_MATH_DEFINES -DOM_STATIC_BUILD)
target_link_libraries(poisson PUBLIC
	OpenMeshCore
)
target_include_directories(poisson PUBLIC
	eigen3
)
