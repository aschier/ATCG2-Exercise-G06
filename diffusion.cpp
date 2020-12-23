#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMeshT.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <Eigen/Eigen>
#include <iostream>
#include <iomanip>

#include "defines.h"
#include "laplace.hpp"

int main(int argc, char **argv) {
	Mesh mesh;
	if(argc != 2) {
		std::cerr << "Syntax: " << argv[0] << " filename.obj" << std::endl;
		exit(1);
	}
	OpenMesh::IO::read_mesh(mesh, argv[1]);

	// find bottom left vertex, independed of the ordering in the input file.
	int start_idx = 0;
	double min_dist = std::numeric_limits<double>::infinity();
	for(auto vh: mesh.vertices()) {
		double dist = mesh.point(vh).norm();
		if(dist < min_dist) {
			start_idx = vh.idx();
			min_dist = dist;
		}
	}

	SparseMatrix laplace = build_laplace(mesh);

	assert(mesh.n_vertices() > 0);
	Eigen::VectorXd u0(mesh.n_vertices());
	u0.setZero();
	u0[start_idx] = 1.0;

	double cfl_timestep = 0.0;
	///// TODO: Calculate and output the maximal time step, for which the CFL condition holds

	/////

	double delta_small = 0.9 * cfl_timestep;
	double delta_large = 1000 * cfl_timestep;
	double time = 20.0;
	int steps_small = time/delta_small;
	int steps_large = time/delta_large;

	std::cerr << "Delta (small): " << delta_small << std::endl;
	std::cerr << "Steps: " << steps_small << std::endl;
	std::cerr << "Delta (large): " << delta_large << std::endl;
	std::cerr << "Steps: " << steps_large << std::endl;

	Eigen::VectorXd u_explicit_small = u0;
	Eigen::VectorXd u_explicit_large = u0;
	Eigen::VectorXd u_implicit_large = u0;

	///// TODO: Implement diffusion using the the explicit euler method with the two different
	//          time step sizes and the implicit euler method with the large time step.

	/////

	mesh.request_vertex_colors();
	OpenMesh::IO::Options wopt;
	wopt += OpenMesh::IO::Options::VertexColor;

	// Write explicit euler result
	double max_value = u_explicit_small.maxCoeff();
	for(auto vh: mesh.vertices()) {
		mesh.set_color(vh, {u_explicit_small[vh.idx()]/max_value * 255, 0, 0});
	}
	OpenMesh::IO::write_mesh(mesh, "explicit_small.ply", wopt);
	std::cout << "Mass (explicit small timesteps): " << u_explicit_small.sum() << std::endl;

	// Write explicit euler result with large timestep
	max_value = u_explicit_large.maxCoeff();
	for(auto vh: mesh.vertices()) {
		mesh.set_color(vh, {u_explicit_large[vh.idx()]/max_value * 255, 0, 0});
	}
	OpenMesh::IO::write_mesh(mesh, "explicit_large.ply", wopt);
	std::cout << "Mass (explicit large timesteps): " << u_explicit_large.sum() << std::endl;

	// Write implicit euler result
	max_value = u_implicit_large.maxCoeff();
	for(auto vh: mesh.vertices()) {
		mesh.set_color(vh, {u_implicit_large[vh.idx()]/max_value * 255, 0, 0});
	}
	OpenMesh::IO::write_mesh(mesh, "implicit_large.ply", wopt);
	std::cout << "Mass (implicit large timestep): " << u_implicit_large.sum() << std::endl;
}
