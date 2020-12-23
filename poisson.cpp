#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMeshT.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <Eigen/Eigen>
#include <iostream>
#include <iomanip>

#include "defines.h"
#include "laplace.hpp"

// Analytic function, used for comparision and for the values on the domain boundary
// defined on the unit square [0,1][0,1]
inline double u_analytic(double x, double y, double max_y) {
	int k = 2;
	return sin(k * M_PI * x / max_y) * sinh(k * M_PI * (1.0 - y / max_y));
}

int main(int argc, char **argv) {
	Mesh mesh;
	if(argc != 2) {
		std::cerr << "Syntax: " << argv[0] << " filename.obj" << std::endl;
		exit(1);
	}
	OpenMesh::IO::read_mesh(mesh, argv[1]);
	SparseMatrix laplace = build_laplace(mesh);

	// We solve for Lu = 0 on the whole domain, so rhs is just 0.
	Eigen::VectorXd rhs(mesh.n_vertices());
	rhs.setZero();

	// Find the dimensions of the square
	double min_y = std::numeric_limits<double>::infinity();
	double max_y = -std::numeric_limits<double>::infinity();
	for(auto vh : mesh.vertices()) {
		min_y = std::min(min_y, mesh.point(vh)[1]);
		max_y = std::max(max_y, mesh.point(vh)[1]);
	}
	// Set the analytic boundary conditions on the boundary of the domain
	std::vector<std::pair<Eigen::Index, double>> boundary_conditions; // Vertex index and fixed value
	for(auto vh : mesh.vertices()) {
		double x = mesh.point(vh)[0];
		double y = mesh.point(vh)[1];
		if(mesh.is_boundary(vh)) {
			boundary_conditions.push_back({vh.idx(), u_analytic(x, y, max_y)});
		}
	}

	// First we substract the contribution of the columns with boundary conditions from the right hand side
	std::set<Eigen::Index> boundary_indices; // used for more efficient lookup which vertex has boundary conditions
	for(auto bc : boundary_conditions) {
		boundary_indices.insert(bc.first);
		rhs -= laplace.col(bc.first) * bc.second;
	}
	// and set the right hand side to the boundary value
	for(auto bc : boundary_conditions) {
		rhs[bc.first] = bc.second;
	}

	// We now eliminate the column entries for the boundary vertices.
	// Because we have a sparse matrix, we use a special iterator, which only
	// visits existing entries and skips sparse zeros.
	for(int k = 0; k < laplace.outerSize(); ++k) {
		for(Eigen::SparseMatrix<double>::InnerIterator it(laplace, k); it; ++it) {
			if(boundary_indices.find(it.col()) != boundary_indices.end() || boundary_indices.find(it.row()) != boundary_indices.end()) {
				it.valueRef() = 0.0;
			}
		}
	}
	// Afterward we set the diagonal entry of the row to 1.0, so the result in the u vector is the boundary value
	for(auto bc : boundary_conditions) {
		laplace.coeffRef(bc.first, bc.first) = 1.0;
	}
	// Compress the laplace, so the new entries set to 0 are eliminated from the sparse matrix
	laplace.makeCompressed();

	Eigen::SparseLU<SparseMatrix> solver;
	solver.compute(laplace);
	if(solver.info() != Eigen::ComputationInfo::Success) {
		// Errors like, e.g., a singular matrix because of a wrong Laplace operator or
		// a correct Laplace operator for a mesh with degenerate edges or triangles.
		std::cerr << "Solver error: " << solver.lastErrorMessage() << std::endl;
	}
	Eigen::VectorXd u_numeric = solver.solve(rhs);

	// Visualize the result
	mesh.request_vertex_colors();
	double max_abs_value = std::max(u_numeric.maxCoeff(), -u_numeric.minCoeff());
	for(auto vh : mesh.vertices()) {
		if(u_numeric[vh.idx()] > 0) {
			// Positive values are red
			mesh.set_color(vh, {u_numeric[vh.idx()] / max_abs_value * 255, 0, 0});
		} else {
			// Negative values are blue
			mesh.set_color(vh, {0, 0, -u_numeric[vh.idx()] / max_abs_value * 255});
		}
	}
	OpenMesh::IO::Options wopt;
	wopt += OpenMesh::IO::Options::VertexColor;
	OpenMesh::IO::write_mesh(mesh, "poisson.ply", wopt);

	// Calculate the numeric error
	double residuum = 0.0;
	for(auto vh : mesh.vertices()) {
		double x = mesh.point(vh)[0];
		double y = mesh.point(vh)[1];
		residuum += pow(u_numeric[vh.idx()] - u_analytic(x, y, max_y), 2);
	}
	residuum /= mesh.n_vertices();
	residuum = sqrt(residuum);

	std::cerr << mesh.n_vertices() << " " << residuum << std::endl;
}