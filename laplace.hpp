#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include "defines.h"

inline OpenMesh::Vec3d from_barycentric_coordinates(OpenMesh::Vec3d v1, OpenMesh::Vec3d v2, OpenMesh::Vec3d v3, OpenMesh::Vec3d bary) {
	assert(abs(1.0 - bary.l1_norm()) < 1e-6);
	return v1 * bary[0] + v2 * bary[1] + v3 * bary[2];
}

inline OpenMesh::Vec3d circumcenter(const OpenMesh::Vec3d v1, const OpenMesh::Vec3d v2, const OpenMesh::Vec3d v3) {
	OpenMesh::Vec3d bary;
	double a = (v2 - v3).norm();
	double b = (v3 - v1).norm();
	double c = (v1 - v2).norm();
	bary[0] = a * a * (b * b + c * c - a * a);
	bary[1] = b * b * (c * c + a * a - b * b);
	bary[2] = c * c * (a * a + b * b - c * c);
	double sum = bary[0] + bary[1] + bary[2];
	bary /= sum;
	return from_barycentric_coordinates(v1, v2, v3, bary);
}

inline SparseMatrix build_laplace(Mesh &mesh) {

	int ill_centered = 0;
	for(auto fh : mesh.faces()) {
		bool found = false;
		for(auto hit = mesh.cfh_ccwbegin(fh); hit != mesh.cfh_ccwend(fh); hit++) {
			if(mesh.calc_sector_angle(*hit) >= M_PI / 2.0) {
				ill_centered++;
				break;
			}
		}
	}
	if(ill_centered > 0) {
	std::cerr << ill_centered << " ill-entered triangle(s) found. Results will be inaccurate." << std::endl;
	}

	SparseMatrix d0(mesh.n_edges(), mesh.n_vertices());
	SparseMatrix d1(mesh.n_faces(), mesh.n_edges());
	DiagonalMatrix star0(mesh.n_vertices());
	DiagonalMatrix star1(mesh.n_edges());
	SparseMatrix laplace;

	///// TODO: calculate the DEC laplacian, by explicitely calculating the neccessary primal and dual
	//          derivative operators and the hodge stars.
	// Hints:
	// - For calculating the dual areas, sum the contributions of the triangles in the one-ring around the primal vertex.
	// - Take care with the sign of the laplacian coming from the signs of the DEC operators,
	// - Take care when handling the mesh boundaries. The diffusion process should have divergence 0 (no sources and sinks),
	//   so each row needs to sum to 0, even on the mesh boundaries where edges have only one neighbor triangle.
	// - Prefer to store Eigen Matrices / Vectors expclicitely with the right result type (e.g. Eigen::VectorXd) to make
	//   sure Eigen dervices the right types. The auto keyword and conversions on the same linke like M1.transpose() * M2
	//   tend to fail to determine the right types in Eigen.

	///// 

	// Sanity checks

	assert(star1.rows() == star1.cols() && star1.rows() == mesh.n_edges());
	assert(star0.rows() == star0.cols() && star0.rows() == mesh.n_vertices());
	assert(d0.rows() == mesh.n_edges() && d0.cols() == mesh.n_vertices());

	double total_area = 0.0;
	for(auto fh : mesh.faces()) {
		total_area += mesh.calc_face_area(fh);
	}

	if(abs(star0.diagonal().sum() - total_area) > 1e-6) {
		std::cerr << "Error: Wrong area in hodge star *_0: " << star0.diagonal().sum() << std::endl;
		std::cerr << "Mesh area: " << total_area << std::endl;
	}

	assert(laplace.rows() == mesh.n_vertices() && laplace.cols() == mesh.n_vertices());
	for(int i = 0; i < laplace.rows(); i++) {
		assert(abs(laplace.row(i).sum()) < 1e-6);
		assert(laplace.coeff(i, i) <= 0);
	}

	return laplace;
}
