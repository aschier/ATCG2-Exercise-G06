#pragma once

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <OpenMesh/Core/Mesh/TriMeshT.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

struct DoublePrecisionTraits: public OpenMesh::DefaultTraits {
  typedef OpenMesh::Vec3d Point;
  typedef OpenMesh::Vec3d Normal;
};

typedef OpenMesh::TriMeshT<OpenMesh::TriMesh_ArrayKernelT<DoublePrecisionTraits>> Mesh;
typedef Eigen::SparseMatrix<double> SparseMatrix;
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic> DiagonalMatrix;
