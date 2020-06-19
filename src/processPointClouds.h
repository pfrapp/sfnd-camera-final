// PCL lib Functions for processing point clouds 

//
// processPointClouds.h
//
// Note: I took this file from my Lidar project, because it
// already contains a nice RANSAC implementation for
// plane fitting.
// That's exactly what is needed here for processing
// the Lidar data.
//
// I removed the PCL dependencies.
//

#ifndef PROCESSPOINTCLOUDS_H_
#define PROCESSPOINTCLOUDS_H_

#include <iostream> 
#include <string>  
#include <vector>
#include <unordered_set>
#include <ctime>
#include <chrono>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "dataStructures.h"

namespace linalg {

template<typename T>
class Vector3 {
    public:
    union {
        struct { T x, y, z; };
        T data[3];
    };

    Vector3(T x, T y, T z) : x(x), y(y), z(z) {}

    Vector3<T> operator+(const Vector3& v_other) const {
        Vector3<T> v_result(this->x + v_other.x, this->y + v_other.y, this->z + v_other.z);
        return v_result;
    }

    Vector3<T> operator-(const Vector3& v_other) const {
        Vector3<T> v_result(this->x - v_other.x, this->y - v_other.y, this->z - v_other.z);
        return v_result;
    }

    T dot(const Vector3& v) const {
        T inner_product = 0.0;
        for (int ii=0; ii<3; ii++) {
            inner_product += this->data[ii] * v.data[ii];
        }
        return inner_product;
    }

    //! Compute the 2-norm
    T norm() const {
        T norm_squared = this->dot(*this);
        return std::sqrt(norm_squared);
    }

    //! Normalize without check for division by zero
    void normalize() {
        T n = this->norm();
        for (int ii=0; ii<3; ii++) {
            this->data[ii] /= n;
        }
    }

    //! Cross product
    Vector3<T> cross(const Vector3& v) const {
        Vector3<T> outer_product(
            this->y * v.z - this->z * v.y,
            this->z * v.x - this->x * v.z,
            this->x * v.y - this->y * v.x
        );
        return outer_product;
    }

};

template<typename T>
std::ostream& operator<<(std::ostream& out, const Vector3<T> v) {
    out << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return out;
}

template<typename T>
class Plane {
    public:
    T a, b, c, d;

    //! Compute the distance of a point to the plane
    T distance(const Vector3<T>& v) {
        T dist = a*v.x + b*v.y + c*v.z + d;
        return fabs(dist);
    }
};

} // namespace linalg


namespace clustering {


// Structure to represent node of kd tree
struct Node
{
	std::vector<float> point;
	int id;
	Node* left;
	Node* right;

	Node(std::vector<float> arr, int setId)
	:	point(arr), id(setId), left(NULL), right(NULL)
	{}
};

enum class KdBoxState : int {
	BOX_IS_EXCLUSIVLY_LEFT = 0,
	BOX_IS_EXCLUSIVLY_RIGHT = 1,
	BOX_GOT_INTERSECTED = 2
};

// A box for a Kdtree.
// Dim is the dimension (i.e., Dim=2 for the quiz and 3 for the real-world).
// T is either float or double.
template<typename T, int Dim>
struct KdBox {
	// Variable for the dimension.
	int N{Dim};

	// Center of the box (the target point).
	std::vector<T> center;

	// Distance tolerance.
	T distanceTolerance;

	// Minimum values for each dimension
	std::array<T, Dim> lower_bounds;
	// Maximum values for each dimension
	std::array<T, Dim> upper_bounds;

	// ctor
	KdBox(const std::vector<T> target, T distTol) : center(target), distanceTolerance(distTol) {
		for (int d = 0; d < Dim; d++) {
			lower_bounds[d] = center[d] - distanceTolerance;
			upper_bounds[d] = center[d] + distanceTolerance;
		}
	}

	// Check if a given point is within the box or not.
	bool isPointWithinBox(const std::vector<T>& point) const {
		for (int d = 0; d < Dim; d++) {
			if (point[d] < lower_bounds[d]) {
				return false;
			}
			if (point[d] > upper_bounds[d]) {
				return false;
			}
		}
		return true;
	}

	// Compute a point's distance to the box center.
	T distanceToBoxCenter(const std::vector<T> &pt) const {
		T distance = 0.0;
		for (int d = 0; d < Dim; d++) {
			T delta = pt[d] - this->center[d];
			distance += delta*delta;
		}
		return std::sqrt(distance);
	}

	// Check if a point is within the distance tolerance.
	bool isPointWithinDistanceTolerance(const std::vector<T>& pt) const {
		T distance = this->distanceToBoxCenter(pt);
		if (distance > this->distanceTolerance) {
			return false;
		} else {
			return true;
		}
	}

	// Check the position of a box wrt/ a plane which is defined by
	// one constant coordinate.
	// For instance, the plane (or 2D-line) could be defined by x = 5.3.
	// If the box is completely within the space with x < 5.3, then the box
	// is said to be on the left.
	// If it is completely within the space with x > 5.3, then the box
	// is said to be on the right.
	// If neither is the case, the plane (or line) intersects the box.
	//
	// The input to this function is the point under interest along with
	// the dimension. For instance, in the example above, the point could
	// be (5.3, 1.8, -3.1) with current dimension = 0 (for x; 1=y; 2=z).
	KdBoxState getBoxState(const std::vector<T>& point_on_separating_plane, int current_dimension) const {
		T plane_coordinate = point_on_separating_plane[current_dimension];
		if (upper_bounds[current_dimension] < plane_coordinate) {
			return KdBoxState::BOX_IS_EXCLUSIVLY_LEFT;
		} else if (lower_bounds[current_dimension] > plane_coordinate) {
			return KdBoxState::BOX_IS_EXCLUSIVLY_RIGHT;
		} else {
			return KdBoxState::BOX_GOT_INTERSECTED;
		}
	}

	// It turns out that this "exclusivly-left" vs. "exclusivly-right" vs. "intersecting"
	// thinking is not good.
	// It is better (see solution) to just be concerned about whether or not
	// a box is (partly) within one half space.
	// That is all we need to know, as we then know that we need to consider this half space.
	bool isBoxInLeftHalfSpace(const std::vector<T>& point_on_separating_plane, int current_dimension) const {
		T plane_coordinate = point_on_separating_plane[current_dimension];
		if (lower_bounds[current_dimension] > plane_coordinate) {
			return false; 	// The box is exclusivly-right
		} else {
			return true;	// The box might at least be partly in the left half space.
		}
	}

	bool isBoxInRightHalfSpace(const std::vector<T>& point_on_separating_plane, int current_dimension) const {
		T plane_coordinate = point_on_separating_plane[current_dimension];
		if (upper_bounds[current_dimension] < plane_coordinate) {
			return false; 	// The box is exclusivly-left
		} else {
			return true;	// The box might at least be partly in the right half space.
		}
	}
};

template<typename T, int Dim>
struct KdTree
{
    // Variable for the dimension.
	int N{Dim};

	Node* root;

	KdTree()
	: root(NULL)
	{}

	void insert(std::vector<T> point, int id)
	{

		if (root == NULL) {
			root = new Node(point, id);
			return;
		}

		Node *n = root;
		Node *parent = NULL;
		int depth = 0;
		bool attach_to_left = false;
		while (n != NULL) {
			parent = n;
			// Current dimension
			// cd == 0: Split along x
			// cd == 1: Split along y
            // cd == 2: Split along z
			int cd = depth % this->N;
			if (point[cd] < n->point[cd]) {
				// Take the left branch
				n = n->left;
				attach_to_left = true;
			} else {
				// Take the right branch
				n = n->right;
				attach_to_left = false;
			}

			// Increase the depth
			depth++;
		}
		// Now n is NULL: create a new node and attach it to the parent.
		n = new Node(point, id);
		if (attach_to_left) {
			parent->left = n;
		} else {
			parent->right = n;
		}

	}

	// \param node Node under consideration. Can be NULL.
	void searchHelper(std::vector<int> *ids, int *nodes_touched, const KdBox<T, Dim>& box, const Node *node, int depth) {
		if (node == NULL) {
			return;
		}
		// std::cout << "Checking node " << node->id << ".\n";
		(*nodes_touched)++;	 // Dereferencing has a lower operator priority than suffix incrementing!
		// std::cout << "*nodes_touched = " << *nodes_touched << ".\n";

		// Easy access to the point.
		const std::vector<T>& point = node->point;

		// First check if the current node is within the box.
		// If so, compute the distance to the box center (i.e., the target),
		// and if it is smaller than the tolerance add it to 'ids'.

		if (box.isPointWithinBox(point)) {
			// Only call this if the point is within the box, as this
			// is cheaper.
			if (box.isPointWithinDistanceTolerance(point)) {
				ids->push_back(node->id);
				// std::cout << "Node " << node->id << " is within the tolerance.\n";
				// Earlier, before refactoring: Check both child nodes.
				// This is actually wrong. Just because this point is within the box,
				// it does not mean that the child nodes need to be.
				// That is because they can be somewhere completely different in the space.
				// All we know about them is that they are on one of either two sides
				// of a plane.
			}
		}

		// Now check the state of the box with respect to the separating plane.
		int current_dimension = depth % box.N;
		if (box.isBoxInLeftHalfSpace(point, current_dimension)) {
			searchHelper(ids, nodes_touched, box, node->left, depth+1);
		}
		if (box.isBoxInRightHalfSpace(point, current_dimension)) {
			searchHelper(ids, nodes_touched, box, node->right, depth+1);
		}

	}

	// return a list of point ids in the tree that are within distance of target
	std::vector<int> search(std::vector<T> target, T distanceTol)
	{
		std::vector<int> ids;

		// Start with the root node.
		Node *node = this->root;

		// std::cout << "Looking for target (" << target[0] << ", " << target[1] << ") with tolerance " << distanceTol << ".\n";
		int nodes_touched = 0;

		// Create a box around the target point.
		KdBox<T, Dim> box(target, distanceTol);

		searchHelper(&ids, &nodes_touched, box, node, 0);
		// std::cout << "Touched " << nodes_touched << " nodes in total.\n";

		return ids;
	}
	

};




// When you call this function, you guarantee that the point (addressed by 'point_idx') is in the vicinity or proximity
// of the cluster 'cluster'.
template<typename T, int Dim>
void proximity(std::vector<int> *cluster, int point_idx, int *remaining_point_count, const std::vector<std::vector<T>>& points, std::vector<bool> *processed, KdTree<T, Dim> *tree, T distanceTol) {
	// Is this necessary? Or not? Or does it harm?
	// It is included in the slides, but not in the pseudo-code in the text.
	// This is actually checked _before_ calling the proximity function.
	// Commenting this code in does not change the (visible?) program behavior.
	// if ((*processed)[point_idx]) {
	// 	return;
	// }

	// Mark the point as processed.
	(*processed)[point_idx] = true;

	// Add the point to the cluster.
	cluster->push_back(point_idx);
	// std::cout << "Added point " << point_idx << " to the current cluster.\n";

	// One point less remaining.
	(*remaining_point_count)--;

	// Find all nearby points (that's where the Kd-tree comes in).
	std::vector<int> nearby = tree->search(points[point_idx], distanceTol);
	// std::cout << "Point " << point_idx << " has " << nearby.size() << " nearby points: ";
	// for (int nearby_point_idx : nearby) {
		// std::cout << nearby_point_idx << ", ";
	// }
	// std::cout << std::endl;

	// Iterate over all nearby points.
	for (int nearby_point_idx : nearby) {
		// Do not consider this point if it has already been processed.
		if ((*processed)[nearby_point_idx]) {
			continue;
		}

		// Find all points that belong to this cluster in a recursive manner.
		// As these points have been obtained by the tree's search method, we know
		// that they are in the proximity.
		proximity<T, Dim>(cluster, nearby_point_idx, remaining_point_count, points, processed, tree, distanceTol);
	}
	return;
}

template<typename T, int Dim>
std::vector<std::vector<int>> euclideanCluster(const std::vector<std::vector<T>>& points, KdTree<T, Dim> *tree, T distanceTol)
{

	std::vector<std::vector<int>> clusters;

	// Hold a boolean for each point which denotes whether or not the
	// point has already been processed.
	int point_count = points.size();
	int remaining_point_count = point_count;
	std::vector<bool> processed(point_count, false);
	// std::cout << "Starting the Euclidean clustering. There are " << point_count << " points in total.\n";

	// Loop over all points
	for (int point_idx = 0; point_idx < point_count; point_idx++) {
		// std::cout << "Looking at point " << point_idx << " at (" << points[point_idx][0] << ", " << points[point_idx][1] << ").\n";
		// If there are no points left, exit early.
		if (remaining_point_count == 0) {
			// std::cout << "No points left -- exiting early." << std::endl;
			break;
		}

		// Do not consider this point if it has already been processed.
		if (processed[point_idx]) {
			continue;
		}

		// Create a new cluster.
		std::vector<int> new_cluster;
		// std::cout << "Creating a new cluster initiated with point " << point_idx << " at (" << points[point_idx][0] << ", " << points[point_idx][1] << ").\n";

		// Find all points that belong to this cluster in a recursive manner.
		// As this point initiates a new cluster, we know that it is in the proximity.
		proximity<T, Dim>(&new_cluster, point_idx, &remaining_point_count, points, &processed, tree, distanceTol);

		// Add the new cluster to our clusters.
		clusters.push_back(new_cluster);
	}

 
	return clusters;

}


} // namespace clustering

// Non-PCL point cloud, with an interface similar to the PCL point cloud
// in order to reduce the code.
// I decided to call the namespace 'pcl' in order to change
// as few as possible of the existing code.
namespace pcl {

// Point cloud.
template<typename PointT>
struct PointCloud {
	using Ptr = std::shared_ptr<PointCloud<PointT>>;

	std::vector<PointT> points;
};

// Point cloud indices.
struct PointIndices {
	using Ptr = std::shared_ptr<PointIndices>;

	std::vector<int> indices;
};

// Extractor.
template<typename PointT>
class ExtractIndices {
	typename PointCloud<PointT>::Ptr point_cloud_ptr_;
	typename PointIndices::Ptr point_indices_ptr_;
	bool select_negative_{false};

	public:
	void setInputCloud(typename PointCloud<PointT>::Ptr cloud) {
		point_cloud_ptr_ = cloud;
	}

	void setIndices(typename PointIndices::Ptr inds) {
		point_indices_ptr_ = inds;
	}

	void setNegative(bool neg) {
		select_negative_ = neg;
	}

	void filter(PointCloud<PointT> &cloud) {
		cloud.points.clear();
		// Number of points in the input cloud
		int n = point_cloud_ptr_->points.size();
		// This is a boolean vector which contains 'true' for every
		// point to extract, and 'false' for every point to discard.
		// We need to use integers (instead of bool) though. See here:
		// https://stackoverflow.com/questions/34079390/range-for-loops-and-stdvectorbool
		std::vector<int> inds_to_extract(n, 0);
		if (select_negative_) {
			// Select negative indices
			for (int &val : inds_to_extract) {
				val = 1;
			}
			for (int ii = 0; ii < point_indices_ptr_->indices.size(); ii++) {
				int idx = point_indices_ptr_->indices[ii];
				if (idx >= 0 && idx < n) {
					inds_to_extract[idx] = 0;
				}
			}
		} else {
			// Select positive indices
			for (int ii = 0; ii < point_indices_ptr_->indices.size(); ii++) {
				int idx = point_indices_ptr_->indices[ii];
				if (idx >= 0 && idx < n) {
					inds_to_extract[idx] = 1;
				}
			}
		}

		for (int ii=0; ii<n; ii++) {
			if (inds_to_extract[ii]) {
				cloud.points.push_back(point_cloud_ptr_->points[ii]);
			}
		}
	}

};

template<typename PointT>
void getMinMax3D(const PointCloud<PointT> &cloud, PointT &minPoint, PointT &maxPoint);

// Specialization for our Lidar point type.
template<>
void getMinMax3D(const PointCloud<LidarPoint> &cloud, LidarPoint &minPoint, LidarPoint &maxPoint);

}

template<typename PointT>
class ProcessPointClouds {
public:

    //constructor
    ProcessPointClouds();
    //deconstructor
    ~ProcessPointClouds();

    void numPoints(typename pcl::PointCloud<PointT>::Ptr cloud);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud);

    //! This is the function that uses my own plane RANSAC
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> MySegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold);

    //! My Plane RANSAC
    std::unordered_set<int> RansacPlane(linalg::Plane<double> *ptr_plane,
                                    typename pcl::PointCloud<PointT>::Ptr cloud,
								    int maxIterations,
									float distanceTol);

    //! My own clustering function
    std::vector<typename pcl::PointCloud<PointT>::Ptr> MyClustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize);

    Box BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster);
  
};
#endif /* PROCESSPOINTCLOUDS_H_ */
