// PCL lib Functions for processing point clouds 

//
// processPointClouds.cpp
//
// Note: I took this file from my Lidar project, because it
// already contains a nice RANSAC implementation for
// plane fitting.
// That's exactly what is needed here for processing
// the Lidar data.
//
// I removed the PCL dependencies.
//

#include <algorithm>    // for sort
#include "processPointClouds.h"

namespace pcl {


// Specialization for our Lidar point type.
template<>
void getMinMax3D(const PointCloud<LidarPoint> &cloud, LidarPoint &minPoint, LidarPoint &maxPoint) {
	// Number of points in the point cloud.
	int n = cloud.points.size();
	if (n > 0) {
		LidarPoint lp = cloud.points[0];
		minPoint = maxPoint = lp;
	}
	if (n > 1) {
		for (int ii=1; ii<n; ii++) {
			const LidarPoint &lp = cloud.points[ii];
			minPoint.x = std::min(minPoint.x, lp.x);
			maxPoint.x = std::max(maxPoint.x, lp.x);
			minPoint.y = std::min(minPoint.y, lp.y);
			maxPoint.y = std::max(maxPoint.y, lp.y);
			minPoint.z = std::min(minPoint.z, lp.z);
			maxPoint.z = std::max(maxPoint.z, lp.z);
		}
	}
}

}

//constructor:
template<typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}


//de-constructor:
template<typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}


template<typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud) 
{
    // Create two new point clouds, one cloud with obstacles and other with segmented plane
    pcl::ExtractIndices<PointT> extract;

    extract.setInputCloud(cloud);
    extract.setIndices(inliers);

    typename pcl::PointCloud<PointT>::Ptr planeCloud(new pcl::PointCloud<PointT>);
    typename pcl::PointCloud<PointT>::Ptr obstacleCloud(new pcl::PointCloud<PointT>);

    // We can fill the plane cloud (which are the inliers) with the following two lines
    // of code:
    // extract.setNegative(false);
    // extract.filter(*planeCloud);
    // However, just for the sake of experimenting, we can also do it by looping over
    // the indices
    for (int index : inliers->indices) {
        planeCloud->points.push_back(cloud->points[index]);
    }

    // Fill the outliers (which are the obstacles)
    extract.setNegative(true);
    extract.filter(*obstacleCloud);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(obstacleCloud, planeCloud);
    return segResult;
}



template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::MySegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold) {
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // Plane coefficients
    linalg::Plane<double> plane;

    // Inlieres
    std::unordered_set<int> inliers_set = RansacPlane(&plane, cloud, maxIterations, distanceThreshold);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    for (auto index : inliers_set) {
        inliers->indices.push_back(index);
    }

    // Plot the coefficients
    std::cout << "Plane coefficients: ";
    std::cout << plane.a << ", " << plane.b << ", " << plane.c << ", " << plane.d;
    std::cout << std::endl;

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "my own plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers,cloud);
    return segResult;
}

template<typename PointT>
std::unordered_set<int> ProcessPointClouds<PointT>::RansacPlane(
                                    linalg::Plane<double> *ptr_plane,
                                    typename pcl::PointCloud<PointT>::Ptr cloud,
								    int maxIterations,
									float distanceTol)
{
	std::unordered_set<int> inliersResult;
	srand(time(NULL));

	int point_count = cloud->points.size();
	std::cout << "Plane RANSAC: There are " << point_count << " points in the PC\n";

    linalg::Plane<double> plane;

	// For max iterations 
	while(maxIterations--) {
		//std::cout << "* Iterations left: " << maxIterations << std::endl;

		// Randomly sample subset and fit plane
		std::unordered_set<int> inliers;
		while (inliers.size() < 3) {
			int r = rand();
			r = r % point_count;
			// Elements in the unordered set are guaranteed to be unique
			inliers.insert(r);
		}
		// Create a vector easy access
		std::vector<int> inliers_vec(inliers.begin(), inliers.end());
		//std::cout << "* Chose indices: " << inliers_vec[0] << ", "
	//			 << inliers_vec[1] << ", " << inliers_vec[2] << std::endl;
		
		// Compute the plane parameters
		// First of all, grab the 3 points on the plane
		linalg::Vector3<double> p1(cloud->points[inliers_vec[0]].x, cloud->points[inliers_vec[0]].y, cloud->points[inliers_vec[0]].z);
		linalg::Vector3<double> p2(cloud->points[inliers_vec[1]].x, cloud->points[inliers_vec[1]].y, cloud->points[inliers_vec[1]].z);
		linalg::Vector3<double> p3(cloud->points[inliers_vec[2]].x, cloud->points[inliers_vec[2]].y, cloud->points[inliers_vec[2]].z);
		auto v1 = p2 - p1;
		auto v2 = p3 - p1;
		auto n = v1.cross(v2);
		n.normalize();

		
		plane.a = n.x;
		plane.b = n.y;
		plane.c = n.z;
		plane.d = -1.0 * (n.dot(p1));
		// The plane is now fit and ready to use

		// Measure distance between every point and the fitted plane
		for (int ii=0; ii<point_count; ii++) {
			// Check if the point was one of the three points to
			// create the plane. If so, continue.
			if (inliers.count(ii) > 0) {
				continue;
			}

			linalg::Vector3<double> pt(cloud->points[ii].x, cloud->points[ii].y, cloud->points[ii].z);
			double distance = plane.distance(pt);

			// If distance is smaller than threshold count it as inlier
			if (distance <= distanceTol) {
				inliers.insert(ii);
			}
		}

		// Check if this is our new consensus set
		if (inliers.size() > inliersResult.size()) {
			//std::cout << "* The consensus set has now " << inliers.size() << " inliers.\n";
			inliersResult = inliers;

            // This is the new best plane.
            if (ptr_plane != nullptr) {
                *ptr_plane = plane;
            }
		}

	}

	//std::cout << "The final consensus set has " << inliersResult.size()
//			<< " inliers.\n";



	// Return indicies of inliers from fitted line with most inliers
	// This is the consensus set
	return inliersResult;

}


template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::MyClustering(
        typename pcl::PointCloud<PointT>::Ptr cloud,
        float clusterTolerance,
        int minSize,
        int maxSize)
{
    // We do not use maxSize in our own custom clustering routine.
    (void) maxSize;

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    // This is a vector of point clouds (in the form of point cloud pointers).
    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    // Number of points in the cloud.
    int total_point_count = cloud->points.size();

    // Copy the data from the cloud into a std::vector<std::vector<float>>.
    std::vector<std::vector<float>> points(total_point_count);
    for (int ii=0; ii<total_point_count; ii++) {
        points[ii].push_back(cloud->points[ii].x);
        points[ii].push_back(cloud->points[ii].y);
        points[ii].push_back(cloud->points[ii].z);
    }

    // Create our own Kd-tree with type float and dimension 3.
    clustering::KdTree<float, 3> *tree = new clustering::KdTree<float, 3>;
    for (int ii=0; ii<total_point_count; ii++) {
        tree->insert(points[ii], ii);
    }

    // Do the Euclidean clustering.
    std::vector<std::vector<int>> cluster_indices = clustering::euclideanCluster<float, 3>(points, tree, clusterTolerance);
    int cluster_count = cluster_indices.size();
    // std::cout << "clustering found " << cluster_count << " clusters.\n";

    // Sort the clusters according to size (because this is what the PCL also seems to do).
    std::vector<std::pair<int,int>> cluster_sizes(cluster_count);
    for (int ii=0; ii<cluster_count; ii++) {
        cluster_sizes[ii].first = cluster_indices[ii].size();
        // Needed for permutation (this is the actual index).
        cluster_sizes[ii].second = ii;
    }

    // Sort in descending order (largest cluster first).
    std::sort(cluster_sizes.begin(), cluster_sizes.end(), [](const std::pair<int,int>& a, const std::pair<int,int>& b) { return (a.first > b.first); });

    // Create the point clouds for the clusters
    for (int ii=0; ii<cluster_count; ii++) {
        // Use sorted sequence.
        // const std::vector<int>& inds_for_this_cluster = cluster_indices[ii];
        const std::vector<int>& inds_for_this_cluster = cluster_indices[cluster_sizes[ii].second];

        if (inds_for_this_cluster.size() >= minSize) {
            typename pcl::PointCloud<PointT>::Ptr clusterCloud(new pcl::PointCloud<PointT>());
            for (int idx : inds_for_this_cluster) {
                clusterCloud->points.push_back(cloud->points[idx]);
            }
            clusters.push_back(clusterCloud);
        }

    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "my own clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}


template<typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}

// Create a template instantiation for the LidarPoint type.
// https://stackoverflow.com/questions/2152002/how-do-i-force-a-particular-instance-of-a-c-template-to-instantiate
template class ProcessPointClouds<LidarPoint>;
