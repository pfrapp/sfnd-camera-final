
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"
#include "processPointClouds.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(const std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait, std::string windowName)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

void showLeadVehicleTailPlane(const std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait, double distance)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // Draw line denoting the tail of the vehicle.
        int y_line = (-distance * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(left, y_line), cv::Point(right, y_line), cv::Scalar(0,0,255), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    std::string windowName = "Lidar points and vehicle tail";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, const std::vector<cv::KeyPoint> &kptsPrev, const std::vector<cv::KeyPoint> &kptsCurr, const std::vector<cv::DMatch> &kptMatches)
{
    // The variable 'boundingBox' is the bounding of of the current frame.
    // In the beginning, when this function is called, both
    // boundingBox.keypoints.size() and boundingBox.kptMatches.size()
    // are zero.

    // The function 'computeTTCCamera' uses the kptMatches which belongs
    // to the current bounding box, not the one which is contained in the
    // data frame.

    // Task: Fill the members 'keypoints' and 'kptMatches' of the boundingBox.

    // Compute the mean of the Euclidean distances of the keypoint
    // matches which belong to this bounding box.
    // The distance is computed based on a vector which points from
    // the previous frame's keypoint to the current frame's keypoint.
    double euclidean_diff_mean = 0.0;
    int N_keypoints = 0;
    for (const cv::DMatch &match : kptMatches) {
        // query is source, train is reference.
        // Here source is the previous, and reference is the current frame.
        const cv::KeyPoint &kpt_previous_frame = kptsPrev[match.queryIdx];
        const cv::KeyPoint &kpt_current_frame = kptsCurr[match.trainIdx];
        if (boundingBox.roi.contains(kpt_current_frame.pt)) {
            N_keypoints++;
            double x_diff = kpt_current_frame.pt.x - kpt_previous_frame.pt.x;
            double y_diff = kpt_current_frame.pt.y - kpt_previous_frame.pt.y;
            // std::cout << "Keypoint " << N_keypoints << ": Previous=(" << kpt_previous_frame.pt.x << ", "
            //     << kpt_previous_frame.pt.y << "), Current=(" << kpt_current_frame.pt.x << ", " <<
            //     kpt_current_frame.pt.y << "), x_diff = " << x_diff << ", y_diff = " << y_diff << "\n";
            double euclidean_diff = std::hypot(x_diff, y_diff);
            euclidean_diff_mean += euclidean_diff;
        }
    }
    euclidean_diff_mean /= N_keypoints;

    // Compute the sample standard deviation.

    double euclidean_diff_std = 0.0;
    for (const cv::DMatch &match : kptMatches) {
        // query is source, train is reference.
        // Here source is the previous, and reference is the current frame.
        const cv::KeyPoint &kpt_previous_frame = kptsPrev[match.queryIdx];
        const cv::KeyPoint &kpt_current_frame = kptsCurr[match.trainIdx];
        if (boundingBox.roi.contains(kpt_current_frame.pt)) {
            double x_diff = kpt_current_frame.pt.x - kpt_previous_frame.pt.x;
            double y_diff = kpt_current_frame.pt.y - kpt_previous_frame.pt.y;
            double euclidean_diff = std::hypot(x_diff, y_diff);
            euclidean_diff_std += (euclidean_diff - euclidean_diff_mean) * (euclidean_diff - euclidean_diff_mean);
        }
    }
    euclidean_diff_std = std::sqrt( euclidean_diff_std / (N_keypoints - 1) );

    // std::cout << "For bbox with id " << boundingBox.boxID << ": Euclidean distance mean=" << euclidean_diff_mean
    //     << ", std=" << euclidean_diff_std << ".\n";

    // Only consider those keypoint matches which have a Euclidean distance
    // deviation which is smaller than two times the standard deviation sigma.

    for (const cv::DMatch &match : kptMatches) {
        // query is source, train is reference.
        // Here source is the previous, and reference is the current frame.
        const cv::KeyPoint &kpt_previous_frame = kptsPrev[match.queryIdx];
        const cv::KeyPoint &kpt_current_frame = kptsCurr[match.trainIdx];
        if (boundingBox.roi.contains(kpt_current_frame.pt)) {
            double x_diff = kpt_current_frame.pt.x - kpt_previous_frame.pt.x;
            double y_diff = kpt_current_frame.pt.y - kpt_previous_frame.pt.y;
            double euclidean_diff = std::hypot(x_diff, y_diff);
            if (std::abs(euclidean_diff - euclidean_diff_mean) < 2.0 * euclidean_diff_std) {
                boundingBox.kptMatches.push_back(match);
                boundingBox.keypoints.push_back(kpt_current_frame);
            } else {
                // std::cout << "Found an outlier with x_diff = " << x_diff << ", y_diff = " << y_diff << "\n";
            }
        }
    }

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC, double &distance)
{
    // According to lesson 2, the formula for the TTC is:
    // TTC = d1 / v0 = d1 * deltaT / (d0 - d1)
    // with
    //  d0      ... Distance from our vehicle to the lead vehicle in the previous frame (index 0)
    //  d1      ... Distance from our vehicle to the lead vehicle in the current frame (index 1)
    //  v0      ... Velocity
    //  deltaT  ... Step size

    // Fit a plane with normal vector n = [1, 0, 0] through both point clouds (previous and current).
    // In order to remove outliers, use RANSAC.
    // The code has been taken from my Lidar submission and has been adjusted
    // in order to account for the data types used in this Camera project.
    linalg::Vector3<double> plane_normal(1.0, 0.0, 0.0);

    //
    // Perform computations for the previous frame.
    //
    ProcessPointClouds<LidarPoint> point_processor;
    pcl::PointCloud<LidarPoint>::Ptr lidar_point_cloud_prev(new pcl::PointCloud<LidarPoint>);
    lidar_point_cloud_prev->points = lidarPointsPrev;
    // Plane coefficients of the lead vehicle's tail for the previous frame.
    linalg::Plane<double> plane_lead_vehicle_previous;

    point_processor.RansacPlaneConstraintNormal(&plane_lead_vehicle_previous, lidar_point_cloud_prev, plane_normal, 200, 0.1);

    // Plot the coefficients
    std::cout << "Plane coefficients of vehicle tail for the previous frame: ";
    std::cout << plane_lead_vehicle_previous.a << ", " << plane_lead_vehicle_previous.b << ", " << plane_lead_vehicle_previous.c << ", " << plane_lead_vehicle_previous.d;
    std::cout << std::endl;

    //
    // Perform computations for the current frame.
    //
    pcl::PointCloud<LidarPoint>::Ptr lidar_point_cloud_current(new pcl::PointCloud<LidarPoint>);
    lidar_point_cloud_current->points = lidarPointsCurr;
    // Plane coefficients of the lead vehicle's tail for the current frame.
    linalg::Plane<double> plane_lead_vehicle_current;

    point_processor.RansacPlaneConstraintNormal(&plane_lead_vehicle_current, lidar_point_cloud_current, plane_normal, 200, 0.1);

    // Plot the coefficients
    std::cout << "Plane coefficients of vehicle tail for the current frame: ";
    std::cout << plane_lead_vehicle_current.a << ", " << plane_lead_vehicle_current.b << ", " << plane_lead_vehicle_current.c << ", " << plane_lead_vehicle_current.d;
    std::cout << std::endl;

    // Compute the distances
    double d1 = plane_lead_vehicle_current.distance(linalg::Vector3<double>(0.0, 0.0, 0.0));
    double d0 = plane_lead_vehicle_previous.distance(linalg::Vector3<double>(0.0, 0.0, 0.0));

    // Compute the time stamp delta.
    double deltaT = 1.0 / frameRate;

    std::cout << "### d0 = " << d0 << ", d1 = " << d1 << ", vehicle approached " << (d0-d1) << ".\n";
    std::cout << "### framerate " << frameRate << " Hz, deltaT = " << deltaT << " s.\n";

    TTC = d1 * deltaT / (d0 - d1);
    distance = d1;
}


void matchBoundingBoxes(const std::vector<cv::DMatch> &matches,
                        std::map<int, int> &bbBestMatches,
                        const DataFrame &prevFrame,
                        const DataFrame &currFrame)
{
    bool bVis = false;

    // Note: When enabling this visualization, make sure to not focus on the ego
    // lane. This can be done by setting maxY to 20.0 (instead of 2.0) in
    // the program's main function.

    if (bVis) {
        // Draw the keypoint matches.
        // query is source, train is reference.
        // Here source is the previous, and reference is the current frame.
        cv::Mat matchImg = prevFrame.cameraImg.clone();
        cv::Mat matchImgEmpty = prevFrame.cameraImg.clone();
        cv::Mat prevImage = prevFrame.cameraImg.clone();
        cv::Mat currImage = currFrame.cameraImg.clone();
        // Also draw the region of interests into the image.
        for (const BoundingBox &bbox : prevFrame.boundingBoxes) {
            const cv::Rect &rt = bbox.roi;
            cv::rectangle(prevImage, rt, cv::Scalar(0, 0, 255), 2);
            cv::putText(prevImage, std::to_string(bbox.boxID), cv::Point(rt.x, rt.y),  cv::FONT_ITALIC, 1.2, cv::Scalar(255, 0, 0), 2);
        }
        for (const BoundingBox &bbox : currFrame.boundingBoxes) {
            const cv::Rect &rt = bbox.roi;
            cv::rectangle(currImage, rt, cv::Scalar(0, 0, 255), 2);
            cv::putText(currImage, std::to_string(bbox.boxID), cv::Point(rt.x, rt.y),  cv::FONT_ITALIC, 1.2, cv::Scalar(255, 0, 0), 2);
        }
        cv::drawMatches(prevImage, prevFrame.keypoints, currImage, currFrame.keypoints, matches,
                        matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS | cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        std::vector<cv::DMatch> empty_matches;
        cv::drawMatches(prevImage, prevFrame.keypoints, currImage, currFrame.keypoints, empty_matches,
                        matchImgEmpty, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS | cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // Overlay
        float opacity = 0.4;
        cv::addWeighted(matchImg, opacity, matchImgEmpty, 1 - opacity, 0, matchImgEmpty);

        string windowName = "Matching keypoints between two camera images";
        cv::namedWindow(windowName, 7);
        cv::imshow(windowName, matchImgEmpty);
    }


    // Create the matrix for the greedy algorithm and fill it.
    // Row count: Number of boxes in the previous frame.
    // Column count: Number of boxes in the current frame.
    int num_boxes_previous_frame = prevFrame.boundingBoxes.size();
    int num_boxes_current_frame = currFrame.boundingBoxes.size();
    // std::cout << "There are " << num_boxes_previous_frame << " boxes in the previous frame.\n";
    // std::cout << "There are " << num_boxes_current_frame << " boxes in the current frame.\n";

    // Note: in cv::Size, the width is the number of columns, and the height is the number of rows.
    cv::Mat keypoint_match_distribution = cv::Mat::zeros(cv::Size(num_boxes_current_frame, num_boxes_previous_frame), CV_32FC1);
    // std::cout << "The matrix has " << keypoint_match_distribution.rows << " rows and " << keypoint_match_distribution.cols << " columns\n";
    
    for (const cv::DMatch &m : matches) {

        for (const BoundingBox &bbox_prev_frame : prevFrame.boundingBoxes) {

            int bbox_prev_frame_id = bbox_prev_frame.boxID;

            // Query is source and the previous frame.
            const cv::Point2f &kpt_prev = prevFrame.keypoints[m.queryIdx].pt;

            // Check if the keypoint is in the previous frame's bounding box
            // which is currently under consideration.
            if (bbox_prev_frame.roi.contains(cv::Point2i((int) kpt_prev.x, (int) kpt_prev.y))) {

                for (const BoundingBox &bbox_curr_frame : currFrame.boundingBoxes) {

                    int bbox_curr_frame_id = bbox_curr_frame.boxID;

                    // Error check.
                    if ((bbox_prev_frame_id < 0) || (bbox_prev_frame_id >= num_boxes_previous_frame)
                        || (bbox_curr_frame_id < 0) || (bbox_curr_frame_id >= num_boxes_current_frame)) {
                        std::cerr << "Error: Invalid box identifier!\n";
                        break;
                    }

                    // Train is reference and the current frame.
                    const cv::Point2f &kpt_curr = currFrame.keypoints[m.trainIdx].pt;
                    if (bbox_curr_frame.roi.contains(cv::Point2i((int) kpt_curr.x, (int) kpt_prev.y))) {
                        keypoint_match_distribution.at<float>(bbox_prev_frame_id, bbox_curr_frame_id) += 1.0f;
                    }

                }

            } // end if (bbox_prev_frame.roi.contains(cv::Point2i((int) kpt_prev.x, (int) kpt_prev.y)))

        }
    }

    // Print the matrix
    auto print_matrix = [](const cv::Mat &mat) {
        for (int rr=0; rr<mat.rows; rr++) {
            for (int cc = 0; cc<mat.cols; cc++) {
                std::cout << mat.at<float>(rr, cc) << ", ";
            }
            std::cout << "\n";
        }
    };
    // std::cout << "keypoint_match_distribution is:\n";
    // print_matrix(keypoint_match_distribution);

    // Helpers to set rows and columns of a float matrix to zero.
    auto zero_row_and_column = [](cv::Mat *mat, int row, int col) {
        // Set the row to zero.
        for (int cc=0; cc<mat->cols; cc++) {
            mat->at<float>(row, cc) = 0.0f;
        }
        // Set the column to zero.
        for (int rr=0; rr<mat->rows; rr++) {
            mat->at<float>(rr, col) = 0.0f;
        }
    };

    // Run the greedy algorithm until the matrix is all zeros.
    while (true) {
        cv::Mat_<typename cv::DataType<float>::value_type>::iterator it
            = std::max_element(keypoint_match_distribution.begin<float>(), keypoint_match_distribution.end<float>());
        // std::cout << "Max element is at x=" << it.pos().x << ", y=" << it.pos().y << " with value " << (*it) << "\n";

        // Number of keypoint matches that support this bounding box match.
        // This value is not used, but might be interesting to look at.
        float number_keypoint_matches = (*it);
        if ((*it) < 1.0f) {
            // If we are here, we have found all bounding box matches.
            break;
        }
        int bbox_prev_frame_id = it.pos().y;
        int bbox_curr_frame_id = it.pos().x;
        // std::cout << "Found a match! Previous frame box ID " << it.pos().y
        //     << " --> Current frame box ID " << bbox_curr_frame_id << ", supported by "
        //     << number_keypoint_matches << " keypoint matches.\n";
        
        // 'first' of the matches-map corresponds to the previous frame.
        // 'second' of the matches-map corresponds to the current frame.
        bbBestMatches.insert(std::pair<int, int>(bbox_prev_frame_id, bbox_curr_frame_id));

        // x is the column, y is the row.
        // row is the previous frame, column is the current frame.
        // Therefore x is the current frame, y is the previous frame.
        zero_row_and_column(&keypoint_match_distribution, it.pos().y, it.pos().x);
    }
    // Done with the greedy bounding box matching.

    if (bVis) {
        show3DObjects(prevFrame.boundingBoxes, cv::Size(20.0, 25.0), cv::Size(2000, 2000), false, "Previous frame");
        show3DObjects(currFrame.boundingBoxes, cv::Size(20.0, 25.0), cv::Size(2000, 2000), true, "Current frame");
    }

}
