// ROS INCLUDES
#include <ros/ros.h>
// Ceres Includes
#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <fstream>
#include <cstdio>
#include "../../../../../cobot/cobot_linux/src/shared/util/helpers.h"

using std::size_t;
using std::vector;
using namespace std;
using Eigen::Vector3d;
using Eigen::Vector2f;

template <class T> T CalculateR(Eigen::Matrix<T,3,1> point) {
  return T(0);
}

// Removes distortion based on the opencv model
template <class T> Eigen::Matrix<T,3,1> Undistort(
    const Eigen::Matrix<T,3,1>& point,
    T r,
    T k1,
    T k2,
    T p1,
    T p2) {
  
  T x = point[0];
  T y = point[1];
  
  x = x * (T(1)+(k1*pow(r,2)) + (k2 * pow(r,4)));
  y = y * (T(1)+(k1*pow(r,2)) + (k2 * pow(r,4)));
  
  x = x + (T(2)*p1*x*y + p2 * (pow(r,2) + T(2) * pow(x,2)));
  y = y + (p1 *(pow(r,2) + T(2) * pow(y,2)) + T(2) * p2 * x * y);
  
  return (Eigen::Matrix<T, 3, 1>(
    x,
    y,
    point[2]));    
}

// Build the instrinc camera matrix from parameters
template <class T> Eigen::Matrix<T,3,3> BuildIntrinsics(
    T f,
    T px,
    T py) {
  
  Eigen::Matrix<T,3,3> K;
  K << f,    T(0), px,
       T(0), f,    py,
       T(0), T(0), T(1);
       
  return K;   
}

// Construct a problem using this
struct ReprojectionError {
  ReprojectionError(const Vector3d& image_point,
                    const Vector3d& world_point) :
      image_point(image_point),
      world_point(world_point) {}

  template <class T>
  bool operator()(const T* const f,
                  const T* const px,
                  const T* const py,
                  const T* const k1,
                  const T* const k2,
                  const T* const p1,
                  const T* const p2,
                  const T* const rotation,
                  const T* const translation,
                  T* residuals) const {
                    
    // Transform by extrinstic matrix transform=
    T point_t[] = {T(world_point.x()), T(world_point.y()), T(world_point.z())};
    Eigen::Matrix<T, 3, 1> world_point_t;
    T transformed_point[3];
    ceres::QuaternionRotatePoint(rotation, point_t, transformed_point);
    transformed_point[0] = transformed_point[0] + translation[0];
    transformed_point[1] = transformed_point[1] + translation[1];
    transformed_point[2] = transformed_point[2] + translation[2];
    world_point_t << transformed_point[0], transformed_point[1], transformed_point[2];
    // Undistort
    T r = CalculateR(world_point_t);
    world_point_t = Undistort(world_point_t, r, k1[0], k2[0], p1[0], p2[0]);
    const Eigen::Matrix<T,3,3> K = BuildIntrinsics(f[0], px[0], py[0]);
    // Apply Intrinsics
    world_point_t = K * world_point_t;
    // The error is the difference between the predicted and observed position.
    const Eigen::Matrix<T, 3, 1> image_point_t = image_point.cast<T>();
    residuals[0] =
        (world_point_t[0] - image_point_t[0]);
    residuals[1] =
        (world_point_t[1] - image_point_t[1]);
    residuals[2] =
        (world_point_t[2] - image_point_t[2]);
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const Vector3d& image_point,
                                     const Vector3d& world_point) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 3, 1,1,1,1,1,1,1,4,3>(
        new ReprojectionError(image_point, world_point)));
  }

  const Vector3d image_point;
  const Vector3d world_point;
};

pair<int, double> LoadImageLocations(const string& tracking_file,
                       vector<pair<Vector2f, int>>* image_locations) {
  
  ScopedFile fid(tracking_file, "r");
  CHECK_NOTNULL(fid());
  Vector2f image_point(0, 0);
  int frame_number = 0;
  pair<Vector2f, int> frame_info = make_pair(image_point, frame_number);

  int frame_rate = 0;
  double total_time = 0.0; 
  //TODO: need code here to read in header, and make it so that the while loop
  //starts in the correct place (after the header)
  //TODO: make sure type mismatch doesn't cause problems
  while (fscanf(fid(), "%d,%d,%d\n", &(image_point(0)), 
                                     &(image_point(1)), 
                                     &(frame_number))) {
    frame_info.first = image_point;
    frame_info.second = frame_number;
    image_locations->push_back(frame_info);
  }
  pair<int, double> header = make_pair(frame_rate, total_time);
  return header; 
}

int main(int argc, char **argv) {
  if (argc < 3) {
    cout << "ERROR: need to specify .csv file and trajectory type (drop or toss)"
         << endl;
  }
  

  return 0;
}
