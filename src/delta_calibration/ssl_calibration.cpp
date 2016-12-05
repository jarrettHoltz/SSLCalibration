// ROS INCLUDES
#include <ros/ros.h>
// Ceres Includes
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "glog/logging.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdio>
//#include "../../../../../cobot/cobot_linux/src/shared/util/helpers.h"


using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using std::size_t;
using std::vector;
using namespace std;
using Eigen::Vector3d;
using Eigen::Vector2d;
using Eigen::Vector2f;
using Eigen::Vector3f;

bool DoubleEquals(double x, double y) {
  return fabs(x-y) < .000005;
}

template <class T> T CalculateR(Eigen::Matrix<T,3,1> point) {
  Eigen::Matrix<T,2,1>point2d;
  point2d[0] = T(point[0]) / T(point[2]);
  point2d[1] = T(point[1]) / T(point[2]);
  return T(point2d.norm());
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

Vector2d WorldToImage(Vector3d point,
                      double f,
                      double px,
                      double py,
                      double k1,
                      double k2,
                      double p1,
                      double p2,
                      double* rotation,
                      double* translation) {
  cout << "World " << endl;
  cout << point << endl;
  Eigen::Quaterniond q = Eigen::Quaterniond(rotation[0],rotation[1],rotation[2],rotation[3]);
  point = q * point;
  point[0] = point[0] + translation[0];
  point[1] = point[1] + translation[1];
  point[2] = point[2] + translation[2];
  
  cout << "Extrinsics" << endl;
  cout << point << endl;
  double r = CalculateR(point);
  point = Undistort(point, r, k1, k2, p1, p2);
  cout << "Undistored" << endl;
  cout << point << endl;
  const Eigen::Matrix<double,3,3> K = BuildIntrinsics(f, px, py);
  point = K * point;
  cout << "Intrinsics" << endl;
  cout << point << endl;
  return (Eigen::Matrix<double, 2, 1>(
    point[0],
    point[1]));
}

// Construct a problem using this
struct ReprojectionError {
  ReprojectionError(const Vector2d& image_point,
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
    const Eigen::Matrix<T, 2, 1> image_point_t = image_point.cast<T>();
    residuals[0] =
        (image_point_t[0] - world_point_t[0]);
    residuals[1] =
        (image_point_t[1] - world_point_t[1]);
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const Eigen::Vector2d& image_point,
                                     const Vector3d& world_point) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 1,1,1,1,1,1,1,4,3>(
        new ReprojectionError(image_point, world_point)));
  }

  const Eigen::Vector2d image_point;
  const Vector3d world_point;
};

void SSLCalibrate(const vector<pair<Vector2f, int>>& image_locations,
                  const vector<Vector3d> world_locations,
                  double* f,
                  double* px,
                  double* py,
                  double* k1,
                  double* k2,
                  double* p1,
                  double* p2,
                  double* rotation,
                  double* translation) {
  
  // Tolerance for RMSE.
  static const double kToleranceError = 5;
  // The maximum number of overall iterations.
  static const int kMaxIterations = 100;
  // The maximum number of repeat iterations while the RMSE is unchanged.
  static const int kMaxRepeatIterations = 20;
  double rmse = 1000000;
  double last_rmse = 1000010;
  vector<double> residuals;
  
  // Loop until no longer changes or error is sufficiently small 
  // (not sure if necessary)
  for (int iteration = 0, repeat_iteration = 0;
       iteration < kMaxIterations &&
       repeat_iteration < kMaxRepeatIterations &&
       rmse > kToleranceError;
       ++iteration) {
    
    if (DoubleEquals(rmse, last_rmse)) {
      repeat_iteration++;
    } else {
      repeat_iteration = 0;
    }
    last_rmse = rmse;
    
    // Construct CERES problem
    ceres::Problem problem;
    
    for(size_t i = 0; i < world_locations.size(); i++) {
      Eigen::Vector2f image_pointf = image_locations[i].first;
      Eigen::Vector2d image_point = image_pointf.cast<double>();
      Vector3d world_point = world_locations[i];
      ceres::CostFunction* cost_function;
      cost_function = ReprojectionError::Create(image_point, world_point);
      problem.AddResidualBlock(cost_function,
                              NULL,
                              f,
                              px,
                              py,
                              k1,
                              k2,
                              p1,
                              p2,
                              rotation,
                              translation);
      if(iteration < 5) {
        problem.SetParameterBlockConstant(rotation);
        problem.SetParameterBlockConstant(translation);
      }
      if(iteration < 2) {
        problem.SetParameterBlockConstant(k1);
        problem.SetParameterBlockConstant(k2);
        problem.SetParameterBlockConstant(p1);
        problem.SetParameterBlockConstant(p2);
      }
    }
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
//     options.minimizer_progress_to_stdout = true;
    options.num_threads = 6;
    options.num_linear_solver_threads = 6;
//     options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
//     std::cout << summary.FullReport() << "\n";
    rmse =
        sqrt(summary.final_cost / static_cast<double>(summary.num_residuals));
    cout << "RMSE: " << rmse << endl;
  }
}

// Begin haresh code for finding point of contact in plane

struct CostFunctor {
    template <typename T>
    bool operator()(const T* const intrinsics,
                    const T* const worldPoint,
                    T* residual) const {

        //x1_i - x_i
        T dIX = T(_deltaImage.x());
        //y1_i - y_i
        T dIY = T(_deltaImage.y());

        T quaternion[4];
        quaternion[0] = T(_quaternion[0]);
        quaternion[1] = T(_quaternion[1]);
        quaternion[2] = T(_quaternion[2]);
        quaternion[3] = T(_quaternion[3]);

        T translation[3];
        translation[0] = T(_translation[0]);
        translation[1] = T(_translation[1]);
        translation[2] = T(_translation[2]);

        T world_known[3];
        world_known[0] = T(_worldPoint.x());
        world_known[1] = T(_worldPoint.y());
        world_known[2] = T(0);

        T cam_known[3];
        ceres::QuaternionRotatePoint(quaternion,world_known,cam_known);
        cam_known[0] = cam_known[0] - translation[0];
        cam_known[1] = cam_known[1] - translation[1];
        cam_known[2] = cam_known[2] - translation[2];

        T world_estimate[3];
        world_estimate[0] = worldPoint[0];
        world_estimate[1] = worldPoint[1];
        world_estimate[2] = T(0);

        T cam_estimate[3];
        ceres::QuaternionRotatePoint(quaternion,world_estimate,cam_estimate);
        cam_estimate[0] = cam_estimate[0] - translation[0];
        cam_estimate[1] = cam_estimate[1] - translation[1];
        cam_estimate[2] = cam_estimate[2] - translation[2];

        T dC[3];
        dC[0] = cam_known[0] - cam_estimate[0];
        dC[1] = cam_known[1] - cam_estimate[1];
        dC[2] = cam_known[2] - cam_estimate[2];

        T eI[2];
        eI[0] = intrinsics[0]*dC[0] + intrinsics[1]*dC[2];
        eI[1] = intrinsics[0]*dC[1] + intrinsics[2]*dC[2];

        residual[0] = eI[0] - dIX;
        residual[1] = eI[1] - dIY;
        return true;
    }

    CostFunctor(Eigen::Vector4d quaternion, Eigen::Vector3d translation,
                Eigen::Vector2d deltaImage, Eigen::Vector2d worldPoint)
       :_quaternion(quaternion), _translation(translation),
        _deltaImage(deltaImage), _worldPoint(worldPoint){}

    Eigen::Vector4d _quaternion;
    Eigen::Vector3d _translation;
    Eigen::Vector2d _deltaImage;
    Eigen::Vector2d _worldPoint;
};

struct IntrinsicEstimator {
    template <typename T>
    bool operator()(const T* const intrinsics,
                    T* residual) const {

        T quaternion[4];
        quaternion[0] = T(_quaternion.w());
        quaternion[1] = T(_quaternion.x());
        quaternion[2] = T(_quaternion.y());
        quaternion[3] = T(_quaternion.z());

        T translation[3];
        translation[0] = T(_translation[0]);
        translation[1] = T(_translation[1]);
        translation[2] = T(_translation[2]);

        T world_known[3];
        world_known[0] = T(_worldPoint.x());
        world_known[1] = T(_worldPoint.y());
        world_known[2] = T(0);

        T cam_known[3];
        ceres::QuaternionRotatePoint(quaternion,world_known,cam_known);
        cam_known[0] = cam_known[0] + translation[0];
        cam_known[1] = cam_known[1] + translation[1];
        cam_known[2] = cam_known[2] + translation[2];

        T estimatedImagePoint[2];
        estimatedImagePoint[0] = intrinsics[0]*(cam_known[0] / cam_known[2]) + intrinsics[1];
        estimatedImagePoint[1] = intrinsics[0]*(cam_known[1] / cam_known[2]) + intrinsics[2];
        residual[0] = estimatedImagePoint[0] - T(_imagePoint.x());
        residual[1] = estimatedImagePoint[1] - T(_imagePoint.y());

        return true;
    }
    IntrinsicEstimator(Eigen::Quaterniond quaternion, Eigen::Vector3d translation,
                       Eigen::Vector2d imagePoint, Eigen::Vector2d worldPoint)
             :_quaternion(quaternion), _translation(translation),
              _imagePoint(imagePoint), _worldPoint(worldPoint){
    }

    Eigen::Quaterniond _quaternion;
    Eigen::Vector3d _translation;
    Eigen::Vector2d _imagePoint;
    Eigen::Vector2d _worldPoint;
};

Eigen::Vector2d getImagePosition(Eigen::Quaterniond q, 
                                 Eigen::Vector3d translation,
                                 double focalLength,
                                 Eigen::Vector2d principalPoints,
                                 Eigen::Vector3d worldLocation) {

//    Eigen::Quaterniond q(0.147620,0.988969,0.011928,-0.002536);
//    double qn = q.norm();
//
//    Eigen::Vector3d translation(-2138.434348,-1911.213759,2707);
//
//    double focalLength = 402.546386;
//    Eigen::Vector2d principalPoints(467.818228,318.845535);

    Eigen::Quaterniond p;
    p.w() = 0;
    p.vec() = worldLocation;
    Eigen::Quaterniond rotatedP = q * p * q.inverse();
    Eigen::Vector3d camCoordinate = rotatedP.vec();
    camCoordinate = camCoordinate + translation;

    double x_i = camCoordinate.x()*focalLength / 
                 camCoordinate.z() + principalPoints.x();
    double y_i = camCoordinate.y()*focalLength / 
                 camCoordinate.z() + principalPoints.y();

    return Eigen::Vector2d(x_i,y_i);
}

Eigen::Vector3d getWorldPosition(Eigen::Quaterniond q, 
                                 Eigen::Vector3d translation,
                                 double focalLength,Eigen::Vector2d principalPoints,
                                 Eigen::Vector2d pixelLocation) {

    q = q.inverse();
    q.normalize();

    Eigen::Quaterniond p;
    p.w() = 0;
    p.vec() = Eigen::Vector3d(0,0,0) - translation;
    Eigen::Quaterniond rotatedP = q * p * q.inverse();
    translation = rotatedP.vec();

    Eigen::Vector3d camCoordinate((pixelLocation.x() - principalPoints.x())/focalLength,
                                  (pixelLocation.y() - principalPoints.y())/focalLength, 1);

    p.w() = 0;
    p.vec() = camCoordinate;
    rotatedP = q * p * q.inverse();
    camCoordinate = rotatedP.vec();

    double t = -translation.z() / camCoordinate.z();
    camCoordinate = camCoordinate * t;

    Eigen::Vector3d worldCoordinate = translation + camCoordinate;

    return worldCoordinate;
}

void UnitTesting(double focalLength = 402.546386, 
                 double pX = 467.818228, 
                 double pY = 318.845535,
                 Eigen::Quaterniond q = Eigen::Quaterniond(0.147620,0.988969,0.011928,-0.002536),
                 Eigen::Vector3d translation = Eigen::Vector3d(-2138.434348,-1911.213759,2707)) {

    //Four pixel locations with known world co-ordinates
    //x_i,y_i...
    double imagePoints[] = {18,500,951,518,813,61,163,49};
    //x_w,y_w...
    double worldPoints[] = {0,-2960,4550,-2960,4550,0,0,0};

    double rootMSE_Image2World = 0, rootMSE_World2Image = 0;
    for (int i = 0; i < 4; ++i) {
      Eigen::Vector2d dIp(imagePoints[i * 2], imagePoints[i * 2 + 1]);
      //dIp = dIp - knownImagePoint;
      Eigen::Vector3d wP(worldPoints[i * 2], worldPoints[i * 2 + 1],0);

      Eigen::Vector2d predictedPixel = getImagePosition(q, translation, focalLength, Eigen::Vector2d(pX, pY),

      Eigen::Vector3d(wP.x(), wP.y(), 0));
      std::cout << "Image X: " << dIp.x()
                << " -> " << predictedPixel.x() << "\n"
                << "Image Y: " << dIp.y()
                << " -> " << predictedPixel.y() << "\n\n";
      Eigen::Vector2d diffPixel = dIp - predictedPixel;
      rootMSE_World2Image += diffPixel.norm();

      Eigen::Vector3d predictedWorld = getWorldPosition(q, translation, focalLength, Eigen::Vector2d(pX, pY),
      Eigen::Vector2d(dIp.x(), dIp.y()));
      std::cout << "World X: " << wP.x()
                << " -> " << predictedWorld.x() << "\n"
                << "World Y: " << wP.y()
                << " -> " << predictedWorld.y() << "\n"
                << "World Z: " << wP.z()
                << " -> " << predictedWorld.z() << "\n\n";
      Eigen::Vector3d diffWorld = wP - predictedWorld;
      rootMSE_Image2World += diffWorld.norm();
    }

    rootMSE_Image2World = rootMSE_Image2World / 4;
    rootMSE_World2Image = rootMSE_World2Image / 4;

    std::cout << "RMSE for image to world coordinates: " << rootMSE_Image2World 
              << " centimeters\n" << "RMSE for world to image coordinates: " 
              << rootMSE_World2Image << " pixels\n\n";
}

Eigen::Vector3d getContactPoint(int px, int py) {

    std::cout << "Unit Testing of utility functions without distortion:\n";
    UnitTesting();


    // The variables to solve for with initial values.
    //double initialIntrinsics[3] = {402.546386,467.818228,318.845535};
    double initialIntrinsics[3] = {100,100,100};
    double *intrinsics= new double[3];
    intrinsics[0] = initialIntrinsics[0];
    intrinsics[1] = initialIntrinsics[1];
    intrinsics[2] = initialIntrinsics[2];

    // Build the problem.
    Problem problem;

    //Four pixel locations with known world co-ordinates
    //x_i,y_i...
    double imagePoints[] = {18,500,951,518,813,61,163,49};
    //x_w,y_w...
    double worldPoints[] = {0,-2960,4550,-2960,4550,0,0,0};

    //Extrinsics
    Eigen::Quaterniond  quaternion(0.147620,0.988969,0.011928,-0.002536);
    Eigen::Vector3d translation(-2138.434348,-1911.213759,2707);
    for (int i = 0; i < 4; ++i) {
      Eigen::Vector2d dIp(imagePoints[i*2], imagePoints[i*2+1]);
      //dIp = dIp - knownImagePoint;
      Eigen::Vector2d wP(worldPoints[i*2], worldPoints[i*2+1]);

      CostFunction* cost_function =
              new AutoDiffCostFunction<IntrinsicEstimator, 2, 3>(new
                   IntrinsicEstimator(quaternion,translation,dIp,wP));

        problem.AddResidualBlock(cost_function, NULL, intrinsics);
    }

    // Run the solver!
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n\n";
    std::cout << "focal: " << initialIntrinsics[0]
              << " -> " << intrinsics[0] << "\n"
              << "pX: " << initialIntrinsics[1]
              << " -> " << intrinsics[1] << "\n"
              << "pY: " << initialIntrinsics[2]
              << " -> " << intrinsics[2] << "\n\n";

    std::cout << "Unit Testing of estimated intrinsics:\n";
    UnitTesting(intrinsics[0], intrinsics[1], intrinsics[2]);

    //Point where ball touches the field
    //Eigen::Vector2d knownImagePoint(554,306);
    Eigen::Vector2d knownImagePoint(px, py);

    Eigen::Vector3d estimatedWorldLocation = getWorldPosition(quaternion,translation,intrinsics[0],
                                                     Eigen::Vector2d(intrinsics[1], intrinsics[2]), knownImagePoint);
    std::cout << "Estimated World Location of "
              << "(" << knownImagePoint.x() << "," << knownImagePoint.y() << ")"
              << ":\n(" << estimatedWorldLocation.x() << ","
              << estimatedWorldLocation.y() << "," << estimatedWorldLocation.z()
              << ")\n";

    Eigen::Vector3d estimated_world_coordinates(estimatedWorldLocation.x(),
                                                estimatedWorldLocation.y(),
                                                estimatedWorldLocation.z());

    return estimated_world_coordinates;
}

// End haresh code for finding point of contact in plane

// Begin samer code for loading csv file and computing set of world points

pair<int, double> LoadImageLocations(const string& tracking_file,
                       vector<pair<Vector2f, int>>* image_locations) {
  
  //ScopedFile fid(tracking_file, "r");
  //CHECK_NOTNULL(fid());
  Vector2f image_point(0, 0);
  int frame_number = 0;
  pair<Vector2f, int> frame_info = make_pair(image_point, frame_number);

  int frame_rate;
  int num_frames;
  string full_line;
  string line;
  
  fstream stream(tracking_file);
  
  getline(stream, line);
  frame_rate = stoi(line);
  getline(stream, line);
  num_frames = stoi(line);

  vector<int> raw_data;
  while (getline(stream, full_line)) {
    stringstream iss;
    iss << full_line;
    while (getline(iss, line, ',')) {
      cout << stoi(line) << endl;
      raw_data.push_back(stoi(line));
    }
  }
  CHECK_GT(raw_data.size(), 0);
  CHECK_EQ(raw_data.size()%3, 0);

  for (size_t i = 0; i < raw_data.size(); i+=3) {
    image_point(0) = raw_data[i];
    image_point(1) = raw_data[i+1];
    frame_number = raw_data[i+2];

    frame_info.first = image_point;
    frame_info.second = frame_number;
    cout << "Here" << endl;
    image_locations->push_back(frame_info);
    
  }

  pair<int, int> header = make_pair(frame_rate, num_frames);
  return header; 
}

void CalculateWorldLocations(pair<int, int> episode_conditions,
                             Vector3d contact_point,
                             vector<Vector3d>* world_locations) {
  double g = 9.81; // m/s^2
  Vector3d world_point = contact_point;
  int frame_rate = episode_conditions.first;
  int numframes = episode_conditions.second;
  double t_f = double(numframes) / double(frame_rate);
  for (int i = 0; i < numframes; ++i) {
    world_point.z() = (g/2.0)*(t_f*t_f - (double(i)/double(frame_rate))*(double(i)/double(frame_rate)));
    world_locations->push_back(world_point);
//     cout << world_point << endl;
  }
}

// End samer code for loading csv file and computing set of world points

int main(int argc, char **argv) {
  if (argc < 3) {
    cout << "ERROR: need to specify .csv file and trajectory type (drop or toss)"
         << endl;
  }
  
  google::InitGoogleLogging(argv[0]);

  pair<int, int> episode_conditions;
  vector<pair<Vector2f, int>> image_locations;
  episode_conditions = LoadImageLocations(argv[1], &image_locations);
//   for(size_t i = 0; i < image_locations.size(); i++) {
//     //     Vector2d i_point = WorldToImage(world_locations[i],
//     //                                     f,
//     //                                     px,
//     //                                     py,
//     //                                     k1,
//     //                                     k2,
//     //                                     p1,
//     //                                     p2, 
//     //                                     extrinsic_rotation,
//     //                                     extrinsic_translation);
//     //     cout << " Projected Point: " << endl <<  i_point << endl;
//     cout << "Image Point: " << endl << image_locations[i].first << endl << endl;
//   }
  int p_x = image_locations.back().first(0);
  int p_y = image_locations.back().first(1);
  Vector3d contact_point = getContactPoint(p_x, p_y);

  vector<Vector3d> world_locations;
  CalculateWorldLocations(episode_conditions, contact_point, &world_locations);
  double* extrinsic_rotation = new double[4];
  double* extrinsic_translation = new double[3];
  extrinsic_rotation[0] = 0.147620;
  extrinsic_rotation[1] = 0.988969;
  extrinsic_rotation[2] = 0.011928;
  extrinsic_rotation[3] = -0.002536;
  extrinsic_translation[0] = -2138.434348;
  extrinsic_translation[1] = -1911.213759;
  extrinsic_translation[2] = 2707;
  double f, px, py, k1, k2, p1, p2;
  //TODO: build and solve intrinsic calibration problem
  SSLCalibrate(image_locations,
               world_locations,
               &f,
               &px,
               &py,
               &k1,
               &k2,
               &p1,
               &p2, 
               extrinsic_rotation,
               extrinsic_translation);
  cout << f << " " << px << " " << py << " " << k1 << " " << k2 << " " << p1 << " " << p2 << endl;
  for(size_t i = 0; i < 5; i++) {
    Vector2d i_point = WorldToImage(world_locations[i],
                                    f,
                                    px,
                                    py,
                                    k1,
                                    k2,
                                    p1,
                                    p2, 
                                    extrinsic_rotation,
                                    extrinsic_translation);
    cout << " Projected Point: " << endl <<  i_point << endl;
    cout << "Image Point: " << endl << image_locations[i].first << endl << endl;
  }
  return 0;
}
