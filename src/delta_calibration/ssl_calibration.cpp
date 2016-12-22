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
#include <random>

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
using namespace cv;

#define BALL_DROPS 4
#define MAX_FRAMES 96

void Visualize(const char* filePath,
               double f,
               double px, double py,
               double k1, double k2,
               double p1, double p2,
               double* rotation, double* translation, vector<Vector3d> world_points, Scalar color, double radius = 3);

bool DoubleEquals(double x, double y) {
    return fabs(x-y) < .000005;
}

// Removes distortion based on the opencv model
template <class T> Eigen::Matrix<T,3,1> Distort(
                                                const Eigen::Matrix<T,3,1>& point,
                                                T k1,
                                                T k2,
                                                T p1,
                                                T p2) {
    
    T x = point[0] / point[2];
    T y = point[1] / point[2];
    
    T r2 = x*x + y*y;
    T x_1 = x * (T(1)+k1*(r2) + k2 * (r2 * r2));
    T y_1 = y * (T(1)+k1*(r2) + k2 * (r2 * r2));
    
    x_1 = x_1 + (T(2)*p1*x*y + p2 * (r2 + T(2) * (x*x)));
    y_1 = y_1 + (p1 * (r2 + T(2) * (y*y)) + T(2) * p2 * x * y);
    x = x_1;
    y = y_1;
    
    return (Eigen::Matrix<T, 3, 1>(
                                   x,
                                   y,
                                   T(1)));
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
    //   cout << "World " << endl;
    //   cout << point << endl;
    Eigen::Quaterniond q = Eigen::Quaterniond(rotation[0],rotation[1],rotation[2],rotation[3]);
    point = q * point;
    point[0] = point[0] + translation[0];
    point[1] = point[1] + translation[1];
    point[2] = point[2] + translation[2];
    
    //   cout << "Extrinsics" << endl;
    //   cout << point << endl;
    point = Distort(point, k1, k2, p1, p2);
    //   cout << "Undistored" << endl;
    //   cout << point << endl;
    const Eigen::Matrix<double,3,3> K = BuildIntrinsics(f, px, py);
    point = K * point;
    //   cout << "Intrinsics" << endl;
    //   cout << point << endl;
    return (Eigen::Matrix<double, 2, 1>(
                                        point[0],
                                        point[1]));
}

template <class T> T GetFallZ(T frame_num, T frame_rate, T z_0, T v_i, T t_0) {
    T time = t_0 + frame_num/frame_rate;
    T z = z_0 + v_i * time + (T(1)/T(2)) * T(-9800) *
    (time * time);
    return z;
}

template <class T> T GetXY(T frame_num, T frame_rate, T xy_0, T v_i, T t_0) {
    T xy =  xy_0 + v_i * (t_0 + (frame_num / frame_rate));
    return xy;
}

struct ReprojectionError {
    ReprojectionError(const Vector2d& image_point,
                      const Vector3d& world_point,
                      const int id,
                      const int frame_num,
                      const int frame_rate,
                      const double* const rotation,
                      const double* const translation) :
    image_point(image_point),
    world_point(world_point),
    id(id),
    frame_num(frame_num),
    frame_rate(frame_rate),
    rotation(rotation),
    translation(translation){}
    
    template <class T>
    bool operator()(const T* const intrinsics,
                    const T* const k1,
                    const T* const k2,
                    const T* const p1,
                    const T* const p2,
                    const T* const X,
                    const T* const Y,
                    const T* const Z,
                    const T* const V,
                    const T* const t0,
                    T* residuals) const {
        
        T x0 = X[0], y0 = Y[0],z0=Z[0];
        T vx = V[0], vy = V[1], vz = V[2];
        //        vy = T(0);
        //        vx = T(0);
        // Transform by extrinstic matrix transform=
        T point_t[] = {
            T(world_point.x()),
            //GetXY(T(frame_num), T(frame_rate), x0, vx, t0[0]),
            
            //T(world_point.y()),
            GetXY(T(frame_num), T(frame_rate), y0, vy, t0[0]),
            
            GetFallZ(T(frame_num), T(frame_rate), z0, vz, t0[0])};
        //T(world_point.z())};
        
        T rot[] = {T(rotation[0]), T(rotation[1]), T(rotation[2]), T(rotation[3])};
        T trans[] = {T(translation[0]), T(translation[1]), T(translation[2])};
        Eigen::Matrix<T, 3, 1> world_point_t;
        T transformed_point[3];
        ceres::QuaternionRotatePoint(rot, point_t, transformed_point);
        transformed_point[0] = transformed_point[0] + trans[0];
        transformed_point[1] = transformed_point[1] + trans[1];
        transformed_point[2] = transformed_point[2] + trans[2];
        world_point_t << transformed_point[0], transformed_point[1], transformed_point[2];
        
        // Distort
        world_point_t = Distort(world_point_t, k1[0], k2[0], p1[0], p2[0]);
        const Eigen::Matrix<T,3,3> K = BuildIntrinsics(intrinsics[0], intrinsics[1], intrinsics[2]);
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
                                       const Vector3d& world_point,
                                       const int id,
                                       const int frame_num,
                                       const int frame_rate,
                                       const double* const rotation,
                                       const double* const translation) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2,3,1,1,1,1,1,1,1,3,1>(
                                                                                          new ReprojectionError(image_point, world_point, id, frame_num, frame_rate, rotation, translation)));
    }
    
    const Eigen::Vector2d image_point;
    const Vector3d world_point;
    const int id;
    const int frame_num;
    const int frame_rate;
    const double* const rotation;
    const double* const translation;
};

void SSLCalibrate(const vector<pair<Vector2d, int>>& image_locations,
                  const vector<Vector3d> world_locations,
                  const int frame_rate,
                  double* intrinsics,
                  double* k1,
                  double* k2,
                  double* p1,
                  double* p2,
                  double* rotation,
                  double* translation,
                  vector<double*> X,
                  vector<double*> Y,
                  vector<double*> Z,
                  vector<double*> V,
                  vector<double*> t_0
                  ) {
    
    // Tolerance for RMSE.
    static const double kToleranceError = 0;
    // The maximum number of overall iterations.
    static const int kMaxIterations = 10000;
    // The maximum number of repeat iterations while the RMSE is unchanged.
    static const int kMaxRepeatIterations = 5;
    double rmse = 1000000;
    double last_rmse = 1000010;
    vector<double> residuals;
    
    // Loop until no longer changes or error is sufficiently small
    // (not sure if necessary)
    for (int iteration = 0, repeat_iteration = 0;
         (iteration < kMaxIterations &&
          repeat_iteration < kMaxRepeatIterations &&
          rmse > kToleranceError) ||
         iteration < 6;
         ++iteration) {
        
        if (DoubleEquals(rmse, last_rmse)) {
            repeat_iteration++;
        } else {
            repeat_iteration = 0;
        }
        last_rmse = rmse;
        
        // Construct CERES problem
        ceres::Problem problem;
        for(size_t i = 0; i < world_locations.size() - 1; i++) {
            Eigen::Vector2d image_point = image_locations[i].first;
            int id = image_locations[i].second;
            Vector3d world_point = world_locations[i];
            ceres::CostFunction* cost_function;
            cost_function = ReprojectionError::Create(image_point, world_point,
                                                      id, (i % MAX_FRAMES), frame_rate, rotation, translation);
            problem.AddResidualBlock(cost_function,
                                     NULL,
                                     intrinsics,
                                     k1,
                                     k2,
                                     p1,
                                     p2,
                                     X[id],
                                     Y[id],
                                     Z[id],
                                     V[id],
                                     t_0[id]);
            
            problem.SetParameterLowerBound(intrinsics, 0, 0);
            problem.SetParameterUpperBound(X[id], 0, 5000);
            problem.SetParameterLowerBound(X[id], 0, 0);
            problem.SetParameterUpperBound(Y[id], 0, 0);
            problem.SetParameterLowerBound(Y[id], 0, -3000);
            
            problem.SetParameterLowerBound(Z[id], 0, 0);
            problem.SetParameterLowerBound(t_0[id], 0, 0);
            problem.SetParameterUpperBound(V[id], 2, 0);
            
            if(iteration < 3) {
                problem.SetParameterBlockConstant(k1);
                problem.SetParameterBlockConstant(k2);
                problem.SetParameterBlockConstant(p1);
                problem.SetParameterBlockConstant(p2);
            }
            if(iteration < 4) {
                problem.SetParameterBlockConstant(Z[id]);
            }
            
            if(iteration < 7) {
                problem.SetParameterBlockConstant(V[id]);
                problem.SetParameterBlockConstant(t_0[id]);
            }
        }
        
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_type = ceres::TRUST_REGION;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        //     options.minimizer_progress_to_stdout = true;
        options.num_threads = 6;
        options.num_linear_solver_threads = 6;
        //     options.linear_solver_type = ceres::SPARSE_SCHUR;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        //     std::cout << summary.FullReport() << "\n";
        rmse = sqrt(summary.final_cost / static_cast<double>(summary.num_residuals));
        cout << "RMSE: " << rmse << ",iteration: " << iteration << endl;
    }
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

void Visualize(const char* filePath,
               double f,
               double px, double py,
               double k1, double k2,
               double p1, double p2,
               double* rotation, double* translation,
               vector<Vector3d> world_points,
               Scalar color,double radius) {
    
    Mat image;
    
    image = imread( filePath, 1 );
    
    if ( !image.data ) {
        printf("No image data \n");
        return;
    }
    
    for(size_t i = 0; i < world_points.size(); ++i) {
        Vector3d wp = world_points[i];
        Vector2d i_point = WorldToImage(wp,
                                        f, px, py,
                                        k1, k2, p1, p2,
                                        rotation,
                                        translation);
        circle(image,Point(i_point.x(),i_point.y()),radius,color);
    }
    
    //Find projection of points on lines
    //(0,0) --> (0,-2960)
    double wx = 10, wy= -200;
    while(wy > -2360) {
        Vector3d wp(wx, wy,0);
        Vector2d i_point = WorldToImage(wp,
                                        f, px, py,
                                        k1, k2, p1, p2,
                                        rotation,
                                        translation);
        circle(image,Point(i_point.x(),i_point.y()),radius,color);
        //circle(image,Point(i_point.x(),i_point.y()),2,Scalar(255,0,0));
        wy = wy - 10;
    }
    //(0,0) --> (4550,0)
    wx = 10, wy= -200;
    while(wx < 4550) {
        Vector3d wp(wx, wy,0);
        Vector2d i_point = WorldToImage(wp,
                                        f, px, py,
                                        k1, k2, p1, p2,
                                        rotation,
                                        translation);
        circle(image,Point(i_point.x(),i_point.y()),radius,color);
        //circle(image,Point(i_point.x(),i_point.y()),2,Scalar(255,0,0));
        wx = wx + 10;
    }
    //(0,-2960) --> (4550,-2960)
    wx = 10, wy= -2360;
    while(wx < 4550) {
        Vector3d wp(wx, wy,0);
        Vector2d i_point = WorldToImage(wp,
                                        f, px, py,
                                        k1, k2, p1, p2,
                                        rotation,
                                        translation);
        circle(image,Point(i_point.x(),i_point.y()),radius,color);
        //circle(image,Point(i_point.x(),i_point.y()),2,Scalar(255,0,0));
        wx = wx + 10;
    }
    //(4550,0) --> (4550,-2960)
    wx = 4550, wy= -200;
    while(wy >= -2360) {
        Vector3d wp(wx, wy,0);
        Vector2d i_point = WorldToImage(wp,
                                        f, px, py,
                                        k1, k2, p1, p2,
                                        rotation,
                                        translation);
        circle(image,Point(i_point.x(),i_point.y()),radius,color);
        //circle(image,Point(i_point.x(),i_point.y()),2,Scalar(255,0,0));
        wy = wy - 10;
    }
    
    IplImage ipltemp=image;
    IplImage* image2=cvCloneImage(&ipltemp);
    //cv::imwrite("../simImage.jpg", image);
    cvSaveImage("../simImage.jpg", image2);
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    waitKey(0);
}


// Begin samer code for loading csv file and computing set of world points

pair<int, double> LoadImageLocations(const string& tracking_file, int dropID,
                                     vector<pair<Vector2d, int>>* image_locations) {
    
    //ScopedFile fid(tracking_file, "r");
    //CHECK_NOTNULL(fid());
    Vector2d image_point(0, 0);
    int frame_number = 0;
    pair<Vector2d, int> frame_info = make_pair(image_point, frame_number);
    
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
            //cout << stoi(line) << endl;
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
        frame_info.second = dropID;
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
    }
}

// End samer code for loading csv file and computing set of world points

double GetZ(int frame_number, double max_time, double frame_rate, double accel) {
    double z = accel/2 * pow((max_time - (frame_number / frame_rate)),2);
    return z;
}

void GenerateBallDrop(vector<pair<Vector2d, int>> *image_locations, vector<Vector3d> *world_locations) {
    
    // Generation Parameters
    double focalLength = 402.546386;
    double pX = 467.818228;
    double pY = 318.845535;
    double rotation[] = {0.147620,0.988969,0.011928,-0.002536};
    double translation[] = {-2138.434348,-1911.213759,2707};
    double g = 9800;
    double k1 = 0.05 , k2 = .02, p1 = -0.01, p2 = 0.01;
    
    double frame_rate = 142;
    double StartPointX[] = {2000,100,4000,3500, 500};
    double StartPointY[] = {-1500,-1000,-250,-1000, -700};
    
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,5.0);
    bool isNoise = false;
    
    for(int pt = 0; pt < BALL_DROPS; ++pt) {
        for(int i = 0; i < MAX_FRAMES; ++i) {
            double x_noise = 0;
            double y_noise = 0;
            if (isNoise) {
                x_noise = distribution(generator);
                y_noise = distribution(generator);
                cout << "x_noise: " << x_noise << " Y_noise: " << y_noise << endl;
            }
            
            // Get the height that the ball will drop from given the MAX_FRAMES
            // and frame rate
            double greatest_z = GetZ(0,  (MAX_FRAMES - 1) / frame_rate,
                                     frame_rate,g);
            double z = GetFallZ(double(i), frame_rate, greatest_z, 0.0, 0.0);
            std::pair<Vector2d,int> temp;
            
            Vector2d image_point = WorldToImage(Vector3d(StartPointX[pt],StartPointY[pt],z),
                                                focalLength,
                                                pX,
                                                pY,
                                                k1,
                                                k2,
                                                p1,
                                                p2,
                                                rotation,
                                                translation);
            
            image_point[0] += x_noise;
            image_point[1] += y_noise;
            temp.first = image_point;
            temp.second = pt;
            image_locations->push_back(temp);
            world_locations->push_back(Vector3d(StartPointX[pt],StartPointY[pt],z));
            
            // Displaying the generated points
            if (pt <  1) {
                cout << "Last z: " << (*world_locations)[i * (pt + 1)] << endl;
                cout << "Calculated z: " << GetFallZ(double(i), 142.0, greatest_z, 0.0, 0.0) << endl;
            }
        }
    }
    cout << "Generating ball drop visualization..." << endl;
    
    IplImage* imgScribble = cvCreateImage(cvSize(1100, 500), 8, 3);
    cvZero(imgScribble);
    cvSaveImage("../simImage.jpg", imgScribble);
    
    Visualize("../simImage.jpg",
              focalLength, pX, pY, k1,k2,p1,p2,
              rotation, translation,*world_locations, Scalar(0,0,255),5);
}

void GenerateParabola(vector<pair<Vector2d, int>> *image_locations, vector<Vector3d> *world_locations) {
    
    // Generation Parameters
    double focalLength = 402.546386;
    double pX = 467.818228;
    double pY = 318.845535;
    double rotation[] = {0.147620,0.988969,0.011928,-0.002536};
    double translation[] = {-2138.434348,-1911.213759,2707};
    double g = 9800;
    double k1 = 0.05 , k2 = .02, p1 = -0.01, p2 = 0.01;
    
    double frame_rate = 142;
    double StartPointX[] = {2000,100,4000,3500, 500};
    double StartPointY[] = {-1500,-1000,-250,-1000, -700};
    
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,5.0);
    bool isNoise = false;
    
    double vx = 100;
    double vy = 100;
    
    for(int pt = 0; pt < BALL_DROPS; ++pt) {
        ofstream outputFile;
        // create a name for the file output
        std::ostringstream filename;
        filename << "parabolaDrop_" << pt << ".csv";
        outputFile.open(filename.str());
        
        for(int i = 0; i < MAX_FRAMES; ++i) {
            double x_noise = 0;
            double y_noise = 0;
            if (isNoise) {
                x_noise = distribution(generator);
                y_noise = distribution(generator);
                cout << "x_noise: " << x_noise << " Y_noise: " << y_noise << endl;
            }
            
            // Get the height that the ball will drop from given the MAX_FRAMES
            // and frame rate
            double x = GetXY(double(i), frame_rate, StartPointX[pt], vx, 0.0);
            double y = GetXY(double(i), frame_rate, StartPointY[pt], vy, 0.0);
            double greatest_z = GetZ(0,  (MAX_FRAMES - 1) / frame_rate,
                                     frame_rate,g);
            double z = GetFallZ(double(i), frame_rate, greatest_z, 0.0, 0.0);
            std::pair<Vector2d,int> temp;
            outputFile << x << "," << y << "," << z << endl;
            
            Vector2d image_point = WorldToImage(Vector3d(x,y,z),
                                                focalLength,
                                                pX,
                                                pY,
                                                k1,
                                                k2,
                                                p1,
                                                p2,
                                                rotation,
                                                translation);
            
            image_point[0] += x_noise;
            image_point[1] += y_noise;
            temp.first = image_point;
            temp.second = pt;
            image_locations->push_back(temp);
            world_locations->push_back(Vector3d(x,y,z));
            
            // Displaying the generated points
            if (pt <  1) {
                cout << "Last z: " << (*world_locations)[i * (pt + 1)] << endl;
                cout << "Calculated z: " << GetFallZ(double(i), 142.0, greatest_z, 0.0, 0.0) << endl;
            }
        }
        outputFile.close();
    }
    cout << "Generating ball drop visualization..." << endl;
    
    IplImage* imgScribble = cvCreateImage(cvSize(1100, 500), 8, 3);
    cvZero(imgScribble);
    cvSaveImage("../simImage.jpg", imgScribble);
    
    Visualize("../simImage.jpg",
              focalLength, pX, pY, k1,k2,p1,p2,
              rotation, translation,*world_locations, Scalar(0,0,255),5);
}

int main(int argc, char **argv) {
    
    google::InitGoogleLogging(argv[0]);
    
    vector<pair<Vector2d, int>> image_locations;
    vector<Vector3d> world_locations;
    pair<int, int> episode_conditions;
    // Block for loading image points from a file, probably outdated
    for(int i=1;i < argc; ++i) {
        char* filePath = argv[i];
        episode_conditions = LoadImageLocations(argv[i], i-1, &image_locations);
        int p_x = (int)image_locations.back().first(0);
        int p_y = (int)image_locations.back().first(1);
        
        Eigen::Quaterniond q = Eigen::Quaterniond(0.147620,0.988969,0.011928,-0.002536);
        Vector3d translation = Vector3d(-2138.434348,-1911.213759,2707);
        
        Vector3d contact_point = getWorldPosition(q,translation,402.546386,Vector2d(467.818228, 318.845535),Vector2d(p_x,p_y));
        
        cout << "Contact Point " << i << ": (" << contact_point.x() << ","
        << contact_point.y() << ","
        << contact_point.z() << ")" << endl;
        CalculateWorldLocations(episode_conditions, contact_point, &world_locations);
    }
    
    double* extrinsic_rotation = new double[4];
    double* extrinsic_translation = new double[3];
    extrinsic_rotation[0] = 0.147620;
    extrinsic_rotation[1] = 0.988969;
    extrinsic_rotation[2] = 0.011928;
    extrinsic_rotation[3] = -0.002536;
    extrinsic_translation[0] = -2138.434348;
    extrinsic_translation[1] = -1911.213759;
    extrinsic_translation[2] = 2707;
    double intrinsics[] = {0, 0, 0};
    double k1, k2, p1, p2;
    k1 = k2 = p1 = p2 = 0;
    
    vector<pair<Vector2d, int>> testImageLocations;
    vector<Vector3d> testWorldLocations;
    
    GenerateBallDrop(&testImageLocations,&testWorldLocations);
    //GenerateParabola(&testImageLocations,&testWorldLocations);
    
    vector<double*> X;
    vector<double*> Y;
    vector<double*> Z;
    vector<double*> V;
    vector<double*> T_0;
    
    double *v_0 = new double[BALL_DROPS];
    double *z_0 = new double[BALL_DROPS];
    double* t_0 = new double[BALL_DROPS];
    // Initializing values of parameters to optimize
    for(int drop = 0; drop < BALL_DROPS; ++drop) {
        double *x = new double(0);
        double *y = new double(0);
        double *z = new double(0);
        double *t0 = new double(0);
        double *v = new double[3];
        v[0] = 100;
        v[1] = 100;
        v[2] = 0;
        X.push_back(x);
        Y.push_back(y);
        Z.push_back(z);
        V.push_back(v);
        T_0.push_back(t0);
    }
    
    SSLCalibrate(testImageLocations,
                 testWorldLocations,
                 142,
                 intrinsics,
                 &k1,
                 &k2,
                 &p1,
                 &p2,
                 extrinsic_rotation,
                 extrinsic_translation,
                 X,
                 Y,
                 Z,
                 V,
                 T_0);
    
    cout << "Generating visualization for estimated parameters..." << endl;
    Visualize("../simImage.jpg",
              intrinsics[0], intrinsics[1], intrinsics[2], k1, k2, p1,p2,
              extrinsic_rotation, extrinsic_translation, testWorldLocations, Scalar(255,255,255),2);
    
    cout << "Intrinsics and Distortion: " << intrinsics[0] << " "
    << intrinsics[1] << " " << intrinsics[2] << " "
    << k1 << " " << k2 << " " << p1 << " " << p2 << endl;
    for(int drop = 0; drop < BALL_DROPS; ++drop) {
        cout << "X: " << X[drop][0] << "\tY: " << Y[drop][0] << "\tZ: " << Z[drop][0] << endl
        << "Vx: " << int(V[drop][0]) << "\tVy: " << int(V[drop][1]) << "\tVz: " << int(V[drop][2]) << endl
        << "\tT_0: " << int(T_0[drop][0]) <<  endl << endl;
    }
    
    return 0;
}
