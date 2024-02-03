#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/so3.hpp"

//编程验证四元数和旋转矩阵的旋转是相同的

using namespace Eigen;
using namespace std;

int main() {
    std::cout << "Hello, World!" << std::endl;


    Vector3d w(0.01, 0.02, 0.03);  //这是so3

    AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1));     //沿 Z 轴旋转 45 度
    Matrix3d rotation_matrix = rotation_vector.toRotationMatrix();
    Sophus::SO3d SO3_R(rotation_matrix);
    cout << "rotation matrix =\n" << rotation_vector.matrix() << endl;   //用matrix()转换成矩阵
    Sophus::SO3d SO3_updated = SO3_R * Sophus::SO3d::exp(w);  //核心1（指数映射）
    cout << "SO(3) from matrix:\n" << SO3_R.matrix() << endl;
    cout << "SO3_updated q:\n" << Quaterniond(SO3_updated.matrix()).coeffs().transpose() << endl;


    Quaterniond q = Quaterniond(rotation_vector);
    Quaterniond wq = Quaterniond(1,0.5*w(0),0.5*w(1), 0.5*w(2));  //核心2（四元数更新，小量theta相当于w）
    cout << "quaternion from rotation vector = " << q.coeffs().transpose()
         << endl;   // 请注意coeffs的顺序是(x,y,z,w),w为实部，前三者为虚部
    q = q * wq;
    cout << "QUaternion updated q: " << q.coeffs().transpose() << endl;

    return 0;
}
