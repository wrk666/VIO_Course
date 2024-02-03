#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include "backend/vertex_pose.h"
#include "backend/vertex_point_xyz.h"
#include "backend/edge_reprojection.h"
#include "backend/eigen_types.h"

#include <iostream>

namespace myslam {
namespace backend {

/*    std::vector<std::shared_ptr<Vertex>> verticies_; // 该边对应的顶点
    VecX residual_;                 // 残差
    std::vector<MatXX> jacobians_;  // 雅可比，每个雅可比维度是 residual x vertex[i]
    MatXX information_;             // 信息矩阵
    VecX observation_;              // 观测信息
    */
//边即是误差项，计算残差：i时刻camera->i时刻imu->j时刻imu->j时刻camera的预测，-j时刻camera观测=residual
void EdgeReprojection::ComputeResidual() {
    double inv_dep_i = verticies_[0]->Parameters()[0];//逆深度

    VecX param_i = verticies_[1]->Parameters();//i时刻camera pose
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);//i时刻camera q
    Vec3 Pi = param_i.head<3>();//i时刻camera t

    VecX param_j = verticies_[2]->Parameters();//j时刻camera pose
    Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
    Vec3 Pj = param_j.head<3>();

    Vec3 pts_camera_i = pts_i_ / inv_dep_i;//归一化(从像素到归一化平面度变化x=zu=u/lambda)
    Vec3 pts_imu_i = qic * pts_camera_i + tic;//imu第i时刻的pose
    Vec3 pts_w = Qi * pts_imu_i + Pi;//i时刻world
    Vec3 pts_imu_j = Qj.inverse() * (pts_w - Pj);//Qj，Pj应该是Qij，tij，都是从j时刻变换到i时刻，要从->j则需要Rij^(-1)(Pi-Pj)，即得j时刻imu坐标
    Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);//利用外参Tic，Tic^(-1)*i_P_j，转换到camera系下c_P_j

    double dep_j = pts_camera_j.z();//取深度
    //归一化之后计算
    residual_ = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();   /// J^t * J * delta_x = - J^t * r
//    residual_ = information_ * residual_;   // remove information here, we multi information matrix in problem solver
}

void EdgeReprojection::SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_) {
    qic = qic_;
    tic = tic_;
}

void EdgeReprojection::ComputeJacobians() {
    double inv_dep_i = verticies_[0]->Parameters()[0];

    VecX param_i = verticies_[1]->Parameters();
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Vec3 Pi = param_i.head<3>();//i时刻的t

    VecX param_j = verticies_[2]->Parameters();
    Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
    Vec3 Pj = param_j.head<3>();//j时刻t

    Vec3 pts_camera_i = pts_i_ / inv_dep_i;//逆深度归一化
    Vec3 pts_imu_i = qic * pts_camera_i + tic;
    Vec3 pts_w = Qi * pts_imu_i + Pi;
    Vec3 pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);

    double dep_j = pts_camera_j.z();

    Mat33 Ri = Qi.toRotationMatrix();//转为旋转矩阵
    Mat33 Rj = Qj.toRotationMatrix();
    Mat33 ric = qic.toRotationMatrix();//外参旋转部分转为旋转矩阵
    Mat23 reduce(2, 3);//误差关于重投影点的导数，是2*3的，不知道为什么没有相机内参(因为耦合在X，Y中了吗？这里看不懂，先放着)
    reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
        0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
//    reduce = information_ * reduce;

    Eigen::Matrix<double, 2, 6> jacobian_pose_i;//针对pose的jacobian(2*3*3*6=2*6)
    Eigen::Matrix<double, 3, 6> jaco_i;
    jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
    jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Sophus::SO3d::hat(pts_imu_i);
    jacobian_pose_i.leftCols<6>() = reduce * jaco_i;

    Eigen::Matrix<double, 2, 6> jacobian_pose_j;
    Eigen::Matrix<double, 3, 6> jaco_j;
    jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
    jaco_j.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_j);
    jacobian_pose_j.leftCols<6>() = reduce * jaco_j;

    Eigen::Vector2d jacobian_feature;
    jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_ * -1.0 / (inv_dep_i * inv_dep_i);

    jacobians_[0] = jacobian_feature;
    jacobians_[1] = jacobian_pose_i;
    jacobians_[2] = jacobian_pose_j;

    ///------------- check jacobians -----------------
//    检查Jacobian计算是否正确，在一个项上添加一个很小的量eps，然后计算(tmp_residual-residual)/eps即可得数值Jacobian可以和我们的解析Jacobian对比看是否接近
    {
        std::cout << "Analytical Jacobian: " << jacobians_[0].transpose()
                  << "size: " << jacobians_[0].size()
                  << std::endl;
        const double eps = 1e-6;
        inv_dep_i += eps;
        Eigen::Vector3d pts_camera_i = pts_i_ / inv_dep_i;
        Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
        Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
        Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
        Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);

        Eigen::Vector2d tmp_residual;
        double dep_j = pts_camera_j.z();
        tmp_residual = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();
        tmp_residual = information_ * tmp_residual;
        std::cout <<"numerical Jacobian: "<<  ((tmp_residual - residual_) / eps).transpose()
                  << " size: " << tmp_residual.size()
                  << "\n" <<std::endl;
    }

}

void EdgeReprojectionXYZ::ComputeResidual() {
    Vec3 pts_w = verticies_[0]->Parameters();

    VecX param_i = verticies_[1]->Parameters();
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Vec3 Pi = param_i.head<3>();

    Vec3 pts_imu_i = Qi.inverse() * (pts_w - Pi);
    Vec3 pts_camera_i = qic.inverse() * (pts_imu_i - tic);

    double dep_i = pts_camera_i.z();
    residual_ = (pts_camera_i / dep_i).head<2>() - obs_.head<2>();
}

void EdgeReprojectionXYZ::SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_) {
    qic = qic_;
    tic = tic_;
}

void EdgeReprojectionXYZ::ComputeJacobians() {

    Vec3 pts_w = verticies_[0]->Parameters();

    VecX param_i = verticies_[1]->Parameters();
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Vec3 Pi = param_i.head<3>();

    Vec3 pts_imu_i = Qi.inverse() * (pts_w - Pi);
    Vec3 pts_camera_i = qic.inverse() * (pts_imu_i - tic);

    double dep_i = pts_camera_i.z();

    Mat33 Ri = Qi.toRotationMatrix();
    Mat33 ric = qic.toRotationMatrix();
    Mat23 reduce(2, 3);
    reduce << 1. / dep_i, 0, -pts_camera_i(0) / (dep_i * dep_i),
        0, 1. / dep_i, -pts_camera_i(1) / (dep_i * dep_i);

    Eigen::Matrix<double, 2, 6> jacobian_pose_i;
    Eigen::Matrix<double, 3, 6> jaco_i;
    jaco_i.leftCols<3>() = ric.transpose() * -Ri.transpose();
    jaco_i.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_i);
    jacobian_pose_i.leftCols<6>() = reduce * jaco_i;

    Eigen::Matrix<double, 2, 3> jacobian_feature;
    jacobian_feature = reduce * ric.transpose() * Ri.transpose();

    jacobians_[0] = jacobian_feature;
    jacobians_[1] = jacobian_pose_i;

}

void EdgeReprojectionPoseOnly::ComputeResidual() {
    VecX pose_params = verticies_[0]->Parameters();
    Sophus::SE3d pose(
        Qd(pose_params[6], pose_params[3], pose_params[4], pose_params[5]),
        pose_params.head<3>()
    );

    Vec3 pc = pose * landmark_world_;
    pc = pc / pc[2];
    Vec2 pixel = (K_ * pc).head<2>() - observation_;
    // TODO:: residual_ = ????
    residual_ = pixel;
}

void EdgeReprojectionPoseOnly::ComputeJacobians() {
    // TODO implement jacobian here
}

}
}