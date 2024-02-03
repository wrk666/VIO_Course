//
// Created by hyj on 18-11-11.
//
#include <iostream>
#include <vector>
#include <random>  
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;

#include <GL/glut.h>

//void display()
//{
//    glClear(GL_COLOR_BUFFER_BIT);
//    glColor3f(1.0, 0.0, 0.0);
//    glBegin(GL_LINE_STRIP);
////    for (float x = -1.0; x <= 1.0; x += 0.01)
////    {
////        float y = sin(x * 10.0);
////        glVertex2f(x, y);
////    }
//
//    glEnd();
//    glutSwapBuffers();
//}


struct Pose
{
    Pose(Eigen::Matrix3d R, Eigen::Vector3d t):Rwc(R),qwc(R),twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;

    Eigen::Vector2d uv;    // 这帧图像观测到的特征坐标
};


int main(int argc, char** argv)
{

    int poseNums = 10;
    double radius = 8;
    double fx = 1.;
    double fy = 1.;
    std::vector<Pose> camera_pose;
    for(int n = 0; n < poseNums; ++n ) {
        double theta = n * 2 * M_PI / ( poseNums * 4); // 1/4 圆弧
        // 绕 z轴 旋转
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        camera_pose.push_back(Pose(R,t));
    }

    // 随机数生成 1 个 三维特征点
    std::default_random_engine generator;
    std::uniform_real_distribution<double> xy_rand(-4, 4.0);
    std::uniform_real_distribution<double> z_rand(8., 10.);
    double tx = xy_rand(generator);
    double ty = xy_rand(generator);
    double tz = z_rand(generator);

    Eigen::Vector3d Pw(tx, ty, tz);
    // 这个特征从第三帧相机开始被观测，i=3
    int start_frame_id = 3;
    //从1帧到8帧来看观测数量和三角化的关系
    int end_frame_id = 9;
    // 添加观测误差
    std::default_random_engine generator_noise;
    vector<pair<double, double>> result_curve;
//    for(int j=1; j<50; ++j) {  //暂时关闭遍历50个noise varince，后面如果再有机会的话，拿着个来进行画图的实验。（也可以用plotjuggler来看啊）
        int j=5;
        std::normal_distribution<double> noise_pdf(0., (double)j / 1000.);  // 2pixel / focal，修改var可改变噪声大小
        for (int i = start_frame_id; i < end_frame_id; ++i) {
            Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
            Eigen::Vector3d Pc = Rcw * (Pw - camera_pose[i].twc);//实际上是Rp+t，拆开来看就是Rwc^T * Pw +(-Rwc * twc)= Rcw * Pw + tcw

            double x = Pc.x();
            double y = Pc.y();
            double z = Pc.z();

            camera_pose[i].uv = Eigen::Vector2d(x/z + noise_pdf(generator),y/z + noise_pdf(generator));//因为camera内参为1，1，所以内参可以忽略
        }

        /// TODO::homework; 请完成三角化估计深度的代码
        // 遍历所有的观测数据，并三角化
        Eigen::Vector3d P_est;           // 结果保存到这个变量
        P_est.setZero();
        /* your code begin */
        //1.构建D
        int D_size = end_frame_id - start_frame_id;
        MatXX D(MatXX::Zero( 2 * D_size, 4));//D维度为2n*4
        for(int i=start_frame_id; i<end_frame_id; ++i) {
            //构建投影矩阵Pk
            MatXX Pi(MatXX::Zero(3,4));
            Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
            Pi.block(0,0,3,3) = Rcw;
            Pi.block(0,3,3,1) = -Rcw * camera_pose[i].twc;//tcw，变换矩阵求逆
            cout << "i = " <<i <<",   Pi_block: \n" << Pi <<endl;
            //构建Dy=0的2n*4的D矩阵快
            D.block((i-start_frame_id) * 2, 0, 2, 4) =
                    camera_pose[i].uv * Pi.block(2,0,1,4) - Pi.block(0,0,2,4);
        }
        cout << "the whole D mat, size: " << D.size() << "\nMat is:\n" << D <<endl;
        //2.对D进行rescale
        MatrixXd::Index maxRow, maxCol;
        double max = D.maxCoeff(&maxRow,&maxCol);
//    max = 1;//取消scale
        cout << "max element of D is: " << max <<endl;
        printf("maxRow: %ld, maxCol: %ld\n", maxRow, maxCol);
        D /= max;
        //3.对D^TD进行SVD（参数有ComputeThinU | ComputeThinV 和 ComputeFullU | ComputeFullV 这个位置不传参就代表你只想计算特征值，不关注左右特征向量(UV矩阵)，
        // 传参就代表你想计算出左右特征向量，而full就是告诉函数计算出来的UV方阵，也就是Matrix3d，计算出来的就是3*3的方阵，thin只在矩阵维度不知道时使用，即n*p的矩阵D，不知道n和p谁更小，假设m=min(n,p),
        // 那么计算结果： U：n*m, V:p*m, 其所代表的特征向量均不是对应实际的\sigma中的特征值的)
        JacobiSVD<MatrixXd> svd(D.transpose() * D, ComputeThinU | ComputeThinV);//D^T*D 进行SVD分解

        cout<< "observe num: " << D_size << endl;
        cout << "D维度： " << D.rows() << "*" << D.cols() <<endl;
        cout << " U matrix:\n" << svd.matrixU() << endl;
        cout << " V matrix:\n" << svd.matrixV() << endl;
        cout << "Its singular values are:\n" << svd.singularValues() << endl;

        //4.判断解的有效性(\sigma_4 / \sigma_3 < 1e-2 ?)
        double judge_value = std::abs(svd.singularValues()(3) / svd.singularValues()(2));
        if(judge_value < 1e-2) {
            Eigen::Vector4d u4 = max * svd.matrixU().rightCols(1);
            cout << "this Triangulation is valid, judge_value:" << judge_value << endl << "u4 is: \n" <<  u4 << endl;//最后一列（为什么是U不是V？）
            //5.对triangulation的结果(4维)进行归一化(最后一维变为1)
            P_est = (u4/u4(3)).head(3);
            result_curve.push_back(make_pair((double)j / 1000., judge_value));
        } else {
            cout << "this Triangulation is NOT valid, judge_value:" << judge_value << endl;
        }
        /* your code end */

        std::cout <<"ground truth: \n"<< Pw.transpose() <<std::endl;
        std::cout <<"your result: \n"<< P_est.transpose() <<std::endl;
//    }


    // TODO:: 请如课程讲解中提到的判断三角化结果好坏的方式，绘制奇异值比值变化曲线




//    glutInit(&argc, argv);
//    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
//    glutInitWindowSize(640, 480);
//    glutCreateWindow("sin curve");
//    glutDisplayFunc(display());
//    glutMainLoop();

    return 0;
}
