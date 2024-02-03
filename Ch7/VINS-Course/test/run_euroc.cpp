
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <thread>
#include <iomanip>

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <eigen3/Eigen/Dense>
#include "System.h"
#include <sophus/se3.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

const int nDelayTimes = 2;
string sData_path = "/home/dataset/EuRoC/MH-05/mav0/";
string sConfig_path = "../config/";
string imu_pose_file_name = "imu_pose.txt";
std::shared_ptr<System> pSystem;

//将IMU数据传入VINS系统
void PubImuData()
{
//	string sImu_data_file = sConfig_path + "MH_05_imu0.txt";//数据中的RSS不知道是什么，但是应该是6轴IMU数据
    string sImu_data_file = sConfig_path + imu_pose_file_name;//数据中的RSS不知道是什么，但是应该是6轴IMU数据

	cout << "1 PubImuData start sImu_data_filea: " << sImu_data_file << endl;
	ifstream fsImu;
	fsImu.open(sImu_data_file.c_str());
	if (!fsImu.is_open())
	{
		cerr << "Failed to open imu file! " << sImu_data_file << endl;
		return;
	}

	std::string sImu_line;
	double dStampNSec = 0.0;
	Vector3d vAcc;
	Vector3d vGyr;
	while (std::getline(fsImu, sImu_line) && !sImu_line.empty()) // read imu data
	{
		std::istringstream ssImuData(sImu_line);
        double tmp_data;//imu的pose不需要，所以不进行读取
        ssImuData >> dStampNSec >> tmp_data >> tmp_data >> tmp_data >> tmp_data >> tmp_data >> tmp_data >> tmp_data
                  >> vGyr.x() >> vGyr.y() >> vGyr.z() >> vAcc.x() >> vAcc.y() >> vAcc.z();//读取时间戳，角速度，加速度
        pSystem->PubImuData(dStampNSec, vGyr, vAcc);//这里读出来的时间直接就是s，不用再转
		usleep(5000*nDelayTimes);
	}
	fsImu.close();
	printf("imu pub finish!!!\n");
}

//将视觉特征传入VINS系统
void PubImageData()
{
	string sImage_file = sConfig_path + "cam_pose.txt";
    printf("start pub img msg...\n");
    printf("sImage_file: %s\n", sImage_file.c_str());
    cout << "1 PubImageData start sImage_file: " << sImage_file << endl;

    ifstream fsImage;
    fsImage.open(sImage_file.c_str());
    if (!fsImage.is_open()) {
        cerr << "Failed to open image file! " << sImage_file << endl;
        return;
    }

    std::string sImage_line;
    double dStampNSec;
    string sImgFileName;

    vector<Matrix<double,6,1>> xyz_uv_sum;
    int point_num = 0;
    // cv::namedWindow("SOURCE IMAGE", CV_WINDOW_AUTOSIZE);
    while (std::getline(fsImage, sImage_line) && !sImage_line.empty())
    {
        std::istringstream ssImgData(sImage_line);
        ssImgData >> dStampNSec;//读取camera时间戳


        Matrix<double,6,1> xyz_uv;
        string KF_PointsFile = sConfig_path + "keyframe/all_points_" + to_string(point_num++) + ".txt";//读all_points_xx.txt
        ifstream fin(KF_PointsFile);
        if(!fin) {
            printf("[ERROR] KF file: %s do not exist", KF_PointsFile.c_str());
        }

        printf("KF_PointsFile: %s\n", KF_PointsFile.c_str());
        while(!fin.eof()) {
            fin >> xyz_uv(0) >> xyz_uv(1) >> xyz_uv(2) >> xyz_uv(3) >> xyz_uv(4) >> xyz_uv(5);
            xyz_uv_sum.push_back(xyz_uv);
        }
        pSystem->PubImageData(dStampNSec, xyz_uv_sum);//
        xyz_uv_sum.clear();//没有清导致的不行！！！CNM，WCNM
        // cv::imshow("SOURCE IMAGE", img);
        // cv::waitKey(0);
        usleep(50000*nDelayTimes);
    }
    fsImage.close();
    printf("pub img finish, point_num: %d\n", point_num);
}

#ifdef __APPLE__
// support for MacOS
void DrawIMGandGLinMainThrd(){
	string sImage_file = sConfig_path + "MH_05_cam0.txt";

	cout << "1 PubImageData start sImage_file: " << sImage_file << endl;

	ifstream fsImage;
	fsImage.open(sImage_file.c_str());
	if (!fsImage.is_open())
	{
		cerr << "Failed to open image file! " << sImage_file << endl;
		return;
	}

	std::string sImage_line;
	double dStampNSec;
	string sImgFileName;

	pSystem->InitDrawGL();
	while (std::getline(fsImage, sImage_line) && !sImage_line.empty())
	{
		std::istringstream ssImuData(sImage_line);
		ssImuData >> dStampNSec >> sImgFileName;
		// cout << "Image t : " << fixed << dStampNSec << " Name: " << sImgFileName << endl;
		string imagePath = sData_path + "cam0/data/" + sImgFileName;

		Mat img = imread(imagePath.c_str(), 0);
		if (img.empty())
		{
			cerr << "image is empty! path: " << imagePath << endl;
			return;
		}
		//pSystem->PubImageData(dStampNSec / 1e9, img);
		cv::Mat show_img;
		cv::cvtColor(img, show_img, CV_GRAY2RGB);
		if (SHOW_TRACK)
		{
			for (unsigned int j = 0; j < pSystem->trackerData[0].cur_pts.size(); j++)
			{
				double len = min(1.0, 1.0 *  pSystem->trackerData[0].track_cnt[j] / WINDOW_SIZE);
				cv::circle(show_img,  pSystem->trackerData[0].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
			}

			cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
			cv::imshow("IMAGE", show_img);
		  // cv::waitKey(1);
		}

		pSystem->DrawGLFrame();
		usleep(50000*nDelayTimes);
	}
	fsImage.close();

} 
#endif

struct MotionData
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double timestamp;
    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb;
    Eigen::Vector3d imu_acc;
    Eigen::Vector3d imu_gyro;

    Eigen::Vector3d imu_gyro_bias;
    Eigen::Vector3d imu_acc_bias;

    Eigen::Vector3d imu_velocity;
};

//输出轨迹结果
void save_pose_asTUM2(const std::string filename, std::vector<MotionData> pose) {
    std::ofstream save_points;
    save_points.setf(std::ios::fixed, std::ios::floatfield);
    save_points.open(filename.c_str());

    for(int i=0; i < pose.size(); ++i) {
        MotionData data = pose[i];
        double time = data.timestamp;
        Eigen::Quaterniond q(data.Rwb);
        Eigen::Vector3d t = data.twb;

        save_points.precision(9);
        save_points << time << " ";
        save_points.precision(5);
        save_points << t(0) << " "
                    << t(1) << " "
                    << t(2) << " "
                    << q.x() << " "
                    << q.y() << " "
                    << q.z() << " "
                    << q.w() << std::endl;
    }
}

void cal_diff() {
    Quaterniond q;
    Eigen::Matrix<double, 3, 1> tmp_mat;
    double tmp_var;

    vector<Sophus::SE3d> vec_T_gt;
    std::string cam_pose_tum = "../config/cam_pose_tum_delete_first11rows.txt";
    ifstream fin(cam_pose_tum);
    if(!fin) {
        printf("[ERROR] cam_pose_tum file: %s do not exist\n", cam_pose_tum.c_str());
    }
    printf("cam_pose_tum file: %s\n", cam_pose_tum.c_str());
    while(!fin.eof()) {
        fin >> tmp_var >>tmp_mat(0) >> tmp_mat(1) >> tmp_mat(2)
            >> q.x() >> q.y() >> q.z() >> q.w();
        vec_T_gt.emplace_back(Sophus::SE3d{q, tmp_mat});
    }

    vector<Sophus::SE3d> vec_T_vins;
    std::string vins_cam_pose_tum = "../bin/VINS_pose_output_asTUM2.txt";
    ifstream fin_vins(vins_cam_pose_tum);
    if(!fin_vins) {
        printf("[ERROR] vins_cam_pose_tum file: %s do not exist\n", vins_cam_pose_tum.c_str());
    }
    printf("vins_cam_pose_tum file: %s\n", vins_cam_pose_tum.c_str());
    while(!fin_vins.eof()) {
        fin_vins >> tmp_var >> q.x() >> q.y() >> q.z() >> q.w() >>
                 tmp_mat(0) >> tmp_mat(1) >> tmp_mat(2);
        vec_T_vins.emplace_back(Sophus::SE3d{q, tmp_mat});
    }

    int index_min = min(vec_T_gt.size(), vec_T_vins.size());
    for(int i = 0; i < index_min; ++i) {
        //Tw_gt^(-1) * Tw_vins = Tgt_vins
        Sophus::SE3d se3_diff = vec_T_gt[i].inverse() * vec_T_vins[i];
        cout << "i = " << i << ",\t se3_diff_ypr = \t" << Utility::R2ypr(se3_diff.rotationMatrix()).transpose()
             << ",\t t_diff = \t" << se3_diff.translation().transpose() << endl;

//        //t_gt_vins
//        Vector3d t_diff = vec_T_vins[i].translation() - vec_T_gt[i].translation();
//        cout << "i = " << i << "\t t_diff = \t" << t_diff.transpose() << endl;
    }
}

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << "./run_euroc PATH_TO_FOLDER/MH-05/mav0 PATH_TO_CONFIG/config \n"
             << "For example: ./run_euroc /home/stevencui/dataset/EuRoC/MH-05/mav0/ ../config/  imu_pose_noise_10.txt"<< endl;
        return -1;
    }


    cal_diff();

	sData_path = argv[1];//数据路径
	sConfig_path = argv[2];//配置路径
    imu_pose_file_name = argv[3];//配置路径

	pSystem.reset(new System(sConfig_path));//读取配置文件，打开输出文件等
    //后端线程
	std::thread thd_BackEnd(&System::ProcessBackEnd, pSystem);//参数：func，param（应该是func的参数）

	//imu线程
	// sleep(5);
	std::thread thd_PubImuData(PubImuData);
    //img线程
	std::thread thd_PubImageData(PubImageData);

	//可视化线程
#ifdef __linux__	
	std::thread thd_Draw(&System::Draw, pSystem);
#elif __APPLE__
	DrawIMGandGLinMainThrd();
#endif

	thd_PubImuData.join();
    thd_PubImageData.join();



	// thd_BackEnd.join();
	// thd_Draw.join();

	cout << "main end... see you ..." << endl;
	return 0;
}
