#include "System.h"

#include <pangolin/pangolin.h>

using namespace std;
using namespace cv;
using namespace pangolin;

System::System(string sConfig_file_)
    :bStart_backend(true)
{
    string sConfig_file = sConfig_file_ + "euroc_config.yaml";

    cout << "1 System() sConfig_file: " << sConfig_file << endl;
    readParameters(sConfig_file);

    trackerData[0].readIntrinsicParameter(sConfig_file);

    estimator.setParameter();
//    ofs_pose.open("./VINS_pose_output_asTUM2_discrete_noise_100.txt", fstream::out);
    ofs_pose.open("./VINS_pose_output_asTUM2_noise_10_distcrete.txt", fstream::out);
    if(!ofs_pose.is_open())
    {
        cerr << "ofs_pose is not open" << endl;
    }
    // thread thd_RunBackend(&System::process,this);
    // thd_RunBackend.detach();
    cout << "2 System() end" << endl;
}

System::~System()
{
    bStart_backend = false;
    
    pangolin::QuitAll();
    
    m_buf.lock();
    while (!feature_buf.empty())
        feature_buf.pop();
    while (!imu_buf.empty())
        imu_buf.pop();
    m_buf.unlock();

    m_estimator.lock();
    estimator.clearState();
    m_estimator.unlock();

    ofs_pose.close();
}

//sys的传入img data
//void System::PubImageData(double dStampSec, Mat &img)
void System::PubImageData(double dStampSec, vector<Matrix<double,6,1>> &img)
{
    PUB_THIS_FRAME = true;//直接pub
    if (PUB_THIS_FRAME)
    {
        pub_count++;
        printf("pub count: %d\n", pub_count);
        shared_ptr<IMG_MSG> feature_points(new IMG_MSG());
        feature_points->header = dStampSec;
        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
//            auto &un_pts = trackerData[i].cur_un_pts;//当前帧去畸变后的点
//            auto &cur_pts = trackerData[i].cur_pts;
//            auto &ids = trackerData[i].ids;
//            auto &pts_velocity = trackerData[i].pts_velocity;
            vector<cv::Point2f> tmp_pts;
            printf("img.size: %d\n", img.size());
            for (unsigned int j = 0; j < img.size(); j++)
            {
                int p_id = j;
                hash_ids[i].insert(p_id);
                double x = img[j][4];
                double y = img[j][5];
                double z = 1;
                double u = 460 * x + 255; //u=fx * x / z + cx 使用内参将归一化坐标转换到像素平面下
                double v = 460 * y + 255; //u=fx * x / z + cx
                feature_points->points.emplace_back(x, y, z);
                feature_points->id_of_point.push_back(p_id * NUM_OF_CAM + i);
                feature_points->u_of_point.push_back(u);//仿真数据直接用xy代替像素坐标
                feature_points->v_of_point.push_back(v);
                tmp_pts.emplace_back(u, v);
                if(prev_pts_.empty()) {
                    feature_points->velocity_x_of_point.push_back(0);  //第一帧速度设为0
                    feature_points->velocity_y_of_point.push_back(0);
                    printf("prev_pts_ is empty set cur volecity to zero, cur_(u,v)=(%f, %f)\n", u, v);
                } else {
                    //由于feature points都是一一对应的，所以直接取对应index的坐标相减然后处以时间即可得速度
                    double v_x = (u - prev_pts_[j].x) / 0.0333333;
                    double v_y = (v - prev_pts_[j].y) / 0.0333333;
                    feature_points->velocity_x_of_point.push_back(v_x);
                    feature_points->velocity_y_of_point.push_back(v_y);
                    printf("prev_(u,v)=(%f, %f), cur_(u,v)=(%f, %f), v_x: %f, v_y: %f\n",
                           prev_pts_[j].x, prev_pts_[j].y, u, v, v_x, v_y);
                }
            }
            prev_pts_.clear();
            prev_pts_ = tmp_pts;
            tmp_pts.clear();
            //}
            // skip the first image; since no optical speed on frist image
            if (!init_pub)
            {
                cout << "4 PubImage init_pub skip the first image!" << endl;
                init_pub = 1;
            }
            else
            {
                m_buf.lock();
                feature_buf.push(feature_points);
                // cout << "5 PubImage t : " << fixed << feature_points->header
                //     << " feature_buf size: " << feature_buf.size() << endl;
                m_buf.unlock();
                con.notify_one();
            }
        }
    }
}

vector<pair<vector<ImuConstPtr>, ImgConstPtr>> System::getMeasurements()
{
    vector<pair<vector<ImuConstPtr>, ImgConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
        {
            // cerr << "1 imu_buf.empty() || feature_buf.empty()" << endl;
            return measurements;
        }

        // 这段是对时间戳的处理吗？怎么看不懂？这个 estimator.td 是什么
        if (!(imu_buf.back()->header > feature_buf.front()->header + estimator.td))
        {
            cerr << "wait for imu, only should happen at the beginning sum_of_wait: " 
                << sum_of_wait << endl;
            sum_of_wait++;
            return measurements;
        }

        if (!(imu_buf.front()->header < feature_buf.front()->header + estimator.td))
        {
            cerr << "throw img, only should happen at the beginning" << endl;
            feature_buf.pop();
            continue;
        }
        ImgConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        vector<ImuConstPtr> IMUs;
        while (imu_buf.front()->header < img_msg->header + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        // cout << "1 getMeasurements IMUs size: " << IMUs.size() << endl;
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty()){
            cerr << "no imu between two image" << endl;
        }
        // cout << "1 getMeasurements img t: " << fixed << img_msg->header
        //     << " imu begin: "<< IMUs.front()->header 
        //     << " end: " << IMUs.back()->header
        //     << endl;
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void System::PubImuData(double dStampSec, const Eigen::Vector3d &vGyr, 
    const Eigen::Vector3d &vAcc)
{
    shared_ptr<IMU_MSG> imu_msg(new IMU_MSG());
	imu_msg->header = dStampSec;
	imu_msg->linear_acceleration = vAcc;
	imu_msg->angular_velocity = vGyr;

	//时间戳应该递增
    if (dStampSec <= last_imu_t)
    {
        cerr << "imu message in disorder!" << endl;
        return;
    }
    last_imu_t = dStampSec;
     cout << "1 PubImuData t: " << fixed << imu_msg->header
         << " acc: " << imu_msg->linear_acceleration.transpose()
         << " gyr: " << imu_msg->angular_velocity.transpose() << endl;
    m_buf.lock();//对IMU进行push操作时需要将 m_buf 上锁
    imu_buf.push(imu_msg);
    // cout << "1 PubImuData t: " << fixed << imu_msg->header
    //     << " imu_buf size:" << imu_buf.size() << endl;
    m_buf.unlock();//解锁
    con.notify_one();//通知什么线程，在哪里设置的？
}

//这看起来应该只有后端处理求解，好像没有前端
// thread: visual-inertial odometry
void System::ProcessBackEnd()
{
    cout << "1 ProcessBackEnd start" << endl;
    while (bStart_backend)
    {
        printf("start to solve problem\n");
        // cout << "1 process()" << endl;
        vector<pair<vector<ImuConstPtr>, ImgConstPtr>> measurements;//很多IMU数据对应一帧Image数据
        
        unique_lock<mutex> lk(m_buf);
        //condition_variable等待，等到获取到measurement之后才能开始后端求解
        con.wait(lk, [&] {
            return (measurements = getMeasurements()).size() != 0;
        });
        if( measurements.size() > 1){
        cout << "1 getMeasurements size: " << measurements.size() 
            << " imu sizes: " << measurements[0].first.size()
            << " feature_buf size: " <<  feature_buf.size()
            << " imu_buf size: " << imu_buf.size() << endl;
        }
        lk.unlock();
        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            //处理IMU数据
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header;
                double img_t = img_msg->header + estimator.td;
                if (t <= img_t)
                {
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    assert(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x();//加计
                    dy = imu_msg->linear_acceleration.y();
                    dz = imu_msg->linear_acceleration.z();
                    rx = imu_msg->angular_velocity.x();//陀螺
                    ry = imu_msg->angular_velocity.y();
                    rz = imu_msg->angular_velocity.z();
                    //进行IMU预积分，从i积分到j
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));//第一个参数都是较小的时间与当前时间的差
                    // printf("1 BackEnd imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
                }
                else
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    assert(dt_1 >= 0);
                    assert(dt_2 >= 0);
                    assert(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x();
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y();
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z();
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x();
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y();
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z();
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            // cout << "processing vision data with stamp:" << img_msg->header 
            //     << " img_msg->points.size: "<< img_msg->points.size() << endl;

            // 处理图像数据，遍历特征点推入
            // TicToc t_s;
            // 遍历该图像上的所有点
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;//打包本帧的观测点信息
            for (unsigned int i = 0; i < img_msg->points.size(); i++) 
            {
                int v = img_msg->id_of_point[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;//VINS-MONO,只有一个camera
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x();
                double y = img_msg->points[i].y();
                double z = img_msg->points[i].z();
                double p_u = img_msg->u_of_point[i];
                double p_v = img_msg->v_of_point[i];
                double velocity_x = img_msg->velocity_x_of_point[i];//点的速度
                double velocity_y = img_msg->velocity_y_of_point[i];
                assert(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;//为什么会有速度？难道是光流得到的像素的速度？看变量名应该是的
                // feature_id map着vector< pair<camera_id, xyz_uv_velocity> >，就是以feature作为主要关注点，这个feature可能在很多camera下都被观测过，
                // 但是这里是单目，所以camera_id只有0，可以看到后面取feature的时候都是只取0
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }
            TicToc t_processImage;
            estimator.processImage(image, img_msg->header);//处理该帧图像：是否需要init，求解后端，并选择帧进行marg（这header有可能是时间戳转的double）

            // 非线性优化
            if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            {
                Vector3d p_wi;
                Quaterniond q_wi;
                q_wi = Quaterniond(estimator.Rs[WINDOW_SIZE]);//Rs即Qbi_bt,取window size即取第j时刻的
                p_wi = estimator.Ps[WINDOW_SIZE];
                vPath_to_draw.push_back(p_wi);
                double dStamp = estimator.Headers[WINDOW_SIZE];
                cout << "1 BackEnd processImage dt: " << fixed << t_processImage.toc() << " stamp: " <<  dStamp << " p_wi: " << p_wi.transpose() << endl;
//                ofs_pose << fixed << dStamp << " " << p_wi.transpose() << " " << q_wi.coeffs().transpose() << endl;
//                ofs_pose << fixed << dStamp << " " << p_wi(0) << " " << p_wi(1) << "" << p_wi(2) << " "
//                         << q_wi.w() << " " << q_wi.x()<< " " << q_wi.y()<< " " << q_wi.z() << endl;
                ofs_pose << fixed << dStamp << " " << p_wi(0) << " " << p_wi(1) << " " << p_wi(2) << " "
                         << q_wi.x()<< " " << q_wi.y()<< " " << q_wi.z() << " " << q_wi.w() << endl;
            }
        }
        m_estimator.unlock();
    }
}

void System::Draw() 
{   
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_cam = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0)
    );

    d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    // pangolin::OpenGlRenderState s_cam(
    //         pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
    //         pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0)
    // );

    // pangolin::View &d_cam = pangolin::CreateDisplay()
    //         .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
    //         .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.75f, 0.75f, 0.75f, 0.75f);
        glColor3f(0, 0, 1);
        pangolin::glDrawAxis(3);
         
        // draw poses
        glColor3f(0, 0, 0);
        glLineWidth(2);
        glBegin(GL_LINES);
        int nPath_size = vPath_to_draw.size();
        for(int i = 0; i < nPath_size-1; ++i)
        {        
            glVertex3f(vPath_to_draw[i].x(), vPath_to_draw[i].y(), vPath_to_draw[i].z());//在求解器里面赋值，在这个线程里面进行绘制
            glVertex3f(vPath_to_draw[i+1].x(), vPath_to_draw[i+1].y(), vPath_to_draw[i+1].z());
        }
        glEnd();
        
        // points
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
        {
            glPointSize(5);
            glBegin(GL_POINTS);
            for(int i = 0; i < WINDOW_SIZE+1;++i)
            {
                Vector3d p_wi = estimator.Ps[i];
                glColor3f(1, 0, 0);
                glVertex3d(p_wi[0],p_wi[1],p_wi[2]);
            }
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

#ifdef __APPLE__
void System::InitDrawGL() 
{   
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_cam = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0)
    );

    d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
}

void System::DrawGLFrame() 
{  

    if (pangolin::ShouldQuit() == false)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.75f, 0.75f, 0.75f, 0.75f);
        glColor3f(0, 0, 1);
        pangolin::glDrawAxis(3);
            
        // draw poses
        glColor3f(0, 0, 0);
        glLineWidth(2);
        glBegin(GL_LINES);
        int nPath_size = vPath_to_draw.size();
        for(int i = 0; i < nPath_size-1; ++i)
        {        
            glVertex3f(vPath_to_draw[i].x(), vPath_to_draw[i].y(), vPath_to_draw[i].z());
            glVertex3f(vPath_to_draw[i+1].x(), vPath_to_draw[i+1].y(), vPath_to_draw[i+1].z());
        }
        glEnd();
        
        // points
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
        {
            glPointSize(5);
            glBegin(GL_POINTS);
            for(int i = 0; i < WINDOW_SIZE+1;++i)
            {
                Vector3d p_wi = estimator.Ps[i];
                glColor3f(1, 0, 0);
                glVertex3d(p_wi[0],p_wi[1],p_wi[2]);
            }
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
#endif

}
