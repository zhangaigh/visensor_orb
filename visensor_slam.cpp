/*!
 * \file visensor_slam.app
 * \author  bo zhang <zhangbo24@lenovo.com>
 * \version 0.1
 * \date   03/27/2017
 * \section LICENSE
 *
 * \section DESCRIPTION
 *
 * This executable is for visensor orbslam. 
 * Visensor is a stereo hardware module which includes two camera and IMU. 
 * 
 * usage:
 *  Examples: ----- ./visensor_orb ../../Vocabulary/ORBvoc.bin visensor.yaml ~/visensor/Loitor_VI_Sensor_SDK_V1.3.1/SDK/Loitor_VISensor_Setups.txt ./
 *            -----visensor.yaml      stereo camera configuration for orbslam
              -----Loitor_VISensor_Setups.txt  stereo camera configuration files
 */

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>

//#include <cv.h>
#include <highgui.h>
#include <sys/time.h>
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <tuple>

#include "loitorusbcam.h"
#include "loitorimu.h"
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <System.h>
using namespace std;
using namespace cv;

void readVIImages(cv::Mat& img_left, cv::Mat& img_right)
{
// 当前左右图像的时间戳
     timeval left_stamp,right_stamp;

    
    if(visensor_cam_selection==2)
    {
        visensor_imudata paired_imu=visensor_get_leftImg((char *)img_left.data,left_stamp);

        // 显示同步数据的时间戳（单位微秒）
        cout<<"left_time : "<<left_stamp.tv_usec<<endl;
        cout<<"paired_imu time ===== "<<paired_imu.system_time.tv_usec<<endl<<endl;

        imshow("left",img_left);
        cvWaitKey(1);
    }
    //Cam2
    else if(visensor_cam_selection==1)
    {
        visensor_imudata paired_imu=visensor_get_rightImg((char *)img_right.data,right_stamp);

        // 显示同步数据的时间戳（单位微秒）
        cout<<"right_time : "<<right_stamp.tv_usec<<endl;
        cout<<"paired_imu time ===== "<<paired_imu.system_time.tv_usec<<endl<<endl;

        imshow("right",img_right);
        cvWaitKey(1);
    }
    // Cam1 && Cam2
    else if(visensor_cam_selection==0)
    {
        visensor_imudata paired_imu=visensor_get_stereoImg((char *)img_left.data,(char *)img_right.data,left_stamp,right_stamp);

        // 显示同步数据的时间戳（单位微秒）
        //cout<<"left_time : "<<left_stamp.tv_usec<<endl;
        //cout<<"right_time : "<<right_stamp.tv_usec<<endl;
        //cout<<"paired_imu time ===== "<<paired_imu.system_time.tv_usec<<endl<<endl;
       // imwrite("left.png",img_left);
        //imwrite("right.png",img_right);
    // cvWaitKey(5);
    }
}

void readIMUData()
{
    if(visensor_imu_have_fresh_data())
            {
         static int   counter = 0;
         counter++;
            // 每隔20帧显示一次imu数据
            if(counter>=20)
            {
                float ax=visensor_imudata_pack.ax;
                float ay=visensor_imudata_pack.ay;
                float az=visensor_imudata_pack.az;
                cout<<"visensor_imudata_pack->a : "<<sqrt(ax*ax+ay*ay+az*az)<<endl;
                //cout<<"visensor_imudata_pack->a : "<<visensor_imudata_pack.ax<<" , "<<visensor_imudata_pack.ay<<" , "<<visensor_imudata_pack.az<<endl;
                //cout<<"imu_time1 : "<<visensor_imudata_pack.imu_time<<endl;
                //cout<<"imu_time2 : "<<visensor_imudata_pack.system_time.tv_usec<<endl;
                counter=0;
            }
        }
        usleep(50);
}


int initVIsensor(const char* cfg)
{
    /************************ Start Cameras ************************/
    visensor_load_settings(cfg);

    // 手动设置相机参数
    //visensor_set_current_mode(5);
    //visensor_set_auto_EG(0);
    //visensor_set_exposure(50);
    //visensor_set_gain(200);
    //visensor_set_cam_selection_mode(0);
    //visensor_set_resolution(false);
    //visensor_set_fps_mode(false);
    // 保存相机参数到原配置文件
    //visensor_save_current_settings();

    int r = visensor_Start_Cameras();
    if(r<0)
    {
        printf("Opening cameras failed...\r\n");
        return r;
    }
    /************************** Start IMU **************************/
    int fd=visensor_Start_IMU();
    if(fd<0)
    {
        printf("visensor_open_port error...\r\n");
        return 0;
    }
    printf("visensor_open_port success...\r\n");
    /************************ ************ ************************/

}

bool CheckFileExistance(const string & fileName)
{
    std::ifstream f(fileName.c_str());
    return f.good();
}


ORB_SLAM2::System *SLAM;
string strMap;
bool map_reuse = false;
std::queue<std::tuple<cv::Mat,cv::Mat>> CameraDataFIFO;
static std::mutex mutexCamera;
cv::Mat M1l, M2l, M1r, M2r;

void slamThread(void)
{
        // Main loop
    cv::Mat imLeft, imRight, imLeftRect, imRightRect;
    
     
    while(1)
    {
        std::cout<<" enter slamthread ..... "<<std::endl;
        //std::unique_lock<std::mutex> lock ( mutexCamera );
        if (!CameraDataFIFO.empty() )
        {
            std::cout<<"..... output image......."<<std::endl;
            imLeft = std::get<0> ( CameraDataFIFO.front() );
            imRight = std::get<1> ( CameraDataFIFO.front() );

            CameraDataFIFO.pop();
            if (visensor_cam_selection == 0)
            {
                cv::remap(imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
                cv::remap(imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);
            }

            double tframe = 0.0;

            // Pass the images to the SLAM system
            //SLAM.TrackStereo(imLeftRect,imRightRect,tframe);
            switch (visensor_cam_selection)
            {
                case 0:
                   // SLAM->TrackMonocularWithStereo(imLeftRect,imRightRect,tframe);
                    SLAM->TrackStereo(imLeftRect,imRightRect,tframe);
                    break;
                case 1:
                    SLAM->TrackMonocular(imLeft, tframe);
                break;
                case 2:
                    SLAM->TrackMonocular(imRight, tframe);
                break;
            }
        }
    }
}


void *addCameraFIFO(void*)
{
    
    cv::Mat img_left, img_right, imLeftRect, imRightRect;
    std::cout<<"visensor_resolution_status "<<visensor_resolution_status<<std::endl;
    if(!visensor_resolution_status)
    {
        img_left.create(cv::Size(640,480),CV_8U);
        img_right.create(cv::Size(640,480),CV_8U);
        img_left.data=new unsigned char[IMG_WIDTH_VGA*IMG_HEIGHT_VGA];
        img_right.data=new unsigned char[IMG_WIDTH_VGA*IMG_HEIGHT_VGA];
    }
    else
    {
        img_left.create(cv::Size(752,480),CV_8U);
        img_right.create(cv::Size(752,480),CV_8U);
        img_left.data=new unsigned char[IMG_WIDTH_WVGA*IMG_HEIGHT_WVGA];
        img_right.data=new unsigned char[IMG_WIDTH_WVGA*IMG_HEIGHT_WVGA];
    }
    while (1)
    {   
        //std::unique_lock<std::mutex> lock ( mutexCamera );
        if(waitKey(30) >= 0)
        {
            break;
        } 
            readVIImages(img_left, img_right);
           // imwrite("left.png", imLeft);
           // imwrite("right.png", imRight);
            if (visensor_cam_selection == 0)
            {
                cv::remap(img_left,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
                cv::remap(img_right,imRightRect,M1r,M2r,cv::INTER_LINEAR);
            }

            double tframe = 0.0;

            // Pass the images to the SLAM system
            //SLAM.TrackStereo(imLeftRect,imRightRect,tframe);
            switch (visensor_cam_selection)
            {
                case 0:
                   // SLAM->TrackMonocularWithStereo(imLeftRect,imRightRect,tframe);
                    SLAM->TrackStereo(imLeftRect,imRightRect,tframe);
                    break;
                case 1:
                    SLAM->TrackMonocular(img_left, tframe);
                break;
                case 2:
                    SLAM->TrackMonocular(img_right, tframe);
                break;
            }
    }

}

bool flag = 0;
int main(int argc, char **argv)
{
    
    if(argc != 5)
    {
        cerr << endl << "Usage: ./stereo_euroc path_to_vocabulary slam_settings vi_setting path_to_map" << endl;
        return 1;
    }

    // Read rectification parameters
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }
    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;

    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;

    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;

    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
            rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
    {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
        return -1;
    }

    cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
    cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);

    // init visensor
    initVIsensor(argv[3]);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    strMap = string(argv[4])+"slam_map.bin";
    map_reuse = CheckFileExistance(strMap);
   // SLAM = new ORB_SLAM2::System(argv[1],argv[2],strMap,ORB_SLAM2::System::STEREO,true,map_reuse); 
    switch (visensor_cam_selection)
    {
        case 0:   //stereo 
        {
            SLAM = new ORB_SLAM2::System(argv[1],argv[2],strMap,ORB_SLAM2::System::STEREO,true,map_reuse);
            break;
        }
        case 1:   // left
        {
            SLAM = new ORB_SLAM2::System(argv[1],argv[2],strMap,ORB_SLAM2::System::MONOCULAR,true,map_reuse);
            break;
        }
        
        case 2:   // right
             SLAM = new ORB_SLAM2::System(argv[1],argv[2],strMap,ORB_SLAM2::System::MONOCULAR,true,map_reuse);
         break;
    }
    


    cout << endl << "-------" << endl;
    cout << "Start read stereo camera ..." << endl;
    //std::thread* thAddIMages = new std::thread ( addCameraFIFO ); 


    //Create img_show thread
    pthread_t add_thread, slam_thread;
    int temp;
    if(temp = pthread_create(&add_thread, NULL, addCameraFIFO, NULL))
        printf("Failed to create thread add_thread\r\n");

    /*while(1)
    {
        // Do - Nothing :)
        //cout<<visensor_get_imu_portname()<<endl;
        //cout<<visensor_get_hardware_fps()<<endl;
        int c;
        if ((c=getchar()) == '\n')
        {
            break;
        }
        usleep(500);
    }  */

    if (add_thread != 0)
    {
        pthread_join(add_thread,NULL);
    }

    SLAM->Shutdown();
            // save map by default
            SLAM->SaveMap(strMap);
            //std::cout << Util::Timer::Summary() << std::endl;            
            delete SLAM;
                visensor_Close_Cameras();
    visensor_Close_IMU();

    /*
    std::thread* thSLAM = new std::thread ( slamThread ); // 


    int c;
    if ((c=getchar()) == '\n')
    {
        visensor_Close_Cameras();
        visensor_Close_IMU();
        if ( thSLAM )
        {
            thSLAM->join();
            delete thSLAM;
            thSLAM = nullptr;
        }
        if ( thAddIMages )
        {
            thAddIMages->join();
            delete thAddIMages;
            thAddIMages = nullptr;
        }
        if ( SLAM )
        {
            SLAM->Shutdown();
            // save map by default
            SLAM->SaveMap(strMap);
            //std::cout << Util::Timer::Summary() << std::endl;            
            delete SLAM;
        } 
    }
    
    */
    return 0;

}
