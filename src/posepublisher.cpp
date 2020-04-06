#include <tf/transform_listener.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <grid_map_msgs/GridMap.h>
//#include <grid_map_msgs/GridMapInfo.h>
#include <tf/transform_broadcaster.h>
#include "BinaryIO.h"
#include "SmartLoader.h"


//double pose(int flag);
geometry_msgs::PoseWithCovarianceStamped pose, sh_pose;
sensor_msgs::PointCloud2 pcloud;
grid_map_msgs::GridMap gmap;

ros::Publisher p_pub, m_pub, sh_pub;


struct DataToCallbackFunction
{
    SmartLoaderConfigParam* configParams;
    std::unique_ptr<SmartLoaderStackData>* SD;
};
DataToCallbackFunction globalDataToCallbackFunction;


void onPoseSet(double x, double y, double theta);


void onPoseSet(double x, double y, double theta)
{   
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    transform.setOrigin( tf::Vector3(x, y, 0.0) );
    tf::Quaternion q;
    q.setRPY(0, 0, theta);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "base_link"));
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "map"));
    std::string fixed_frame = "/base_link";
    pose.header.frame_id = fixed_frame;
    pose.header.stamp = ros::Time::now();

    // set x,y coord
    pose.pose.pose.position.x = x;
    pose.pose.pose.position.y = y;
    pose.pose.pose.position.z = 0.0;

    // set theta
    tf::Quaternion quat;
    quat.setRPY(0.0, 0.0, theta);
    tf::quaternionTFToMsg(quat, pose.pose.pose.orientation);
    pose.pose.covariance[6*0+0] = 0.5 * 0.5;
    pose.pose.covariance[6*1+1] = 0.5 * 0.5;
    pose.pose.covariance[6*5+5] = M_PI/12.0 * M_PI/12.0;

    // Grid Map
    std::string second_fixed_frame = "/map";
    gmap.info.header.frame_id = second_fixed_frame;
    gmap.info.header.stamp = pose.header.stamp;


    // publish
    ROS_INFO("x: %f, y: %f, z: 0.0, theta: %f",x,y,theta);
    p_pub.publish(pose);
    m_pub.publish(gmap);
    //p_pub.publish(pose.pose.pose);
   

}

void dealWithPCloudCB(sensor_msgs::PointCloud2 pc)
{
   ROS_INFO("Got Point Cloud message %x, %d", &pc, pc.header.seq);
   //onPoseSet(0.968507917975, -4.61491396765, 1);

   std::vector<float> xyz, x,y,z,intensity;
   auto numPoints = pc.height * pc.width;

    x.reserve(numPoints);
    y.reserve(numPoints);
    z.reserve(numPoints);
    xyz.reserve(numPoints);
    intensity.reserve(numPoints);

    for (auto j = 0; j < pc.height; j++)
    {
        for (auto i = 0; i < pc.width; i++)
        {
            auto loc = j * pc.row_step + i * pc.point_step;
            // float *xCorPtr = *((float*)&pc.data[loc]);
            x.push_back(*((float*)&pc.data[loc]));
            y.push_back(*((float*)&pc.data[loc + 4]));
            z.push_back(*((float*)&pc.data[loc + 8]));

            xyz.push_back(*((float*)&pc.data[loc]));
            xyz.push_back(*((float*)&pc.data[loc + 4]));
            xyz.push_back(*((float*)&pc.data[loc + 8]));
            intensity.push_back(*((float*)&pc.data[loc + 16]));    
        }
    }

    // TODO: save to file system 
    printf("size of vector %d", x.size());

    if (x.size() > 0)
    {
        static int counter = 0;
        std::string dummyString, st, path = "/home/sload/Downloads/temp/1/";

        //st = path + "x" + std::to_string(counter) + ".bin";
        st = path + "x" + std::to_string(counter);
        IAIRoboticsAlgorithms::BinaryIO::WriteBinary(&x[0], 1, sizeof(float), x.size(), 1, "x");

        st = path + "y" + std::to_string(counter) + ".bin";
        IAIRoboticsAlgorithms::BinaryIO::WriteBinary(&y[0], 1, sizeof(float), y.size(), 1, "y");

        st = path + "z" + std::to_string(counter) + ".bin";
        IAIRoboticsAlgorithms::BinaryIO::WriteBinary(&z[0], 1, sizeof(float), z.size(), 1, "z");

        st = path + "intensity_" + std::to_string(counter) + ".bin";
        IAIRoboticsAlgorithms::BinaryIO::WriteBinary(&intensity[0], 1, sizeof(float), intensity.size(), 1, "i");
    }
    
    // TODO: apply the algorithm
    // Udpate the time tag in the config 
    // // // // auto& t = pc.header.stamp; 
    // // // // // ros::Time t = pc.header.stamp; 
    // // // // auto tt = t.toNSec();
    auto& configParams = *(globalDataToCallbackFunction.configParams);
    configParams.timeTagMs = pc.header.stamp.nsec * 1000 * 1000;


    // uint64_t toNSec() const {return static_cast<uint64_t>(sec)*1000000000ull + static_cast<uint64_t>(nsec); }
    // T& fromNSec(uint64_t t);
    
    // double toSec()  const { return static_cast<double>(sec) + 1e-9*static_cast<double>(nsec); };
    // T& fromSec(double t);

    int xyz_size[2] = { numPoints, 3 }, intensity_size[2] = { numPoints, 1 };
	
    std::vector<double> xyzDouble(xyz.begin(), xyz.end());
    std::vector<double> intensityDouble(intensity.begin(), intensity.end());

    SmartLoaderStruct smartLoaderStruct;

    std::vector<float> heightMap_res_data;
	heightMap_res_data.resize(1024 * 1024);
	int heightMap_res_size[2] = { 0,0 };

    SmartLoader(globalDataToCallbackFunction.SD->get(), &configParams, (double*)&xyzDouble[0], xyz_size, (double*)&intensityDouble[0], intensity_size,
        &smartLoaderStruct, &heightMap_res_data[0], heightMap_res_size);

    //Build GMap topic
    // Grid Map
    std::string second_fixed_frame = "/map";
    gmap.info.header.frame_id = second_fixed_frame;
    gmap.info.header.stamp = ros::Time::now();

    // TODO: Shahar  
    gmap.info.resolution = 0; 

    // TODO: shshar x,y directions 
    gmap.info.length_x = 0;
    gmap.info.length_y = 0;
    memset(&gmap.info.pose.orientation, 0x00, sizeof(gmap.info.pose.orientation));
    memset(&gmap.info.pose.position, 0x00, sizeof(gmap.info.pose.position));
    

    gmap.layers.push_back(std::string("smartload"));
    gmap.data.resize(heightMap_res_size[0] * heightMap_res_size[1]);
    memcpy(&gmap.data[0], &heightMap_res_data[0], sizeof(float) * heightMap_res_size[0] * heightMap_res_size[1]);

    // Row start index (default 0).
    gmap.inner_start_index = 0;
    // Column start index (default 0).
    gmap.outer_start_index = 0;

    // , heightMap_res_size
    // gmap.layers[0] ="smartload";
    //gmap.layers


    //Topic localization of the vehicle
    std::string fixed_frame = "/base_link";
    pose.header.frame_id = fixed_frame;
    pose.header.stamp = gmap.info.header.stamp;

    // set x,y coord

    // TODO : memset all ros topics before usage --> TOOD Michele 

    // tood = shshar chagne to loader status - TOOD 
    if (smartLoaderStruct.status)
    {
        pose.pose.pose.position.x = smartLoaderStruct.loaderLoc[0];
        pose.pose.pose.position.y = smartLoaderStruct.loaderLoc[1];
        pose.pose.pose.position.z = smartLoaderStruct.loaderLoc[2];
        //TODO SHAHAR pose.pose.pose.orientation
    }
    else 
    {
        // TOOD Michele  
        pose.pose.covariance[0] = 0;
    }
    
    

    //Topic localization of the shovel
    std::string fixed_frame = "/base_link";
    sh_pose.header.frame_id = fixed_frame;
    sh_pose.header.stamp = gmap.info.header.stamp;

    if (smartLoaderStruct.status)
    {
        sh_pose.pose.pose.position.x = smartLoaderStruct.shvelLoc[0];
        sh_pose.pose.pose.position.y = smartLoaderStruct.shvelLoc[1];
        sh_pose.pose.pose.position.z = smartLoaderStruct.shvelLoc[2];
        //TODO SHAHAR pose.pose.pose.orientation
    }
    else 
    {
        // TOOD Michele  
        sh_pose.pose.covariance[0] = 0;
    }
    
    m_pub.publish(gmap);
    p_pub.publish(pose);
    sh_pub.publish(sh_pose);
    
}


void GetPerceptionConfig(SmartLoaderConfigParam& configParams)
{
	memset(&configParams, 0x00, sizeof(configParams));
	configParams.timeTagMs = 1;
	configParams.planeModelParameters[0] = -0.028009425848722;
	configParams.planeModelParameters[1] = -5.510213086381555e-04;
	configParams.planeModelParameters[2] = 0.999607503414154;
	configParams.planeModelParameters[3] = 2.846343278884888;
	configParams.useExternalProjectionMatrix = false;
	// configParams.externalProjectionMatrix is already set to zero
	double xyzLimits[6]= { -1.681562873721123, 1.918437126278877, -9999, 9999, -3.361619710922240, -2.361619710922240};
	memcpy(configParams.xyzLimits, xyzLimits, sizeof(configParams.xyzLimits));
	configParams.minNumPointsInPc = 64;
	configParams.minimumDistanceFromLoaderToPlaneMeter = 0.2;
	configParams.minPointsForReflector = 5;
	configParams.maximumTimeTagDiffMs = 3000;
	configParams.minimumIntensityReflectorValue = 100;
	configParams.loaderReflectorDiameterMeter = 0.18;
	configParams.reflectorMaxZaxisDistanceForOutlierMeter = 0.06;
	configParams.previousLoaderLocationToCurrentLocationMaximumDistanceMeter = 0.18;
	configParams.loaderReflectorMaxZaxisDistanceForOutlierMeter = 0.04;
	configParams.maxDistanceBetweenEachRayMeter = 0.07;
	configParams.debugMode = false;
}


int main(int argc, char** argv)
{
    // Initialize perception code
    SmartLoaderConfigParam configParams;
    GetPerceptionConfig(configParams);
    auto SD = std::make_unique<SmartLoaderStackData>();
	auto pd = std::make_unique<SmartLoaderPersistentData>();
	SD->pd = pd.get();

	SmartLoader_initialize(SD.get());

    // setting
    ros::init(argc, argv, "sl_pose");
    ros::NodeHandle nh;
    ros::NodeHandle nh_priv("~");
    std::string map_frame, base_frame;
    double publish_frequency = 10;
    bool is_stamped = false;
    ros::Subscriber s_sub;

    globalDataToCallbackFunction.configParams = &configParams;
    globalDataToCallbackFunction.SD = &SD;

    s_sub = nh.subscribe("/velodyne_points", 1000, dealWithPCloudCB);
   
    m_pub = nh.advertise<grid_map_msgs::GridMap>("sl_map", 1);

    p_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("sl_pose", 1);
    sh_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("shovel_pose", 1);

    ros::Rate rate(publish_frequency);
    while (nh.ok()){
#ifdef MoreThanOneThread
          onPoseSet(0.968507917975, -4.61491396765, 1); 
          p_pub.publish(pose);
          m_pub.publish(gmap);
#endif        
          ros::spinOnce();
          rate.sleep(); 
   } 
  

  SmartLoader_terminate();

   return 0;
}