#include <tf/transform_listener.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <grid_map_msgs/GridMap.h>
//#include <grid_map_msgs/GridMapInfo.h>
#include <tf/transform_broadcaster.h>
#include <std_msgs/Float32MultiArray.h>
#include "BinaryIO.h"
#include "SmartLoader.h"


//double pose(int flag);
geometry_msgs::PoseWithCovarianceStamped pose, sh_pose;
sensor_msgs::PointCloud2 pcloud;
grid_map_msgs::GridMap gmap;

ros::Publisher p_pub, m_pub, sh_pub;

std::string datasetPath = "/home/sload/Downloads/WorkingVerForUTWindows/SmartLoader/SmartLoaderDataset/Test2/";
std::string xyzPath = "xyz$$c1_d_w3_h17235.bin", intensityPath = "intensity$$c1_d_w1_h17235.bin";
int xyzNumBands, xyzNumBytePerBand, xyzWidth, xyzHeight;
std::vector<unsigned char> xyzData;
int intensityNumBands, intensityNumBytePerBand, intensityWidth, intensityHeight;
std::vector<unsigned char> intensityData;

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

    // Print the size of the preloaded buffers 
   std::vector<float> xyz, x,y,z,intensity;
   auto numPoints = pc.height * pc.width;
   auto isWorkingWithFSdata = false;

    if (!isWorkingWithFSdata)
    {  
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
        printf("size of intensity vector %d, x %d, y %d, z 5d, xyz %d", intensity.size(), x.size(), y.size(), z.size(), xyz.size());
    }
    else 
    {
        printf("\nxyz loaded size %d\n", xyzHeight * xyzWidth );
        printf("\nintensity loaded size %d\n", intensityWidth * intensityHeight );
        numPoints = intensityWidth * intensityHeight;
    }
    printf("\nnumPoints = %d\n", numPoints);

#ifdef SAVE_LOGS
    // printf("x.size=%d", x.size());
    if (x.size() > 0)
    {
        static int counter = 0;
        /*** Note that the path for saving file is hardcoded in BinaryIO.h***/
        std::string dummyString, st, path = "/home/sload/Downloads/temp/3/";

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
#endif    
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

    printf("\nBefore SmartLoader call\n");
    if (isWorkingWithFSdata)
    {
        SmartLoader(globalDataToCallbackFunction.SD->get(), &configParams, (double*)&xyzData[0], xyz_size,
         (double*)&intensityData[0], intensity_size,
        &smartLoaderStruct, &heightMap_res_data[0], heightMap_res_size);
    }
    else 
    {
        SmartLoader(globalDataToCallbackFunction.SD->get(), &configParams, (double*)&xyzDouble[0], xyz_size, (double*)&intensityDouble[0], intensity_size,
        &smartLoaderStruct, &heightMap_res_data[0], heightMap_res_size);
    }
    //printf("\nAfter SmartLoader call\n");
    
    printf("\nSmart loader status %d\nMap Size %d %d\n", smartLoaderStruct.status, 
        heightMap_res_size[0], heightMap_res_size[1]);
    
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    transform.setOrigin( tf::Vector3(0.0, 0.0, 0.0) );
    tf::Quaternion q;
    q.setRPY(0, 0, 0);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "base_link"));
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "map"));
    
    
    //Build GMap topic
    // Grid Map
    std::string second_fixed_frame = "/map";
    gmap.info.header.frame_id = second_fixed_frame;
    // gmap.info.header.seq = pose.header.seq;
    gmap.info.header.stamp = ros::Time::now();

   // printf("\nPrint1\n");

    if (isWorkingWithFSdata) 
    {
        gmap.info.resolution = 0.1; 
        gmap.info.length_x = 10;
        gmap.info.length_y = 10;
        gmap.info.pose.orientation.x = gmap.info.pose.orientation.y = gmap.info.pose.orientation.z = 
            gmap.info.pose.orientation.w = 0;
        // memset(&gmap.info.pose.orientation, 0x00, sizeof(gmap.info.pose.orientation));
        
        gmap.info.pose.position.x = gmap.info.pose.position.y = gmap.info.pose.position.z = 0;
        // memset(&gmap.info.pose.position, 0x00, sizeof(gmap.info.pose.position));
        gmap.basic_layers.push_back(std::string("smartloadMap"));
        gmap.layers.push_back(std::string("smartloadMap"));
        // gmap.data.resize(gmap.info.length_x * gmap.info.resolution 
        //     * gmap.info.length_y * gmap.info.resolution);
        // int numElements = gmap.info.length_x / gmap.info.resolution 
        //      * gmap.info.length_y / gmap.info.resolution;

       int numElements = heightMap_res_size[0] * heightMap_res_size[1];

        std_msgs::Float32MultiArray float32MultiArray; 

        // MultiArrayLayout 
        std_msgs::MultiArrayDimension multiArrayDimensionCol,multiArrayDimensionRow;
        multiArrayDimensionCol.label = std::string("column_index");
        //multiArrayDimensionCol.size = gmap.info.length_x / gmap.info.resolution;
        multiArrayDimensionCol.size = heightMap_res_size[1];
        multiArrayDimensionCol.stride = sizeof(float); 

        multiArrayDimensionRow.label = std::string("row_index");
        // multiArrayDimensionRow.size = gmap.info.length_y / gmap.info.resolution;
        multiArrayDimensionRow.size = heightMap_res_size[0];
        multiArrayDimensionRow.stride = multiArrayDimensionCol.size * sizeof(float); 

        float32MultiArray.layout.dim.push_back(multiArrayDimensionCol);
        float32MultiArray.layout.dim.push_back(multiArrayDimensionRow);
        
        float32MultiArray.layout.data_offset = 0;

        float32MultiArray.data.resize(numElements);

        size_t tt = numElements * sizeof(float);
        float32MultiArray.data = std::vector<float>(heightMap_res_data.begin(), heightMap_res_data.end());

        // memcpy(float32MultiArray.data[0], heightMap_res_data)
        // memcpy((void*)&(float32MultiArray.data[0]), 
        //     (void*)&(heightMap_res_data.data[0]), tt);

        // for (int i = 0; i < numElements; i++)
        // {
        //     float32MultiArray.data.push_back(rand());
        //     //float32MultiArray.data.push_back(0.1 * i);
        // }

        gmap.data.push_back(float32MultiArray);

        // // TODO: Shahar  // TODO: shshar
        // gmap.info.resolution = 0.1; 

        // // TODO: shshar x,y directions 
        // // TODO: shshar
        // gmap.info.length_x = 10;
        // // TODO: shshar
        // gmap.info.length_y = 10;
        // memset(&gmap.info.pose.orientation, 0x00, sizeof(gmap.info.pose.orientation));
        // memset(&gmap.info.pose.position, 0x00, sizeof(gmap.info.pose.position));
        // gmap.info.pose.position.x = 5;
        // gmap.info.pose.position.y = 5;

        // if (heightMap_res_size[0] * heightMap_res_size[1] > 0)
        // {
        //     gmap.layers.push_back(std::string("smartload"));
        //     gmap.data.resize(heightMap_res_size[0] * heightMap_res_size[1]);
        //     memcpy(&gmap.data[0], &heightMap_res_data[0], sizeof(float) * heightMap_res_size[0] * heightMap_res_size[1]);
        // }
        // else 
        // {
        //     gmap.layers.push_back(std::string("smartload"));
        //     gmap.data.resize(1024 * 1024);
        //     memset(&gmap.data[0], 0x00, 1024 * 1024 * sizeof(float));
        // }
    }
    else 
    {
        gmap.info.resolution = 0.1; 
        gmap.info.length_x = 100;
        gmap.info.length_y = 100;
        memset(&gmap.info.pose.orientation, 0x00, sizeof(gmap.info.pose.orientation));
        memset(&gmap.info.pose.position, 0x00, sizeof(gmap.info.pose.position));
        gmap.basic_layers.push_back(std::string("smartloadMap"));
        gmap.layers.push_back(std::string("smartloadMap"));
        // gmap.data.resize(gmap.info.length_x * gmap.info.resolution 
        //     * gmap.info.length_y * gmap.info.resolution);
        int numElements = gmap.info.length_x / gmap.info.resolution 
             * gmap.info.length_y / gmap.info.resolution;
        
        std_msgs::Float32MultiArray float32MultiArray; 

        // MultiArrayLayout 
        std_msgs::MultiArrayDimension multiArrayDimensionCol,multiArrayDimensionRow;
        multiArrayDimensionCol.label = std::string("column_index");
        multiArrayDimensionCol.size = gmap.info.length_x / gmap.info.resolution;
        multiArrayDimensionCol.stride = sizeof(float); 

        multiArrayDimensionRow.label = std::string("row_index");
        multiArrayDimensionRow.size = gmap.info.length_y / gmap.info.resolution;
        multiArrayDimensionRow.stride = multiArrayDimensionCol.size * sizeof(float); 

        float32MultiArray.layout.dim.push_back(multiArrayDimensionCol);
        float32MultiArray.layout.dim.push_back(multiArrayDimensionRow);
        
        float32MultiArray.layout.data_offset = 0;

        float32MultiArray.data.resize(numElements);
        for (int i = 0; i < numElements; i++)
        {
            float32MultiArray.data.push_back(rand());
            //float32MultiArray.data.push_back(0.1 * i);
        }

        gmap.data.push_back(float32MultiArray);
    }

    //printf("\nPrint2\n");

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
    printf("\nLoader location (x=%f, y=%f, z=%f)\n", 
        smartLoaderStruct.loaderLoc[0], smartLoaderStruct.loaderLoc[1], smartLoaderStruct.loaderLoc[2]);
    
    printf("\nShovel location (x=%f, y=%f, z=%f)\n", 
        smartLoaderStruct.shvelLoc[0], smartLoaderStruct.shvelLoc[1], smartLoaderStruct.shvelLoc[2]);

    //Topic localization of the shovel
    //std::string fixed_frame = "/base_link";
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

    //printf("\nPrint3\n");
    sh_pub.publish(sh_pose);
    //printf("\nPrint4\n");
    p_pub.publish(pose);
   // printf("\nPrint5\n");

    if (1) 
    {
        gmap.info.header.seq = pose.header.seq;
        static uint32_t counterTemp = 0;
        gmap.info.header.seq = counterTemp++;
    }

    printf("\nmap seq %d\n", gmap.info.header.seq);

    m_pub.publish(gmap);
    //printf("\nPrint6\n");
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

    //////////////////////////
	auto retVal = IAIRoboticsAlgorithms::BinaryIO::ReadBinary((datasetPath + xyzPath).c_str(), xyzNumBands, xyzNumBytePerBand, xyzWidth, xyzHeight, xyzData);
	//assert(retVal); if (!retVal) return;

	retVal = IAIRoboticsAlgorithms::BinaryIO::ReadBinary((datasetPath + intensityPath).c_str(), intensityNumBands, intensityNumBytePerBand, intensityWidth, intensityHeight, intensityData);
	//assert(retVal); if (!retVal) return;
    /////////////////////////

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