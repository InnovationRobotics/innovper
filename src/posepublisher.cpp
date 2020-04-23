#include <tf/transform_listener.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <grid_map_msgs/GridMap.h>
//#include <grid_map_msgs/GridMapInfo.h>
#include <tf/transform_broadcaster.h>
#include <std_msgs/Float32MultiArray.h>
#include "BinaryIO.h"
#include "PerceptionSmartLoader.h"


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
    PerceptionSmartLoaderConfigParam* configParams;
    std::unique_ptr<PerceptionSmartLoaderStackData>* SD;
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

const char* GetPerceptionSmartLoaderStatusString(PerceptionSmartLoaderReturnValue status)
{
    if (status == PerceptionSmartLoaderReturnValue::PerceptionSmartLoaderReturnValue_eSuccess)
        return "eSuccess";

    else if (status == PerceptionSmartLoaderReturnValue::PerceptionSmartLoaderReturnValue_eFailed)
        return "eFailed";

    else if (status == PerceptionSmartLoaderReturnValue::PerceptionSmartLoaderReturnValue_eFailedNotEnoughPoints)
        return "eFailedNotEnoughPoints";

    else if (status == PerceptionSmartLoaderReturnValue::PerceptionSmartLoaderReturnValue_eFailedNotEnoughReflectorPoints)
        return "eFailedNotEnoughReflectorPoints";
            
    else if (status == PerceptionSmartLoaderReturnValue::PerceptionSmartLoaderReturnValue_eFailedLoaderLocation)
        return "eFailedLoaderLocation";
    
    else 
        throw("unsupported return value");
}


std::vector<double> globalxyz, globalIntensity;

void dealWithPCloudCB(sensor_msgs::PointCloud2 pc)
{
   ROS_INFO("Got Point Cloud message %x, %d", &pc, pc.header.seq);
   //onPoseSet(0.968507917975, -4.61491396765, 1);

    // Print the size of the preloaded buffers 
   globalxyz.resize(0);
   globalIntensity.resize(0);

   auto numPoints = pc.height * pc.width;
   auto isWorkingWithFSdata = false;

    if (!isWorkingWithFSdata)
    {  
        globalxyz.reserve(numPoints);
        globalIntensity.reserve(numPoints);

        for (auto j = 0; j < pc.height; j++)
        {
            for (auto i = 0; i < pc.width; i++)
            {
                auto loc = j * pc.row_step + i * pc.point_step;

                globalxyz.push_back(double(*((float*)&pc.data[loc])));
                globalxyz.push_back(double(*((float*)&pc.data[loc + 4])));
                globalxyz.push_back(double(*((float*)&pc.data[loc + 8])));
                globalIntensity.push_back(double(*((float*)&pc.data[loc + 12])));    
            }
        }
        //printf("size of intensity vector %d, x %d, y %d, z %d, xyz %d", intensity.size(), x.size(), y.size(), z.size(), xyz.size());
    }
    else 
    {
        printf("\nxyz loaded size %d\n", xyzHeight * xyzWidth );
        printf("\nintensity loaded size %d\n", intensityWidth * intensityHeight );
        numPoints = intensityWidth * intensityHeight;
    }
    //printf("\nnumPoints = %d\n", numPoints);

// #define SAVE_LOGS 
#ifdef SAVE_LOGS
    // printf("x.size=%d", x.size());
    if (xyz.size() > 0)
    {
        static int counter = 0;
        /*** Note that the path for saving file is hardcoded in BinaryIO.h***/
        IAIRoboticsAlgorithms::BinaryIO::WriteBinary(&xyz[0], 1, sizeof(float), xyz.size()/3, 3, std::to_string(counter) + "xyz");
        IAIRoboticsAlgorithms::BinaryIO::WriteBinary(&intensity[0], 1, sizeof(float), intensity.size(), 1, std::to_string(counter) + "i");
    }
#endif    
    // TODO: apply the algorithm
    // Udpate the time tag in the config 
    // // // // auto& t = pc.header.stamp; 
    // // // // // ros::Time t = pc.header.stamp; 
    // // // // auto tt = t.toNSec();
    auto& configParams = *(globalDataToCallbackFunction.configParams);
    configParams.timeTagMs = pc.header.stamp.nsec * 1000 * 1000;

    int xyz_size[2] = { numPoints, 3 }, intensity_size[2] = { numPoints, 1 };
	
    PerceptionSmartLoaderStruct smartLoaderStruct;

    std::vector<float> heightMap_res_data;
	heightMap_res_data.resize(1024 * 1024);
	int heightMap_res_size[2] = { 0,0 };

    // printf("\nBefore SmartLoader call\n");
    if (isWorkingWithFSdata)
    {
        // PerceptionSmartLoader(globalDataToCallbackFunction.SD->get(), &configParams, (double*)&xyzData[0], xyz_size,
        //  (double*)&intensityData[0], intensity_size,
        // &smartLoaderStruct, &heightMap_res_data[0], heightMap_res_size);
    }
    else 
    {
        PerceptionSmartLoader(globalDataToCallbackFunction.SD->get(), &configParams, 
            (double*)&globalxyz[0], xyz_size, (double*)&globalIntensity[0], intensity_size,
            &smartLoaderStruct, &heightMap_res_data[0], heightMap_res_size);
    }
    //printf("\nAfter SmartLoader call\n");
    
    printf("Smart loader status %s\tMap Size %d %d\n", GetPerceptionSmartLoaderStatusString(smartLoaderStruct.status), heightMap_res_size[0], heightMap_res_size[1]);
    
    // @Shahar - what are these used for? 
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
    }
    else if (smartLoaderStruct.heightMapStatus)
    {
        
        // # Resolution of the grid [m/cell].
        gmap.info.resolution = configParams.heightMapResolutionMeterToPixel; 
        // # Length in x-direction [m].
        gmap.info.length_x = heightMap_res_size[1] * configParams.heightMapResolutionMeterToPixel;
        // # Length in y-direction [m].
        gmap.info.length_y = heightMap_res_size[0] * configParams.heightMapResolutionMeterToPixel;
   

        // # Pose of the grid map center in the frame defined in `header` [m].
        memset(&gmap.info.pose.orientation, 0x00, sizeof(gmap.info.pose.orientation));
        memset(&gmap.info.pose.position, 0x00, sizeof(gmap.info.pose.position));

        //# Grid map basic layer names (optional). The basic layers
        //# determine which layers from `layers` need to be valid
        //# in order for a cell of the grid map to be valid.
        gmap.basic_layers.push_back(std::string("smartloadMap"));

        // # Grid map layer names.
        gmap.layers.push_back(std::string("smartloadMap"));

//         # Accessors should ALWAYS be written in terms of dimension stride
// # and specified outer-most dimension first.
// # 
// # multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
// #
// # A standard, 3-channel 640x480 image with interleaved color channels
// # would be specified as:
// #
// # dim[0].label  = "height"
// # dim[0].size   = 480
// # dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
// # dim[1].label  = "width"
// # dim[1].size   = 640
// # dim[1].stride = 3*640 = 1920
// # dim[2].label  = "channel"
// # dim[2].size   = 3
// # dim[2].stride = 3
// #
// # multiarray(i,j,k) refers to the ith row, jth column, and kth channel.

        // # Array of dimension properties
        std_msgs::MultiArrayDimension multiArrayDimensionCol,multiArrayDimensionRow;

        // rvis naming
        //multiArrayDimensionCol.label = std::string("column_index");
        multiArrayDimensionCol.label = std::string("width");
        multiArrayDimensionCol.size = heightMap_res_size[1];
        multiArrayDimensionCol.stride = multiArrayDimensionCol.size * sizeof(float);

        // rvis naming
        //multiArrayDimensionRow.label = std::string("row_index");
        multiArrayDimensionRow.label = std::string("height");
        multiArrayDimensionRow.size = heightMap_res_size[0];
        multiArrayDimensionRow.stride = multiArrayDimensionCol.size * multiArrayDimensionRow.size * sizeof(float); 
       
        // # specification of data layout

        // # The multiarray declares a generic multi-dimensional array of a
        // # particular data type.  Dimensions are ordered from outer most
        // # to inner most.
        // # Grid map data.
        std_msgs::Float32MultiArray float32MultiArray; 

        // Shahar order
        float32MultiArray.layout.dim.push_back(multiArrayDimensionRow);
        float32MultiArray.layout.dim.push_back(multiArrayDimensionCol);
        // rvis order        
        //float32MultiArray.layout.dim.push_back(multiArrayDimensionCol);
        //float32MultiArray.layout.dim.push_back(multiArrayDimensionRow);

        float32MultiArray.layout.data_offset = 0;

        auto sz = heightMap_res_size[0] * heightMap_res_size[1];
        float32MultiArray.data.resize(sz);

        memcpy(&(float32MultiArray.data[0]), &(heightMap_res_data[0]), size_t(sz * sizeof(float)));

        gmap.data.push_back(float32MultiArray);

        // Row start index (default 0).
        gmap.inner_start_index = 0;
        // Column start index (default 0).
        gmap.outer_start_index = 0;
    }

    //Topic localization of the vehicle
    std::string fixed_frame = "/base_link";
    pose.header.frame_id = fixed_frame;
    pose.header.stamp = gmap.info.header.stamp;

    // set x,y coord

    // TODO : memset all ros topics before usage --> TOOD Michele 

    // tood = shshar chagne to loader status - TOOD 
    if (smartLoaderStruct.loaderLocStatus)
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

    bool isPrintVersion = false;
    if (isPrintVersion)
    {
        printf("\nLoader location (x=%f, y=%f, z=%f)\n", 
        smartLoaderStruct.loaderLoc[0], smartLoaderStruct.loaderLoc[1], smartLoaderStruct.loaderLoc[2]);
    
        printf("\nShovel location (x=%f, y=%f, z=%f)\n", 
        smartLoaderStruct.shovelLoc[0], smartLoaderStruct.shovelLoc[1], smartLoaderStruct.shovelLoc[2]);
    }
    
    //Topic localization of the shovel
    //std::string fixed_frame = "/base_link";
    sh_pose.header.frame_id = fixed_frame;
    sh_pose.header.stamp = gmap.info.header.stamp;

    if (smartLoaderStruct.shovelLocStatus)
    {
        sh_pose.pose.pose.position.x = smartLoaderStruct.shovelLoc[0];
        sh_pose.pose.pose.position.y = smartLoaderStruct.shovelLoc[1];
        sh_pose.pose.pose.position.z = smartLoaderStruct.shovelLoc[2];
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

    // printf("\nmap seq %d\n", gmap.info.header.seq);
    m_pub.publish(gmap);
    //printf("\nPrint6\n");
}


void SetDefaultSmartLoaderConfigParams(PerceptionSmartLoaderConfigParam& configParams)
{
    memset(&configParams, 0x00, sizeof(configParams));
	configParams.timeTagMs = 1;

	configParams.maxDistanceToPlaneMeter = 0.04;
	configParams.minNumPointsInPc = 64;
	configParams.minimumDistanceFromLoaderToPlaneMeter = 0.2;
	configParams.minPointsForReflector = 5;
	configParams.maximumTimeTagDiffMs = 1000;
	configParams.minimumIntensityReflectorValue = 100;
	configParams.loaderReflectorDiameterMeter = 0.18;
	configParams.loaderWhiteHatMeter = 0.15;
	configParams.loaderCenterToBackwardPointMeter = 0.2850;
	configParams.locationsBiasMeter = 0.05;
	configParams.loaderWidthMeter = 0.233;
	configParams.reflectorMaxZaxisDistanceForOutlierMeter = 0.06;
	configParams.previousLoaderLocationToCurrentLocationMaximumDistanceMeter = 0.18;
	configParams.loaderReflectorMaxZaxisDistanceForOutlierMeter = 0.04;
	configParams.maxDistanceBetweenEachRayMeter = 0.07;
	configParams.heightMapResolutionMeterToPixel = 0.04;
	configParams.maxDistanceFromThePlaneForLoaderYawCalculation = 0.100000000000000;
	configParams.yawEstimationMinPercentageOfPointsInLoaderBody = 0.6;
	configParams.yawEstimationMinNumPointsInLoaderBody = 40;
	configParams.loaderYawAngleSmoothWeight = 0.6;
	configParams.loaderToShovelYawAngleSmoothWeight = 0.6;

	configParams.debugMode = false;

    {
		double planeModelParameters[4] = { 0.0090749003000000008634096104742638999596238136291503906250000000,0.0167408289999999987385237432135909330099821090698242187500000000,0.9998186799999999596622046738048084080219268798828125000000000000,-0.0239179950000000009213696472443189122714102268218994140625000000 };
		memcpy(&configParams.planeModelParameters[0], planeModelParameters, sizeof(planeModelParameters));
	}

	{
		double pcAlignmentProjMat[12] = { -0.0523359562429437999431236505643028067424893379211425781250000000,0.9986295347545740552774873322050552815198898315429687500000000000,0.0000000000000000064093061293237101741858507329833845334833240759,1.3367783418562400044038440682925283908843994140625000000000000000,-0.0000000000000001224646799147350002426336569097937119478395663359,0.0000000000000000000000000000000000000000000000000000000000000000,-1.0000000000000000000000000000000000000000000000000000000000000000,0.8069977760314940296026975374843459576368331909179687500000000000,-0.9986295347545740552774873322050552815198898315429687500000000000,-0.0523359562429437999431236505643028067424893379211425781250000000,0.0000000000000001222968463271199898913999796741335001694767059804,2.6988332719881098498149185616057366132736206054687500000000000000 };
		memcpy(&configParams.pcAlignmentProjMat[0], pcAlignmentProjMat, sizeof(pcAlignmentProjMat));
	}
	
	{
		double xyzLimits[6] = { 0.0000000000000000000000000000000000000000000000000000000000000000,2.6000000000000000888178419700125232338905334472656250000000000000,0.0000000000000000000000000000000000000000000000000000000000000000,1.6000000000000000888178419700125232338905334472656250000000000000,-0.2999999999999999888977697537484345957636833190917968750000000000,1.0000000000000000000000000000000000000000000000000000000000000000 };
		memcpy(configParams.xyzLimits, xyzLimits, sizeof(configParams.xyzLimits));
	}
}


int main(int argc, char** argv)
{
    // Initialize perception code
    PerceptionSmartLoaderConfigParam configParams;
    SetDefaultSmartLoaderConfigParams(configParams);
    auto SD = std::make_unique<PerceptionSmartLoaderStackData>();
	auto pd = std::make_unique<PerceptionSmartLoaderPersistentData>();
	SD->pd = pd.get();

	PerceptionSmartLoader_initialize(SD.get());

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
  

  PerceptionSmartLoader_terminate(SD.get());

   return 0;
}