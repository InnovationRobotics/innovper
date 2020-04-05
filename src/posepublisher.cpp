#include <tf/transform_listener.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <grid_map_msgs/GridMap.h>
//#include <grid_map_msgs/GridMapInfo.h>
#include <tf/transform_broadcaster.h>


//double pose(int flag);
geometry_msgs::PoseWithCovarianceStamped pose;
sensor_msgs::PointCloud2 pcloud;
grid_map_msgs::GridMap gmap;

ros::Publisher p_pub, m_pub;

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
   onPoseSet(0.968507917975, -4.61491396765, 1); 
   

}

int main(int argc, char** argv)
{
    // setting
    ros::init(argc, argv, "sl_pose");
    ros::NodeHandle nh;
    ros::NodeHandle nh_priv("~");
    std::string map_frame, base_frame;
    double publish_frequency = 10;
    bool is_stamped = false;
    ros::Subscriber s_sub;
    
    s_sub = nh.subscribe("/velodyne_points", 1000, dealWithPCloudCB);
   
    m_pub = nh.advertise<grid_map_msgs::GridMap>("sl_map", 1);

    p_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("sl_pose", 1);
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
  
   return 0;
}