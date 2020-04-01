#include <tf/transform_listener.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

double pose(int flag);
ros::Publisher p_pub;

void onPoseSet(double x, double y, double theta);


void onPoseSet(double x, double y, double theta)
{   
    std::string fixed_frame = "map";
    geometry_msgs::PoseWithCovarianceStamped pose;
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

    // publish
    ROS_INFO("x: %f, y: %f, z: 0.0, theta: %f",x,y,theta);
    p_pub.publish(pose.pose.pose);
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
 
    p_pub = nh.advertise<geometry_msgs::Pose>("sl_pose", 1);
    ros::Rate rate(publish_frequency);
    while (nh.ok()){
          onPoseSet(0.968507917975, -4.61491396765, 1); 
          rate.sleep(); 
          }    
   return 0;
}