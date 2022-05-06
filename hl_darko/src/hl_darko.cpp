#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>

#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>

#include <ros/ros.h>
#include <geometry_msgs/Pose.h>

#include <std_msgs/Float64.h>

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");


#define NUM_ROWS 24
#define NUM_COLS 128

std::string matrix_name = "/home/mb/catkin_ws/src/hl_darko/src/matrice_pc.csv";
std::ifstream file_stream(matrix_name);
std::string data_row_str;
std::vector<std::string> data_row;
Eigen::MatrixXd data_matrix;

Eigen::Matrix3d pre_rot;

Eigen::MatrixXd ee_trajectory(6, NUM_COLS);

double safe_velocity = 1000;
Eigen::RowVectorXd max_norm_velocity(NUM_COLS-1);
Eigen::RowVectorXd max_norm_velocity_backward(NUM_COLS-1);

Eigen::VectorXd initial_EE_point(6);
Eigen::VectorXd final_EE_point(6);
bool initial_pose_init = false;
bool desired_pose_init = false;
geometry_msgs::Pose msg_pose;

void load_fpc()
{
    if (!file_stream.is_open())
    {
        std::cout << "ERROR OPENING .csv\n";
    }

    data_matrix.resize(NUM_ROWS, NUM_COLS);

    for (int i = 0; i < NUM_ROWS; ++i)
    {
        std::getline(file_stream, data_row_str);
        boost::algorithm::split(data_row, data_row_str, boost::is_any_of(","), boost::token_compress_on);
        for (int j = 0; j < NUM_COLS; ++j)
        {
            data_matrix(i,j) = std::stod(data_row[j]);
        }
    }
}

void single_dof(double starting_point, double ending_point, double starting_vel, double ending_vel, int i, double dt)
{
    //Starting and ending values of fPCs
    double fPC0_start = data_matrix(i*4,0);
    double fPC1_start = data_matrix(i*4+1,0);
    double fPC2_start = data_matrix(i*4+2,0);
    double fPC3_start = data_matrix(i*4+3,0);
    double fPC0_end = data_matrix(i*4,NUM_COLS-1);
    double fPC1_end = data_matrix(i*4+1,NUM_COLS-1);
    double fPC2_end = data_matrix(i*4+2,NUM_COLS-1);
    double fPC3_end = data_matrix(i*4+3,NUM_COLS-1);

    //Starting and ending values of first derivative of fPCs
    double fPC0dot_start = (data_matrix(i*4,1) - data_matrix(i*4,0))/dt;
    double fPC1dot_start = (data_matrix(i*4+1,1) - data_matrix(i*4+1,0))/dt;
    double fPC2dot_start = (data_matrix(i*4+2,1) - data_matrix(i*4+2,0))/dt;
    double fPC3dot_start = (data_matrix(i*4+3,1) - data_matrix(i*4+3,0))/dt;
    double fPC0dot_end = (data_matrix(i*4,NUM_COLS-1) - data_matrix(i*4,NUM_COLS-2))/dt;
    double fPC1dot_end = (data_matrix(i*4+1,NUM_COLS-1) - data_matrix(i*4+1,NUM_COLS-2))/dt;
    double fPC2dot_end = (data_matrix(i*4+2,NUM_COLS-1) - data_matrix(i*4+2,NUM_COLS-2))/dt;
    double fPC3dot_end = (data_matrix(i*4+3,NUM_COLS-1) - data_matrix(i*4+3,NUM_COLS-2))/dt;

    //Definition of linear system for trajectory computation
    Eigen::Vector4d b;
    b << starting_point-fPC0_start, ending_point-fPC0_end, starting_vel-fPC0dot_start, ending_vel-fPC0dot_end;

    Eigen::Matrix4d A;
    Eigen::Matrix4d A_I;
    A << 1, fPC1_start, fPC2_start, fPC3_start,
         1, fPC1_end, fPC2_end, fPC3_end,
         0, fPC1dot_start, fPC2dot_start, fPC3dot_start,
         0, fPC1dot_end, fPC2dot_end, fPC3dot_end;
    
    A_I = A.inverse();

    //fPCs' weight computation
    Eigen::Vector4d x;
    x = A_I*b;

    //Single DoF trajectory computation
    Eigen::RowVectorXd trajectory_single_dof;
    trajectory_single_dof = x(0)*Eigen::RowVectorXd::Ones(NUM_COLS) + data_matrix.row(i*4) + x(1)*data_matrix.row(i*4+1) + x(2)*data_matrix.row(i*4+2) + x(3)*data_matrix.row(i*4+3);

    ee_trajectory.block(i,0,1,NUM_COLS) = trajectory_single_dof;
}

// Convert xyzrpy vector to geometry_msgs Pose
geometry_msgs::Pose convert_vector_to_pose(Eigen::VectorXd input_vec){
    
    // Creating temporary variables
    geometry_msgs::Pose output_pose;
    Eigen::Affine3d output_affine;

    // Getting translation and rotation
    Eigen::Vector3d translation(input_vec[0], input_vec[1], input_vec[2]);
    output_affine.translation() = translation;
    Eigen::Matrix3d rotation = Eigen::Matrix3d(Eigen::AngleAxisd(input_vec[5], Eigen::Vector3d::UnitZ())
        * Eigen::AngleAxisd(input_vec[4], Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(input_vec[3], Eigen::Vector3d::UnitX()));
    rotation = pre_rot*rotation;
    output_affine.linear() = rotation;    
    
    // Converting to geometry_msgs and returning
    tf::poseEigenToMsg(output_affine, output_pose);
    return output_pose;
}

void robotPoseCallback(const geometry_msgs::PoseStamped& msg)
{
    if (!initial_pose_init)
    {
        msg_pose = msg.pose;
        std::cout << "Message received" << std::endl;
        //std::cout << msg << std::endl;
        Eigen::Affine3d input_affine;
        Eigen::Vector3d traslazione;
        Eigen::Vector3d rpy;
        Eigen::Matrix3d mat_rotazione;
        tf::poseMsgToEigen(msg_pose,input_affine);
        traslazione = input_affine.translation();
        mat_rotazione = input_affine.rotation();
        mat_rotazione = pre_rot*mat_rotazione;
        rpy = mat_rotazione.eulerAngles(2, 1, 0);

        
        initial_EE_point << traslazione[0], traslazione[1], traslazione[2], rpy[2], rpy[1], rpy[0];
        // CHECK ON THE FINAL POSE LISTENING
        //std::cout << "Starting pose read from topic:" << std::endl;
        //std::cout << initial_EE_point << std::endl;

        initial_pose_init = true;
    }
    
}

void desiredPoseCallback(const geometry_msgs::PoseStamped& msg)
{
    if (!desired_pose_init)
    {
        msg_pose = msg.pose;
        std::cout << "Message received" << std::endl;
        std::cout << msg << std::endl;
        Eigen::Affine3d input_affine;
        Eigen::Vector3d traslazione;
        Eigen::Vector3d rpy;
        Eigen::Matrix3d mat_rotazione;
        tf::poseMsgToEigen(msg_pose,input_affine);
        traslazione = input_affine.translation();
        mat_rotazione = input_affine.rotation();
        mat_rotazione = pre_rot*mat_rotazione;
        rpy = mat_rotazione.eulerAngles(2, 1, 0);

        
        final_EE_point << traslazione[0], traslazione[1], traslazione[2], rpy[2], rpy[1], rpy[0];
        // CHECK ON THE FINAL POSE LISTENING
        //std::cout << "Desired pose read from topic:" << std::endl;
        //std::cout << final_EE_point << std::endl;

        desired_pose_init = true;
    }
    
}

void safeVelocityCallback(const std_msgs::Float64& msg)
{
    safe_velocity = msg.data;
}

void trajectory_maximum_velocity()
{
    Eigen::RowVectorXd norm_velocity(NUM_COLS-1);

    for (int i = 0; i < NUM_COLS-1 ; i++)
    {
        Eigen::Vector3d aux = ee_trajectory.block(0, i+1, 3, 1) - ee_trajectory.block(0, i, 3, 1);
        norm_velocity[i] = aux.norm();
    }
    double max_aux = 0;
    double max_aux_bkw = 0;
    for (int i = NUM_COLS-2; i >= 0; i--)
    {
        if (max_aux > norm_velocity[i])
        {
            max_norm_velocity[i] = max_aux;
        }else{
            max_norm_velocity[i] = norm_velocity[i];
            max_aux = norm_velocity[i];
        } 
    }
    for (int i = 0; i < NUM_COLS -1; i++)
    {
        if (max_aux_bkw > norm_velocity[i])
        {
            max_norm_velocity_backward[i] = max_aux;
        }else{
            max_norm_velocity_backward[i] = norm_velocity[i];
            max_aux = norm_velocity[i];
        } 
    }
}


int main(int argc, char **argv)
{
    pre_rot << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    //pre_rot << 1, 0, 0, 0, -1, 0, 0, 0, -1;
    //Initialize the node
    ROS_INFO("NODE INITIALIZATION");
    ros::init(argc, argv, "HL_planner");
    ros::NodeHandle node;

    //Initialize frame trajectory publisher
    ROS_INFO("PUBLISHER INITIALIZATION");
    ros::Publisher pub = node.advertise<geometry_msgs::PoseStamped>("/manipulation/equilibrium_pose", 1);

    //Initialize starting pose subscriber
    ros::Subscriber robot_pose_sub = node.subscribe("/manipulation/franka_ee_pose", 1, robotPoseCallback);

    //Initialize desired pose subscriber
    ros::Subscriber desired_pose_sub = node.subscribe("/manipulation/desired_ee_pose", 1, desiredPoseCallback);

    //Initialize maximum velocity
    ros::Subscriber safe_velocity_sub = node.subscribe("/manipulation/safe_velocity", 1, safeVelocityCallback);
    
    //Load the fPCs
    ROS_INFO("LOAD fPCs");
    load_fpc();

    ROS_INFO("Variable definitions");
    //Starting and ending times definition
    double t_start = 0;
    double t_end;
    //double t_end = 5;

    std::cout << "Input time to perform trajectory [s]";
    std::cin >> t_end;

    //Time axis definition
    Eigen::VectorXd t;
    t = Eigen::VectorXd::LinSpaced(NUM_COLS, t_start, t_end);

    //Velocities constrains definition
    Eigen::VectorXd initial_velocity(6);
    Eigen::VectorXd final_velocity(6);

    initial_velocity << 0, 0, 0, 0, 0, 0;
    
    final_velocity << 0, 0, 0, 0, 0, 0;


    //Sampling time definition
    double dt_min;
    double dt;
    double dt_orig = t(2)-t(1);
    dt = dt_orig;

    while (ros::ok())
    {
        ros::spinOnce();
        if (initial_pose_init)
        {
            if (desired_pose_init)
            {
            ROS_INFO("Trajectory Computation");
            //Trajectory computation
                for (int i = 0; i < 6; i++)
                {
                    single_dof(initial_EE_point(i), final_EE_point(i), initial_velocity(i), final_velocity(i), i, dt);
                }

                trajectory_maximum_velocity();                
                
                Eigen::VectorXd actual_pose;
                geometry_msgs::Pose actual_pose_msg;
                geometry_msgs::PoseStamped actual_posestamped_msg;

                ros::Rate rate(1/dt);

                ROS_INFO("Trajectory Publishing");
                // Reaching the target
                for (int i = 0; i < NUM_COLS; i++)
                {
                    actual_pose = ee_trajectory.col(i);
                    actual_pose_msg = convert_vector_to_pose(actual_pose);
                    actual_posestamped_msg.pose = actual_pose_msg;
                    // CHECK ON THE INITIAL POSE LISTENING
                    //if (i==0)
                    //{
                    //    std::cout << "First Frame" << std::endl;
                    //    std::cout << actual_pose_msg << std::endl;
                    //}
                    actual_posestamped_msg.header.stamp = ros::Time::now();
                    pub.publish(actual_posestamped_msg);
                    if (i < (NUM_COLS-1))
                    {
                        //std::cout << i << std::endl;
                        dt_min = max_norm_velocity[i]/safe_velocity;
                        if (dt < dt_min)
                        {
                            //ROS_INFO("CAMBIO dt");
                            dt = dt_min;
                            rate = ros::Rate (1/dt);
                        }
                    }

                    rate.sleep();
                }

                //PUT A COMMAND TO CLOSE THE GRIPPER

                //Come back to the initial position

                dt = dt_orig;
                rate = ros::Rate(1/dt);

                for (int i = NUM_COLS-1; i > 0; i--)
                {
                    actual_pose = ee_trajectory.col(i);
                    actual_pose_msg = convert_vector_to_pose(actual_pose);
                    actual_posestamped_msg.pose = actual_pose_msg;
                    actual_posestamped_msg.header.stamp = ros::Time::now();
                    pub.publish(actual_posestamped_msg);
                    if (i < (NUM_COLS-1)) //to be corrected for backward trajectory
                    {
                        //std::cout << i << std::endl;
                        dt_min = max_norm_velocity[i]/safe_velocity;
                        if (dt < dt_min)
                        {
                            ROS_INFO("CAMBIO dt");
                            dt = dt_min;
                            rate = ros::Rate(1/dt);
                        }
                    }

                    rate.sleep();
                }

                break;
            }
            ROS_INFO("WAITING DESIRED POSE");
        }
        if (!desired_pose_init)
        {
            ROS_INFO("WAITING INITIAL POSE");
        }   
    }
    
    ROS_INFO("TASK COMPLETED");

}