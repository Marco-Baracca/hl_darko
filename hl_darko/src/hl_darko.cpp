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

Eigen::VectorXd initial_EE_point(6);
bool initial_pose_init = false;
geometry_msgs::Pose msg_pose;

void load_fpc()
{
    if (!file_stream.is_open())
    {
        std::cout << "ERRORE APERTURA .csv\n";
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

// Convert xyzrpy vector to geometry_msgs Pose (PRESA DA PANDA-SOFTHAND -> TaskSequencer.cpp)
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
    tf::poseEigenToMsg(output_affine, output_pose);  //CONTROLLARE SE METTERE #include <eigen_conversions/eigen_msg.h>
    return output_pose;
}

void robotPoseCallback(const geometry_msgs::PoseStamped& msg)
{
    if (!initial_pose_init)
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

        
        initial_EE_point << traslazione[0], traslazione[1], traslazione[2], rpy[2], rpy[1], rpy[0];
        std::cout << "Starting pose read from topic:" << std::endl;
        std::cout << initial_EE_point << std::endl;

        initial_pose_init = true;
    }
    
}


int main(int argc, char **argv)
{
    pre_rot << 1, 0, 0, 0, -1, 0, 0, 0, -1;
    //Initialize the node
    ROS_INFO("NODE INITIALIZATION");
    ros::init(argc, argv, "HL_planner");
    ros::NodeHandle node;

    //Initialize frame trajectory publisher
    ROS_INFO("PUBLISHER INITIALIZATION");
    ros::Publisher pub = node.advertise<geometry_msgs::PoseStamped>("/franka/equilibrium_pose", 1); //DA SISTEMARE SIA PER IL NOME CHE PER IL TIPO DI MSG

    //Initialize starting pose subscriber
    ros::Subscriber robot_pose_sub = node.subscribe("/franka_ee_pose", 1, robotPoseCallback);
    
    //Load the fPCs
    ROS_INFO("LOAD fPCs");
    load_fpc();

    ROS_INFO("Variable definitions");
    //Starting and ending times definition
    double t_start = 0;
    double t_end;

    std::cout << "Input time to perform trajectory [s]";
    std::cin >> t_end;

    //Time axis definition
    Eigen::VectorXd t;
    t = Eigen::VectorXd::LinSpaced(NUM_COLS, t_start, t_end);

    //Velocities constrains definition
    Eigen::VectorXd initial_velocity(6);
    Eigen::VectorXd final_velocity(6);

    initial_velocity << 0, 0, 0, 0, 0, 0; //POI DA SOSTITUIRE CON QUELLA ALL'ISTANTE INIZIALE DEL ROBOT
    
    double aux;
    //std::cout << "Input final velocity vector element by element (xyzrpy)";
    //for (int i = 0; i < 6; i++)
    //{
    //    std::cin >> aux;
    //    final_velocity(i) = aux;
    //}
    
    final_velocity << 0, 0, 0, 0, 0, 0;

    //Cartesian constrains definition

    Eigen::VectorXd final_EE_point(6);

    std::cout << "Input final pose vector element by element (xyzrpy)";
    
    for (int i = 0; i < 6; i++)
    {
        std::cin >> aux;
        final_EE_point(i) = aux;
    }

    ROS_INFO("INPUT COMPLETED");


    double dt;
    dt = t(2)-t(1);

    while (ros::ok())
    {
        ros::spinOnce();
        if (initial_pose_init)
        {
            ROS_INFO("Trajectory Computation");
            //Trajectory computation
            for (int i = 0; i < 6; i++)
            {
                single_dof(initial_EE_point(i), final_EE_point(i), initial_velocity(i), final_velocity(i), i, dt);
            }
            
            Eigen::VectorXd actual_pose;
            geometry_msgs::Pose actual_pose_msg;
            geometry_msgs::PoseStamped actual_posestamped_msg;

            ros::Rate rate(1/dt);

            ROS_INFO("Trajectory Publishing");
            for (int i = 0; i < NUM_COLS; i++)
            {
                actual_pose = ee_trajectory.col(i);
                actual_pose_msg = convert_vector_to_pose(actual_pose);
                actual_posestamped_msg.pose = actual_pose_msg;
                if (i==0)
                {
                    std::cout << "First Frame" << std::endl;
                    std::cout << actual_pose_msg << std::endl;
                }
                actual_posestamped_msg.header.stamp = ros::Time::now();
                pub.publish(actual_posestamped_msg);
                rate.sleep();
            }

            break;
        }
        ROS_INFO("WAITING INITIAL POSE");
        
    }
    
    ROS_INFO("TASK COMPLETED");

    std::ofstream myfile_1;
    myfile_1.open("/home/mb/catkin_ws/src/hl_planning/src/traiettoria_calcolata.csv");
    myfile_1 << ee_trajectory.format(CSVFormat);
    myfile_1.close();


}