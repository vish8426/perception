#! /usr/bin/env python
#
# Author: Tejaswi Digumarti (tejaswi.digumarti@sydney.edu.au)
# Description: This code defines the MoveGroupPythonInterface class to control the UR5e arm using ROS and MoveIt!

import rospy
import sys
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_commander.conversions import pose_to_list
from create_stage import CreateStage
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    :param goal: A list of floats, a Pose or a PoseStamped
    :param actual: A list of floats, a Pose or a PoseStamped
    :param tolerance: A float
    :return: Boolean indicating if the actual value is close to the goal w.r.t. the specified tolerance
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True


class MoveGroupPythonInteface(object):
    """
    Class that implements an interface to the MoveGroupCommander, PlanningScene and RobotCommander.

    The MoveGroupCommander object interacts with the robot's joints - the planning group.
    The PlanningScene object interacts with objects/obstacles in the scene.
    The RobotCommander object provides the kinematic model and the state of the robot.
    """

    def __init__(self):
        super(MoveGroupPythonInteface, self).__init__()

        # First initialize moveit_commander and a rospy node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('calib_node', anonymous=True)

        # Instantiate a RobotCommander object. Provides information such as the robot's
        # kinematic model and the robot's current joint states
        self.robot = moveit_commander.RobotCommander()

        # Instantiate a PlanningSceneInterface object.  This provides a remote interface
        # for getting, setting, and updating the robot's internal understanding of the
        # surrounding world:
        self.scene_interface = moveit_commander.PlanningSceneInterface()
        self.scene = moveit_commander.PlanningScene()

        # Instantiate a MoveGroupCommander object.
        # This object is an interface to a planning group (group of joints).
        # In our case the group consists of the primary joints in the UR5e robot, defined as "manipulator".
        # This interface can be used to plan and execute motions.
        group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        # Create a DisplayTrajectory ROS publisher which is used to display trajectories in Rviz:
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)

        # Allow re planning to increase the odds of finding a solution
        self.move_group.allow_replanning(True)
        # Set the number of planning attempts - By default it is 1, we are increasing it to 10
        self.move_group.set_num_planning_attempts(10)
        # Set the amount of time allocated for generating a plan - By default it is 5s
        self.move_group.set_planning_time(20)

        # Misc variables
        self.planning_frame = self.move_group.get_planning_frame()
        self.eef_link = self.move_group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()
        self.home_joint_angles = self.assign_home()

        # # Build the stage for RViz (not necessary if RViz is not used)
        self.stage = CreateStage(self.scene_interface, self.robot, self.eef_link)
        # Block [0-4] and goal [5] positions:
        self.stage.build_stage()

    def print_useful_info(self):
        """
        Prints some useful information such as the reference frame, end effector link, available planning groups
        """
        # Reference frame for this robot:
        print("Reference frame used for planning: %s" % self.planning_frame)
        # End-effector link for this group:
        print("End effector link: %s" % self.eef_link)
        # We can get a list of all the groups in the robot:
        print("Available Planning Groups:", self.group_names)

    def print_robot_state(self):
        """
        Prints the current state of the robot
        """
        print("Robot State")
        print(self.robot.get_current_state())

    def move_to_joint_state(self, joint_goal):
        """
        Moves the robot to a desired joint state

        :param joint_goal: A list of 6 values specifying the angles (in radians) to which the 6 joints have to move
        :type joint_goal: list
        """
        self.move_group.go(joint_goal, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group.stop()

    def get_current_joint_state(self):
        """
        Returns the current state of the 6 joints of the robot, in radians

        :return: List of current joint state in radians
        :rtype: list
        """
        return self.move_group.get_current_joint_values()

    def move_eef_to_pose(self, pose_goal):
        """
        Moves the end-effector of the robot to a desired pose. Note that the entire robot moves to achieve.

        :param pose_goal: The desired pose to which the end effector of the robot has to move to.
        :type pose_goal: geometry_msgs.msg.Pose
        """
        # Set the desired goal
        self.move_group.set_pose_target(pose_goal)
        # Move the end effector to the desired pose.
        self.move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        self.move_group.clear_pose_targets()

    def get_current_eef_pose(self):
        """
        Returns the current state of the end effector

        :return: Current pose of the end effector
        :rtype: geometry_msgs.msg.PoseStamped
        """
        return self.move_group.get_current_pose()

    def plan_cartesian_path(self, waypoints, eef_step=0.01, jump=0.0, max_attempts=200):
        """
        Plans a Cartesian path for the end-effector of the robot following the specified waypoints.

        This function only plans the path (if it exists).
        Use execute_plan() to execute the planned path and display_trajectory() to display the planed path
        """
        fraction = 0.0
        attempts = 0
        while fraction < 1.0 and attempts < max_attempts:
            (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, eef_step, jump, True)
            # Increment the number of attempts
            attempts += 1

            # Print out a progress message
            if attempts % 10 == 0:
                rospy.loginfo("Still trying after " + str(attempts) + " attempts...")

            if fraction == 1.0:
                rospy.loginfo("Path computed successfully. Moving the arm.")
                return plan, fraction
            else:
                rospy.loginfo("Path planning failed with only " +
                              str(fraction) + " success after " +
                              str(max_attempts) + " attempts.")
                return plan, fraction

    def display_trajectory(self, plan):
        """
        Visualizes a planned trajectory in RViz.

        move_group.plan() calls this function automatically, but in case you want to visualize a plan,
        you can use this function.

        :param plan: The plan to visualize
        :type plan: List
        """
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish plan to RViz
        self.display_trajectory_publisher.publish(display_trajectory)

    def execute_plan(self, plan):
        """
        Executes a plan and moves the robot such that the it's end effector follows the specified plan

        :param plan: The plan to execute. Use plan_cartesian_path to plan the path
        :type plan: List
        """
        self.move_group.execute(plan, wait=True)
        # **Note:** The robot's current joint state must be within some tolerance of the
        # first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail

    def assign_home(self):
        """
        Convenience function to assign home joint angles.

        This "home" state is defined in universal_robot/ur5_e_moveit_config/config/ur5e.srdf
        # Please DO NOT edit that file.
        :return: A list of 6 joint angles in radians
        :rtype: List
        """
        home_joint_angle_dict = self.move_group.get_named_target_values("home")
        return [home_joint_angle_dict['shoulder_pan_joint'],
                home_joint_angle_dict['shoulder_lift_joint'],
                home_joint_angle_dict['elbow_joint'],
                home_joint_angle_dict['wrist_1_joint'],
                home_joint_angle_dict['wrist_2_joint'],
                home_joint_angle_dict['wrist_3_joint']]
