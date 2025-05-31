#! /usr/bin/env python

# Author: Tejaswi Digumarti (tejaswi.digumarti@sydney.edu.au)
# Description: This is the file that creates the robot arena for assignment 1.

import rospy
import geometry_msgs.msg
import numpy as np

class CreateStage(object):
	"""
	This class contains functions to create the stage for the planning scene.
	It also has utilities to to attach and detach objects in the planning scene and in Gazebo.
	"""

	def __init__(self, scene_interface, robot, eef_link):
		"""
		Initializes the object

		:param scene_interface: A PlanningSceneInterface object
		:param robot: The robot
		:param eef_link: Name of the end effector link
		"""
		super(CreateStage, self).__init__()
		self.scene_interface = scene_interface 			# PlanningSceneInterface
		self.robot = robot 			# Robot
		self.eef_link = eef_link 	# Name of the end effector link

	def wait_for_scene_update(self, name, box_is_known=False, box_is_attached=False, timeout=4.0):
		"""
		Ensures that collision updates are received.

		If the Python node dies before publishing a collision object update message, the message
		could get lost and the box will not appear. To ensure that the updates are 
		made, we wait until we see the changes reflected in the 
		``get_attached_objects()`` and ``get_known_object_names()`` lists. 
		For the purpose of this tutorial, we call this function after adding, 
		removing, attaching or detaching an object in the planning scene. We then wait 
		until the updates have been made or ``timeout`` seconds have passed
		
		:param name: Name of the object that is being updated in the scene
		:param box_is_known: Boolean indicating if the box is part of the known objects
		:param box_is_attached: Boolean indicating if the box is attached
		:param timeout: Number of seconds to wait for scene update
		"""
		start = rospy.get_time()
		seconds = rospy.get_time()
		while (seconds - start < timeout) and not rospy.is_shutdown():
			# Test if the box is in attached objects
			attached_objects = self.scene_interface.get_attached_objects([name])
			is_attached = len(attached_objects.keys()) > 0
			# Test if the box is in the scene.
			# Note that attaching the box will remove it from known_objects
			is_known = name in self.scene_interface.get_known_object_names()
			# Test if we are in the expected state
			if (box_is_attached == is_attached) and (box_is_known == is_known):
				return True
			# Sleep so that we give other threads time on the processor
			rospy.sleep(0.1)
			seconds = rospy.get_time()

		# If we exited the while loop without returning then we timed out
		return False

	def add_block(self, size, pose_stamped, name, timeout=4):
		"""
		Adds a block to the scene.

		:param size: Dimensions of the block in metres as a tuple (sx, sy, sz)
		:type size: tuple
		:param pose_stamped: Pose of the block in 3D wrt
		:type pose_stamped: geometry_msgs.msg.PoseStamped
		:param name: Name of the block
		:type name: string
		:param timeout: Number of seconds to wait for scene update
		:type timeout: float
		"""
		self.scene_interface.add_box(name, pose_stamped, size)
		return self.wait_for_scene_update(name, box_is_known=True, timeout=timeout)

	def remove_object(self, name, timeout=4):
		"""
		Removes specified object from the scene

		:param name: Name of the object to remove
		:type name: string
		:param timeout: Number of seconds to wait for scene update
		:type timeout: float
		"""
		# if the box it attached throw warning and detach the object
		if len(self.scene_interface.get_attached_objects([name]).keys()) > 0:
			print("Object named ", name, " is stilled attached. Detaching it.")
			self.scene_interface.detach_box(name)

		# **Note:** The object must be detached before we can remove it from the world
		self.scene_interface.remove_world_object(name)

		# We wait for the planning scene to update.
		return self.wait_for_scene_update(name, box_is_attached=False, box_is_known=False, timeout=timeout)

	def build_stage(self):
		# Randomly build stage, return x,y,z positions for blocks and goal
		block_dim = 0.08									# length of the side of the block in m
		block_size = (block_dim, block_dim, block_dim)		# size of the block as a 3-tuple
		# cylinder_radius = 0.04							# radius of the obstacle cylinder
		# cylinder_height = 1.0								# height of the obstacle cylinder
		table_width = 0.75									# width of the table
		table_length = 1.2									# length of the table
		small_z = 0.001										# just a small number to use boxes as planes

		obj_pose = geometry_msgs.msg.PoseStamped()
		obj_pose.header.frame_id = "world"
		obj_pose.pose.orientation.w = 1.0

		rospy.loginfo("Creating the table...")

		# add the table surface
		obj_pose.pose.position.x = 0
		obj_pose.pose.position.y = 0.525		# do not modify this value
		obj_pose.pose.position.z = small_z/2
		self.add_block((table_width, table_length, small_z), obj_pose, "table")

		rospy.loginfo("Creating the calibration board...")
		# add the target position
		obj_pose.pose.position.x = 0
		obj_pose.pose.position.y = 0.5
		obj_pose.pose.position.z = 0.005
		self.add_block((0.25, 0.3, 0.01), obj_pose, "calib_board")
