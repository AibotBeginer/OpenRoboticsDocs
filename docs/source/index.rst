Welcome to OpenRobotic's documentation
=========================================

OpenRobotic开源机器人文档索引页面, 由 **AibotBeginer** 团队创建。2024

`OpenRobotic Code <https://github.com/duyongquan/OpenRobotics>`_

 .. note ::

| **唯天下之至诚能胜天下之至伪**
| **唯天下之至拙能胜天下之至巧**


工程下载
----------------

.. code-block:: bash

   git clone https://github.com/AibotBeginer/OpenRoboticsDocs.git


Python依赖安装
----------------

.. code-block:: bash

   pip install Sphinx
   pip install recommonmark
   pip install sphinx_rtd_theme
   pip install mathjax


工程编译
----------------

.. code-block:: bash

   cd OpenRoboticsDocs
   mkdir build
   cd build
   cmake ..
   make -j6
   
查看文档
----------------

.. code-block:: bash

   cd OpenRoboticsDocs/build
   使用浏览器打开文件: html/index.html

Contents
----------------

.. toctree::
   :maxdepth: 1

   Chapter 01 - 工程介绍 <_source/Chapter-01/project_introduction>
   Chapter 02 - ROS2之系统System <_source/Chapter-02/ros2_system>
   Chapter 03 - ROS2之话题Topic <_source/Chapter-03/ros2_topic>
   Chapter 04 - ROS2之服务Sevice <_source/Chapter-04/ros2_service>
   Chapter 05 - ROS2之动作Action <_source/Chapter-05/ros_action>
   Chapter 06 - ROS2之Lifecycle <_source/Chapter-06/ros_lifecycle>
   Chapter 07 - ROS2之坐标系变换TF2 <_source/Chapter-07/ros2_tf2>
   Chapter 08 - Dijkstra算法 <_source/Chapter-08/dijkstra>
   Chapter 09 - A*算法 <_source/Chapter-09/a_star>
   Chapter 10 - Theta*算法 <_source/Chapter-10/theta_star>
   Chapter 11 - RRT*算法 <_source/Chapter-11/rrt_star>
   Chapter 12 - 纯跟踪控制算法 <_source/Chapter-12/pure_pursuit>
   Chapter 13 - DWA控制算法 <_source/Chapter-13/dwa>
   Chapter 14 - TEB控制算法 <_source/Chapter-14/teb>
   Chapter 15 - MPC控制算法 <_source/Chapter-15/mpc>
   Chapter 16 - LQR控制算法 <_source/Chapter-16/lqr>
   Chapter 17 - 最优化方法 <_source/Chapter-17/optimization_method>
   Chapter 18 - 非线性最小二乘 <_source/Chapter-18/nonlinear_least_squares>
   Chapter 19 - C++ Eigen库 <_source/Chapter-19/optimization_method>
   Chapter 20 - RVIZ使用 <_source/Chapter-20/optimization_method>
   Chapter 21 - ROS2 Navigation2 <_source/Chapter-21/optimization_method>
   Chapter 22 - 激光SLAM <_source/Chapter-22/optimization_method>
   Chapter 23 - 论文推荐<_source/Chapter-23/optimization_method>
