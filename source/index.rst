Welcome to OpenRobotic Documents
=========================================

OpenRobotic开源机器人文档索引页面, 由 **AibotBeginer** 团队创建。2024

`OpenRobotic Code <https://github.com/duyongquan/OpenRobotics>`_

 .. note ::

   唯天下之至诚能胜天下之至伪, 
   唯天下之至拙能胜天下之至巧


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

.. toctree::
   :maxdepth: 1

   Chapter 01 - 工程介绍 <_source/Chapter-01/project_introduction.md>
   Chapter 02 - ROS2之系统Filesystem <_source/Chapter-02/ROS2-Parameters.md>
   Chapter 03 - ROS2之话题Topic <_source/Chapter-03/ROS2-Parameters.md>
   Chapter 04 - ROS2之服务Sevice <_source/Chapter-04/ROS2-Parameters.md>
   Chapter 05 - ROS2之动作Action <_source/Chapter-05/ROS2-Parameters.md>
   Chapter 06 - ROS2之坐标系变换TF2 <_source/Chapter-06/ROS2-Parameters.md>
   Chapter XX - 最优化方法 <_source/Chapter-XX/optimization_method>


   

