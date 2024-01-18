=================
工程介绍
=================


1 OpenRobotics源码下载
======================

`OpenRobotics源码 <https://github.com/duyongquan/OpenRobotics>`_

.. code-block:: bash

    git clone https://github.com/duyongquan/OpenRobotics.git


2 OpenRobotics源码的编译和安装
================================

.. code-block:: bash

    mkdir -p OpenRobotics/src
    cd OpenRobotics/src
    git clone https://github.com/duyongquan/OpenRobotics.git
    cd ..
    colcon build --symlink-install --packages-up-to XXXX


3 OpenRobotics文档的编译和安装
===============================
    
.. code-block:: bash

    git clone https://github.com/AibotBeginer/OpenRoboticsDocs.git
    cd OpenRoboticsDocs/docs
    cmake .. && make -j6



