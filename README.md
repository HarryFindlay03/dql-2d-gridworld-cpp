# dql-2d-gridworld-cpp

This is a repository containing all the cpp code for a deep-Q-learning reinforcement learning agent to learn how to find some 'food' in a 2D gridworld environment.

This is built all from scratch in c++ utilising only the Eigen linear algebra library for the matrix multiplications found within the multi-layer perceptron network (a.k.a. neural network a.k.a. deep neural network).

Please feel free to do whatever you want with this code, for me this has been a helpful project to fine tune the algorithms and implementation for use within some slightly more serious dissertation work applying the same methods found here to compiler optimisation (coming soon!).

Thanks, Harry 🕺

### Installation

1. Please first download the Eigen Linear Algebra header library found here: https://eigen.tuxfamily.org/index.php?title=Main_Page, v3.40 has been used for this work.

2. Run the install shell script - this simply creates two folders used for aesthetics during the build and linking stage of compilation, this file will also ensure that the Eigen library has been installed correctly !

### Compiling

A Makefile has been included, this will build a driver.cpp program that you write if it is placed in src/driver.cpp. However of course you may need to edit the Makefile slightly for your system.

Have fun !