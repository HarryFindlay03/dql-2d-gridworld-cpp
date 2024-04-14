/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3
 * FILE START: 04/02/2024
 * FILE LAST UPDATED: 25/04/2024
 * 
 * REQUIREMENTS: Eigen v3.4.0, src: https://eigen.tuxfamily.org/index.php?title=Main_Page
 * REFERENCES: (Heavy) Inspiration taken from Brian Dolhansky's similar implementation in Python, go check it out!, src: https://github.com/bdol/bdol-ml
 * 
 * DESCRIPTION: Class definition for a simple, extensible implementation of a multi layer artificial neural network for use within a deep q-learning agent.
*/

#ifndef NETWORK_H
#define NETWORK_H

#include <iostream>
#include <random>
#include <ctime>

#include "Eigen/Dense"

#include "Layer.h"
#include "ML_ANN.h"

/* GLOBAL ACTIVATION FUNCTIONS */

Eigen::MatrixXd vector_f_sigmoid_rl(const Eigen::MatrixXd& in, bool deriv);

#endif /* NETWORK_H */