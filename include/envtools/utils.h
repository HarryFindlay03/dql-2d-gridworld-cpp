#include "cpp-nn/network.h"
#include "dqn/Agent.h"

#include "RandHelper.h"

#include <vector>


Eigen::MatrixXd construct_env(int rows, int cols, RandHelper* agent_rnd);

std::tuple<int, int> get_init_agent_pos(const Eigen::MatrixXd& env, RandHelper* agent_rnd);

ML_ANN* construct_network_from_env(const Eigen::MatrixXd& env, int num_hidden_layer);

std::vector<double> get_state_vector(const Eigen::MatrixXd& env, const std::tuple<int, int>& agent_pos);

bool check_goal(const Eigen::MatrixXd& env, const std::tuple<int, int>& agent_pos);