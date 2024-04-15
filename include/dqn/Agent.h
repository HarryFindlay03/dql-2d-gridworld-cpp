/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 25/04/2024
 * FILE LAST UPDATED: 11/05/2024
 * 
 * REQUIREMENTS: Eigen v3.4.0, src: https://eigen.tuxfamily.org/index.php?title=Main_Page
 * REFERENCES: Volodymyr Mnih et al. "Human-level control through deep reinforcement learning."
 * 
 * DESCRIPTION: Class definition for deep Q-learning agent following DeepMind's paper.
*/

#ifndef AGENT_H
#define AGENT_H

#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <map>
#include <tuple>

#include "cpp-nn/network.h"
#include "envtools/utils.h"
#include "envtools/RandHelper.h"
#include "BufferItem.h"

#define NOP (std::string)""


/* HELPER FUNCTIONS */

/**
 * @brief squared loss function defined to be passed to ML_ANN
 * 
 * @param output 
 * @param target 
 * @return Eigen::MatrixXd 
 */
Eigen::MatrixXd l2_loss(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target);


/* AGENT CLASS DEFINITION */


class Agent
{
    /* NETWORKS */
public:
    ML_ANN* Q;

private:
    ML_ANN* Q_hat;

    /* REPLAY BUFFER */
    std::vector<BufferItem*> buff;

    /* STARTING ACTION SPACE */
    std::vector<int> actions;

    /* CURRENT ENVIRONMENT */
    Eigen::MatrixXd curr_env;

    /* AGENT POSITION (part of env) */
    std::tuple<int, int> agent_pos;

    /* PARAMS */
    unsigned int buffer_size;
    unsigned int copy_period;
    unsigned int number_of_episodes;
    unsigned int episode_length;
    double discount_rate;
    double eta;

    /* PRIVATE VALUES */
    unsigned int curr_buff_pos;
    RandHelper* rnd;


public:
    Agent
    (
        int env_rows,
        int env_cols,
        const unsigned int buffer_size, 
        const unsigned int copy_period,
        const unsigned int number_of_episodes,
        const unsigned int episode_length,
        const double discount_rate,
        const double eta
    );

    ~Agent()
    {
        // delete each buffer item
        for(auto it = buff.begin(); it != buff.end(); ++it)
            delete *it;

        delete Q;
        delete Q_hat;
        delete rnd;
    };

    /* TRAINING FUNCTIONS */

    void train_optimiser(const double epsilon);

    void sampling(const double epsilon, bool terminate);

    void train_phase();

    /* TRAINED AGENT USER FUNCTIONS */


    /**
     * @brief Function that is used once the agent has been trained - constructs an environment
     * and attemps to find the 'food' within the environment, if the food has been found then report successful.
     */
    void find_food();

    /* HELPER FUNCTIONS*/

    int epsilon_greedy_action(const std::vector<double>& st, const double epsilon);

    bool move_agent_in_env(int action_pos);

    bool check_bounds(const std::tuple<int, int>& new_pos);

    void copy_network_weights();

    double get_reward(const double new_runtime);

    int get_num_features() { return Q->get_layers()[0]->W.rows(); };

    /* DEBUG HELPER FUNCTIONS */

    void print_networks();

    void print_environment();
};

#endif /* AGENT_H */