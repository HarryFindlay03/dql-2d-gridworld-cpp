/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 25/04/2024
 * FILE LAST UPDATED: 09/05/2024
 * 
 * REQUIREMENTS: Eigen v3.4.0, src: https://eigen.tuxfamily.org/index.php?title=Main_Page
 * REFERENCES: Volodymyr Mnih et al. "Human-level control through deep reinforcement learning."
 * 
 * DESCRIPTION: An implementation of a deep Q-learning agent following DeepMind's paper.
*/

/* TODO */
    // implement global static random number generation


#include "dqn/Agent.h"


/* HELPER FUNCTIONS */


Eigen::MatrixXd l2_loss(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target)
{
    Eigen::MatrixXd diff = (output - target);

    int i;
    for(i = 0; i < diff.size(); i++)
        *(diff.data() + i) = std::pow(*(diff.data() + i), 2);

    return diff;
}


/* AGENT CLASS*/


Agent::Agent
(
    int env_rows,
    int env_cols,
    const unsigned int buffer_size,
    const unsigned int copy_period,
    const unsigned int number_of_episodes,
    const unsigned int episode_length,
    const double discount_rate,
    const double eta
)
:
    buffer_size(buffer_size), 
    copy_period(copy_period), 
    number_of_episodes(number_of_episodes), 
    episode_length(episode_length),
    discount_rate(discount_rate),
    eta(eta)
{
    // instatiate random helper
    rnd = new RandHelper();

    // create agent's environment
    curr_env = construct_env(env_rows, env_cols, rnd);

    // get agent position
    agent_pos = get_init_agent_pos(curr_env, rnd);

    // instatiate both networks
    Q = construct_network_from_env(curr_env, 2);
    Q_hat = construct_network_from_env(curr_env, 2);

    // initially set both network weights as equal
    copy_network_weights();

    // setting agent action space - 4 moves representing up, down, left and right
    actions = {1, 2, 3, 4};

    // resize buffer and set curr pos
    buff.resize(buffer_size);
    curr_buff_pos = 0;

    // instatiante random generator

    return;
}


void Agent::train_optimiser(const double epsilon)
{
    int i, j, curr_itr;
    bool terminate;

    int num_features = get_num_features();
    
    curr_itr = 0;

    for(i = 0; i < number_of_episodes; i++)
    {
        // reset environment
        if(i != 0)
            curr_env = construct_env(curr_env.rows(), curr_env.cols(), rnd);

        for(j = 0; j < episode_length; j++)
        {
            /* sampling */
            terminate = (((j+1) == episode_length) ? true : false);
            sampling(epsilon, terminate);

            /* sample from replay buffer and train */
            train_phase();

            /* copy network weights */
            if(!((curr_itr++) % copy_period))
                copy_network_weights();

            if(terminate)
                break;
        }
    }
}


void Agent::sampling(const double epsilon, bool terminate)
{
    std::vector<double> curr_st;
    std::vector<double> next_st;

    curr_st = get_state_vector(curr_env, agent_pos);
    int action_pos = epsilon_greedy_action(curr_st, epsilon);

    // execute in emulator and observe reward (moving the agent)
    double reward = 0;

    bool move_validity = move_agent_in_env(action_pos); /* moving agent */
    bool goal_status = check_goal(curr_env, agent_pos); /* checking goal */

    next_st = get_state_vector(curr_env, agent_pos);

    if(goal_status) /* found food */
    {
        terminate = true;
        reward = 10;
    }
    if((!goal_status) && (!terminate)) /* found food but episode not over - negative reward to push for faster finding */
    {
        reward = -1;
    }
    if((!goal_status) && terminate) /* found food and episode over - worst case */
    {
        reward = -10;
    }

    // save to replay buffer
    buff[(curr_buff_pos++) % buffer_size] = new BufferItem(curr_st, action_pos, reward, next_st, terminate);

    return;
}


void Agent::train_phase()
{
    int max_size = (buff[(curr_buff_pos) % buffer_size] == NULL) ? (curr_buff_pos-1) : (buffer_size-1);
    BufferItem* b = buff[rnd->random_int_range(0, max_size)];

    double y_j;

    if(b->get_terminate())
    {
        y_j = b->get_reward();
    }
    else
    {
        // find the best action value with Q_hat
        // forward prop the preprocessed state
        Eigen::MatrixXd out = Q_hat->forward_propogate_rl(b->get_next_st());

        int i;
        double best_pos = 0;

        for(i = 1; i < out.rows(); i++)
            if(out.row(i)[0] > out.row(best_pos)[0])
                best_pos = i;

        y_j = b->get_reward() + (discount_rate * out.row(best_pos)[0]);
    }

    // gradient descent step only on output node j for action j.
    Eigen::MatrixXd out_Q = Q->forward_propogate_rl(b->get_curr_st());

    // need the action pos of j - this is where we set yj
    Eigen::MatrixXd out_yj = out_Q;
    out_yj(b->get_action_pos(), 0) = y_j;

    Q->back_propogate_rl(out_yj, out_Q);
    Q->update_weights_rl(eta);

    return;    
}


/* TRAINED AGENT USER FUNCTIONS */


void Agent::find_food()
{
    /* construct new environment */
    curr_env = construct_env(curr_env.rows(), curr_env.cols(), rnd);
    agent_pos = get_init_agent_pos(curr_env, rnd);

    // choose action based on Q network, get curr
    // get current state choose action based on Q network then continue until goal found
    
    int iter_pos = 0;
    bool found = false;
    while(!found)
    {
        std::cout << "Iteration " << iter_pos << std::endl;

        print_environment();

        // get state
        std::vector<double> st = get_state_vector(curr_env, agent_pos);

        // get q vals from Q net and choose best action
        Eigen::MatrixXd q_vals = Q->forward_propogate_rl(st);

        std::cout << "Qvals: \n";
        std::cout << q_vals << std::endl;
        
        int i;
        int best_pos = 0;
        for(i = 1; i < actions.size(); i++)
            if(q_vals(i, 0) > q_vals(best_pos, 0))
                best_pos = i;


        move_agent_in_env(best_pos);

        if(check_goal(curr_env, agent_pos))
        {
            std::cout << "FOOD FOUND!\n";
            std::cout << iter_pos << " steps taken.\n";
            found = true;
            break;
        }

        if(iter_pos == 20)
            break;

        iter_pos++;
    }

    return;
}


/* HELPER FUNCTIONS */


int Agent::epsilon_greedy_action(const std::vector<double>& st, const double epsilon)
{
    std::tuple<int, int> action_res;

    double r = rnd->random_double_range(0.0, 1.0);

    if(r > (1 - epsilon))
    {
        int pos = rnd->random_int_range(0, actions.size()-1);
        return pos;
    }

    Eigen::MatrixXd q_vals = Q->forward_propogate_rl(st);

    // find the best available action (remember to penalise if the action takes out of bounds)
    int best_pos = 0;
    int i;
    for(i = 1; i < actions.size(); i++)
        if((q_vals(i, 0) > q_vals(best_pos, 0)))
            best_pos = i;

    return best_pos;
}


bool Agent::move_agent_in_env(int action_pos)
{
    switch(actions[action_pos])
    {
    case 1: // left
        if(!(check_bounds(std::make_tuple(std::get<0>(agent_pos), std::get<1>(agent_pos)-1))))
            return false;
        std::get<1>(agent_pos) -= 1;
        break;
    case 2: // right
        if(!(check_bounds(std::make_tuple(std::get<0>(agent_pos), std::get<1>(agent_pos)+1))))
            return false;
        std::get<1>(agent_pos) += 1;
        break;
    case 3: // up
        if(!(check_bounds(std::make_tuple(std::get<0>(agent_pos)+1, std::get<1>(agent_pos)))))
            return false;
        std::get<0>(agent_pos) += 1;
        break;
    case 4: // down
        if(!(check_bounds(std::make_tuple(std::get<0>(agent_pos)-1, std::get<1>(agent_pos)))))
            return false;
        std::get<0>(agent_pos) -= 1;
        break;
    default:
        break;
    }

    return true;
}


bool Agent::check_bounds(const std::tuple<int, int>& new_pos)
{
    int row = std::get<0>(new_pos);
    int col = std::get<1>(new_pos);

    if(row < 0 || row >= curr_env.rows())
        return false;
    if(col < 0 || col >= curr_env.cols())
        return false;

    return true;
}

void Agent::copy_network_weights()
{
    // set Q_hat to Q (weights)

    // for each layer copy the weights matrix (excluding last)
    int i;
    for(i = 0; i < ((Q->get_num_layers())-1); i++)
        Q_hat->set_weight_matrix((Q->get_layers())[i]->W, i);

    return;
}


void Agent::print_networks()
{
    std::cout << "Q network:\n";

    for(auto l : Q->get_layers())
        std::cout << l->W << "\n\n";
    std::cout << std::endl;

    std::cout << "Q_hat network:\n";

    for(auto l : Q_hat->get_layers())
        std::cout << l->W << "\n\n";
    std::cout << std::endl;

    return;
}


void Agent::print_environment()
{
    std::cout << "Agent environment:\n";

    int i, j;
    for(i = 0; i < curr_env.rows(); i++)
    {
        for(j = 0; j < curr_env.cols(); j++)
        {
            if((std::get<0>(agent_pos) == i) && (std::get<1>(agent_pos) == j))
                std::cout << "A ";
            else
                std::cout << curr_env(i, j) << " ";
        }
        std::cout << std::endl;
    }
}