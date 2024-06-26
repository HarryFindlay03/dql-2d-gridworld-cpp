#include "envtools/utils.h"


Eigen::MatrixXd construct_env(int rows, int cols, RandHelper* agent_rnd)
{
    Eigen::MatrixXd ret_env(rows, cols);

    // get random position for 'food' and place
    int food_x = agent_rnd->random_int_range(0, rows-1);
    int food_y = agent_rnd->random_int_range(0, cols-1);

    ret_env(food_x, food_y) = 1;

    return ret_env;
}


std::tuple<int, int> get_init_agent_pos(const Eigen::MatrixXd& env, RandHelper* agent_rnd)
{
    int ran_x = agent_rnd->random_int_range(0, env.cols()-1);
    int ran_y = agent_rnd->random_int_range(0, env.rows()-1);

    return std::make_tuple(ran_x, ran_y);
}


ML_ANN* construct_network_from_env(const Eigen::MatrixXd& env, int num_hidden_layer)
{
    int n = 2 + num_hidden_layer;
    std::vector<size_t> network_config(n);

    int norm_layer_size = env.rows() * env.cols();

    int i;
    for(i = 0; i < n; i++)
        network_config[i] = (i != n - 1) ? norm_layer_size : 4;

    // 4 represents action space up, down, left, right

    ML_ANN* ptr = new ML_ANN(network_config, l2_loss);
    return ptr;
}


std::vector<double> get_state_vector(const Eigen::MatrixXd& env, const std::tuple<int, int>& curr_pos)
{
    std::vector<double> res_state(env.size());

    // unwrap each row (or col depending on compiler orientation) and return as a single vector
    int pos = 0;
    for(auto const& row : env.rowwise())
        for(auto const& v : row)
            res_state[pos++] = v;

    // place agent pos in correct position
    res_state[((std::get<0>(curr_pos) * env.cols()) + std::get<1>(curr_pos))] = 1;

    return res_state;
}


bool check_goal(const Eigen::MatrixXd& env, const std::tuple<int, int>& agent_pos)
{
    return (env(std::get<0>(agent_pos), std::get<1>(agent_pos)) == 1);
}


void save_Q_network(ML_ANN* Q, const std::string& filename)
{
    std::ofstream output_file(filename.c_str());

    if(!(output_file.is_open()))
    {
        std::cerr << "FILENAME ENTERED IS NOT VALID! CANNOT WRITE NETWORK TO FILE\n";
        return;
    }

    /* file opened successfully */
    for(auto const& l : Q->get_layers())
    {
        for(auto const& r : (l->W).rowwise())
        {
            for(auto const& v : r)
                output_file << v << '\n';
        }
        output_file << "<br>\n"; /* helper to reconstruct network*/
    }
}


void read_Q_weights(ML_ANN* Q, const std::string& filename)
{
    std::ifstream input_file(filename.c_str());

    if(!(input_file.is_open()))
    {
        std::cerr << "FILENAME ENTERED IS NOT VALID! CANNOT READ NETWORK FROM FILE\n";
        return;
    }

    /* file open success */
    int layer_pos = 0;
    int col = 0;
    int row = 0;

    std::string line;
    while(getline(input_file, line))
    {
        if(layer_pos >= Q->get_num_layers())
            break;

        if(!(strcmp(line.c_str(), "<br>")))
        {
            layer_pos++;
            col = 0;
            row = 0;
            continue;
        }

        ((Q->get_layers())[layer_pos]->W)(row, col++) = std::stod(line);

        if((col == (Q->get_layers()[layer_pos]->W.cols())))
        {
            row++;
            col = 0;
        }
    }

    return;
}