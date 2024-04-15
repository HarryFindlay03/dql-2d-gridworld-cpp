/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3
 * FILE START: 04/02/2024
 * FILE LAST UPDATED: 08/05/2024
 * 
 * REQUIREMENTS: Eigen v3.4.0, src: https://eigen.tuxfamily.org/index.php?title=Main_Page
 * REFERENCES: (Heavy) Inspiration taken from Brian Dolhansky's similar implementation in Python, go check it out!, src: https://github.com/bdol/bdol-ml
 * 
 * DESCRIPTION: Class definition for a simple, extensible implementation of a multi layer artificial neural network for use within a deep q-learning agent.
*/


#ifndef ML_ANN_H
#define ML_ANN_H


class ML_ANN
{
    std::vector<Layer*> layers;
    size_t num_layers;
    size_t minibatch_size;

    // loss function
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target)> loss_func;

public:
    ML_ANN(const std::vector<size_t>& layer_config, size_t minibatch_size);

    ML_ANN
    (
        const std::vector<size_t>& layer_config, 
        std::function<Eigen::MatrixXd(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target)> loss_func
    );

    ~ML_ANN();

    static Eigen::MatrixXd elem_wise_product(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs);

    Eigen::MatrixXd forward_propogate_rl(const std::vector<double>& data);

    void back_propogate_rl(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target);
    void back_propogate_rl(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target, int action_pos);

    void update_weights_rl(const double eta);

    void set_weight_matrix(const Eigen::MatrixXd& new_weight, const size_t layer_pos) { layers[layer_pos]->set_weight(new_weight); };

    inline size_t get_num_layers() { return num_layers; };

    inline const std::vector<Layer*>& get_layers() { return layers; };
};

#endif