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

#include "envtools/RandHelper.h"

class ML_ANN
{
    std::vector<Layer*> layers;
    size_t num_layers;
    size_t minibatch_size;

    // loss function
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target)> loss_func;

    // RandHelper
    RandHelper* rnd;

    // gradient clip values
    double min_clip; 
    double max_clip;

public:
    ML_ANN(const std::vector<size_t>& layer_config, size_t minibatch_size);

    ML_ANN
    (
        const std::vector<size_t>& layer_config, 
        std::function<Eigen::MatrixXd(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target)> loss_func
    );

    ~ML_ANN();


    /* MAIN NN FUNCTIONS */

    Eigen::MatrixXd forward_propogate_rl(const std::vector<double>& data);

    void back_propogate_rl(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target);
    void back_propogate_rl(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target, int action_pos);

    void update_weights_rl(const double eta);

    Eigen::MatrixXd gradient_clip_by_val(const Eigen::MatrixXd& in);

    /* CLASS HELPER FUNCTIONS */

    void set_weight_matrix(const Eigen::MatrixXd& new_weight, const size_t layer_pos) { layers[layer_pos]->set_weight(new_weight); };

    inline size_t get_num_layers() { return num_layers; };

    inline std::vector<Layer*>& get_layers() { return layers; };

    /* STATIC HELPER FUNCTIONS */

    static Eigen::MatrixXd elem_wise_product(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs);

    /**
     * @brief Guassian distribution G(0.0, sqrt(2/n)) with mean zero and s.d. sqrt(2/n) where n is the number of inputs into a node.
     * 
     * @param net
     */
    static void he_weight_init(ML_ANN* net, RandHelper* rnd);

    static void small_weight_init(ML_ANN* net, RandHelper* rnd);
};

#endif