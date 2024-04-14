/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3
 * FILE START: 04/02/2024
 * FILE LAST UPDATED: 27/04/2024
 * 
 * REQUIREMENTS: Eigen v3.4.0, src: https://eigen.tuxfamily.org/index.php?title=Main_Page
 * REFERENCES: (Heavy) Inspiration taken from Brian Dolhansky's similar implementation in Python, go check it out!, src: https://github.com/bdol/bdol-ml
 * 
 * DESCRIPTION: A simple, extensible implementation of a multi layer artificial neural network for use within a deep q-learning agent.
*/


#include "cpp-nn/network.h"


/******************************/
/* GLOBAL ACTIVATION FUCNTIONS */
/******************************/


Eigen::MatrixXd vector_f_sigmoid_rl(const Eigen::MatrixXd& in, bool deriv)
{
    int i;
    if(!deriv)
    {
        Eigen::MatrixXd res(in.rows(), in.cols());

        for(i = 0; i < in.size(); i++)
            *(res.data() + i) = 1 / (1 + exp(-(*(in.data() + i))));

        return res;
    }

    Eigen::MatrixXd temp = vector_f_sigmoid_rl(in, false);
    for(i = 0; i < temp.size(); i++)
        *(temp.data() + i) *= 1 - *(temp.data() + i);

    return temp;
}


/******************************/
/* LAYER CLASS IMPLEMENTATION */
/******************************/


Layer::Layer(int curr_size, int next_size, bool is_input, bool is_output, std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, bool)> activation_func) 
    : is_input(is_input), is_output(is_output), activation_func(activation_func)
{
    if(!count)
    {
        random_engine.seed(RANDOM_SEED);
        count++;
    }

    // all layers have a Z matrix
    Z = Eigen::MatrixXd::Zero(curr_size, 1);

    // output and hidden layers
    if(!is_input) 
    {
        S = Eigen::MatrixXd::Zero(curr_size, 1);
        G = Eigen::MatrixXd::Zero(curr_size, 1);
    }

    // input layer and hidden layer
    if(!is_output)
    {
        // random generator - seeded with static random_engine
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        auto uni = [&]() { return distribution(random_engine); };

        W = Eigen::MatrixXd::NullaryExpr(curr_size, next_size, uni);
    }

    // hidden layer only
    if((!is_input) && (!is_output))
    {
        Fp = Eigen::MatrixXd::Zero(curr_size, 1);
    }
}


Eigen::MatrixXd Layer::forward_propogate_rl()
{
    /* remark: output is handled seperately in ML_ANN::forward_propogate_rl()*/

    // input layer - no activation func return data only
    if(is_input)
        return Z;

    Z = activation_func(S, false);

    // output is handled seperately in ML_ANN, here just in case. 
    if(is_output)
        return Z;

    // adding bias row to weights and Z
    Eigen::MatrixXd W_bias = W;
    W_bias.conservativeResize(W_bias.rows() + 1, W_bias.cols());
    W_bias.row(W_bias.rows()-1) = Eigen::MatrixXd::Ones(1, W.cols());

    Eigen::MatrixXd Z_bias = Z;
    Z_bias.conservativeResize(Z_bias.rows() + 1, Z_bias.cols());
    Z_bias.row(Z_bias.rows()-1) = Eigen::MatrixXd::Ones(1, 1);

    // storing f'(S^(i)) for backpropogation step
    Fp = activation_func(S, true);

    return (W_bias.transpose().eval()) * Z_bias;
}


void Layer::set_weight(const Eigen::MatrixXd& new_weight)
{
    if(!((W.rows() == new_weight.rows()) && (W.cols() == new_weight.cols())))
    {
        std::cout << "DIMENSION OF MATRIX TO COPY INCORRECT! (" << W.rows() << "x" << W.cols() << ") required, exiting!";
        std::exit(-1);
    }

    int i;
    for(i = 0; i < W.size(); i++)
        *(W.data() + i) = *(new_weight.data() + i);

    return;
}


/******************************/
/* ML_ANN CLASS IMPLEMENTATION */
/******************************/


ML_ANN::ML_ANN(const std::vector<size_t>& layer_config, std::function<Eigen::MatrixXd(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target)> loss_func)
: loss_func(loss_func)
{
    num_layers = layer_config.size();
    layers.resize(num_layers);

    // input layer
    layers[0] = new Layer(layer_config[0], layer_config[1], true, false, vector_f_sigmoid_rl);

    // hidden layers
    int i;
    for(i = 1; i < (num_layers-1); i++)
        layers[i] = new Layer(layer_config[i], layer_config[i+1], false, false, vector_f_sigmoid_rl);

    // output layer
    layers[num_layers-1] = new Layer(layer_config[num_layers-1], 0, false, true, vector_f_sigmoid_rl); // activation function not used in Layer - potentially dangerours
}


ML_ANN::~ML_ANN()
{
    for(auto it = layers.begin(); it != layers.end(); ++it)
        delete *it;
}


/* remark: static function */
Eigen::MatrixXd ML_ANN::elem_wise_product(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs)
{
    // if lhs & rhs do not have same dimensions - todo throw exception
    if(!((lhs.cols() == rhs.cols()) && (lhs.rows() == rhs.rows())))
    {
        std::cout << "ERROR: elem wise multiplication not possible";
        std::cout << " matrix dimensions are not equal!" << std::endl;
        std::exit(-1);
    }

    Eigen::MatrixXd res(lhs.rows(), lhs.cols());
    int i, j;
    for(i = 0; i < lhs.size(); i++)
        *(res.data() + i) = *(lhs.data() + i) * *(rhs.data() + i);

    return res;
}


Eigen::MatrixXd ML_ANN::forward_propogate_rl(const std::vector<double>& data)
{
    auto l_ptr_0 = layers[0];

    // checking input data is correct size
    if(!(data.size() == l_ptr_0->Z.rows()))
    {
        std::cout << "INPUT DATA NOT OF CORRECT LENGTH: INPUT(" << data.size() << ") REQUIRED(" << l_ptr_0->Z.rows() << ")" << std::endl;
        std::exit(-1);
    }

    // input layer - set Z to data
    int i;
    for(i = 0; i < l_ptr_0->Z.size(); i++)
        *(l_ptr_0->Z.data() + i) = data[i];

    // forward propogate through hidden layers
    for(i = 1; i < (num_layers-1); i++)
        layers[i]->S = layers[i-1]->forward_propogate_rl(); 

    // get output
    layers[num_layers-1]->S = layers[num_layers-2]->forward_propogate_rl();
    return vector_f_sigmoid_rl(layers[num_layers-1]->S, false);
}


void ML_ANN::back_propogate_rl(const double output, const double target)
{
    // output layer
    // output matrix for G
    Eigen::MatrixXd t_out(1, 1);
    t_out << (output - target);

    layers[num_layers-1]->G = t_out;

    // backwards through the remaining layers excluding input
    int i;
    for(i = (num_layers-2); i > 0; i--)
    {
        Eigen::MatrixXd W_nbias = layers[i]->W;
        layers[i]->G = ML_ANN::elem_wise_product(layers[i]->Fp, (W_nbias * layers[i+1]->G));
    }
}


void ML_ANN::back_propogate_rl(const Eigen::MatrixXd& output, const std::vector<double>& targets)
{
    int n = output.rows();
    if(!(n == targets.size()))
    {
        std::cout << "ERROR: output and targets not the same dimension, skipping BP step!" << std::endl;
        return;
    }

    // output layer - (output.size() x 1) dimension
    Eigen::MatrixXd net_output(n, 1);

    int i;
    for(i = 0; i < n; i++)
        net_output(i, 0) = (output(i, 0) - targets[i]);

    layers[num_layers-1]->G = net_output;

    // BP through remaining layers excluding input
    for(i = (num_layers-2); i > 0; i--)
        layers[i]->G = ML_ANN::elem_wise_product(layers[i]->Fp, (layers[i]->W * layers[i+1]->G));

    return;
}

void ML_ANN::back_propogate_rl(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target)
{
    if(!(output.rows() == target.rows()))
    {
        std::cout << "ERROR: output and targets not the same dimension, skipping BP step!" << std::endl;
        return;
    }

    layers[num_layers-1]->G = loss_func(output, target);

    // BP through remaining layers excluding input
    int i;
    for(i = (num_layers-2); i > 0; i--)
        layers[i]->G = ML_ANN::elem_wise_product(layers[i]->Fp, (layers[i]->W * layers[i+1]->G));

    return;
}


void ML_ANN::update_weights_rl(const double eta)
{
    int i;
    for(i = 0; i < (num_layers-1); i++)
        layers[i]->W += -(eta) * (layers[i+1]->G * layers[i]->Z.transpose().eval()).transpose().eval();
}