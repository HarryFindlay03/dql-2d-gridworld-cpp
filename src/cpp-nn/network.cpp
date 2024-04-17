/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3
 * FILE START: 04/02/2024
 * FILE LAST UPDATED: 15/04/2024
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


Eigen::MatrixXd vector_ReLU(const Eigen::MatrixXd& in, bool deriv)
{
    Eigen::MatrixXd res = in;

    int i;
    if(!deriv)
    {
        for(i = 0; i < res.size(); i++)
        {
            if(*(res.data() + i) < 0)
                *(res.data() + i) = 0;
        }

        return res;
    }

    for(i = 0; i < res.size(); i++)
    {
        if(*(res.data() + i) > 0)
            *(res.data() + i) = 1;
        else
            *(res.data() + i) = 0;
    }

    return res;
}


Eigen::MatrixXd vector_linear(const Eigen::MatrixXd& in, bool deriv)
{
    Eigen::MatrixXd res;
    if(!deriv)
    {
        res = in;
        return res;
    }

    // derivative of linear activation function is always one
    res = Eigen::MatrixXd::Ones(in.rows(), in.cols());
    return res;
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
        Fp = Eigen::MatrixXd::Zero(curr_size, 1);
    }

    // input layer and hidden layer
    if(!is_output)
    {
        // random generator - seeded with static random_engine
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        auto uni = [&]() { return distribution(random_engine); };

        W = Eigen::MatrixXd::NullaryExpr(curr_size, next_size, uni);
    }
}


// Eigen::MatrixXd Layer::forward_propogate_rl()
// {
//     /* remark: output is handled seperately in ML_ANN::forward_propogate_rl()*/

//     // if input layer then run Z through activation as this has been set prior
//     // else run input of layer through activation to get Z the layer output (like normal)
//     // if(is_input)
//     //     Z = activation_func(Z, false);
//     // else
//     //     Z = activation_func(S, false);

//     if(!is_input)
//         Z = activation_func(S, false); // applying activation function to inputs

//     // output is handled seperately in ML_ANN, here just in case. 
//     if(is_output)
//     {
//         Fp = activation_func(S, true);
//         return Z;
//     }

//     // adding bias row to weights and Z
//     Eigen::MatrixXd W_bias = W;
//     W_bias.conservativeResize(W_bias.rows() + 1, W_bias.cols());
//     W_bias.row(W_bias.rows()-1) = Eigen::MatrixXd::Ones(1, W.cols());

//     Eigen::MatrixXd Z_bias = Z;
//     Z_bias.conservativeResize(Z_bias.rows() + 1, Z_bias.cols());
//     Z_bias.row(Z_bias.rows()-1) = Eigen::MatrixXd::Ones(1, Z.cols());

//     // storing f'(S^(i)) for backpropogation step
//     Fp = activation_func(S, true);
    
//     // todo NOTE REMOVED bias values here - bias values caused an explosion of gradient - may need to decrease the bias values etc
//     return (W.transpose().eval()) * Z;
//     // return (W_bias.transpose().eval()) * Z_bias;

// }

Eigen::MatrixXd Layer::forward_propogate_rl()
{
    // input layer does not have activation function - Z has been set prior to this function 
    if(!is_input)
    {
        Z = activation_func(S, false);

        // storing derivative of activation func for later
        Fp = activation_func(S, true);
    }


    if(is_output)
        return Z;

    return (W.transpose().eval()) * Z;
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
    // instantiation random number generator
    rnd = new RandHelper();

    // setting clip values
    min_clip = -0.5;
    max_clip = 0.5;

    num_layers = layer_config.size();
    layers.resize(num_layers);

    // input layer
    layers[0] = new Layer(layer_config[0], layer_config[1], true, false, vector_ReLU);

    // hidden layers
    int i;
    for(i = 1; i < (num_layers-1); i++)
        layers[i] = new Layer(layer_config[i], layer_config[i+1], false, false, vector_ReLU);

    // output layer
    layers[num_layers-1] = new Layer(layer_config[num_layers-1], 0, false, true, vector_linear);
}


ML_ANN::~ML_ANN()
{
    for(auto it = layers.begin(); it != layers.end(); ++it)
        delete *it;

    delete rnd;
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


/* MAIN NN FUNCTIONS */


Eigen::MatrixXd ML_ANN::forward_propogate_rl(const std::vector<double>& data)
{
    auto l_ptr_0 = layers[0];

    // checking input data is correct size
    if(!(data.size() == l_ptr_0->Z.rows()))
    {
        std::cout << "INPUT DATA NOT OF CORRECT LENGTH: INPUT(" << data.size() << ") REQUIRED(" << l_ptr_0->Z.rows() << ")" << std::endl;
        std::exit(-1);
    }

    // input layer - set Z to data - no activation function here
    int i;
    for(i = 0; i < l_ptr_0->Z.size(); i++)
        *(l_ptr_0->Z.data() + i) = data[i];

    // layers[1]->S = layers[0]->Z;

    // forward propogate through to the output layer - setting input to next as output of prev
    for(i = 1; i < num_layers; i++)
    {
        layers[i]->S = layers[i-1]->forward_propogate_rl();
        std::cout << "S (" << i << "):\n" << layers[i]->S << std::endl;
    }

    // process output layer
    Eigen::MatrixXd res = layers[num_layers-1]->forward_propogate_rl();
    return res;
}


void ML_ANN::back_propogate_rl(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target)
{
    if(!(output.rows() == target.rows()))
    {
        std::cout << "ERROR: output and targets not the same dimension, skipping BP step!" << std::endl;
        return;
    }

    // output
    Eigen::MatrixXd loss = loss_func(output, target);
    layers[num_layers-1]->G = ML_ANN::elem_wise_product(layers[num_layers-1]->Fp, loss);
    std::cout << "G(Output: " << num_layers-1 << "):\n" << layers[num_layers-1]->G << std::endl;

    // BP through remaining layers excluding input
    int i;
    for(i = (num_layers-2); i > 0; i--)
    {
        layers[i]->G = ML_ANN::elem_wise_product(layers[i]->Fp, (layers[i]->W * layers[i+1]->G));
        std::cout << "G(" << i << "):\n"
                  << layers[i]->G << std::endl;
    }

    return;
}

// todo this function just seems a bit wrong !
void ML_ANN::update_weights_rl(const double eta)
{
    int i, x, y;
    for(i = 0; i < (num_layers-1); i++)
    {
        // std::cout << "G: \n" << layers[i+1] ->G << std::endl;
        // Eigen::MatrixXd res = -(eta) * (layers[i+1]->G * layers[i]->Z.transpose().eval()).transpose().eval();
        Eigen::MatrixXd res = eta * ((layers[i]->Z) * (layers[i+1]->G.transpose().eval()));
        layers[i]->W += res;
    }
}


Eigen::MatrixXd ML_ANN::gradient_clip_by_val(const Eigen::MatrixXd& g_mat)
{
    Eigen::MatrixXd res_mat = g_mat;

    int i;
    for(i = 0; i < g_mat.rows(); i++)
    {
        if(g_mat(i, 0) > max_clip)
            res_mat(i, 0) = max_clip;
        if(g_mat(i, 0) < min_clip)
            res_mat(i, 0) = min_clip;
    }

    return res_mat;
}


/* WEIGHT INITIALISATION */


void ML_ANN::he_weight_init(ML_ANN* net, RandHelper* rnd)
{
    // for each weight network apply He's rule

    int i, j;
    for(i = 0; i < net->num_layers; i++)
    {
        // for each value in w_mat look at the dimension of the previous layer
        // the number of inputs is the number of rows

        Eigen::MatrixXd w_mat(net->get_layers()[i]->W.rows(), net->get_layers()[i]->W.cols());
        int prev_rows = (i != 0) ? net->get_layers()[i-1]->W.rows() : 1;

        for(j = 0; j < w_mat.size(); j++)
            *(w_mat.data() + j) = rnd->normal_distribution(0.0, std::sqrt(2.0 / (double)prev_rows));

        net->get_layers()[i]->set_weight(w_mat);
    }

    return;
}


void ML_ANN::small_weight_init(ML_ANN* net, RandHelper* rnd)
{

    int i, j;
    for(i = 0; i < net->num_layers; i++)
    {
        // for each value in w_mat look at the dimension of the previous layer
        // the number of inputs is the number of rows
        Eigen::MatrixXd w_mat(net->get_layers()[i]->W.rows(), net->get_layers()[i]->W.cols());

        for(j = 0; j < w_mat.size(); j++)
            *(w_mat.data() + j) = rnd->random_double_range(0.0, 0.1);

        net->get_layers()[i]->set_weight(w_mat);
    }

    return;
}
