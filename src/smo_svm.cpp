#include <iostream> 
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

vector<vector<double>> kernel_table(const vector<vector<double>>&data){
    if(data.empty()){
        return {};
    }
    size_t rows = data.size();
    size_t cols = data[0].size();
    
    vector<vector<double>> kernel_matrix(rows,vector<double>(rows,0.0));
    for(size_t i = 0; i<rows;i++){
        for(size_t j = 0; j<rows;j++){
            for(size_t k = 0; k<cols; k++){
                kernel_matrix[i][j] += data[i][k]*data[j][k];
            }
        }
    }
    return kernel_matrix;
}

vector<float> error_store(const vector<vector<double>> & kernel_matrix, double bias, const float & error_tolerance, const vector<float> & labels){
    if(kernel_matrix.empty()){
        return {};
    }
    
    size_t rows = kernel_matrix.size();
    size_t cols = kernel_matrix[0].size();
    vector<float> alphas(rows,0.0); // shape = 1 by sample size all intiliazed to 0
    vector<float> error_vector(rows,0.0);
    // check kkt conditions 
    float model_output = 0;
    float model_error = 0;
    for(int i = 0; i<rows; i++){
        for(int j = 0; j<rows;j++){
            model_output += alphas[i] * labels[i] * kernel_matrix[i][j];
        }
        model_error = model_output -labels[i];
        error_vector.push_back(model_error);
    }
    return error_vector;
}

vector<double> compute_alpha(const vector<vector<double>> & kernel_matrix, 
double bias, const float & error_tolerance, 
const vector<float> & labels, const float & c, 
vector<float> error_vector
,const int & max_it
){
    if(kernel_matrix.empty()){
        return {};
    }
    if(max_it == 0){
        return {};
    }
    vector<double> error_cache;
    size_t rows = kernel_matrix.size();
    vector<double> alphas(rows,0.0); // shape = 1 by sample size all intiliazed to 0
    // check kkt conditions 
    double model_output = 0.0;
    double model_error = 0.0;
    double margin_validation = 0.0;
    double error = 0.0;
    double alpha_a = 0.0;
    int alpha_a_index = 0;
    // chose alpha a ( first alpha of the pair)
    int operation = 0;
    int max_op = max_it*rows*kernel_matrix[0].size();
    while(operation<max_op){

    
    for(int i = 0; i<rows; i++){
        for(int j = 0; j<rows;j++){
            model_output += alphas[i] * labels[i] * kernel_matrix[i][j];
            operation++;
        }
        model_error = model_output -labels[i];
        margin_validation = model_error*labels[i];
        if(margin_validation>1*(error_tolerance)){
            if(alphas[i]>c){
                alpha_a = alphas[i];
                alpha_a_index = i;
                break;

            }
        }
        else if(margin_validation<-1*(error_tolerance)){
            if(alphas[i]<0){
                alpha_a = alphas[i];
                alpha_a_index = i;
                break;
            }
        }
    }
    int index = 0;

    //chose alpha b 
    if(model_error<0){
        auto max_it = max_element(error_vector.begin(), error_vector.end());
        index = max_it - error_vector.begin();
        error = abs(model_error - error_vector[index]);
    }
    else{
        auto min_it = min_element(error_vector.begin(), error_vector.end());
        index = min_it - error_vector.begin();
        error = abs(model_error - error_vector[index]);
    }
    
    // compute clips of [L,H]
    double L = max(0.0,(alphas[index] - alpha_a));
    double H = min(c,(c + alphas[index] - alpha_a));
      

    // calculate eta ( pronounce AY-THA or EE-THA)/ basically caculating second derivative formula is K_aa + K_bb - 2 * K_ab where a = index of first alpha and b of second alpha of the pair
    
    double eta = kernel_matrix[alpha_a_index][alpha_a_index] + kernel_matrix[index][index] - 2*kernel_matrix[alpha_a_index][index];
    
    // update alpha_b
    double alpha_b_new = alphas[index] + (labels[index]*error)/eta;

    // clip eta
    if(alpha_b_new>L){
        alpha_b_new = L;
    }
    else if(alpha_b_new<H){
        alpha_b_new = H;
    }

    // update alpha_a 
    double alpha_a_new = alpha_a + labels[alpha_a_index]*labels[index]*(alphas[index]-alpha_b_new);

    // update bias
    bias = bias + error_vector[index] + labels[alpha_a_index]*(alpha_a_new-alpha_b_new) + labels[index]*(alpha_b_new - alphas[index]);

    //update alpha vector 
    alphas[alpha_a_index] = alpha_a_new;
    alphas[index] = alpha_b_new;

    }
}   