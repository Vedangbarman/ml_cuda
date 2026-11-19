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

vector<double> error_store(const vector<vector<double>> & kernel_matrix, const vector<double> & labels){
    if(kernel_matrix.empty()){
        return {};
    }
    double bias = 0.0;
    size_t rows = kernel_matrix.size();
    vector<double> alphas(rows,0.0); // shape = 1 by sample size all intiliazed to 0
    vector<double> error_vector(rows,0.0);
    // check kkt conditions 
    double model_output = 0;
    double model_error = 0;
    for(size_t i = 0; i<rows; i++){
        model_output = 0;
        for(size_t j = 0; j<rows;j++){
            model_output += alphas[j] * labels[j] * kernel_matrix[j][i];
        }
        model_error = model_output + bias - labels[i];
        error_vector[i] = model_error;
    }
    return error_vector;
}

vector<double> compute_alpha(const vector<vector<double>> & kernel_matrix, 
const double & error_tolerance, 
const vector<double> & labels, const double & c, 
vector<double> error_vector
,const int & max_it
){
    if(kernel_matrix.empty()){
        return {-1};
    }
    if(max_it == 0){
        return {-1};
    }
    size_t rows = kernel_matrix.size();
    vector<double> alphas(rows+1,0.0); // shape = 1 by sample size all intiliazed to 0
    vector<bool> skip(rows, false);
    // check kkt conditions 
    double bias = 0.0;
    double model_output = 0.0;
    double model_error = 0.0;
    double margin_validation = 0.0;
    double error = 0.0;
    double alpha_a = 0.0;
    int alpha_a_index = -1;
    // chose alpha a ( first alpha of the pair)
    int operation = 0;
    while(operation<max_it){  
        alpha_a_index = -1;  
        for(size_t i = 0; i<rows; i++){
            if(skip[i]){
                continue;
            }
            model_output = 0; 
            for(size_t j = 0; j<rows;j++){
                model_output += alphas[j] * labels[j] * kernel_matrix[j][i];
                
            }
            model_error = model_output + bias - labels[i];
            error_vector[i] = model_error;
            margin_validation = model_error*labels[i];
            if(margin_validation > error_tolerance){
                if(alphas[i] > 0){
                    alpha_a = alphas[i];
                    alpha_a_index = i;
                    break;
                }
            }
            else if(margin_validation < -error_tolerance){
                if(alphas[i] < c){
                    alpha_a = alphas[i];
                    alpha_a_index = i;
                    break;
                }
            }
        }
        if (alpha_a_index == -1){
            break;
        }
        size_t index = 0;

        //chose alpha b 
        if(model_error < 0){
            auto max_pos = max_element(error_vector.begin(), error_vector.end());
            index = max_pos - error_vector.begin();
        }
        else{
            auto min_pos = min_element(error_vector.begin(), error_vector.end());
            index = min_pos - error_vector.begin();
        }
        error = model_error - error_vector[index];
        
        // compute clips of [L,H]
        double L = 0.0;
        double H = 0.0;
        if(labels[alpha_a_index] != labels[index]){
            L = max(0.0, alphas[index] - alpha_a);
            H = min((double)c, c + alphas[index] - alpha_a);
        }
        else{
            L = max(0.0, alpha_a + alphas[index] - c);
            H = min((double)c, alpha_a + alphas[index]);
        }
        if(L >= H){
            skip[alpha_a_index] = true;
            continue;
        }
        // calculate eta ( pronounce AY-THA or EE-THA)/ basically caculating second derivative formula is K_aa + K_bb - 2 * K_ab where a = index of first alpha and b of second alpha of the pair
        
        double eta = kernel_matrix[alpha_a_index][alpha_a_index] + kernel_matrix[index][index] - 2*kernel_matrix[alpha_a_index][index];
        if(index == alpha_a_index){
            skip[alpha_a_index] = true;
            continue;
        }
        if(eta <= 0) {
            skip[alpha_a_index] = true;
            continue;
        }
        // update alpha_b
        double alpha_b_new = alphas[index] + (labels[index]*error)/eta;

        // clip eta
        if(alpha_b_new > H) {
            alpha_b_new = H;
        }
        else if(alpha_b_new < L) {
            alpha_b_new = L;
        }
        // update alpha_a 
        double alpha_a_new = alpha_a + labels[alpha_a_index]*labels[index]*(alphas[index]-alpha_b_new);

        // update bias
        double da = alpha_a_new - alpha_a;
        double db = alpha_b_new - alphas[index];
        double b1 = bias - model_error - labels[alpha_a_index]*da*kernel_matrix[alpha_a_index][alpha_a_index]
                                    - labels[index]*db*kernel_matrix[alpha_a_index][index];
        double b2 = bias - error_vector[index] - labels[alpha_a_index]*da*kernel_matrix[alpha_a_index][index]
                                            - labels[index]*db*kernel_matrix[index][index];
        if(alpha_a_new > 0 && alpha_a_new < c){
            bias = b1;
        } 
        else if(alpha_b_new > 0 && alpha_b_new < c){ 
            bias = b2;
        }
        else {
            bias = (b1 + b2) / 2;
        }
        alphas[rows] = bias;
        //update alpha vector 
        alphas[alpha_a_index] = alpha_a_new;
        alphas[index] = alpha_b_new;
        for(size_t k = 0; k<rows; k++){
            double out = 0;
            for(size_t j = 0; j<rows; j++){
                out += alphas[j] * labels[j] * kernel_matrix[j][k];
                }
            error_vector[k] = out + bias - labels[k];
        }
        fill(skip.begin(), skip.end(), false);   
        operation++;
    }
    return alphas;
}   