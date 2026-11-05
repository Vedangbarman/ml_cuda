#include <cmath>
#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

using namespace std;

// vector<vector<int>> matrix = {
//         {1, 2, 3},
//         {4, 5, 6},
//         {7, 8, 9}
//     }

vector<double> compute_mean(const vector<vector<double>> matrix){
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    vector<double> mean(cols,0.0);

    for(size_t i = 0;i<rows;++i){
        for(size_t j = 0;j<cols;++j){
            mean[j] += matrix[i][j];
        }
    }

    for(size_t j = 0; j<cols;++j){
        mean[j] /= static_cast<double>(rows);
    }

    return mean;

}

vector<vector<double>> center_data(const vector<vector<double>> matrix, const vector<double> mean ){
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    vector<vector<double>> centered_matrix(rows,vector<double>(cols));

    for(size_t i = 0; i<rows; ++i){
        for(size_t j = 0;j<cols; ++j){
            centered_matrix[i][j] = (matrix[i][j] - mean[j]);
        }
    }

    return centered_matrix;

}


vector<double> compute_variance(const vector<vector<double>> centered_matrix){
    size_t rows = centered_matrix.size();
    size_t cols = centered_matrix[0].size();

    vector<double> variance(0.0);

    for(size_t i = 0; i<rows; i++){
        for(size_t j = 0; j<cols; ++j){
            variance[j] += pow(centered_matrix[i][j],2);
        }
    }
    for(size_t j = 0; j<variance.size(); j++){
        variance[j] = variance[j]/(rows-1);
    }

    return variance;

}


vector<double> compute_covariance(const vector<vector<double>> centered_matrix){
    size_t rows = centered_matrix.size();
    size_t cols = centered_matrix[0].size();

    vector<double> covariance(0.0);

    for(size_t i = 0; i<rows; i++){
        for(size_t j = 0; j<cols ; j++){
            if(i<cols){
                covariance[j] += pow((centered_matrix[i][j] + centered_matrix[i][j+1]),2);
            }
        }
    }

    for(size_t j = 0; j<rows; j++){
        covariance[j] = covariance[j]/(rows-1);
    }

    return covariance;

}

vector<vector<double>> create_matrix(const vector<vector<double>> matrix){

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    vector<vector<double>> cov_matrix(rows,vector<double>(cols,0.0));
    vector<double> mean = compute_mean(vector<vector<double>> (matrix));

    vector<vector<double>> center = center_data(vector<vector<double>> (matrix), vector<double> (mean));

    vector<double> variance = compute_variance(vector<vector<double>> (center));

    vector<double> covariance = compute_covariance(vector<vector<double>> (center));

    for(int i = 0 ; i<rows; i++){
        cov_matrix[i][i] = variance[i];
        for(int j = 0; j<cols; j++){
            if(i != j){
                cov_matrix[i][j] = covariance[j];
            }           
            if(i == j){
                cov_matrix[i][j+1] = covariance[j];
            }
        }
    }

    return cov_matrix;

}

vector<double> csr_values_finder(vector<vector<double>>(cov_matrix)){
    size_t rows = cov_matrix.size();
    size_t cols = cov_matrix[0].size();

    vector<double> csr_values(0.0);

    for(int i = 0; i<rows; i++){
        for(int j = 0; j<cols; j++){
            if(cov_matrix[i][j] != 0 ){
                csr_values[i] = cov_matrix[i][j];
            }
        }
    }
    return csr_values;

}

vector<double> csr_row_finder(vector<vector<double>>(cov_matrix)){
    size_t rows = cov_matrix.size();
    size_t cols = cov_matrix[0].size();

    vector<double> csr_rows(0.0);
    for(int i = 0; i<rows; i++){
        for(int j = 0; j<cols; j++){
            if(cov_matrix[i][j] != 0 ){
                csr_rows[i] = i;
            }
        }
    }
    return csr_rows;

}


vector<double> csr_col_finder(vector<vector<double>>(cov_matrix)){
    size_t rows = cov_matrix.size();
    size_t cols = cov_matrix[0].size();

    vector<double> csr_cols(0.0);
    for(int i = 0; i<rows; i++){
        for(int j = 0; j<cols; j++){
            if(cov_matrix[i][j] != 0 ){
                csr_cols[i] = j;
            }
        }
    }
    return csr_cols;

}


vector<double> compute_eigen_values(){
    
}