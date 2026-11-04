#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>

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

    for(size_t i = 0;i<cols;++i){
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


vector<double> variance(const vector<vector<double>>& centered_matrix){
    size_t rows = centered_matrix.size();
    size_t cols = centered_matrix[0].size();

    vector<double> variance;

    for(size_t i = 0; i<rows; i++){
        for(size_t j = 0; j<cols; ++j){
            variance[j] += centered_matrix[i][j];
        }
    }
    for(size_t j = 0; j<= variance.size(); j++){
        variance[j] = pow(variance[j],2);
        variance[j] = variance[j]/rows-1;
    }

    return variance;

}


vector<double> covariance(const vector<vector<double>> centered_matrix){
    size_t rows = centered_matrix.size();
    size_t cols = centered_matrix[0].size();

    vector<double> covariance;

    for(size_t i = 0; i<rows; i++){
        for(size_t j = 0; j<cols ; j++){
            if(i<cols){
                covariance[j] += (centered_matrix[i][j] + centered_matrix[i+1][j]);
            }
        }
    }

    for(size_t j = 0; j<rows; j++){
        covariance[j] = pow(covariance[j],2);
        covariance[j] = covariance[j]/rows-1;
    }

    return covariance;

}