#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <future>
#include <chrono>
using namespace std;

int raise_to_the_power (int value, int power) {
    int result;
    if (power < 0) {
        result = 0;
        std::cout << "Not positive power" << std::endl;
    }
    else if (power == 0) result = 1;
    else result = value;

    for (int i = 0; i < power-1; i++) {
        result = result * value;
    }
    
    return result;
}

template <class T>
class Matrix {
private:
    T** M; 
    size_t height; 
    size_t width;
    const char* name; 
public:
    Matrix(const char* Matrix_Name) {
        height  = 0;
        width   = 0;
        name    = Matrix_Name;
        M       = nullptr;
    }
    
    Matrix(const char* Matrix_Name, int n, int m) {
        height  = n;
        width   = m;
        name    = Matrix_Name;

        M = (T**) new T*[height];
        for (int i = 0; i < height; i++) M[i] = (T*) new T[width];

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                M[i][j] = 0;
            }
        }
    }

    Matrix(const char* Matrix_Name, const char* File_Name) {
        name    = Matrix_Name;
        std::ifstream in(File_Name);
        if (in.is_open()) {
            in >> height >> width;
            M = (T**) new T*[height];
            for (int i = 0; i < height; i++) M[i] = (T*) new T[width];

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    in >> M[i][j];
                }
            }
        }
        else std::cout << "File cannot be read" << std::endl;
        in.close();
    }

    void set_matrix() {
        if ((height == 0)&&(width == 0)) {
            std::cout << "Enter the height and width of the matrix " << name << ": " << std::endl;
            std::cin >> height >> width;

            M = (T**) new T*[height];
            for (int i = 0; i < height; i++) M[i] = (T*) new T[width];
        }
        std::cout << "Enter the elements of the matrix " << name << ": " << std::endl;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                std::cin >> M[i][j];
            }
        }
    }

    void set_fromfile_matrix(const char* File_Name) {
        std::ifstream in(File_Name);
        if (in.is_open()) {
            if ((height == 0)&&(width == 0)) {
                in >> height >> width;

                M = (T**) new T*[height];
                for (int i = 0; i < height; i++) M[i] = (T*) new T[width];
            }
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    in >> M[i][j];
                }
            }
        }
        else std::cout << "File cannot be read" << std::endl;
        in.close();
    }

    void set_infile_matrix(const char* File_Name) {
        std::ofstream out(File_Name);
        if (out.is_open()) {
            out << height << " " << width << std::endl;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    out << M[i][j] << " ";
                }
                out << std::endl;
            }
        }
        else std::cout << "File cannot be written" << std::endl;
        out.close();
    }

    T** get_matrix() {
        return M;
    }

    int get_size() {
        if (height != width) {
            std::cout << "The matrix is not square" << std::endl;
            return 0;
        }
        return height;
    }

    int get_row() { return height;  }
    int get_col() { return width;   }

    void print_matrix() {
        std::cout << "Matrix " << name << ": " << std::endl;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                std::cout << M[i][j] << '\t';
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void set_zero_matrix(int n, int m) {
        if ((n == 0)||(m == 0)) std::cout << "The size are not suitable" << std::endl;
        else {
            if (width > 0) {
                for (int i = 0; i < height; i++) {
                    delete[] M[i];
                }
            }
            if (height > 0) delete[] M;

            height  = n;
            width   = m;

            M = (T**) new T*[height];
            for (int i = 0; i < height; i++) M[i] = (T*) new T[width];
            
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    M[i][j] = 0;
                }
            }
        }
    }

    void set_identity_matrix(int n, int m) {
        if ((n != m)||(n == 0)||(m == 0)) std::cout << "The size are not suitable" << std::endl;
        else {
            if (width > 0) {
                for (int i = 0; i < height; i++) {
                    delete[] M[i];
                }
            }
            if (height > 0) delete[] M;

            height  = n;
            width   = m;

            M = (T**) new T*[height];
            for (int i = 0; i < height; i++) M[i] = (T*) new T[width];
            
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    if (i == j) M[i][j] = 1;
                    else M[i][j] = 0;
                }
            }
        }
    }

    void reformed_matrix(T **base_matr, T **matr, int sz, int row, int col) {
        int new_row = 0;
        for (int i = 0; i < sz; i++) {
            if (i != row) {
                int new_col = 0;
                for (int j = 0; j < sz; j++) {
                    if (j != col) {
                        matr[new_row][new_col] = base_matr[i][j];
                        new_col ++;
                    }
                }
                new_row ++;
            }
        }
    }

    void transpose_matrix(T **matr, int row, int col) {
        
        // creating an auxiliary matrix
        T **new_matr = new T* [col];
        for (int i = 0; i < col; i++) {
            new_matr[i] = new T [row];
        }

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                new_matr[j][i] = matr[i][j];
            }
        }

        if (row != col) {
            if (col > 0) {
                for (int i = 0; i < row; i++) {
                    delete[] matr[i];
                }
            }
            if (row > 0) delete[] matr;

            height  = col;
            width   = row;

            matr = (T**) new T*[col];
            for (int i = 0; i < col; i++) matr[i] = (T*) new T[row];
        }

        for (int i = 0; i < col; i++) {
            for (int j = 0; j < row; j++) {
                matr[i][j] = new_matr[i][j];
            }
        }
        // deleting an auxiliary matrix
        if (col > 0) {
            for (int i = 0; i < col; i++) {
                delete[] new_matr[i];
            }
        }
        if (row > 0) delete[] new_matr;
    }

    int Determinant(T **matrix, int size) {
        int value = 0;
        if (size == 0) std::cout << "The size is not suitable" << std::endl;
        else if (size == 1) value = matrix[0][0];
        else if (size == 2) value = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        else {
            int k = 1;
            for (int i = 0; i < size; i++) {

                T **new_matrix = new T* [size-1];
                for (int j = 0; j < size-1; j++) {
                    new_matrix[j] = new T [size-1];
                }

                reformed_matrix(matrix, new_matrix, size, 0, i);

                value = value + k * matrix[0][i] * Determinant(new_matrix, size-1);
                k = (-1) * k;

                if (size-1 > 0) {
                    for (int k = 0; k < size-1; k++) {
                        delete[] new_matrix[k];
                    }
                }
                if (size-1 > 0) delete[] new_matrix;
            }
        }
        return value;
    }
    
    void get_matrix_inverse() {
        int det = Determinant(M, height);
        if (det == 0) {
            std::cout << "Matrix is not invertible" << std::endl;
        }
        else {
            // creating an auxiliary matrix
            T **inv_mat = new T* [height];
            for (int i = 0; i < height; i++) {
                inv_mat[i] = new T [width];
            }

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    T **new_matrix = new T* [height-1];
                    for (int k = 0; k < height-1; k++) {
                        new_matrix[k] = new T [width-1];
                    }

                    reformed_matrix(M, new_matrix, height, i, j);

                    inv_mat[i][j] = raise_to_the_power(-1, (i+j+2)) * Determinant(new_matrix, height-1) / det;

                    if (height-1 > 0) {
                        for (int k = 0; k < height-1; k++) {
                            delete[] new_matrix[k];
                        }
                    }
                    if (height-1 > 0) delete[] new_matrix;
                }
            }

            transpose_matrix(inv_mat, height, width);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    M[i][j] = inv_mat[i][j];
                }
            }

            // deleting an auxiliary matrix
            if (width > 0) {
                for (int i = 0; i < height; i++) {
                    delete[] inv_mat[i];
                }
            }
            if (height > 0) delete[] inv_mat;
        }
    }

    bool operator!() {
        if (Determinant(M, height) == 0) {
            std::cout << "Matrix is not invertible" << std::endl;
            return false;
        }
        return true;
    }
 
    Matrix(size_t height, size_t width, T val) {
        this->height = height;
        this->width = width;
        M = new T*[height];
        for (size_t i = 0; i < height; i++)
        M[i] = new T[width];
        for (size_t i = 0; i < height; i++)
            for (size_t j = 0; j < width; j++)
                M[i][j] = val;
    }

    Matrix(size_t height, size_t width) {
        this->height = height;
        this->width = width;
        M = new T*[height]; 
        for (size_t i = 0; i < height; i++)
        M[i] = new T[width];
        for (size_t i = 0; i < height; i++)
            for (size_t j = 0; j < width; j++)
                std::cin >> M[i][j];
    }


    Matrix(const Matrix& M1) {
        height = M1.height;
        width = M1.width;
        M = (T**) new T*[height];
        for (size_t i = 0; i < height; i++)
        M[i] = (T*) new T[width];
        for (size_t i = 0; i < height; i++)
        for (size_t j = 0; j < width; j++)
            M[i][j] = M1.M[i][j];
    }

    Matrix operator=(const Matrix& M2) {
        if (*this == M2) {
            return *this;
        }
        else {
            if (width > 0)
            {
                for (size_t i = 0; i < height; i++)
                    delete[] M[i];
            }
            if (height > 0)
            {
            delete[] M;
            }
            height = M2.height;
            width = M2.width;
            M = (T**) new T*[height]; 
            for (size_t i = 0; i < height; i++)
                M[i] = (T*) new T[width];
            for (size_t i = 0; i < height; i++)
                for (size_t j = 0; j < width; j++)
                    M[i][j] = M2.M[i][j];
            return *this;
        }
    }

    ~Matrix() {
        if (width > 0)
        {
        for (size_t i = 0; i < height; i++)
            delete[] M[i];
        }
        if (height > 0) {
            delete[] M;
        }
    }
  
    Matrix operator+(const Matrix& m1) const { 
        if ((height != m1.height)||(width != m1.width)) {
            std::cerr << "The operation cannot be performed due to different Matrix sizes" << std::endl;
            return *this;
        }
        else {
            Matrix b = Matrix(height, width, 0);
            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    b.M[i][j] = M[i][j] + m1.M[i][j];
                }
            }
            return b;
        }
    }

    Matrix operator-(const Matrix& m2) const { 
        if ((height != m2.height)||(width != m2.width)) {
            std::cerr << "The operation cannot be performed due to different Matrix sizes" << std::endl;
            return *this;
        }
        else {
            Matrix b = Matrix(height, width, 0);
            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    b.M[i][j] = M[i][j] - m2.M[i][j];
                }
            }
            return b;
        }
    }
    Matrix operator*(T num) const { 
        Matrix b = Matrix(height, width, 0);
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                b.M[i][j] = (T) M[i][j] * num;
            }
        }
        return b;
    }
    Matrix operator*(const Matrix& a) const{ 
        if (width != a.height ) {
            std::cerr << "It is impossible to perform the operation due to the inequality of the number of columns of the first Matrix and the number of rows of the second Matrix" << std::endl;
            return *this;
        }
        else {
            Matrix b = Matrix(height, a.width, 0);
            for (size_t i = 0; i < height; ++ i)
                for (size_t j = 0; j < a.width; ++ j)
                    for (size_t k = 0; k < width; ++ k)
                        b.M[i][j] += M[i][k] * a.M[k][j];
            return b;
        }
    }
    
    bool operator==(const Matrix& m3) const { 
        if ((height != m3.height) || (width != m3.width)) {
            return false;
        }
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                if (M[i][j] != m3.M[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }
    
    bool operator!=(const Matrix& m4) const { 
        return !(*this == m4);
    }
    Matrix add(const Matrix& b, int Treads_col) { //std::thread version of matrix addition
        if ((height != b.height)||(width != b.width)) {
            std::cerr << "The operation cannot be performed to different Matrix sizes" << std::endl;
            return *this;
        }
        else {
            Matrix result(height, width, 0);

            //creating a vector of threds
            std::vector<std::thread> threads(Treads_col);

            int sz_ch = height / Treads_col;
            int sz_ch_l = sz_ch + height % Treads_col;

            int cur_row = 0;
            for (int t = 0; t < Treads_col; t++) {
                int start_row = cur_row;
                int end_row = cur_row + (t == Treads_col - 1 ? sz_ch_l : sz_ch);
                cur_row = end_row;
                threads[t] = thread([start_row, end_row, this, &b, &result]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < width; j++) {
                        result.M[i][j] = M[i][j] + b.M[i][j];
                    }
                }
                });
            }

            for (int i = 0; i < Treads_col; i++) {
            threads[i].join();
            }

            return result;
        }
    }
    Matrix async_add(const Matrix& b, int blockSize) const { //std::async version of matrix addition
        if ((height != b.height)||(width != b.width)) {
            std::cerr << "The operation cannot be performed due to different Matrix sizes" << std::endl;
            return *this;
        }
        else {

            Matrix result(height, width, 0);
            std::vector<future<void>> futures;
            for (int i = 0; i < height; i += blockSize) {
                for (int j = 0; j < width; j += blockSize) {
                        futures.push_back(async([this, &b, &result, blockSize, i, j]() {
                            int blockEndI = min(i + blockSize, (int) height);
                            int blockEndJ = min(j + blockSize, (int) width);
                            for (int ii = i; ii < blockEndI; ++ii) {
                                for (int jj = j; jj < blockEndJ; ++jj) {
                                    result.M[ii][jj] = M[ii][jj] + b.M[ii][jj];
                                }
                            }
                        }));
                }
            }

            for (auto& f : futures) f.wait();

            return result;
        }
    }
    Matrix sub(const Matrix& b, int Treads_col) const { //std::thread version of matrix subtraction
        if ((height != b.height)||(width != b.width)) {
            std::cerr << "The operation cannot be performed due to different Matrix sizes" << std::endl;
            return *this;
        }
        else {
            Matrix result(height, width, 0);

            //creating a vector of threds
            vector<thread> threads(Treads_col);

            int sz_ch = height / Treads_col;
            int sz_ch_l = sz_ch + height % Treads_col;

            int cur_row = 0;
            for (int t = 0; t < Treads_col; t++) {
                int start_row = cur_row;
                int end_row = cur_row + (t == Treads_col - 1 ? sz_ch_l : sz_ch);
                cur_row = end_row;
                threads[t] = thread([start_row, end_row, this, &b, &result]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < width; j++) {
                        result.M[i][j] = M[i][j] - b.M[i][j];
                    }
                }
                });
            }

            for (int i = 0; i < Treads_col; i++) {
            threads[i].join();
            }

            return result;
        }
    }
    Matrix async_sub(const Matrix& b, int blockSize) const { //std::async version of matrix subtraction
        if ((height != b.height)||(width != b.width)) {
            std::cerr << "The operation cannot be performed due to different Matrix sizes" << std::endl;
            return *this;
        }
        else {

            Matrix result(height, width, 0);
            vector<future<void>> futures;
            for (int i = 0; i < height; i += blockSize) {
                for (int j = 0; j < width; j += blockSize) {
                        futures.push_back(async([this, &b, &result, blockSize, i, j]() {
                            int blockEndI = min(i + blockSize, (int) height);
                            int blockEndJ = min(j + blockSize, (int) width);
                            for (int ii = i; ii < blockEndI; ++ii) {
                                for (int jj = j; jj < blockEndJ; ++jj) {
                                    result.M[ii][jj] = M[ii][jj] - b.M[ii][jj];
                                }
                            }
                        }));
                }
            }

            for (auto& f : futures) f.wait();

            return result;
        }
    }
    Matrix multiplic_scal(T num, int Treads_col) const { //std::thread version of matrix*(int)
            Matrix result(height, width, 0);
            //creating a vector of threds
            vector<thread> threads(Treads_col);

            int sz_ch = height / Treads_col;
            int sz_ch_l = sz_ch + height % Treads_col;

            int cur_row = 0;
            for (int t = 0; t < Treads_col; t++) {
                int start_row = cur_row;
                int end_row = cur_row + (t == Treads_col - 1 ? sz_ch_l : sz_ch);
                cur_row = end_row;
                threads[t] = thread([start_row, end_row, this, num, &result]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < width; j++) {
                        result.M[i][j] = (T) M[i][j] * num;
                    }
                }
                });
            }

            for (int i = 0; i < Treads_col; i++) {
            threads[i].join();
            }

            return result;
    }
    Matrix async_mult1(T num, int blockSize) const { //std::async version of mmatrix*(int)

            Matrix result(height, width, 0);
            vector<future<void>> futures;
            for (int i = 0; i < height; i += blockSize) {
                for (int j = 0; j < width; j += blockSize) {
                        futures.push_back(async([this, num, &result, blockSize, i, j]() {
                            int blockEndI = min(i + blockSize, (int) height);
                            int blockEndJ = min(j + blockSize, (int) width);
                            for (int ii = i; ii < blockEndI; ++ii) {
                                for (int jj = j; jj < blockEndJ; ++jj) {
                                    result.M[ii][jj] = (T) M[ii][jj] * num;
                                }
                            }
                        }));
                }
            }

            for (auto& f : futures) f.wait();

            return result;
    }
    Matrix mult2(const Matrix& b, int Treads_col) const{ //std::thread version of matrix multiplication
        if (width != b.height ) {
            std::cerr << "It is impossible to perform the operation due to the inequality of the number of columns of the first Matrix and the number of rows of the second Matrix" << std::endl;
            return *this;
        }
        else {

            Matrix result(height, b.width, 0);
            //creating a vector of threds
            vector<thread> threads(Treads_col);
            
            int sz_ch = height / Treads_col;
            int sz_ch_l = sz_ch + height % Treads_col;

            int cur_row = 0;
            for (int t = 0; t < Treads_col; t++) {
                int start_row = cur_row;
                int end_row = cur_row + (t == Treads_col - 1 ? sz_ch_l : sz_ch);
                cur_row = end_row;
                threads[t] = thread([start_row, end_row, this, &b, &result]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < b.width; j++) {
                        for (int k = 0; k < width; ++k)
                            result.M[i][j] += M[i][k] * b.M[k][j];
                    }
                }
                });
            }

            for (int i = 0; i < Treads_col; i++) {
            threads[i].join();
            }

            return result;
        }
    }
    Matrix async_multiplic_matr(const Matrix& b, int blockSize) { //std::async version of matrix multiplication
        if (width != b.height ) {
            std::cerr << "It is impossible to perform the operation due to the inequality of the number of columns of the first Matrix and the number of rows of the second Matrix" << std::endl;
            return *this;
        }
        else {
            Matrix result(height, width, 0);
            vector<future<void>> futures;
            for (int i = 0; i < height; i += blockSize) {
                for (int j = 0; j < b.width; j += blockSize) {
                    for (int k = 0; k < width; k += blockSize) {
                        futures.push_back(async([this, &b, &result, blockSize, i, j, k]() {
                            int blockEndI = min(i + blockSize, (int) height);
                            int blockEndJ = min(j + blockSize, (int) b.width);
                            int blockEndK = min(k + blockSize, (int) width);
                            for (int ii = i; ii < blockEndI; ++ii) {
                                for (int jj = j; jj < blockEndJ; ++jj) {
                                    for (int kk = k; kk < blockEndK; ++kk) {
                                        result.M[ii][jj] += M[ii][kk] * b.M[kk][jj];
                                    }
                                }
                            }
                        }));
                    }
                }
            }
            for (auto& f : futures) f.wait();
            return result;
        }
    }
    bool compare(const Matrix& b, int Treads_col) const { //std::thread version of matrix equality check
        if ((height != b.height) || (width != b.width)) {
            return false;
        }
        bool flag = true;
        //creating a vector of threds
        vector<thread> threads(Treads_col);

        int sz_ch = height / Treads_col;
        int sz_ch_l = sz_ch + height % Treads_col;

        int cur_row = 0;
        for (int t = 0; t < Treads_col; t++) {
            int start_row = cur_row;
            int end_row = cur_row + (t == Treads_col - 1 ? sz_ch_l : sz_ch);
            cur_row = end_row;
            threads[t] = thread([start_row, end_row, this, &b, &flag]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < width; j++) {
                        if (M[i][j] != b.M[i][j]) {
                            flag = false;
                            break;
                        }
                    }
                }
            });
        }

        for (int i = 0; i < Treads_col; i++) {
            threads[i].join();
        }

        return flag;
    }
    bool async_compare(const Matrix& b, int blockSize) const { //std::async version of matrix equality check
        if ((height != b.height)||(width != b.width)) {
            return false;
        }
        else {
            bool flag = true;
            vector<future<void>> futures;
            for (int i = 0; i < height; i += blockSize) {
                for (int j = 0; j < width; j += blockSize) {
                        futures.push_back(async([this, &b, &flag, blockSize, i, j]() {
                            int blockEndI = min(i + blockSize, (int) height);
                            int blockEndJ = min(j + blockSize, (int) width);
                            for (int ii = i; ii < blockEndI; ++ii) {
                                for (int jj = j; jj < blockEndJ; ++jj) {
                                    if (M[ii][jj] != b.M[ii][jj]) {
                                        flag = false;
                                    }
                                }
                            }
                        }));
                }
            }

            for (auto& f : futures) f.wait();

            return flag;
        }
    }
    bool notcompare(const Matrix& b, int Treads_col) const { //std::thread version of matrix unequality check
        return !(this->compare(b));
    }
    bool async_notcompare(const Matrix& b, int blockSize) const { //std::async version of matrix unequality check
        return !(this->compare(b, blockSize));
    }
    static Matrix zero_matr(size_t r1, size_t c1) {
        Matrix b(r1, c1, 0);
        return b;
    }
    static Matrix one_matr(size_t size) {
        Matrix a(size, size, 0);

        for (size_t i = 0; i < size; i++)
            a.M[i][i] = 1;

        return a;
    }
    bool operator==(int val) const {
        if (val == 0) {
            Matrix q = Matrix::zero_matr(height, width);
            if (*this == q) {
                return true;
            }
            else {
                return false;
            }
        }
        else if (val == 1) {
            Matrix w = Matrix::one_matr(height);
            if (*this == w) {
                return true;
            }
            else {
                return false;
            }
        }
        else {
            std::cerr << "Matrix can only be compared with 0 or 1" << std::endl;
            return false;
        }
    }
    
    bool operator!=(int val) const {
        if (val == 0) {
            Matrix q = Matrix::zero_matr(height, width);
            if (*this == q) {
                return false;
            }
            else {
                return true;
            }
        }
        else if (val == 1) {
            Matrix w = Matrix::one_matr(height);
            if (*this == w) {
                return false;
            }
            else {
                return true;
            }
        }
        else {
            std::cerr << "Matrix can only be compared with 0 or 1" << std::endl;
            return false;
        }
    }

    template <typename T1>
    friend std::ostream& operator<<(std::ostream& out, const Matrix<T1>& mtr);
};

template <class T>
std::ostream& operator<<(std::ostream& out, const Matrix<T>& mtr) {
    for (size_t i = 0; i < mtr.height; i++)
    {
        for (size_t j = 0; j < mtr.width; j++)
            out << mtr.M[i][j] << '\t';
        out << std::endl;
    }

    return out;
}

int main() {
    Matrix<int> M_1("M_1");
    M_1.print_matrix();

    Matrix<int> M_2("M_2", 7, 3);
    M_2.print_matrix();

    Matrix<int> M_3("M_3");
    //M_3.set_matrix();
    M_3.print_matrix();

    Matrix<int> M_4("M_4", 2, 2);
    //M_4.set_matrix();
    M_4.print_matrix();  
    

    Matrix<int> M_5("M_5");
    M_5.set_fromfile_matrix("matrix_files/M_5.txt");
    M_5.print_matrix();
    M_5.set_infile_matrix("matrix_files/M_5_in.txt");

    Matrix<int> M_6("M_6", 2, 5);
    M_6.set_fromfile_matrix("matrix_files/M_6.txt");
    M_6.print_matrix();
    M_6.set_infile_matrix("matrix_files/M_6_in.txt");


    // set matrix from file constructor check
    Matrix<int> M_7("M_7", "matrix_files/M_7.txt");
    M_7.print_matrix();
    

    // assignment operator check
    Matrix<int> M_8("M_8", 3, 3);
    Matrix<int> M_9("M_9");
    M_8 = M_9 = M_6;
    M_8.print_matrix();
    M_9.print_matrix();


    // self-copying check
    M_8 = M_8;
    M_8.print_matrix();


    // arithmetic operator+ check (Вопрос об изменении матрицы, к которой применяется прибавление в формулле с =; можно создать доп поле M_dop или просто матрицу)
    M_8 + M_9;
    M_8.print_matrix();

    M_8 + M_9 + M_9;
    M_8.print_matrix();
    M_9.print_matrix();

    M_8 = M_9;
    Matrix<int> M_10("M_10");
    M_10 = M_8 + M_9;
    M_10.print_matrix();
    M_8.print_matrix();
    M_9.print_matrix();
    

    // arithmetic operator- check
    M_8 - M_9 - M_9;
    M_8.print_matrix();
    M_9.print_matrix();


    // arithmetic operator* check
        //square matrix
    Matrix<int> M_11("M_11", "matrix_files/M_umn_1.txt");
    Matrix<int> M_12("M_12", "matrix_files/M_umn_2.txt");
    M_11.print_matrix();
    M_12.print_matrix();
    M_11 * M_12;
    M_11.print_matrix();
        //rectangle matrix
    Matrix<int> M_13("M_13", "matrix_files/M_umn_3.txt");
    Matrix<int> M_14("M_14", "matrix_files/M_umn_4.txt");
    M_13.print_matrix();
    M_14.print_matrix();
    M_13 * M_14;
    M_13.print_matrix();


    // arithmetic operator* check
    M_14.print_matrix();
    M_14 * 5;
    M_14.print_matrix();


    // comparison operator check
    Matrix<int> M_15("M_15", "matrix_files/M_4.txt");
    Matrix<int> M_16("M_16", "matrix_files/M_5.txt");
    if (M_15 == M_16) std::cout << "equal" << std::endl;
    else std::cout << "NOT equal" << std::endl;
    if (M_15 != M_16) std::cout << "NOT equal" << std::endl;
    else std::cout << "equal" << std::endl;


    // comparison operator check - Identity/Zero matrix
    Matrix<int> M_17("M_17", "matrix_files/M_17.txt");
    if (M_17 == 1) std::cout << "Identity matrix" << std::endl;
    else std::cout << "Not Identity matrix" << std::endl;

    Matrix<int> M_18("M_18", "matrix_files/M_18.txt");
    if (M_18 != 0) std::cout << "Not Zero matrix" << std::endl;
    else std::cout << "Zero matrix" << std::endl;
    

    // creating Identity/Zero matrix check
    Matrix<int> M_19("M_19");
    M_19.set_zero_matrix(3, 5);
    M_19.print_matrix();
    
    Matrix<int> M_20("M_20");
    M_20.set_identity_matrix(5, 5);
    M_20.print_matrix();


    // searching inverse matrix method check
    Matrix<int> M_21("M_21", "matrix_files/M_21.txt");
    M_21.print_matrix();
    std::cout << "Determinant: " << M_21.Determinant(M_21.get_matrix(), M_21.get_size()) << std::endl;

    Matrix<int> M_22("M_22", "matrix_files/M_22.txt");
    M_22.print_matrix();
    std::cout << "Determinant: " << M_22.Determinant(M_22.get_matrix(), M_22.get_size()) << std::endl;

    if (!M_22) std::cout << "Matrix is invertible" << std::endl;
    !M_21;

    Matrix<int> M_23("M_23", "matrix_files/M_23.txt");
    M_23.print_matrix();
    std::cout << "Determinant: " << M_23.Determinant(M_23.get_matrix(), M_23.get_size()) << std::endl;
    M_23.get_matrix_inverse();
    M_23.print_matrix();

    Matrix<float> M_24("M_24", "matrix_files/M_24.txt");
    M_24.print_matrix();
    M_24.get_matrix_inverse();
    M_24.print_matrix();

    Matrix<double> M_25("M_25", "matrix_files/M_25.txt");
    M_25.print_matrix();


    std::cout << "lab_15 std::thread/std::async" << std::endl;
    int tr_col = 12;
    int block = 2;
    auto my_mtrx = Matrix<int>::one_matr(5);
    Matrix<int> mtrx_1(5, 5, 3);
    std::cout << mtrx_1;
    std::cout << std::endl << std::endl;
    auto mtrx_2 = my_mtrx.multiplic_scal(2, tr_col);
    std::cout << mtrx_2;
    std::cout << std::endl << std::endl;
    std::cout << mtrx_2.add(mtrx_1, tr_col) << std::endl;
    std::cout << std::endl << std::endl;
    std::cout << mtrx_2.async_add(mtrx_1, block) << std::endl;
    std::cout << std::endl << std::endl;
    if (mtrx_2.async_notcompare(mtrx_1, block)) {
        std::cout << "not equal" << std::endl;
    }
    std::cout << std::endl << std::endl;
    std::cout << mtrx_2.async_multiplic_matr(mtrx_1, block) << std::endl;

    std::cout << "lab_16" << std::endl;
    std::cout << "With fixed threads" << std::endl << "Time taken: " << std::endl;
    Matrix<int> m1(100, 1000, 4);
    Matrix<int> m2(100, 1000, 5);
    auto startTime = std::chrono::high_resolution_clock::now();
    m1.add(m2, tr_col);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "100*1000: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() << " microseconds" << std::endl;
    Matrix<int> m3(1000, 1000, 4);
    Matrix<int> m4(1000, 1000, 5);
    auto startTime1 = std::chrono::high_resolution_clock::now();
    m3.add(m4, tr_col);
    auto endTime1 = std::chrono::high_resolution_clock::now();
    std::cout << "1000*1000: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTime1 - startTime1).count() << " microseconds" << std::endl;
    Matrix<int> m5(10000, 1000, 4);
    Matrix<int> m6(10000, 1000, 5);
    auto startTime2 = std::chrono::high_resolution_clock::now();
    m5.add(m6, tr_col);
    auto endTime2 = std::chrono::high_resolution_clock::now();
    std::cout << "10000*1000: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - startTime2).count() << " microseconds" << std::endl;
    Matrix<int> m7(100000, 1000, 4);
    Matrix<int> m8(100000, 1000, 5);
    auto startTimee = std::chrono::high_resolution_clock::now();
    m7.add(m8, tr_col);
    auto endTimee = std::chrono::high_resolution_clock::now();
    std::cout << "100000*1000: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTimee - startTimee).count() << " microseconds" << std::endl;
    Matrix<int> m9(250000, 1000, 4);
    Matrix<int> m10(250000, 1000, 5);
    auto startTimee1 = std::chrono::high_resolution_clock::now();
    m9.add(m10, tr_col);
    auto endTimee1 = std::chrono::high_resolution_clock::now();
    std::cout << "250000*1000: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTimee1 - startTimee1).count() << " microseconds" << std::endl;
    auto startTimesq1 = std::chrono::high_resolution_clock::now();
    m1 + m2;
    auto endTimesq1 = std::chrono::high_resolution_clock::now();
    std::cout << "100*1000" << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTimesq1 - startTimesq1).count() << " microseconds" << std::endl;
    auto startTimesq2 = std::chrono::high_resolution_clock::now();
    m3 + m4;
    auto endTimesq2 = std::chrono::high_resolution_clock::now();
    std::cout << "1000*1000: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTimesq2 - startTimesq2).count() << " microseconds" << std::endl;
    auto startTimesq3 = std::chrono::high_resolution_clock::now();
    m5 + m6;
    auto endTimesq3 = std::chrono::high_resolution_clock::now();
    std::cout << "10000*1000: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTimesq3 - startTimesq3).count() << " microseconds" << std::endl;
    auto startTimesq4 = std::chrono::high_resolution_clock::now();
    m7 + m8;
    auto endTimesq4 = std::chrono::high_resolution_clock::now();
    std::cout << "100000*1000: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTimesq4 - startTimesq4).count() << " microseconds" << std::endl;
    auto startTimesq5 = std::chrono::high_resolution_clock::now();
    m9 + m10;
    auto endTimesq5 = std::chrono::high_resolution_clock::now();
    std::cout << "250000*1000: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTimesq5 - startTimesq5).count() << " microseconds" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    std::cout << "With fixed matrix" << std::endl << "Time taken: " << std::endl;
    int tr1 = 4;
    int tr2 = 10;
    int tr3 = 12;
    int tr4 = 50;
    int tr5 = 100;
    int tr6 = 500;
    int tr7 = 1000;
    int tr8 = 1025;
    int tr9 = 1050;
    int tr10 = 2000;
    int tr11 = 3000;
    Matrix<int> m11(1000, 1000, 4);
    Matrix<int> m12(1000, 1000, 5);
    auto startTime4 = std::chrono::high_resolution_clock::now();
    m11.mult2(m12, tr1);
    auto endTime4 = std::chrono::high_resolution_clock::now();
    std::cout << "4: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTime4 - startTime4).count() << " microseconds" << std::endl;
    auto startTime6 = std::chrono::high_resolution_clock::now();
    m11.mult2(m12, tr2);
    auto endTime6 = std::chrono::high_resolution_clock::now();
    std::cout << "10: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTime6 - startTime6).count() << " microseconds" << std::endl;
    auto startTime5 = std::chrono::high_resolution_clock::now();
    m11 * m12;
    auto endTime5 = std::chrono::high_resolution_clock::now();
    std::cout << "1: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTime5 - startTime5).count() << " microseconds" << std::endl;
    auto startTime12 = std::chrono::high_resolution_clock::now();
    m11.mult2(m12, tr3);
    auto endTime12 = std::chrono::high_resolution_clock::now();
    std::cout << "12: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTime12 - startTime12).count() << " microseconds" << std::endl;
    auto startTime50 = std::chrono::high_resolution_clock::now();
    m11.mult2(m12, tr4);
    auto endTime50 = std::chrono::high_resolution_clock::now();
    std::cout << "50: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTime50 - startTime50).count() << " microseconds" << std::endl;
    auto startTime100 = std::chrono::high_resolution_clock::now();
    m11.mult2(m12, tr5);
    auto endTime100 = std::chrono::high_resolution_clock::now();
    std::cout << "100: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTime100 - startTime100).count() << " microseconds" << std::endl;
    auto startTime500 = std::chrono::high_resolution_clock::now();
    m11.mult2(m12, tr6);
    auto endTime500 = std::chrono::high_resolution_clock::now();
    std::cout << "500: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTime500 - startTime500).count() << " microseconds" << std::endl;
    auto startTime1000 = std::chrono::high_resolution_clock::now();
    m11.mult2(m12, tr7);
    auto endTime1000 = std::chrono::high_resolution_clock::now();
    std::cout << "1000: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTime1000 - startTime1000).count() << " microseconds" << std::endl;
    auto startTime1100 = std::chrono::high_resolution_clock::now();
    m11.mult2(m12, tr8);
    auto endTime1100 = std::chrono::high_resolution_clock::now();
    std::cout << "1025: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTime1100 - startTime1100).count() << " microseconds" << std::endl;
    auto startTime1101 = std::chrono::high_resolution_clock::now();
    m11.mult2(m12, tr9);
    auto endTime1101 = std::chrono::high_resolution_clock::now();
    std::cout << "1050: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTime1101 - startTime1101).count() << " microseconds" << std::endl;
    auto startTime1107 = std::chrono::high_resolution_clock::now();
    m11.mult2(m12, tr10);
    auto endTime1107 = std::chrono::high_resolution_clock::now();
    std::cout << "2000: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTime1107 - startTime1107).count() << " microseconds" << std::endl;
    auto startTime1111 = std::chrono::high_resolution_clock::now();
    m11.mult2(m12, tr11);
    auto endTime1111 = std::chrono::high_resolution_clock::now();
    std::cout << "3000: " << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(endTime1111 - startTime1111).count() << " microseconds" << std::endl;
    
    
    return 0;
}
