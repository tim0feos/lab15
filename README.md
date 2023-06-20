# lab15-16
# matrix_class tim0feos
## Matrices and all kinds of operations on them 
In the project released the following methods:
```
Matrix(const char* Matrix_Name) { }

Matrix(const char* Matrix_Name, int n, int m) { }

Matrix(const char* Matrix_Name, const char* File_Name) { }

void set_matrix() { }

void set_fromfile_matrix(const char* File_Name) { }

void set_infile_matrix(const char* File_Name) { }

T** get_matrix() { }

int get_size() { }

int get_row() { }

int get_col() { }

void print_matrix() { }

Matrix& operator=(const Matrix& matrix) { }

Matrix& operator+(const Matrix& matrix) { }

Matrix& operator-(const Matrix& matrix) { }

Matrix& operator*(const Matrix& matrix) { }

Matrix& operator*(int value) { }

bool operator==(const Matrix& matrix) { }

bool operator!=(const Matrix& matrix) { }

bool operator==(int value) { }

bool operator!=(int value) { }

void set_zero_matrix(int n, int m) { }

void set_identity_matrix(int n, int m) { }

void reformed_matrix(T **base_matr, T **matr, int sz, int row, int col) { }

void transpose_matrix(T **matr, int row, int col) { }

int Determinant(T **matrix, int size) { }

void get_matrix_inverse() { }

bool operator!() { }

~Matrix() { }
```
