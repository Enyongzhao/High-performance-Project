#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define ROWS 1000 // Number of rows in matrix X (adjust as needed)
#define COLS 1000 // Number of columns in matrix X (adjust as needed)
//#define PROBABILITY 0.2 // Probability that an element in X is non-zero

void generate_matrix_X(int** X, double probability) {
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			if (rand() % 100 < probability * 100) {
				X[i][j] = rand() % 10 + 1; // Random non-zero integer between 1 and 10
			}
			else {
				X[i][j] = 0; // Zero entry
			}
		}
	}
}

void generate_matrices_B_and_C(int** X, int** B, int** C) {
	for (int i = 0; i < ROWS; i++) {
		int bIndex = 0; // Index for B and C matrices

		for (int j = 0; j < COLS; j++) {
			if (X[i][j] != 0) {
				B[i][bIndex] = X[i][j];
				C[i][bIndex] = j;
				bIndex++;
			}
		}

		// If no non-zero elements were found in the row, store two consecutive 0s
		if (bIndex == 0) {
			B[i][0] = 0;
			B[i][1] = 0;
			C[i][0] = 0;
			C[i][1] = 0;
		}
		else {
			// If there are remaining spaces, fill them with 0s
			for (int k = bIndex; k < COLS; k++) {
				B[i][k] = 0;
				C[i][k] = 0;
			}
		}
	}
}

void ordinary_multiply_matrices(int** A, int** B, int** C) {

	// Perform matrix multiplication
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			for (int k = 0; k < COLS; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

void multiply_sparse_matrices(int** B_X, int** C_X, int** B_Y, int** C_Y, int** Z) {

	// Perform matrix multiplication
//#pragma omp parallel for schedule(guided)   //static, dynamic, guided, runtime
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			if (B_X[i][j] != 0) {  // If B_X[i][j] is non-zero
				int col = C_X[i][j];  // Get the column index from C_X

				int m = 0;
				int k = C_Y[col][m];
				//#pragma omp atomic
				Z[i][k] += B_X[i][j] * B_Y[col][m];
				m++;
				for (; m < COLS; m++) {
					k = C_Y[col][m];
					if (k == 0)
						break;
					//#pragma omp atomic
					Z[i][k] += B_X[i][j] * B_Y[col][m];
				}
			}
		}
	}

}

void print_matrix(int** matrix, const char* name) {
	printf("Matrix %s:\n", name);
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			printf("%d ", matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

bool are_matrices_equal(int** matrix1, int** matrix2) {
	for (int i = 0; i < ROWS; ++i) {
		for (int j = 0; j < COLS; ++j) {
			if (matrix1[i][j] != matrix2[i][j]) {
				return false; // 如果有任意一对元素不相等，则矩阵不同
			}
		}
	}
	return true; // 如果所有元素都相等，则矩阵相同
}


int main8() {
	srand(time(NULL)); // Seed the random number generator

	//int X[ROWS][COLS]; // Original matrix X
	//int Y[ROWS][COLS]; // Original matrix Y
	//int XB[ROWS][COLS] = { 0 }; // Matrix B storing non-zero elements
	//int XC[ROWS][COLS] = { 0 }; // Matrix C storing indices of non-zero elements
	//int YB[ROWS][COLS] = { 0 }; // Matrix B storing non-zero elements
	//int YC[ROWS][COLS] = { 0 }; // Matrix C storing indices of non-zero elements
	//int Z[ROWS][COLS] = { 0 };
	//int correct_Z[ROWS][COLS] = { 0 };

	// 动态分配数组
	int** X = (int**)malloc(ROWS * sizeof(int*));
	int** Y = (int**)malloc(ROWS * sizeof(int*));
	int** XB = (int**)malloc(ROWS * sizeof(int*));
	int** XC = (int**)malloc(ROWS * sizeof(int*));
	int** YB = (int**)malloc(ROWS * sizeof(int*));
	int** YC = (int**)malloc(ROWS * sizeof(int*));
	int** Z = (int**)malloc(ROWS * sizeof(int*));
	int** correct_Z = (int**)malloc(ROWS * sizeof(int*));

	// 为每一行分配内存
	for (int i = 0; i < ROWS; ++i) {
		X[i] = (int*)malloc(COLS * sizeof(int));
		Y[i] = (int*)malloc(COLS * sizeof(int));
		XB[i] = (int*)calloc(COLS, sizeof(int)); // calloc 初始化为 0
		XC[i] = (int*)calloc(COLS, sizeof(int)); // calloc 初始化为 0
		YB[i] = (int*)calloc(COLS, sizeof(int)); // calloc 初始化为 0
		YC[i] = (int*)calloc(COLS, sizeof(int)); // calloc 初始化为 0
		Z[i] = (int*)calloc(COLS, sizeof(int));  // calloc 初始化为 0
		correct_Z[i] = (int*)calloc(COLS, sizeof(int)); // calloc 初始化为 0
	}

	generate_matrix_X(X, 0.01);
	generate_matrix_X(Y, 0.01);
	generate_matrices_B_and_C(X, XB, XC);
	generate_matrices_B_and_C(Y, YB, YC);

	//ordinary_multiply_matrices(X, Y, correct_Z);
	//omp_set_num_threads(16);
	printf("OpenMP running with %d threads\n", omp_get_max_threads());

	clock_t begin = clock();

	multiply_sparse_matrices(XB, XC, YB, YC, Z);


	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("omp Time spent: %f\n", time_spent);
	// Print the matrices
	//print_matrix(X, "X");
	//print_matrix(Y, "Y");
	//print_matrix(XB, "XB");
	//print_matrix(XC, "XC");
	//print_matrix(YB, "YB");
	//print_matrix(YC, "YC");

	// Print the result
	//print_matrix(correct_Z, "correct_Z");
	//print_matrix(Z, "Z");

	//if (are_matrices_equal(correct_Z, Z)) {
	//    printf("The matrices are equal.\n");
	//}
	//else {
	//    printf("The matrices are not equal.\n");
	//}

	// 使用完毕后释放内存
	for (int i = 0; i < ROWS; ++i) {
		free(X[i]);
		free(Y[i]);
		free(XB[i]);
		free(XC[i]);
		free(YB[i]);
		free(YC[i]);
		free(Z[i]);
		free(correct_Z[i]);
	}

	free(X);
	free(Y);
	free(XB);
	free(XC);
	free(YB);
	free(YC);
	free(Z);
	free(correct_Z);

	return 0;
}
