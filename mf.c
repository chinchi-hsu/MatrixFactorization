#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>

typedef struct DICT{
	int key;
	double value;
} Dict;

int dictCompareAscendingKeys(const void *a, const void *b){
	Dict p = *(Dict*)a;
	Dict q = *(Dict*)b;

	if(p.key < q.key){
		return -1;
	}
	if(p.key > q.key){
		return 1;
	}
	return 0;
}
//==============================================================================================
double vectorCalculateDotProduct(double *vector1, double *vector2, int length){
	double dotProduct = 0.0;
	for(int d = 0; d < length; d ++){
		dotProduct += vector1[d] * vector2[d];
	}
	return dotProduct;
}
//==============================================================================================
typedef struct LIST{
	int rowCount;
	int *columnCounts;
	Dict **entries;
} List;

void listInitialize(List *list, int rowCount){
	list -> rowCount = rowCount;
	list -> columnCounts = (int*)malloc(sizeof(int) * rowCount);
	list -> entries = (Dict**)malloc(sizeof(Dict*) * rowCount);
	for(int row = 0; row < rowCount; row ++){
		list -> columnCounts[row] = 0;
		list -> entries[row] = (Dict*)malloc(sizeof(Dict));
	}
}

void listReleaseSpace(List *list){
	for(int row = 0; row < list -> rowCount; row ++){
		free(list -> entries[row]);
	}
	free(list -> entries);
	free(list -> columnCounts);
	list -> entries = NULL;
	list -> columnCounts = NULL;
}

void listSortRows(List *list){
	for(int row = 0; row < list -> rowCount; row ++){
		qsort(list -> entries[row], list -> columnCounts[row], sizeof(Dict), dictCompareAscendingKeys);
	}
}

int listCountEntries(List *list){
	int entryCount = 0;
	for(int row = 0; row < list -> rowCount; row ++){
		entryCount += list -> columnCounts[row];
	}
	return entryCount;
}
//==============================================================================================
typedef struct MATRIX{
	int rowCount;
	int columnCount;
	double **entries;
} Matrix;

void matrixInitialize(Matrix *matrix, int rowCount, int columnCount){
	matrix -> rowCount = rowCount;
	matrix -> columnCount = columnCount;
	matrix -> entries = (double**)malloc(sizeof(double*) * rowCount);
	for(int row = 0; row < rowCount; row ++){
		matrix -> entries[row] = (double*)malloc(sizeof(double) * columnCount);
	}	
}

void matrixReleaseSpace(Matrix *matrix){
	for(int row = 0; row < matrix -> rowCount; row ++){
		free(matrix -> entries[row]);
	}
	free(matrix -> entries);
	matrix -> entries = NULL;
}

void matrixAssignRandomValues(Matrix *matrix, double minValue, double maxValue){
	for(int row = 0; row < matrix -> rowCount; row ++){
		for(int column = 0; column < matrix -> columnCount; column ++){
			double uniform = (double)rand() / RAND_MAX;
			matrix -> entries[row][column] = minValue + (maxValue - minValue) * uniform;
		}
	}
}

// Source and target should contain the same rows and columns
void matrixCopyEntries(Matrix *source, Matrix *target){
	for(int row = 0; row < source -> rowCount; row ++){
		for(int column = 0; column < source -> columnCount; column ++){
			target -> entries[row][column] = source -> entries[row][column];
		}
	}
}
//==============================================================================================
void ratingFetchUserItemCount(char *ratingFilePath, int *userCount, int *itemCount){
	FILE *inFile = fopen(ratingFilePath, "r");
	char line[10000];
	int lastUser = 0;
	int lastItem = 0;

	while(fgets(line, 10000, inFile)){
		int user, item;
		sscanf(line, "%d%d", &user, &item);
		if(lastUser < user){
			lastUser = user;
		}
		if(lastItem < item){
			lastItem = item;
		}
	}

	fclose(inFile);
	*userCount = lastUser + 1;
	*itemCount = lastItem + 1;
}

// User index [0, userCount - 1], item index [0, itemCount - 1];
void ratingReadFromFile(char *ratingFilePath, List *ratings){
	FILE *inFile = fopen(ratingFilePath, "r");
	char line[10000];

	// Assume no repeated (user, item) pairs
	while(fgets(line, 10000, inFile)){
		int user, item;
		double rating;
		sscanf(line, "%d%d%lf", &user, &item, &rating);
	
		int itemCount = ratings -> columnCounts[user];
		ratings -> entries[user] = (Dict*)realloc(ratings -> entries[user], sizeof(Dict) * (itemCount + 1));
		ratings -> entries[user][itemCount].key = item;
		ratings -> entries[user][itemCount].value = rating;
		ratings -> columnCounts[user] += 1;
	}

	fclose(inFile);
}
//==============================================================================================
typedef struct MATRIX_FACTORIZATION{
	int latentFactorCount;				// K
	double learningRate;				// alpha
	double userRegularizationRate;		// lambda_1
	double itemRegularizationRate;		// lambda_2
	double unitConvergenceThreshold;
	double learningRateEncouragingRatio;
	double learningRateDiscouragingRatio;
	int maxSGDIterationCount;
	Matrix *userMatrix;
	Matrix *itemMatrix;
} MatrixFactorization;

void matrixFactorizationRunSGDStep(MatrixFactorization *model, List *ratings, Matrix *userMatrix, Matrix *itemMatrix){
	for(int user = 0; user < ratings -> rowCount; user ++){
		for(int j = 0; j < ratings -> columnCounts[user]; j ++){
			int item = ratings -> entries[user][j].key;
			double trueRating = ratings -> entries[user][j].value;

			double predictedRating = vectorCalculateDotProduct(userMatrix -> entries[user], itemMatrix -> entries[item], model -> latentFactorCount);
			double ratingError = trueRating - predictedRating;					// e_ij
			
			for(int k = 0; k < model -> latentFactorCount; k ++){
				double userFactor = userMatrix -> entries[user][k];
				double itemFactor = itemMatrix -> entries[item][k];
				userMatrix -> entries[user][k] += model -> learningRate * (ratingError * itemFactor - model -> userRegularizationRate * userFactor);
				itemMatrix -> entries[item][k] += model -> learningRate * (ratingError * userFactor - model -> itemRegularizationRate * itemFactor);
			}
		}
	}
}

double matrixFactorizationCalculateCost(MatrixFactorization *model, List *ratings, Matrix *userMatrix, Matrix *itemMatrix){
		double totalRatingCost = 0.0;
		for(int user = 0; user < ratings -> rowCount; user ++){
			for(int j = 0; j < ratings -> columnCounts[user]; j ++){
				int item = ratings -> entries[user][j].key;
				double trueRating = ratings -> entries[user][j].value;
				double predictedRating = vectorCalculateDotProduct(userMatrix -> entries[user], itemMatrix -> entries[item], model -> latentFactorCount);
				totalRatingCost += (trueRating - predictedRating) * (trueRating - predictedRating);
			}
		}
		double userRegularizationCost = 0.0;
		for(int user = 0; user < userMatrix -> columnCount; user ++){
			userRegularizationCost += vectorCalculateDotProduct(userMatrix -> entries[user], userMatrix -> entries[user], userMatrix -> columnCount);
		}
		double itemRegularizationCost = 0.0;
		for(int item = 0; item < itemMatrix -> columnCount; item ++){
			itemRegularizationCost += vectorCalculateDotProduct(itemMatrix -> entries[item], itemMatrix -> entries[item], itemMatrix -> columnCount);
		}
		return 0.5 * (totalRatingCost + model -> userRegularizationRate * userRegularizationCost + model -> itemRegularizationRate * itemRegularizationCost);
}

void matrixFactorizationLearn(MatrixFactorization *model, List *ratings){
	double lastTotalCost = DBL_MAX;
	double totalCost = 0.0;
	int totalRatingCount = listCountEntries(ratings);
	double convergenceThreshold = model -> unitConvergenceThreshold * totalRatingCount;
	Matrix *newUserMatrix = (Matrix*)malloc(sizeof(Matrix));
	Matrix *newItemMatrix = (Matrix*)malloc(sizeof(Matrix));
	matrixInitialize(newUserMatrix, model -> userMatrix -> rowCount, model -> userMatrix -> columnCount);
	matrixInitialize(newItemMatrix, model -> itemMatrix -> rowCount, model -> itemMatrix -> columnCount);

	printf("Total %d ratings\n", totalRatingCount);

	for(int iteration = 0; iteration < model -> maxSGDIterationCount; iteration ++){
		matrixCopyEntries(model -> userMatrix, newUserMatrix);
		matrixCopyEntries(model -> itemMatrix, newItemMatrix);
		matrixFactorizationRunSGDStep(model, ratings, newUserMatrix, newItemMatrix);

		totalCost = matrixFactorizationCalculateCost(model, ratings, newUserMatrix, newItemMatrix);
		printf("Iteration %d\tCost %f\tLearningRate %f\tCostDescent %f\tConvergenceThreshold %f\n", iteration + 1, totalCost, model -> learningRate, (lastTotalCost < DBL_MAX) ? lastTotalCost - totalCost : 0.0, convergenceThreshold);	
		
		if(lastTotalCost >= totalCost){				// if this gradient descend does reduce the overall cost
			matrixCopyEntries(newUserMatrix, model -> userMatrix);
			matrixCopyEntries(newItemMatrix, model -> itemMatrix);
			
			if(iteration >= 1 && lastTotalCost - totalCost < convergenceThreshold){
				break;
			}
			model -> learningRate *= model -> learningRateEncouragingRatio;		// raises the learning rate to save learning time
		}
		else{
			model -> learningRate *= model -> learningRateDiscouragingRatio;	// reduces the learning rate to learn more precisely
		
		}
		
		lastTotalCost = totalCost;
	}
	
	matrixReleaseSpace(newUserMatrix);
	matrixReleaseSpace(newItemMatrix);
	free(newUserMatrix);
	free(newItemMatrix);
}

double matrixFactorizationPredict(MatrixFactorization *model, int user, int item){
	double prediction = vectorCalculateDotProduct(model -> userMatrix -> entries[user], model -> itemMatrix -> entries[item], model -> latentFactorCount);
	return prediction;
}


double matrixFactorizationEvaluateRMSE(MatrixFactorization *model, List *ratings){
	double rmse = 0.0;
	int totalCount = 0;

	for(int user = 0; user < ratings -> rowCount; user ++){
		for(int j = 0; j < ratings -> columnCounts[user]; j ++){
			int item = ratings -> entries[user][j].key;
			double trueRating = ratings -> entries[user][j].value;

			double predictedRating = matrixFactorizationPredict(model, user, item);
			double ratingError = trueRating - predictedRating;
			
			rmse += ratingError * ratingError;
			totalCount += 1;
		}
	}

	rmse = sqrt(rmse / totalCount);
	return rmse;
}

double matrixFactorizationEvaluateMAE(MatrixFactorization *model, List *ratings){
	double mae = 0.0;
	int totalCount = 0;
	
	for(int user = 0; user < ratings -> rowCount; user ++){
		for(int j = 0; j < ratings -> columnCounts[user]; j ++){
			int item = ratings -> entries[user][j].key;
			double trueRating = ratings -> entries[user][j].value;

			double predictedRating = matrixFactorizationPredict(model, user, item);
			double ratingError = trueRating - predictedRating;
			
			mae += fabs(ratingError);
			totalCount += 1;
		}
	}

	mae /= totalCount;
	return mae;
}

double matrixFactorizationEvaluate(MatrixFactorization *model, List *ratings, int type){
	switch(type){
		case 1:
			return matrixFactorizationEvaluateRMSE(model, ratings);
		case 2:
			return matrixFactorizationEvaluateMAE(model, ratings);
	}
}
//==============================================================================================
typedef struct CROSS_VALIDATION{
	int foldCount;
	int evaluationTypeCount;
	int* evaluationTypes;
} CrossValidation;

// Group index [0, groupCount - 1]
int crossValidationDetermineGroup(int groupCount){
	double uniform;
	do{
		uniform = (double)rand() / RAND_MAX;
	}while(uniform == 1.0);
	return (int)(uniform * groupCount);
}

void crossValidationGroupRatings(CrossValidation *validation, List *ratings, List *groupMarkers){
	for(int user = 0; user < ratings -> rowCount; user ++){
		groupMarkers -> entries[user] = (Dict*)realloc(groupMarkers -> entries[user], sizeof(Dict) * ratings -> columnCounts[user]);
		groupMarkers -> columnCounts[user] = ratings -> columnCounts[user];

		for(int j = 0; j < ratings -> columnCounts[user]; j ++){
			int item = ratings -> entries[user][j].key;
			groupMarkers -> entries[user][j].key = item;
			groupMarkers -> entries[user][j].value = crossValidationDetermineGroup(validation -> foldCount);
		}
	}
}

void crossValidationSplitRatings(List *ratings, List *groupMarkers, List *trainingRatings, List *validationRatings, int validationGroup){
	for(int user = 0; user < ratings -> rowCount; user ++){
		for(int j = 0; j < ratings -> columnCounts[user]; j ++){
			int item = ratings -> entries[user][j].key;
			double rating = ratings -> entries[user][j].value;
			int group = groupMarkers -> entries[user][j].value;

			if(group == validationGroup){
				int itemCount = validationRatings -> columnCounts[user];
				validationRatings -> entries[user] = (Dict*)realloc(validationRatings -> entries[user], sizeof(Dict) * (itemCount + 1));
				validationRatings -> entries[user][itemCount].key = item;
				validationRatings -> entries[user][itemCount].value = rating;
				validationRatings -> columnCounts[user] += 1;
			}
			else{
				int itemCount = trainingRatings -> columnCounts[user];
				trainingRatings -> entries[user] = (Dict*)realloc(trainingRatings -> entries[user], sizeof(Dict) * (itemCount + 1));
				trainingRatings -> entries[user][itemCount].key = item;
				trainingRatings -> entries[user][itemCount].value = rating;
				trainingRatings -> columnCounts[user] += 1;	
			}
		}
	}
}

void crossValidationRun(CrossValidation *validation, MatrixFactorization *model, List *ratings){
	List groupMarkers;
	listInitialize(&groupMarkers, ratings -> rowCount);
	crossValidationGroupRatings(validation, ratings, &groupMarkers);
	double* performanceMeans = (double*)malloc(sizeof(double) * validation -> evaluationTypeCount);

	for(int validedFold = 0; validedFold < validation -> foldCount; validedFold ++){
		List trainingRatings, validationRatings;
		listInitialize(&trainingRatings, ratings -> rowCount);
		listInitialize(&validationRatings, ratings -> rowCount);
	
		crossValidationSplitRatings(ratings, &groupMarkers, &trainingRatings, &validationRatings, validedFold);

		matrixAssignRandomValues(model -> userMatrix, 0, 1);
		matrixAssignRandomValues(model -> itemMatrix, 0, 1);
		matrixFactorizationLearn(model, &trainingRatings);

		printf("Cross validation %d\n", validedFold + 1);
		for(int e = 0; e < validation -> evaluationTypeCount; e ++){	
			double performance = matrixFactorizationEvaluate(model, &validationRatings, validation -> evaluationTypes[e]);
			performanceMeans[e] += performance;
			printf("\tPerformance %d %f\n", validedFold + 1, validation -> evaluationTypes[e], performance);
		}

		listReleaseSpace(&trainingRatings);
		listReleaseSpace(&validationRatings);
	}

	for(int e = 0; e < validation -> evaluationTypeCount; e ++){	
		performanceMeans[e] /= validation -> foldCount;
		printf("Average performance %d %f\n", validation -> evaluationTypes[e], performanceMeans[e]);
	}

	listReleaseSpace(&groupMarkers);
	free(performanceMeans);
}
//==============================================================================================
int main(int argc, char *argv[]){
	srand(time(NULL));
	
	char *ratingFilePath = argv[1];

	// Read rating data from some file
	int userCount, itemCount;
	ratingFetchUserItemCount(ratingFilePath, &userCount, &itemCount);
	
	List ratings;
	listInitialize(&ratings, userCount);
	ratingReadFromFile(ratingFilePath, &ratings);
	listSortRows(&ratings);
	printf("%d users, %d items\n", userCount, itemCount);

	// Set matrix factorization
	MatrixFactorization mf = {
		.latentFactorCount = 10,
		.learningRate = 0.005,
		.userRegularizationRate = 0.01,
		.itemRegularizationRate = 0.01,
		.unitConvergenceThreshold = 1e-5,
		.maxSGDIterationCount = 5000,
		.learningRateEncouragingRatio = 1.05,
		.learningRateDiscouragingRatio = 0.5,
	};
	Matrix userMatrix;
	Matrix itemMatrix;
	matrixInitialize(&userMatrix, userCount, mf.latentFactorCount);
	matrixInitialize(&itemMatrix, itemCount, mf.latentFactorCount);
	printf("User matrix %d %d\n", userMatrix.rowCount, userMatrix.columnCount);
	printf("Item matrix %d %d\n", itemMatrix.rowCount, itemMatrix.columnCount);
	matrixAssignRandomValues(&userMatrix, 0, 1);
	matrixAssignRandomValues(&itemMatrix, 0, 1);
	mf.userMatrix = &userMatrix;
	mf.itemMatrix = &itemMatrix;	

	// Run cross validation	
	CrossValidation cv = {
		.foldCount = 5,
		.evaluationTypeCount = 2
	};
	cv.evaluationTypes = (int*)malloc(sizeof(int) * cv.evaluationTypeCount);
	cv.evaluationTypes[0] = 1;
	cv.evaluationTypes[1] = 2;
	crossValidationRun(&cv, &mf, &ratings);

	// Release space
	listReleaseSpace(&ratings);
	matrixReleaseSpace(&userMatrix);
	matrixReleaseSpace(&itemMatrix);
	free(cv.evaluationTypes);

	printf("OK\n");
	return 0;
}
