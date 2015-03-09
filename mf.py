import sys;
import numpy;
import math;

class MatrixFactorization():
	def __init__(self, userCount, itemCount, latentFactorCount):
		self.learningRate = 0.005;
		self.userRegularizationRate = 0.01;
		self.itemRegularizationRate = 0.01;
		self.maxSGDIterationCount = 5000;
		self.learningRateEncouragingRatio = 1.05;
		self.learningRateDiscouragingRatio = 0.5;
		self.unitConvergenceThreshold = 1e-5;
		self.userCount = userCount;							# N
		self.itemCount = itemCount;							# M
		self.latentFactorCount = latentFactorCount;			# K

		self.resetLearnedParameters();

	def resetLearnedParameters(self):
		self.userMatrix = numpy.random.rand(self.userCount, self.latentFactorCount);	# N * K matrix, each entry is set a random value
		self.itemMatrix = numpy.random.rand(self.itemCount, self.latentFactorCount);	# M * K matrix

	def runSGDStep(self, ratingList, userMatrix, itemMatrix):
		for (user, item, trueRating) in ratingList:
			userVector = userMatrix[user, :].copy();			# vectors should be copied to make the following update formula correct
			itemVector = itemMatrix[item, :].copy();

			predictedRating = userVector.dot(itemVector);
			ratingError = trueRating - predictedRating;

			userMatrix[user, :] += self.learningRate * (ratingError * itemVector - self.userRegularizationRate * userVector);	# update formula
			itemMatrix[item, :] += self.learningRate * (ratingError * userVector - self.itemRegularizationRate * itemVector);

	def calculateCost(self, ratingList, userMatrix, itemMatrix):
		totalRatingCost = 0.0;
		for (user, item, trueRating) in ratingList:
			predictedRating = userMatrix[user, :].dot(itemMatrix[item, :]);
			ratingError = trueRating - predictedRating;
			totalRatingCost += ratingError * ratingError;

		userRegularizationCost = numpy.linalg.norm(userMatrix) ** 2;
		itemRegularizationCost = numpy.linalg.norm(itemMatrix) ** 2;

		return 0.5 * (totalRatingCost + self.userRegularizationRate * userRegularizationCost + self.itemRegularizationRate * itemRegularizationCost);

	def learn(self, ratingList):
		ratingCount = len(ratingList);
		convergenceThreshold = self.unitConvergenceThreshold * ratingCount;
		lastTotalCost = sys.float_info.max;
		successfulLearning = False;
		newUserMatrix = None;
		newItemMatrix = None;

		print("{0:d} training rating examples".format(ratingCount));

		for iteration in range(self.maxSGDIterationCount):
			if successfulLearning == False:
				newUserMatrix = self.userMatrix.copy();
				newItemMatrix = self.itemMatrix.copy();

			self.runSGDStep(ratingList, newUserMatrix, newItemMatrix);

			totalCost = self.calculateCost(ratingList, newUserMatrix, newItemMatrix);

			print("Iteration {0:d}\tCost {1:f}\tLearningRate {2:f}\tCostDescent {3:f}\tConvergenceThreshold {4:f}".format(iteration + 1, totalCost, \
					self.learningRate, lastTotalCost - totalCost if iteration > 0 else 0.0, convergenceThreshold));

			if lastTotalCost >= totalCost:
				self.userMatrix = newUserMatrix.copy();
				self.itemMatrix = newItemMatrix.copy();

				if iteration > 0 and lastTotalCost - totalCost <= convergenceThreshold:
					break;

				self.learningRate *= self.learningRateEncouragingRatio;
				successfulLearning = True;
			else:
				self.learningRate *= self.learningRateDiscouragingRatio;
				successfulLearning = False;

			lastTotalCost = totalCost;

	def predict(self, user, item):
		return self.userMatrix[user, : ].dot(self.itemMatrix[item, : ]);

	def evaluate(self, ratingList, type):
		if type == 1:
			return self.evaluateRMSE(ratingList);
		elif type == 2:
			return self.evaluateMAE(ratingList);

	def evaluateRMSE(self, ratingList):
		rmse = 0.0;

		for (user, item, trueRating) in ratingList:
			predictedRating = self.predict(user, item);
			ratingError = trueRating - predictedRating;
			rmse += ratingError * ratingError;

		rmse = math.sqrt(rmse / len(ratingList));
		return rmse;

	def evaluateMAE(self, ratingList):
		mae = 0.0;

		for (user, item, trueRating) in ratingList:
			predictedRating = self.predict(user, item);
			ratingError = trueRating - predictedRating;
			mae += abs(ratingError);

		mae /= len(ratingList);
		return mae;
#==============================================================================
class CrossValidation:
	def __init__(self, foldCount, evaluationTypes):
		self.foldCount = foldCount;
		self.evaluationTypes = evaluationTypes;

	def splitRatingList(self, ratingList, groupMarkers, validationGroup):
		trainingList = list();
		validationList = list();

		for (i, element) in enumerate(ratingList):
			if groupMarkers[i] == validationGroup:
				validationList.append(element);
			else:
				trainingList.append(element);

		return (trainingList, validationList);

	def run(self, mfModel, ratingList):
		groupMarkers = numpy.random.randint(self.foldCount, size = len(ratingList));
		evaluationTypeCount = len(self.evaluationTypes);
		performanceMeans = [0.0 for e in range(evaluationTypeCount)];

		for validedFold in range(self.foldCount):
			(trainingList, validationList) = self.splitRatingList(ratingList, groupMarkers, validedFold);

			mfModel.resetLearnedParameters();
			mfModel.learn(trainingList);

			print("Cross validation {0:d}".format(validedFold + 1));
			for e in range(evaluationTypeCount):
				performance = mfModel.evaluate(validationList, self.evaluationTypes[e]);
				print("\tPerformance {0:d} {1:f}".format(e + 1, performance));
				performanceMeans[e] += performance;

		for e in range(evaluationTypeCount):
			performanceMeans[e] /= self.foldCount;
			print("Average performance {0:d} {1:f}".format(e + 1, performanceMeans[e]));

def ratingReadFromFile(filePath):
	ratingList = list();
	maxUser = 0;
	maxItem = 0;

	with open(filePath, "r") as inFile:
		for line in inFile:
			row = line.strip().split();

			user = int(row[0]);
			item = int(row[1]);
			rating = float(row[2]);

			ratingList.append((user, item, rating));
			if maxUser < user:
				maxUser = user;
			if maxItem < item:
				maxItem = item;

	userCount = maxUser + 1;
	itemCount = maxItem + 1;
	return (ratingList, userCount, itemCount);

def main():
	ratingFilePath = sys.argv[1];

	ratingDict = dict();
	userCount = 0;
	itemCount = 0;

	print("Read data");
	(ratingList, userCount, itemCount) = ratingReadFromFile(ratingFilePath);

	print("{0:d} users, {1:d} items".format(userCount, itemCount));
	mf = MatrixFactorization(userCount, itemCount, 10);

	cv = CrossValidation(5, [1, 2]);
	cv.run(mf, ratingList);
	print("OK");

if __name__ == "__main__":
	main();
