import csv
import random

def labelToNum(s):
	if s == 'N':
		return 0.0
	elif s == 'O':
		return 1.0
	elif s == 'A':
		return 2.0

def main():
	N = 8528 # Number of data points

	M = 1500 # Input size

	# Get the data from the file
	inputData = []
	inputRef = []
	perm = list(range(N))

	for i in range(N):
		name = str(i + 1)
		path = "D:/ECG/ECG/Data/CSV/A" + ("0" * (5 - len(name))) + name + ".csv"
		with open(path, 'r', newline='') as csvfile:
			csvreader = csv.reader(csvfile)
			for row in csvreader:
				inputData.append([float(v) / 1000.0 for v in row])
	
	with open("D:/ECG/ECG/Data/REFERENCE.txt", "r", newline='') as file:
		for line in file:
			inputRef.append(line.split()[1])

	random.seed(74)
	random.shuffle(inputData)
	random.seed(74)
	random.shuffle(inputRef)
	random.seed(74)
	random.shuffle(perm)
	
	# process the data into a format planned for the models
	trainData, trainRef = [], []
	validData, validRef = [], []
	testData, testMap, testRef = [], [], []

	# about 80-10-10 split

	# training data
	for i in range(int(N * 0.8)):
		if inputRef[i] == '~':
			continue

		shift = 1000
		if inputRef[i] == 'O':
			shift = 750
		elif inputRef[i] == 'A':
			shift = 625

		pos = 0
		while pos + M <= len(inputData[i]):
			trainData.append(inputData[i][pos:pos + M])
			trainRef.append(labelToNum(inputRef[i]))
			pos += shift

	# validation data
	for i in range(int(N * 0.8), int(N * 0.9)):
		if inputRef[i] == '~':
			continue

		shift = 1000
		if inputRef[i] == 'O':
			shift = 750
		elif inputRef[i] == 'A':
			shift = 625

		pos = 0
		while pos + M <= len(inputData[i]):
			validData.append(inputData[i][pos:pos + M])
			validRef.append(labelToNum(inputRef[i]))
			pos += shift

	# test data
	for i in range(int(N * 0.9), N):
		if inputRef[i] == '~':
			continue

		shift = 750
		testRef.append((perm[i], labelToNum(inputRef[i])))

		pos = 0
		while pos + M <= len(inputData[i]):
			testData.append(inputData[i][pos:pos + M])
			testMap.append(perm[i])
			pos += shift
	
	# Save the data into a new file
	with open('../../data/trainData.csv', 'w', newline='') as file:
		csvwriter = csv.writer(file)
		for row in trainData:
			csvwriter.writerow(row)
	
	with open('../../data/trainRef.csv', 'w', newline='') as file:
		csvwriter = csv.writer(file)
		csvwriter.writerow(trainRef)

	with open('../../data/validData.csv', 'w', newline='') as file:
		csvwriter = csv.writer(file)
		for row in validData:
			csvwriter.writerow(row)
	
	with open('../../data/validRef.csv', 'w', newline='') as file:
		csvwriter = csv.writer(file)
		csvwriter.writerow(validRef)

	with open('../../data/testData.csv', 'w', newline='') as file:
		csvwriter = csv.writer(file)
		for row in testData:
			csvwriter.writerow(row)
	
	with open('../../data/testMap.csv', 'w', newline='') as file:
		csvwriter = csv.writer(file)
		csvwriter.writerow(testMap)

	with open('../../data/testRef.csv', 'w', newline='') as file:
		csvwriter = csv.writer(file)
		for _tuple in testRef:
			csvwriter.writerow(list(_tuple))

if __name__ == "__main__":
	# main()
	# call this function only once, when dataset is created,
	# and then use the same dataset with the same input to train all the models
	pass
