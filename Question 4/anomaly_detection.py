import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import euclidean_distances

def create_rand_array() -> None:
    normal = np.random.randint(low=0, high=10, size=(100, 1000))
    abnormal = np.random.randint(low=5, high=15, size=(100, 1000))

    normal.astype(dtype='int16').tofile(file='normal.bin')
    abnormal.astype(dtype='int16').tofile(file='abnormal.bin')

create_rand_array()

def load_arrays():# -> tuple[np.ndarray, np.ndarray]:
    normal = np.fromfile(file='normal.bin', dtype='int16').reshape((100, 1000))
    abnormal = np.fromfile(file='abnormal.bin', dtype='int16').reshape((100, 1000))
    return normal, abnormal

# Load 'normal' and 'abnormal' from the binary files
normal, abnormal = load_arrays()

def get_train_test_set(normal, abnormal):
    # Split the normal data into training and test sets
    training, normal_test = train_test_split(normal, test_size=0.1, random_state=1)

    # Split the abnormal data into training and test sets
    _, abnormal_test = train_test_split(abnormal, test_size=0.1, random_state=1)
    # abnormal_test = abnormal[:10]  # Selecting 10% of abnormal data
    
    # Concatenate 10% of abnormal_test with 10% of normal_test data
    test = np.concatenate([normal_test, abnormal_test], axis=0)

    # Return the training and test sets
    return training, test

training, test = get_train_test_set(normal=normal, abnormal=abnormal)

# print("Training set shape:", training.shape)
# print("Test set shape:", test.shape)


"""
For the first element (row) in the “training” set calculate the euclidean
distance of that element to all the rest of the elements in that set (each
other row). 
Retrieve the top five distances and sum them. This is the
dissimilarity score of that element to rest of the training set. Repeat the
process for all other elements. The scores from this process should be
stored in a vector named “baseline”.
"""

def get_training_dissimilarity_scores(training):
    # Initialize an empty array to store dissimilarity scores
    baseline = np.zeros(len(training))

    # Calculate Euclidean distances for each row in the training set
    for i, row in enumerate(training):
        # Calculate Euclidean distances to all other rows
        distances = euclidean_distances(row.reshape(1, -1), training).flatten()
        
        # Exclude distance to itself
        distances = np.delete(distances, i)
        
        # Get the top five distances and sum them
        top_five_distances_sum = np.sum(np.sort(distances)[:5])
        
        # Store the sum in the baseline array
        baseline[i] = top_five_distances_sum
    
    return baseline

baseline = get_training_dissimilarity_scores(training=training)
print("Baseline dissimilarity scores:", baseline)



"""
For the first element of the “test” set, calculate the euclidean distance of
that element to all elements contained in the “training” set. Retrieve the
top 5 distances and sum them. That is the dissimilarity score for that
element. If the score is between the min-max values of the “baseline” flag
that element as normal else flag it as abnormal. Repeat the process for
each element in the test set and print the algorithm’s predictions.
"""
def predict_test_set_anomaly(training, test, baseline):
    # Calculate Euclidean distances between test set and training set
    dissimilarity_scores = []

    for test_sample in test:
        distances = euclidean_distances(test_sample.reshape(1, -1), training).flatten()
        top_five_distances_sum = np.sum(np.sort(distances)[:5])
        dissimilarity_scores.append(top_five_distances_sum)

    # Convert dissimilarity_scores list to numpy array
    dissimilarity_scores = np.array(dissimilarity_scores)

    # Determine the range of dissimilarity scores in the baseline
    min_baseline = np.min(baseline)
    max_baseline = np.max(baseline)

    # Flag elements in the test set as normal or abnormal
    predictions = []

    for score in dissimilarity_scores:
        if min_baseline <= score <= max_baseline:
            predictions.append("Normal")
        else:
            predictions.append("Abnormal")

    return predictions


predictions = predict_test_set_anomaly(training=training, test=test, baseline=baseline)

# Print the algorithm's predictions
print("Predictions:")
for i, prediction in enumerate(predictions):
    print(f"Element {i + 1}: {prediction}")