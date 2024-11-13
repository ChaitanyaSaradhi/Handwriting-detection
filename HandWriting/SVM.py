import numpy as np
from cvxopt import matrix, solvers

class SupportVectorMachine:
    def __init__(self, regularization_parameter, kernel_function, tolerance_level, max_iterations):
        self.regularization_parameter = regularization_parameter
        self.kernel_function = kernel_function
        self.tolerance_level = tolerance_level
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_indices = None

    def fit(self, feature_data, labels):
        number_of_samples, number_of_features = feature_data.shape
        
        kernel_matrix = self.compute_kernel_matrix(feature_data)
        
        P = matrix(np.outer(labels, labels) * kernel_matrix)
        q = matrix(-np.ones(number_of_samples))
        G = matrix(np.vstack((-np.eye(number_of_samples), np.eye(number_of_samples))))
        h = matrix(np.hstack((np.zeros(number_of_samples), np.ones(number_of_samples) * self.regularization_parameter)))
        A = matrix(labels, (1, number_of_samples), 'd')
        b = matrix(0.0)

        solution = solvers.qp(P, q, G, h, A, b)
        
        lagrange_multipliers = np.ravel(solution['x'])
        
        support_vector_indices = lagrange_multipliers > self.tolerance_level
        self.support_vectors = feature_data[support_vector_indices]
        self.support_vector_labels = labels[support_vector_indices]
        self.support_vector_indices = lagrange_multipliers[support_vector_indices]
        
        self.weights = np.sum(
            self.support_vector_indices[:, None] * self.support_vector_labels[:, None] * self.support_vectors,
            axis=0
        )
        self.bias = np.mean(
            self.support_vector_labels - np.dot(self.support_vectors, self.weights)
        )

    def predict(self, feature_data):
        return np.sign(np.dot(feature_data, self.weights) + self.bias)
    
    def compute_kernel_matrix(self, feature_data):
        number_of_samples = feature_data.shape[0]
        kernel_matrix = np.zeros((number_of_samples, number_of_samples))
        for i in range(number_of_samples):
            for j in range(number_of_samples):
                kernel_matrix[i, j] = self.kernel_function(feature_data[i], feature_data[j])
        return kernel_matrix

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3):
    return (1 + np.dot(x1, x2)) ** degree

def gaussian_kernel(x1, x2, sigma=5.0):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

def generate_synthetic_data(number_of_samples_per_class, mean_class_1, mean_class_2, covariance_matrix, random_seed=None):
    np.random.seed(random_seed)
    class_1_samples = np.random.multivariate_normal(mean_class_1, covariance_matrix, number_of_samples_per_class)
    class_2_samples = np.random.multivariate_normal(mean_class_2, covariance_matrix, number_of_samples_per_class)
    feature_data = np.vstack((class_1_samples, class_2_samples))
    labels = np.hstack((np.ones(number_of_samples_per_class), -np.ones(number_of_samples_per_class)))
    return feature_data, labels

def accuracy_score(true_labels, predicted_labels):
    return np.mean(true_labels == predicted_labels)

number_of_samples_per_class = 100
mean_class_1 = [2, 2]
mean_class_2 = [4, 4]
covariance_matrix = [[1, 0.8], [0.8, 1]]
feature_data, labels = generate_synthetic_data(
    number_of_samples_per_class, mean_class_1, mean_class_2, covariance_matrix
)

svm_classifier = SupportVectorMachine(
    regularization_parameter=1.0,
    kernel_function=linear_kernel,
    tolerance_level=1e-5,
    max_iterations=1000
)

svm_classifier.fit(feature_data, labels)
predictions = svm_classifier.predict(feature_data)

