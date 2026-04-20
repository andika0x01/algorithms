import numpy as np


def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def initialize_parameters(n_x, n_h, seed=1):
	np.random.seed(seed)
	W1 = np.random.randn(n_h, n_x) * 0.01
	b1 = np.zeros((n_h, 1))
	W2 = np.random.randn(1, n_h) * 0.01
	b2 = np.zeros((1, 1))
	return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward_propagation(X, parameters):
	W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
	Z1 = W1.dot(X.T) + b1
	A1 = np.tanh(Z1)
	Z2 = W2.dot(A1) + b2
	A2 = sigmoid(Z2)
	cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
	return A2, cache


def compute_loss(A2, Y):
	m = Y.shape[1]
	A2 = np.clip(A2, 1e-12, 1 - 1e-12)
	loss = -1.0 / m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
	return np.squeeze(loss)


def backward_propagation(parameters, cache, X, Y):
	m = X.shape[0]
	W2 = parameters['W2']
	A1, A2 = cache['A1'], cache['A2']
	dZ2 = A2 - Y
	dW2 = 1.0 / m * dZ2.dot(A1.T)
	db2 = 1.0 / m * np.sum(dZ2, axis=1, keepdims=True)
	dZ1 = W2.T.dot(dZ2) * (1 - np.power(A1, 2))
	dW1 = 1.0 / m * dZ1.dot(X)
	db1 = 1.0 / m * np.sum(dZ1, axis=1, keepdims=True)
	grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
	return grads


def update_parameters(parameters, grads, lr):
	parameters['W1'] -= lr * grads['dW1']
	parameters['b1'] -= lr * grads['db1']
	parameters['W2'] -= lr * grads['dW2']
	parameters['b2'] -= lr * grads['db2']
	return parameters


def predict(parameters, X):
	A2, _ = forward_propagation(X, parameters)
	preds = (A2 > 0.5).astype(int)
	return preds.ravel()


def generate_house_dataset(n_samples=1000, seed=1):
	np.random.seed(seed)
	size = np.random.normal(loc=1500, scale=500, size=n_samples)
	bedrooms = np.random.randint(1, 6, size=n_samples)
	age = np.random.randint(0, 50, size=n_samples)
	distance = np.random.exponential(scale=5, size=n_samples)
	price = size * 200 + bedrooms * 10000 - age * 300 + (30 - distance) * 5000
	price += np.random.normal(0, 20000, size=n_samples)
	threshold = np.median(price)
	labels = (price > threshold).astype(int)
	X = np.vstack([size, bedrooms, age, distance]).T
	return X, labels


def train_model(X_train, y_train, n_h=8, epochs=2000, lr=0.1, print_every=200, seed=1):
	n_x = X_train.shape[1]
	parameters = initialize_parameters(n_x, n_h, seed)
	Y = y_train.reshape(1, -1)
	for epoch in range(1, epochs + 1):
		A2, cache = forward_propagation(X_train, parameters)
		loss = compute_loss(A2, Y)
		grads = backward_propagation(parameters, cache, X_train, Y)
		parameters = update_parameters(parameters, grads, lr)
		if epoch % print_every == 0 or epoch == 1:
			print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")
	return parameters


def accuracy(y_pred, y_true):
	return float(np.mean(y_pred == y_true)) * 100


def main():
	X, y = generate_house_dataset(n_samples=1000, seed=2)
	# shuffle and split
	idx = np.random.permutation(X.shape[0])
	X, y = X[idx], y[idx]
	split = int(0.8 * X.shape[0])
	X_train, X_test = X[:split], X[split:]
	y_train, y_test = y[:split], y[split:]
	# standardize using training stats
	mu = X_train.mean(axis=0)
	sigma = X_train.std(axis=0) + 1e-8
	X_train = (X_train - mu) / sigma
	X_test = (X_test - mu) / sigma

	print("Training model..")
	params = train_model(X_train, y_train, n_h=10, epochs=2000, lr=0.5, print_every=400, seed=3)

	y_pred_train = predict(params, X_train)
	y_pred_test = predict(params, X_test)

	print(f"Akurasi train: {accuracy(y_pred_train, y_train):.2f}%")
	print(f"Akurasi test: {accuracy(y_pred_test, y_test):.2f}%")

	print("Contoh prediksi (actual -> predicted):")
	for i in range(10):
		actual = 'Mahal' if y_test[i] == 1 else 'Murah'
		pred = 'Mahal' if y_pred_test[i] == 1 else 'Murah'
		print(f"  {i+1}. {actual} -> {pred}")


if __name__ == '__main__':
	main()


