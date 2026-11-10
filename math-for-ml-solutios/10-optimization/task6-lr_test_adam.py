import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data.astype(np.float32) / 255.0, mnist.target.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_dim, hidden_dim, output_dim = 784, 128, 10

def loss_and_grad(theta, X_batch, y_batch):
    bs = len(y_batch)
    w1 = theta[:input_dim*hidden_dim].reshape(input_dim, hidden_dim)
    b_1 = theta[input_dim*hidden_dim:input_dim*hidden_dim+hidden_dim]
    w2 = theta[input_dim*hidden_dim+hidden_dim:input_dim*hidden_dim+hidden_dim+hidden_dim*output_dim].reshape(hidden_dim, output_dim)
    b_2 = theta[input_dim*hidden_dim+hidden_dim+hidden_dim*output_dim:]

    z1 = X_batch @ w1 + b_1
    a1 = np.maximum(0, z1)
    z2 = a1 @ w2 + b_2
    exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    loss = -np.mean(np.log(probs[np.arange(bs), y_batch] + 1e-12))

    dz2 = probs.copy()
    dz2[np.arange(bs), y_batch] -= 1
    dz2 /= bs
    
    dw2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)
    da1 = dz2 @ w2.T
    dz1 = da1 * (z1 > 0)
    dw1 = X_batch.T @ dz1
    db1 = np.sum(dz1, axis=0)
    
    return loss, np.concatenate([dw1.ravel(), db1.ravel(), dw2.ravel(), db2.ravel()])

learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
all_results = {}

for lr in learning_rates:
    print(f"\n=== lr = {lr} ===")
    
    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(output_dim)
    params = np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])

    m = np.zeros_like(params)
    v = np.zeros_like(params)
    t = 0

    batch_size, epochs = 128, 10
    n_train = len(X_train)
    
    history_loss = []
    history_acc = []

    for epoch in range(epochs):
        idx = np.random.permutation(n_train)
        X_shuf, y_shuf = X_train[idx], y_train[idx]
        
        epoch_loss = 0
        n_batches = 0
        
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            X_batch, y_batch = X_shuf[start:end], y_shuf[start:end]
            
            loss, g = loss_and_grad(params, X_batch, y_batch)
            epoch_loss += loss
            n_batches += 1
            
            t += 1
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g**2
            m_hat = m / (1 - 0.9**t)
            v_hat = v / (1 - 0.999**t)
            params -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        w1 = params[:input_dim*hidden_dim].reshape(input_dim, hidden_dim)
        b_1 = params[input_dim*hidden_dim:input_dim*hidden_dim+hidden_dim]
        w2 = params[input_dim*hidden_dim+hidden_dim:input_dim*hidden_dim+hidden_dim+hidden_dim*output_dim].reshape(hidden_dim, output_dim)
        b_2 = params[input_dim*hidden_dim+hidden_dim+hidden_dim*output_dim:]
        
        z1_test = X_test @ w1 + b_1
        a1_test = np.maximum(0, z1_test)
        z2_test = a1_test @ w2 + b_2
        preds = np.argmax(z2_test, axis=1)
        test_acc = np.mean(preds == y_test)
        
        history_loss.append(epoch_loss / n_batches)
        history_acc.append(test_acc)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/n_batches:.4f} | Test Acc: {test_acc*100:.2f}%")
    
    all_results[lr] = (history_loss, history_acc)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
for lr in learning_rates:
    plt.plot(all_results[lr][0], label=f'lr={lr:.0e}', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Влияние Learning Rate на Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()