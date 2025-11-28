import numpy as np

def softmax_with_temperature(probs, temperature=1.0):
    """
    probs: numpy array, исходные вероятности MCTS
    temperature: float, T < 1 → усиление топов, T > 1 → сглаживание
    """
    probs = np.array(probs)
    probs /= np.sum(probs)
    # безопасный softmax: вычитаем максимум для численной стабильности
    scaled = probs / temperature
    exp_probs = np.exp(scaled - np.max(scaled))
    return exp_probs / np.sum(exp_probs)

# пример использования
action_probs = np.array([0.02, 0.018, 0.01, 0.005, 0.001] + [0.01 for i in range(20)])
adjusted_probs = softmax_with_temperature(action_probs, temperature=0.01)
print(list(map(float, list(adjusted_probs))))
print("Сумма:", np.sum(adjusted_probs))