import torch

class DAM:
    def __init__(self, n_power=20, m_power=30, k_memories=100, learningrate=1e-2, damp_f=0.998):
        self.n_power = n_power
        self.m_power = m_power
        self.k_memories = k_memories
        self.lr = learningrate
        self.damp_f = damp_f
        self.beta = 1.0
        self.v_node = None
        self.weight_memory = None

    def initialize_memory(self, input_dim):
        self.v_node = input_dim
        self.weight_memory = torch.normal(-0.1, 0.1, (self.k_memories, self.v_node * 2))

    def train(self, patterns, epochs=10):
        criterion = torch.nn.MSELoss()
        for epoch in range(epochs):
            act_lr = self.lr * (self.damp_f ** epoch)
            target_output = torch.reshape(patterns.T, (-1,))
            extend_img = torch.cat((patterns.T, -torch.ones((self.v_node, patterns.shape[0]))), axis=0)
            target_u = torch.tile(extend_img, (1, self.v_node))
            WMvv = torch.maximum(self.weight_memory @ target_u, torch.tensor(0))
            Y = torch.tanh(self.beta * torch.sum(WMvv ** self.n_power, axis=0))
            print(f"Epoch {epoch}, MSE: {criterion(Y, target_output).item()}")

            diff_WM = ((target_output - Y) ** (2 * self.m_power - 1) * (1 - Y ** 2)).reshape(1, -1) @ target_u.T
            self.weight_memory += act_lr * diff_WM / (torch.amax(torch.abs(diff_WM), axis=1, keepdim=True) + 1e-10)
