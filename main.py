import numpy as np

class Perceptron:
    def __init__(self, w : list[float], theta):
        self.w = np.array(w).reshape(-1, 1)
        self.theta = theta
        self.output : Perceptron | None = None
        self.f = lambda mu : 1 if mu >= 0 else 0

    def forward(self, s : list[float]) -> int:
        if self.w.shape[0] != len(s):
            raise Exception("w と s の次元が一致しません")
        weighted_sum = np.array(s) @ self.w
        v = self.f(weighted_sum - self.theta)
        return v

#パラメータの設定
w_1 = [1,1,1]
theta_1 = 2
w_2 = [-1,-1,-1]
theta_2 = -2
w_3 = [1,1]
theta_3 = 2

p1 = Perceptron(w_1, theta_1)
p2 = Perceptron(w_2, theta_2)
p3 = Perceptron(w_3, theta_3)

x = [1, 0, 1]


def net(x):
    z_1 = p1.forward(x)
    z_2 = p2.forward(x)
    y = p3.forward([z_1, z_2])
    return y

#出力を確認 1で成功
print(net(x))

#全パターンをテスト，失敗するとエラーがでる．ALl Passedで成功
def test():
    for x_1 in range(2):
        for x_2 in range(2):
            for x_3 in range(3):
                x = [x_1, x_2, x_3]
                correct = 1 if x_1 + x_2 + x_3 == 2 else 0
                if correct != net(x):
                    raise Exception(f"Test Failed at x = {x}")
                else:
                    print(f"Pass at x = {x}")

    print("All Passed!")

test()
