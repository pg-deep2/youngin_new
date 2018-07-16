# 라이브러리
import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)

# 데이터 세팅
idx2char = ['h', 'i', 'e', 'l', 'o']

# Teach hihell -> ihello
x_data = [[0, 1, 0, 2, 3, 3]]   # hihell
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3

y_data = [1, 0, 1, 0, 1, 1]    # ihello

# array -> tensor 변환
inputs = Variable(torch.Tensor(x_one_hot))
labels = Variable(torch.LongTensor(y_data))
print(inputs)
print(labels)

num_classes = 2 # class 개수
input_size = 5  # one-hot size
hidden_size = 2  # 바로 one hot으로 예측하기 위해 5로 설정
batch_size = 1   # one sentence
sequence_length = 6  # |ihello| == 6
num_layers = 1  # one-layer rnn


# RNN 모델 정의
class RNN(nn.Module):
    # RNN 구성 인자 세팅
    # 여기선 FC layer 안씀
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.rnn = nn.RNN(input_size=5, hidden_size=2, batch_first=True)

    # forwarding 과정
    def forward(self, x):
        # hidden layer 초기화
        # (layer 개수, batch size, hidden size)
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Reshape input (batch size, sequence length, input size)
        # x.view(x.size(0), x.size(1), self.input_size)

        # Propagate input through RNN
        out, _ = self.rnn(x, h_0)
        return out.view(-1, num_classes)

# RNN 객체 정의
rnn = RNN(num_classes, input_size, hidden_size, num_layers)
print(rnn)

# loss, optimizer 정의
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.1)

# 모델 학습 과정
for epoch in range(100):
    # gradient -> zero 과정
    optimizer.zero_grad()

    # forwarding
    outputs = rnn(inputs)

    # loss 계산, backwarding 과정
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # class중 가장 높은 score 가진 index 뽑아내기
    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    result_str = [idx2char[c] for c in idx.squeeze()]
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.item()))
    print("Predicted string: ", ''.join(result_str))

print("Learning finished!")

# test
outputs = rnn(inputs)
_, idx = outputs.max(1)
idx = idx.data.numpy()
result_str = [idx2char[c] for c in idx.squeeze()]
print("Predicted string: ", ''.join(result_str))