import os
import random
import sys
import time
import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import re
import heapq

#from tensorboardX import SummaryWriter

from game.flappy_bird import GameState

#writer = SummaryWriter()

class NeuralNetwork(nn.Module):

    def __init__(self, matrix_size):
        super(NeuralNetwork, self).__init__()
        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 500000
        self.replay_memory_size = 10000
        self.minibatch_size = 32
        self.fc6 = None
        
	#IMAGE 54
        if matrix_size == 54:
            self.conv1 = nn.Conv2d(4, 32, 6, 2)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(32, 64, 5, 2)
            self.bn2 = nn.BatchNorm2d(64)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(64, 64, 3, 2)
            self.bn3 = nn.BatchNorm2d(64)
            self.relu3 = nn.ReLU(inplace=True)
            self.fc4 = nn.Linear(1600, 512)
            self.relu4 = nn.ReLU(inplace=True)
            self.fc5 = nn.Linear(512, self.number_of_actions)
       
        #IMAGE 64
        if matrix_size == 64:
                self.conv1 = nn.Conv2d(4, 32, 8, 2)
                self.bn1 = nn.BatchNorm2d(32)
                self.relu1 = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(32, 64, 4, 2)
                self.bn2 = nn.BatchNorm2d(64)
                self.relu2 = nn.ReLU(inplace=True)
                self.conv3 = nn.Conv2d(64, 64, 3, 1)
                self.bn3 = nn.BatchNorm2d(64)
                self.relu3 = nn.ReLU(inplace=True)
                self.fc4 = nn.Linear(7744, 1232)
                self.relu4 = nn.ReLU(inplace=True)
                self.fc5 = nn.Linear(1232, 196)
                self.relu5 = nn.ReLU(inplace=True)
                self.fc6 = nn.Linear(196, self.number_of_actions)

	#IMAGE 75
        if matrix_size == 75:
            self.conv1 = nn.Conv2d(4, 32, 9, 3)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(32, 64, 5, 3)
            self.bn2 = nn.BatchNorm2d(64)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(64, 64, 3, 1)
            self.bn3 = nn.BatchNorm2d(64)
            self.relu3 = nn.ReLU(inplace=True)
            self.fc4 = nn.Linear(1600, 512)
            self.relu4 = nn.ReLU(inplace=True)
            self.fc5 = nn.Linear(512, self.number_of_actions)
       
        #IMAGE 84
        if matrix_size == 84:
            self.conv1 = nn.Conv2d(4, 32, 8, 4)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(32, 64, 4, 2)
            self.bn2 = nn.BatchNorm2d(64)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(64, 64, 3, 1)
            self.bn3 = nn.BatchNorm2d(64)
            self.relu3 = nn.ReLU(inplace=True)
            self.fc4 = nn.Linear(3136, 512)
            self.relu4 = nn.ReLU(inplace=True)
            self.fc5 = nn.Linear(512, self.number_of_actions)

	#IMAGE 87
        if matrix_size == 87:
            self.conv1 = nn.Conv2d(4, 32, 7, 4)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(32, 64, 5, 2)
            self.bn2 = nn.BatchNorm2d(64)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(64, 64, 3, 2)
            self.bn3 = nn.BatchNorm2d(64)
            self.relu3 = nn.ReLU(inplace=True)
            self.fc4 = nn.Linear(1024, 512)
            self.relu4 = nn.ReLU(inplace=True)
            self.fc5 = nn.Linear(512, self.number_of_actions)
     
 	

	#IMAGE 104
        if matrix_size == 104:
            self.conv1 = nn.Conv2d(4, 32, 9, 5)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(32, 64, 4, 2)
            self.bn2 = nn.BatchNorm2d(64)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(64, 64, 3, 1)
            self.bn3 = nn.BatchNorm2d(64)
            self.relu3 = nn.ReLU(inplace=True)
            self.fc4 = nn.Linear(3136, 512)
            self.relu4 = nn.ReLU(inplace=True)
            self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        if self.fc6 is not None:
            out = self.fc6(self.relu5(out))

        return out


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)



def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)

    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()

    #1*84*84 size of imagetensor
    return image_tensor


def resize_and_bgr2gray(image, matrix_size):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (matrix_size, matrix_size)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (matrix_size, matrix_size, 1))
    return image_data

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def train(model, start, filename, matrix_size, pid):
    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    game_state = GameState()

    # initialize replay memory
    replay_memory = []
    heap = []
    heapq.heapify(heap)

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data, matrix_size)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    # keep track of wins per game
    max_wins = 0
    global_max_wins = 0

    # used for the csv output file
    outputlog = []

    # counter of the +1 reward
    reward_counter = 0
    # main infinite loop
    #state = state.cuda()
    while iteration < model.number_of_iterations:
        # get output from the neural network
        output = model(state)[0]

        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        #if random_action:
            #print("Performed random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1

        # get next state and reward
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1, matrix_size)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1, terminal))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()


        # get output for the next state
        output_1_batch = model(state_1_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        # extract Q-value
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()


        # calculate loss
        loss = criterion(q_value, y_batch)
  
        # do backward pass
        loss.backward()

        optimizer.step()


        # set state to be state_1
        state = state_1
        iteration += 1


        qmax = np.max(output.cpu().detach().numpy())

        if reward.numpy()[0][0] == 1:
            reward_counter += 1
            max_wins += 1
            if global_max_wins < max_wins:
                global_max_wins = max_wins


        if reward.numpy()[0][0] == -1:
            max_wins = 0


        if iteration % 500 == 0:
            outputlog.append('%s,%s,%s,%s,%s,%s\n' % (iteration, epsilon, qmax, loss.item(), reward_counter, global_max_wins))
            global_max_wins = 0
            printProgressBar(iteration, model.number_of_iterations, prefix = 'Progress:', suffix = 'Complete', length = 100)
            with open(filename, 'a') as f:
                while len(outputlog) > 0:
                    f.write(outputlog.pop(0))
                f.close()

        if iteration % 250000 == 0:
            torch.save(model, "pretrained_model/" + str(matrix_size) + "_" + pid + "_current_model_" + str(iteration) + ".pth")


def test(model):
    game_state = GameState()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1


def main(mode, matrix_size):
    cuda_is_available = torch.cuda.is_available()
    matrix_size = int(matrix_size)
    if mode == 'test':
        model = torch.load(
            'pretrained_model/current_model_2000000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')
        if not os.path.exists('logs/'):
            os.mkdir('logs/')

        now = str(datetime.datetime.now())
        filename = re.sub(r"(-|\s|:|\W)*","", now)
        pid = filename
        filename = "logs/" + str(matrix_size) + "_" + filename + ".csv"

        model = NeuralNetwork(matrix_size)

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()
        #pass also the file for logs
        train(model, start, filename, matrix_size, pid)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

