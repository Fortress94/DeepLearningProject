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
import pickle
from torch.autograd import Variable
#from tensorboardX import SummaryWriter

from game.flappy_bird import GameState
from maxHeap import MaxHeap
from zipf import Quantile

#writer = SummaryWriter()

class NeuralNetwork(nn.Module):

    def __init__(self, matrix_size):
        super(NeuralNetwork, self).__init__()
        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        # Parameter for memory
        self.alpha = 0.7
        self.beta = 0.5
        self.number_of_iterations = 500000
        #self.replay_memory_size = 20000
        self.replay_memory_size = 10000

        self.minibatch_size = 32

        #IMAGE 54
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


def train(model, start, filename, matrix_size, pid, weight_yes):
    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    game_state = GameState()

    # initialize replay memory
    replay_memory = []

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
    state = state.cuda()
    mem_id = 0
    delete_id = 0
    memory_map = {}
    memory_content = {}
    rank_heap = MaxHeap()
    rank_heap.init_alpha(model.replay_memory_size, model.alpha)



    delete_id = 0
    # Get precomputed quatiles from file
    N_list, quantiles = Quantile.load_quantiles()
    current_quantile = 0
    beta_annealing = (1 - model.beta) / model.number_of_iterations
    beta = model.beta
    sorted_mem = []
    loss = None
    loss_previous_state = torch.from_numpy(np.zeros([32], dtype=np.float32))

    while iteration < model.number_of_iterations:
        # get output from the neural network
        output = model(state)[0]

        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
    
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

        # Add element to Hash maps for manage indixiziation problem
        memory_map[mem_id] = (state, action, reward, state_1, terminal)
        memory_content[(state, action, reward, state_1, terminal)] = mem_id
        rank_heap.insert((mem_id, 1000))
        mem_id = mem_id + 1

        # Add new element to the heap with max td-error


        # if replay memory is full, remove the oldest transition
        if len(memory_map) > model.replay_memory_size:
            value = memory_map[delete_id]
            del memory_content[value]
            del memory_map[delete_id]
            rank_heap.delete(delete_id)
            delete_id = delete_id + 1
            #replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = []
        ranking = []
        batch_weights = []
        # If the memory_map is lower than the first set of quantiles then it does uniform sample
        if len(memory_map) < N_list[0]:
            random_indexes = random.sample(range(0, mem_id), min(model.minibatch_size, len(memory_map)))
            for i in random_indexes:
                minibatch.append(memory_map[i])
        else:
            # Sample minibatch using a sorted array by TD-error
            # and using quantiles as ranges
            if iteration % 1000 == 0:
                rank_heap.sort()
                rank_heap.update_weights(beta)

  
            # Pool is a set of 32 elements(minibatch_size). An elment is composed by (memory_map index, td_error)
            # Indexes are random numbers generated from each quantile. They correspond to the position of the sorted vector, so the ranking.
            pool, indexes = Quantile.sample(model.minibatch_size, quantiles[current_quantile], rank_heap.lemon_tree)
            for j in range(0, len(pool)):
                ranking.append(rank_heap.ranking[indexes[j]]) # pj
                batch_weights.append(rank_heap.weights[indexes[j]]) # wj
                minibatch.append(memory_map[pool[j][0]])



        #minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

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
        # TD-error at step t
        td_error = abs(y_batch - q_value)

        delta = np.zeros(len(minibatch), dtype=np.float32)
        for i in range(0, len(minibatch)):
            element = minibatch[i]
            element_td_error = td_error[i].item()
            key = memory_content[element]
            if iteration > N_list[0]:
                normalizaition = None
                max_weight = None
                if iteration < model.replay_memory_size:
                    normalizaition = rank_heap.normalizaition[iteration]
                    max_weight = rank_heap.weights[iteration]
                else:
                    normalizaition = rank_heap.normalizaition[-1]
                    max_weight = rank_heap.weights[-1]

                p_i = ranking[i] / normalizaition # P(j) = pj / SUM pi
                weight = batch_weights[i] / max_weight # wj / max wi
                rank_heap.logical_update((key, element_td_error))# Update the priority
                delta[i] = element_td_error * weight # Compute the delta
               	rank_heap.update_all()



        # calculate loss
        loss = criterion(q_value, y_batch)
        if iteration > N_list[0] and weight_yes:
            delta = torch.from_numpy(np.array([delta], dtype=np.float32)).cuda()
            loss = loss.squeeze() * delta
            loss = loss.mean()
            #print(loss)

        # do backward pass
        loss.backward()

        #loss.backward()


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
            if iteration >= 1000 and iteration < model.replay_memory_size:
                current_quantile = current_quantile + 1

            printProgressBar(iteration, model.number_of_iterations, prefix = 'Progress:', suffix = 'Complete', length = 100)
            with open(filename, 'a') as f:
                while len(outputlog) > 0:
                    f.write(outputlog.pop(0))
                f.close()

        if iteration % 250000 == 0:
            torch.save(model, "pretrained_model/" + str(matrix_size) + "_" + pid + "_current_model_" + str(iteration) + ".pth")


        beta = beta + beta_annealing
        '''
            #writing live graph with tensorboard
            #writer.add_scalar('data/epsilon', epsilon, iteration)
            #writer.add_scalar('data/loss', loss, iteration)
            #writer.add_scalar('data/q-max', np.max(output.cpu().detach().numpy()), iteration)
            #writer.add_scalar('data/reward', reward.numpy()[0][0], iteration)

            print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
                  action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
                  np.max(output.cpu().detach().numpy()))
         '''


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


def main(mode, matrix_size, weight_yes):
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
        train(model, start, filename, matrix_size, pid, weight_yes)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
