import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import Input, Model
from PIL import Image
import pathlib
import numpy as np
import os
import random
from random import shuffle
import cv2

tf.keras.backend.set_floatx('float64')

NUM_GAMES = 20000
NUM_STEPS = 4
NUM_EPOCHS = 2

MINI_BATCH = 10
POOL_SIZE = 60
IM_WIDTH = 160
IM_HEIGHT = 210

IM_WIDTH_RESIZE = int(IM_WIDTH/2)
IM_HEIGHT_RESIZE = int(IM_HEIGHT/2)

OUTPUT_POLICY = 9

# Gamma is future reward decay
GAMMA = .95
# Epsilon is chance to take a random move
EPSILON = 1.0

PRINT_OUT = False

data_track = "policy_loss.csv"
checkpoint_path_policy = "pacsave/policy/policy.ckpt"

def show_im(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class MsPacman:
    def __init__(self, epsilon = EPSILON):
        self.games_played = 0
        self.epsilon = epsilon
        self.experience_pool = []
        # When self.train_on_samples() is called, uses data from the self.training_pool
        self.training_pool = []
        # Memory allocated for passing into training
        self.minibatch = (np.zeros((MINI_BATCH,IM_HEIGHT_RESIZE,IM_WIDTH_RESIZE,NUM_STEPS)),      # obs
                            np.zeros((MINI_BATCH,OUTPUT_POLICY)),              # action
                            np.zeros((MINI_BATCH,IM_HEIGHT_RESIZE,IM_WIDTH_RESIZE,NUM_STEPS)),    # next obs
                            np.zeros((MINI_BATCH)),                          # reward
                            np.zeros((MINI_BATCH)),                          # not_done
                            np.zeros((MINI_BATCH)))                          # future reward
        self.current_save_path = 'mspacman/'
        # Create optimizer
        self.opt = tf.keras.optimizers.Adam(2e-5)
        # Create models
        self.policy = self._create_q_net()
        # Keep track of old network
        self.policy_old = self._create_q_net()
        # Sets states of policy and clears save folder
        self.reset_policy()
        print("Successfully created policy and critic")

    # Evaluates a frame of pacman to get move probabilities
    def _create_q_net(self):
        image_input = Input(shape=(IM_HEIGHT_RESIZE,IM_WIDTH_RESIZE,NUM_STEPS))
        # TO DO: Add padding
        # Object convolution
        x = layers.Conv2D(32, (8, 8), strides=4, activation='relu')(image_input)
        x = layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(OUTPUT_POLICY)(x)
        model = Model(inputs=image_input, outputs=x)
        if PRINT_OUT:
            model.summary()
        try:
            model.load_weights(checkpoint_path_policy)
        except:
            print('Error loading weights from file')
        return model

    # This function will be called every frame to make moves and save results
    def eval_policy(self, images, reward, not_done):
        obs = np.zeros((IM_HEIGHT_RESIZE,IM_WIDTH_RESIZE,NUM_STEPS))
        # Grayscale, resize, add to buffer
        for i in range(NUM_STEPS):
            im = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
            im = cv2.resize(im, (IM_WIDTH_RESIZE,IM_HEIGHT_RESIZE))
            #show_im(im)
            obs[:,:,i] = im
        #show_im(obs)
        obs = tf.image.convert_image_dtype(obs, tf.float64)
        # Evaluate frame on model
        qvals = self.policy(np.expand_dims(obs, 0))
        # Selects the move with the largest qval
        if random.random() > self.epsilon:
            chosen_btn_return = tf.math.argmax(tf.squeeze(qvals))
        else:
            chosen_btn_return = random.randint(0, OUTPUT_POLICY - 1)
        chosen_btn = tf.one_hot(chosen_btn_return, OUTPUT_POLICY)
        if self.internal_clock != 0:
            # Save experience in live pool
            self.experience_pool.append((self.previous_obs, self.previous_move, obs, reward, not_done))
        # Update previous frame
        self.previous_obs = obs
        # Update previous move
        self.previous_move = chosen_btn
        # Update clock
        self.internal_clock += 1
        # Return chosen moves
        return tf.squeeze(chosen_btn_return)
    
    # Uses self.training_pool to do training for NUM_EPOCHS
    def train_on_samples(self):
        if len(self.training_pool) == 0:
            return
        for _ in range(NUM_EPOCHS):
            indices = list(range(len(self.training_pool)))
            random.shuffle(indices)
            for x in range(len(indices)):
                # If can't fully fill minibatch, old data is fine
                i = indices.pop()
                for y in range(5):
                    self.minibatch[y][x % MINI_BATCH] = self.training_pool[i][y]
                if (x % MINI_BATCH) - MINI_BATCH == -1 or len(indices) == 0:
                    loss = self._train_step(*self.minibatch)
                    self.loss += loss.numpy()
        # Clear training pool
        self.training_pool = []

    # This function will be called after a game is over
    def fit_policy(self):
        # Empty the experience pool
        while len(self.experience_pool) != 0:
            experience = self.experience_pool.pop(0)
            # Monte Carlo future reward
            future_reward = np.sum([self.experience_pool[x][3]*pow(GAMMA,x) for x in range(len(self.experience_pool))])
            self.training_pool.append((*experience, future_reward))
        self.train_on_samples()
        with open(data_track, 'a+') as f:
            f.write(str(self.loss) + '\n')
            print('Loss:', self.loss, 'Current epsilon:', self.epsilon)
        self.policy.save_weights(checkpoint_path_policy)
        self.reset_policy()
        self.epsilon -= EPSILON * (1/NUM_GAMES)
        # Epsilon greedy
        if self.epsilon < .05:
            self.epsilon = .05
        self.policy_old.set_weights(self.policy.get_weights())
        self.games_played += 1

    def reset_policy(self):
        self.loss = 0
        self.internal_clock = 0

    @tf.function
    def _train_step(self, state, move_chosen, next_state, reward, not_done, future_reward):
        with tf.GradientTape() as grad_tape:
            grad_tape.watch(self.policy.trainable_variables)
            # Evaluate on models
            qvals = self.policy(state)
            q_next = tf.reduce_max(self.policy_old(next_state), axis=-1)
            loss = not_done*GAMMA*q_next + reward - tf.boolean_mask(qvals, move_chosen)
            policy_loss = tf.reduce_mean(tf.square(loss))
        gradients_of_policy = grad_tape.gradient(policy_loss, self.policy.trainable_variables)
        self.opt.apply_gradients(zip(gradients_of_policy, self.policy.trainable_variables))
        return policy_loss
