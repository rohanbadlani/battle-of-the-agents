import argparse
import sys
import pdb
sys.path.append('../meta-keras-rl/keras-rl/')
from PIL import Image
import numpy as np
import gym
from keras.models import Model
from keras.layers import Flatten, Input, Dense, Concatenate, Reshape, Conv2D
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K
from rl.agents.dqn import DQNAgent, DQfDAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import PartitionedMemory, PrioritizedMemory
from rl.core import Processor
from rl.callbacks import TrainEpisodeLogger, ModelIntervalCheckpoint
from rl.util import load_demo_data_from_file
from record_demonstrations import demonstrate, reward_threshold_subset
from keras.layers import ELU
from keras.activations import relu
from keras.utils import plot_model
import pdb

from rl.agents.curious_agent import CuriousDQNAgent, CuriousDQfDAgent

parser = argparse.ArgumentParser()
parser.add_argument('--mode',choices=['train','test','demonstrate'],default='test')
parser.add_argument('--filename_append',default='')
parser.add_argument('--model',choices=['student','expert','curious_expert','curious_student'],default='expert')
parser.add_argument('--env',choices=['LunarLander', 'Breakout'],default='LunarLander')
parser.add_argument('--curiosity_during_expert_phase',choices=['True', 'False'],default='True')
args = parser.parse_args()

ATARI_INPUT_SHAPE = (84, 84)

class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        #img = img.resize(ATARI_INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        img = img.resize(ATARI_INPUT_SHAPE)
        processed_observation = np.array(img)
        #assert processed_observation.shape == ATARI_INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

class RocketProcessor(Processor):
    def process_observation(self, observation):
        return np.array(observation, dtype='float32')

    def process_state_batch(self, batch):
        return np.array(batch).astype('float32')

    def process_reward(self, reward):
        return np.sign(reward) * np.log(1 + abs(reward))

    def process_demo_data(self, demo_data):
        for step in demo_data:
            step[0] = self.process_observation(step[0])
            step[2] = self.process_reward(step[2])
        return demo_data

env = None
curiosity_input = None
sensors = None
q_network_input = None
demonstrations_file = None
plot_file_prefix = None
processor = None

if args.env == "Breakout":
    env = gym.make("Breakout-v0")
    demonstrations_file = "breakout_demos.npy"
    plot_file_prefix = "breakout_"
    processor = AtariProcessor()
    np.random.seed(231)
    env.seed(123)

    #FIXME: setting window length to 1 as then the convolution does not work if shape is (4, 210, 160, 3)
    #FIXME: Try to see if we can make this 4 as Mnih et al had done. Will have to implement tiling of images I think.
    WINDOW_LENGTH = 1
    LL_state_size = 288 #after convolution, this is the size of the state.

    #state vector is 210, 160, 3 dimensional in Atari Breakout.
    #conv_input_shape = (WINDOW_LENGTH, 210, 160, 3)
    conv_input_shape = (WINDOW_LENGTH, 84, 84, 3)
    
    #reference: https://github.com/pathak22/noreward-rl/blob/master/src/model.py - uses 4 conv layers for Doom/Mario. But ours is Breakout. (can try this but commented for now)
    #OpenAI Curiosity Large Scale uses only 2 conv layers for Breakout. reference: https://arxiv.org/pdf/1312.5602.pdf Page 5 end, Page 6 beginning.
    sensors = Input(shape=(conv_input_shape))
    
    image_shape = tuple([x for x in sensors.shape.as_list() if x != WINDOW_LENGTH and x is not None])

    reshaped_sensors = Reshape(image_shape, input_shape=conv_input_shape)(sensors)
    conv_layer1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu', use_bias=True, input_shape=image_shape)(reshaped_sensors)
    conv_layer2 = Conv2D(filters=32, kernel_size=(3, 3) , padding='same', strides=(2, 2), activation='relu', use_bias=True)(conv_layer1)
    conv_layer3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu', use_bias=True)(conv_layer2)
    conv_out = Conv2D(filters=8, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu', use_bias=True)(conv_layer3)

    q_network_input = conv_out

    #Define inputs to curiosity network; s1 represents state at time t, s2 represents state at time t+1
    curiosity_s1 = Input(shape=(conv_input_shape))
    curiosity_s2 = Input(shape=(conv_input_shape))


    curiosity_reshape = Reshape(image_shape, input_shape=conv_input_shape)

    #rehsape both s1 and s2 with the same layer
    curiosity_reshaped_sensors_phi1 = curiosity_reshape(curiosity_s1)
    curiosity_reshaped_sensors_phi2 = curiosity_reshape(curiosity_s2)

    #define the convolution layers
    curiosity_conv_l1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(2, 2), activation=ELU(alpha=1.0), use_bias=True)
    curiosity_conv_l2 = Conv2D(filters=32, kernel_size=(3, 3) , padding='same', strides=(2, 2), activation=ELU(alpha=1.0), use_bias=True)
    curiosity_conv_l3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(2, 2), activation=ELU(alpha=1.0), use_bias=True)
    curiosity_conv_l4 = Conv2D(filters=8, kernel_size=(3, 3), padding='same', strides=(2, 2), activation=ELU(alpha=1.0), use_bias=True)

    #Apply those convolution layers to our inputs
    #layer 1
    curiosity_conv_l1_s1 = curiosity_conv_l1(curiosity_reshaped_sensors_phi1)
    curiosity_conv_l1_s2 = curiosity_conv_l1(curiosity_reshaped_sensors_phi2)

    #layer 2
    curiosity_conv_l2_s1 = curiosity_conv_l2(curiosity_conv_l1_s1)
    curiosity_conv_l2_s2 = curiosity_conv_l2(curiosity_conv_l1_s2)

    #layer 3
    curiosity_conv_l3_s1 = curiosity_conv_l3(curiosity_conv_l2_s1)
    curiosity_conv_l3_s2 = curiosity_conv_l3(curiosity_conv_l2_s2)

    #layer 4
    curiosity_conv_l4_s1 = curiosity_conv_l4(curiosity_conv_l3_s1)
    curiosity_conv_l4_s2 = curiosity_conv_l4(curiosity_conv_l3_s2)


else:
    #defaulting to LunarLander
    env = gym.make("LunarLander-v2")
    demonstrations_file = "lunar_lander_demos.npy"
    plot_file_prefix = "lunar_lander_"
    processor = RocketProcessor()
    np.random.seed(231)
    env.seed(123)

    WINDOW_LENGTH = 2
    #state vector is 8 dimensional in lunar lander (x,y,x_vel,y_vel,theta,theta_vel,leg1_touched,leg2_touched)
    LL_state_size = 8
    input_shape = (WINDOW_LENGTH, LL_state_size)
    sensors = Input(shape=(input_shape))

    #input for the curiosity
    curiosity_s1 = Input(shape=(input_shape))
    curiosity_s2 = Input(shape=(input_shape))

    q_network_input = sensors

nb_actions = env.action_space.n

# DQfD "student" model architecture
#sensors = Input(shape=(input_shape))
s_dense = Flatten()(q_network_input)
s_dense = Dense(64, activation='relu', kernel_regularizer=l2(.0001))(s_dense)
s_dense2= Dense(128, activation='relu', kernel_regularizer=l2(.0001))(s_dense)
s_dense3 = Dense(64, activation='relu', kernel_regularizer=l2(.0001))(s_dense2)
s_actions = Dense(nb_actions, activation='linear', name="student_output", kernel_regularizer=l2(.0001))(s_dense3)
student_model = Model(inputs=sensors, outputs=s_actions)
# "Expert" (regular dqn) model architecture
e_dense = Flatten()(q_network_input)
e_dense = Dense(64, activation='relu')(e_dense)
e_dense2= Dense(128, activation='relu')(e_dense)
e_dense3 = Dense(64, activation='relu')(e_dense2)
e_actions = Dense(nb_actions, activation='linear', name="expert_output")(e_dense3)
expert_model = Model(inputs=sensors, outputs=e_actions)
plot_model(expert_model, show_shapes=True, to_file=plot_file_prefix + 'expert_model.png')

#### FORWARD MODEL ########
#Window length is number of consecutive states to capture in a single observation
curious_forward_flatten_state = None
if(args.env == "Breakout"):
    curious_forward_flatten_state = Flatten()(curiosity_conv_l4_s1)
else:
    curious_forward_flatten_state = Flatten()(curiosity_s1)

#single action so shape is just 1
curious_forward_inputs_action = Input(shape=(nb_actions,))
#input to the forward model is observation (size 16) plus action (size 1) so total length=17
curious_forward_concat = Concatenate()([curious_forward_flatten_state, curious_forward_inputs_action])

curious_fw_dense_hidden_units =  min(256, K.int_shape(curious_forward_flatten_state)[1] ) 
curious_forward_fc1 = Dense(curious_fw_dense_hidden_units, activation='relu', kernel_regularizer=l2(.0001))(curious_forward_concat)

curious_forward_fc2 = Dense(K.int_shape(curious_forward_flatten_state)[1], activation='linear', kernel_regularizer=l2(.0001))(curious_forward_fc1)

if(args.env == "Breakout"):
    curious_reshape_output = Reshape( K.int_shape(curiosity_conv_l4_s1)[1:], name="curious_forward_output" )(curious_forward_fc2)
else:
    curious_reshape_output = Reshape( K.int_shape(curiosity_s1)[1:], name="curious_forward_output" )(curious_forward_fc2)

curious_fw_input_list = [curiosity_s1, curious_forward_inputs_action]
curiosity_forward_model = Model(inputs=curious_fw_input_list, outputs=curious_reshape_output)
plot_model(curiosity_forward_model, show_shapes=True, to_file=plot_file_prefix + 'curiosity_forward_model.png')
########## END FORWARD MODEL ##########


############ INVERSE MODEL ##########
curious_inverse_flatten_st = curious_forward_flatten_state
curious_inverse_flatten_next_st_layer = Flatten(name="flattened_phi_next_state")
if(args.env == "Breakout"):
    curious_inverse_flatten_next_st = curious_inverse_flatten_next_st_layer(curiosity_conv_l4_s2)
else:
    curious_inverse_flatten_next_st = curious_inverse_flatten_next_st_layer(curiosity_s2)

curious_inverse_fullinput = Concatenate()([curious_inverse_flatten_st, curious_inverse_flatten_next_st])

curious_fw_dense_hidden_units =  min(256, K.int_shape(curious_inverse_fullinput)[1] ) 

curious_inverse_fc1 = Dense(curious_fw_dense_hidden_units, activation='relu', kernel_regularizer=l2(.0001))(curious_inverse_fullinput)
curious_inverse_fc2 = Dense(nb_actions, activation='softmax', name="curious_inverse_output", kernel_regularizer=l2(.0001))(curious_inverse_fc1)

curious_inv_input_list = [curiosity_s1, curiosity_s2]

curiosity_inverse_model = Model(inputs=curious_inv_input_list, outputs=curious_inverse_fc2)
plot_model(curiosity_inverse_model, show_shapes=True, to_file=plot_file_prefix + 'curiosity_inverse_model.png')
######## END INVERSE MODEL ################

model_saves = './demonstrations/'

if __name__ == "__main__":
    if args.model == 'student':
        # load expert data
        expert_demo_data = load_demo_data_from_file(model_saves + demonstrations_file)
        expert_demo_data = reward_threshold_subset(expert_demo_data,0)
        print(expert_demo_data.shape)
        expert_demo_data = processor.process_demo_data(expert_demo_data)
        # memory
        memory = PartitionedMemory(limit=500000, pre_load_data=expert_demo_data, alpha=.6, start_beta=.4, end_beta=.4, window_length=WINDOW_LENGTH)
        # policy
        policy = EpsGreedyQPolicy(.01)
        # agent
        dqfd = DQfDAgent(model=student_model, nb_actions=nb_actions, policy=policy, memory=memory,
                       processor=processor, enable_double_dqn=True, enable_dueling_network=True, gamma=.99, target_model_update=10000,
                       train_interval=1, delta_clip=1., pretraining_steps=15000, n_step=10, large_margin=.8, lam_2=1)

        lr = .00025
        dqfd.compile(Adam(lr), metrics=['mae'])
        weights_filename = model_saves + 'student_lander15k_weights.h5f'
        checkpoint_weights_filename = model_saves +'student_lander15k_weights{step}.h5f'
        log_filename = model_saves + 'student_lander15k_REWARD_DATA.txt'
        callbacks = [TrainEpisodeLogger(log_filename),
                        ModelIntervalCheckpoint(checkpoint_weights_filename, interval=1000000)
                    ]
        if args.mode == 'train':
            dqfd.fit(env, callbacks=callbacks, nb_steps=4250000, verbose=0, nb_max_episode_steps=1500)
            dqfd.save_weights(weights_filename, overwrite=True)
        if args.mode == 'test':
            dqfd.load_weights(model_saves + 'student_lander15k_weights.h5f')
            dqfd.test(env, nb_episodes=12, visualize=True, verbose=2, nb_max_start_steps=30)
        if args.mode == 'demonstrate':
            print("DQfD cannot demonstrate.")

    ##keeping this as originally
    if args.model == 'expert':
            # memory
            memory = PrioritizedMemory(limit=500000, alpha=.6, start_beta=.4, end_beta=.4, steps_annealed=5000000, window_length=WINDOW_LENGTH)
            # policy
            policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.02, value_test=.01,
                                          nb_steps=1000000)
            # agent
            dqn = DQNAgent(model=expert_model, nb_actions=nb_actions, policy=policy, memory=memory,
                           processor=processor, enable_double_dqn=True, enable_dueling_network=True, gamma=.99, target_model_update=10000,
                           train_interval=1, delta_clip=1., nb_steps_warmup=50000)

            lr = .00025
            dqn.compile(Adam(lr), metrics=['mae'])
            weights_filename = model_saves + args.filename_append + 'expert_lander_weights.h5f'
            checkpoint_weights_filename = model_saves + args.filename_append + 'expert_lander_weights{step}.h5f'
            log_filename = model_saves + args.filename_append + 'expert_lander_REWARD_DATA.txt'
            callbacks = [TrainEpisodeLogger(log_filename),
                            ModelIntervalCheckpoint(checkpoint_weights_filename, interval=1000000)
                        ]
            if args.mode == 'train':
                dqn.fit(env, callbacks=callbacks, nb_steps=4250000, verbose=0, nb_max_episode_steps=1500)
                dqn.save_weights(weights_filename, overwrite=True)
            if args.mode == 'test':
                dqn.load_weights(model_saves + args.filename_append + 'expert_lander_weights.h5f')
                dqn.test(env, nb_episodes=5, visualize=True, verbose=2, nb_max_start_steps=30)
            if args.mode == 'demonstrate':
                dqn.load_weights(model_saves + args.filename_append + 'expert_lander_weights.h5f')
                demonstrate(dqn, env, 75000, model_saves + demonstrations_file)


    if args.model == 'curious_expert':
        # memory
        memory = PrioritizedMemory(limit=500000, alpha=.6, start_beta=.4, end_beta=.4, steps_annealed=5000000, window_length=WINDOW_LENGTH)
        # policy
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.02, value_test=.01,
                                      nb_steps=1000000)
        # agent
        dqn = CuriousDQNAgent(model=expert_model, curiosity_forward_model=curiosity_forward_model, curiosity_inverse_model=curiosity_inverse_model, nb_actions=nb_actions, policy=policy, memory=memory,
                       processor=processor, enable_double_dqn=True, enable_dueling_network=True, gamma=.99, target_model_update=10000,
                       train_interval=1, delta_clip=1., nb_steps_warmup=50000)

        lr = .00025
        dqn.compile(Adam(lr), metrics=['mae'])

        #plot_model(dqn.trainable_model, show_shapes=True, to_file='full_trainable_model.png')
        

        weights_filename = model_saves + 'expert_lander_weights.h5f'
        checkpoint_weights_filename = model_saves +'expert_lander_weights{step}.h5f'
        log_filename = model_saves + 'expert_lander_REWARD_DATA.txt'
        callbacks = [TrainEpisodeLogger(log_filename),
                        ModelIntervalCheckpoint(checkpoint_weights_filename, interval=1000000)
                    ]
        if args.mode == 'train':
            dqn.fit(env, callbacks=callbacks, nb_steps=4250000, verbose=0, nb_max_episode_steps=1500)
            dqn.save_weights(weights_filename, overwrite=True)
        if args.mode == 'test':
            dqn.load_weights(model_saves + 'expert_lander_weights.h5f')
            dqn.test(env, nb_episodes=5, visualize=True, verbose=2, nb_max_start_steps=30)
        if args.mode == 'demonstrate':
            dqn.load_weights(model_saves + 'expert_lander_weights.h5f')
            demonstrate(dqn, env, 75000, model_saves + demonstrations_file)

    if args.model == 'curious_student':
        # load expert data
        expert_demo_data = load_demo_data_from_file(model_saves + demonstrations_file)
        expert_demo_data = reward_threshold_subset(expert_demo_data,0)
        print(expert_demo_data.shape)
        expert_demo_data = processor.process_demo_data(expert_demo_data)
        # memory
        memory = PartitionedMemory(limit=500000, pre_load_data=expert_demo_data, alpha=.6, start_beta=.4, end_beta=.4, window_length=WINDOW_LENGTH)
        # policy
        policy = EpsGreedyQPolicy(.01)
        # agent
        dqfd = CuriousDQfDAgent(model=student_model, curiosity_forward_model=curiosity_forward_model, curiosity_inverse_model=curiosity_inverse_model, nb_actions=nb_actions, policy=policy, memory=memory,
                       processor=processor, enable_double_dqn=True, enable_dueling_network=True, gamma=.99, target_model_update=10000,
                       train_interval=1, delta_clip=1., pretraining_steps=15000, n_step=10, large_margin=.8, lam_2=1)

        lr = .00025
        dqfd.compile(Adam(lr), metrics=['mae'])
        weights_filename = model_saves + 'student_lander15k_weights.h5f'
        checkpoint_weights_filename = model_saves +'student_lander15k_weights{step}.h5f'
        log_filename = model_saves + 'student_lander15k_REWARD_DATA.txt'
        callbacks = [TrainEpisodeLogger(log_filename),
                        ModelIntervalCheckpoint(checkpoint_weights_filename, interval=1000000)
                    ]
        if args.mode == 'train':
            dqfd.fit(env, callbacks=callbacks, nb_steps=4250000, verbose=0, nb_max_episode_steps=1500)
            dqfd.save_weights(weights_filename, overwrite=True)
        if args.mode == 'test':
            dqfd.load_weights(model_saves + 'student_lander15k_weights.h5f')
            dqfd.test(env, nb_episodes=12, visualize=True, verbose=2, nb_max_start_steps=30)
        if args.mode == 'demonstrate':
            print("Curious DQfD cannot demonstrate.")

