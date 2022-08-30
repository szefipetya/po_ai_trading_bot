#================================================================
#
#   File name   : multiprocessing_env.py
#   Author      : PyLessons
#   Created date: 2021-02-08
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/RL-Bitcoin-trading-bot
#   Description : functions to train/test multiple custom BTC trading environments
#
#================================================================
from collections import deque
from multiprocessing import Process, Pipe
import numpy as np
from datetime import datetime
import pandas as pd
import json
import copy

from tensorflow.keras.optimizers import Adam

class Environment(Process):
    def __init__(self, env_idx, child_conn, env, training_batch_size, visualize, isTest=False):
        super(Environment, self).__init__()
        self.env = env
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.training_batch_size = training_batch_size
        self.visualize = visualize
        self.isTest=isTest

    def run(self):
        super(Environment, self).run()
        state = self.env.reset(env_steps_size = self.training_batch_size)
        self.child_conn.send(state)
        while True :
            reset, net_worth, episode_orders, buy_and_hold_minus_net_worth_relative_to_initial_in_percent, avg_price_minus_avg_net_worth_relative_to_initial_in_percent = 0, 0, 0, 0, 0
            action = self.child_conn.recv()
            #if self.env_idx == 0:
            #   self.env.render(self.visualize)
            state, reward, realaction, done, date, currentprice = self.env.step(action,isTest=self.isTest)
            net_worth = self.env.net_worth
            if done or self.env.current_step == self.env.end_step:
                episode_orders = self.env.episode_orders
                buy_and_hold_minus_net_worth_relative_to_initial_in_percent, avg_price_minus_avg_net_worth_relative_to_initial_in_percent = self.env.get_statistics()
                state = self.env.reset(env_steps_size = self.training_batch_size)
                reset = 1
                

            self.child_conn.send([state, reward, realaction, done, reset, net_worth, episode_orders, date, currentprice, buy_and_hold_minus_net_worth_relative_to_initial_in_percent, avg_price_minus_avg_net_worth_relative_to_initial_in_percent])

def train_multiprocessing(CustomEnv, agent, train_df, train_df_nomalized, num_worker=4, training_batch_size=500, visualize=False, EPISODES=10000, smoothing=20):

    train_state={
        'episode':0,
        'total_average':deque(maxlen=100),
        'best_average':0,
        'average':0,
        'buy_and_hold_minus_net_worth_relative_to_initial_in_percent_queue':deque(maxlen=smoothing),
        'avg_price_minus_avg_net_worth_relative_to_initial_in_percent_queue':deque(maxlen=smoothing)
    }
  

    #agent.end_training_log()
    # terminating processes after while loop

    works=train_multiprocessing_core(CustomEnv, agent, train_df, train_df_nomalized, train_state, num_worker, training_batch_size, visualize, EPISODES,smoothing=smoothing)
    works.append(work)
    for work in works:
        work.terminate()
        print('TERMINATED:', work)
        work.join()

def continue_train_multiprocessing(CustomEnv, agent, train_df, train_df_nomalized,train_state, num_worker=4, training_batch_size=500, visualize=False, EPISODES=10000, smoothing=20):

    #agent.end_training_log()
    # terminating processes after while loop
    train_state['total_average']=deque(train_state['total_average'],maxlen=100)
    train_state['buy_and_hold_minus_net_worth_relative_to_initial_in_percent_queue']=deque(train_state['buy_and_hold_minus_net_worth_relative_to_initial_in_percent_queue'],maxlen=smoothing)
    train_state['avg_price_minus_avg_net_worth_relative_to_initial_in_percent_queue']=deque(train_state['avg_price_minus_avg_net_worth_relative_to_initial_in_percent_queue'],maxlen=smoothing)

    works=train_multiprocessing_core(CustomEnv, agent, train_df, train_df_nomalized,train_state, num_worker, training_batch_size, visualize, EPISODES, smoothing=smoothing)
    works.append(work)
    for work in works:
        work.terminate()
        print('TERMINATED:', work)
        work.join()


def train_multiprocessing_core(CustomEnv, agent, train_df, train_df_nomalized,train_state, num_worker, training_batch_size, visualize, EPISODES, smoothing=20):
    works, parent_conns, child_conns = [], [], []

    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        env = CustomEnv(train_df,train_df_nomalized, lookback_window_size=agent.lookback_window_size)
        work = Environment(idx, child_conn, env, training_batch_size, visualize)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    agent.create_writer(env.initial_balance, env.normalize_value, EPISODES) # create TensorBoard writer


    states =        [[] for _ in range(num_worker)]
    next_states =   [[] for _ in range(num_worker)]
    actions =       [[] for _ in range(num_worker)]
    rewards =       [[] for _ in range(num_worker)]
    dones =         [[] for _ in range(num_worker)]
    predictions =   [[] for _ in range(num_worker)]

    state = [0 for _ in range(num_worker)]
    for worker_id, parent_conn in enumerate(parent_conns):
        state[worker_id] = parent_conn.recv()

    while train_state['episode'] < EPISODES:
        predictions_list = agent.Actor.actor_predict(np.reshape(state, [num_worker]+[_ for _ in state[0].shape]))
        actions_list = [np.random.choice(agent.action_space, p=i) for i in predictions_list]

        for worker_id, parent_conn in enumerate(parent_conns):
            parent_conn.send(actions_list[worker_id])
            action_onehot = np.zeros(agent.action_space.shape[0])
            action_onehot[actions_list[worker_id]] = 1
            actions[worker_id].append(action_onehot)
            predictions[worker_id].append(predictions_list[worker_id])

        for worker_id, parent_conn in enumerate(parent_conns):
            next_state, reward, realaction, done, reset, net_worth, episode_orders, date, currentprice, buy_and_hold_minus_net_worth_relative_to_initial_in_percent, avg_price_minus_avg_net_worth_relative_to_initial_in_percent = parent_conn.recv()
            states[worker_id].append(np.expand_dims(state[worker_id], axis=0))
            next_states[worker_id].append(np.expand_dims(next_state, axis=0))
            rewards[worker_id].append(reward)
            dones[worker_id].append(done)
            state[worker_id] = next_state

            if reset:
                train_state['episode'] += 1
                a_loss, c_loss = agent.replay(states[worker_id], actions[worker_id], rewards[worker_id], predictions[worker_id], dones[worker_id], next_states[worker_id])
                train_state['total_average'].append(net_worth)
                train_state['average'] = np.average(train_state['total_average'])
                
                agent.writer.add_scalar('Data/average net_worth', train_state['average'], train_state['episode'])
                agent.writer.add_scalar('Data/episode_orders', episode_orders, train_state['episode'])
                train_state['buy_and_hold_minus_net_worth_relative_to_initial_in_percent_queue'].append(buy_and_hold_minus_net_worth_relative_to_initial_in_percent)
                train_state['avg_price_minus_avg_net_worth_relative_to_initial_in_percent_queue'].append(avg_price_minus_avg_net_worth_relative_to_initial_in_percent)

                if(train_state['episode']%smoothing==0):
                    agent.writer.add_scalar('Data/buy_and_hold_minus_net_worth_relative_to_initial_in_percent_smoothed',np.average(train_state['buy_and_hold_minus_net_worth_relative_to_initial_in_percent_queue']), train_state['episode'])
                    agent.writer.add_scalar('Data/avg_price_minus_avg_net_worth_relative_to_initial_in_percent_smoothed', np.average(train_state['avg_price_minus_avg_net_worth_relative_to_initial_in_percent_queue']), train_state['episode'])

                agent.writer.add_scalar('Data/buy_and_hold_minus_net_worth_relative_to_initial_in_percent',buy_and_hold_minus_net_worth_relative_to_initial_in_percent, train_state['episode'])
                agent.writer.add_scalar('Data/avg_price_minus_avg_net_worth_relative_to_initial_in_percent', avg_price_minus_avg_net_worth_relative_to_initial_in_percent, train_state['episode'])
                print("episode: {:<5} worker: {:<1} net worth: {:<7.2f} average: {:<7.2f} orders: {}".format(train_state['episode'], worker_id, net_worth, train_state['average'], episode_orders))
                if train_state['episode'] > len(train_state['total_average']):
                    if train_state['best_average'] < train_state['average']:
                        train_state['best_average'] = train_state['average']
                        print("Saving model")
                        agent.save(score="{:.2f}".format(train_state['best_average']), args=[train_state['episode'], train_state['average'], episode_orders, a_loss, c_loss],train_state=train_state)
                    if(train_state['episode']%smoothing==0): #save agent every 20 episodes to speed up training
                        save_state=copy.copy(train_state)
                        save_state['total_average']=list(train_state['total_average'])
                        agent.save(train_state=save_state)
                        print("updated base model")


                
                states[worker_id] = []
                next_states[worker_id] = []
                actions[worker_id] = []
                rewards[worker_id] = []
                dones[worker_id] = []
                predictions[worker_id] = []
    return works


def load_agent(CustomAgent, folder, name="_Crypto_trader"):
    with open(folder+"/Parameters.json", "r") as json_file:
        params = json.load(json_file)
        if name != "":
            params["Actor name"] = f"{name}_Actor.h5"
            params["Critic name"] = f"{name}_Critic.h5"
        name = params["Actor name"][:-9]

        agent = CustomAgent(lookback_window_size=params["lookback window size"], optimizer=Adam,
        model=params["model"],log_name=folder,lr=params["lr"],epochs=params["epochs"], batch_size=params["batch size"], depth=params["depth"])

        agent.load(folder, name)
    with open(folder+"/training_state.json", "r") as json_file:
         train_state = json.load(json_file)
    return agent, train_state

def test_multiprocessing(CustomEnv, CustomAgent, test_df,test_df_nomalized, num_worker = 4, visualize=False, test_episodes=1000, folder="", name="_Crypto_trader", comment="", initial_balance=1000):
    
 with open(folder+"/Parameters.json", "r") as json_file:
    params = json.load(json_file)
    if name != "":
        params["Actor name"] = f"{name}_Actor.h5"
        params["Critic name"] = f"{name}_Critic.h5"
    name = params["Actor name"][:-9]

    agent = CustomAgent(lookback_window_size=params["lookback window size"], optimizer=Adam, depth=params["depth"], model=params["model"])

    agent.load(folder, name)
    works, parent_conns, child_conns = [], [], []
    average_net_worth = 0
    average_orders = 0
    no_profit_episodes = 0
    episode = 0
    rewards =        [[] for _ in range(num_worker)]
    net_worths =        [[] for _ in range(num_worker)]
    actions =        [[] for _ in range(num_worker)]
    dates =        [[] for _ in range(num_worker)]
    prices =        [[] for _ in range(num_worker)]

    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        #env = CustomEnv(test_df, initial_balance=initial_balance, lookback_window_size=agent.lookback_window_size)
        env = CustomEnv(df=test_df, df_normalized=test_df_nomalized, initial_balance=initial_balance, lookback_window_size=agent.lookback_window_size)

        work = Environment(idx, child_conn, env, training_batch_size=0, visualize=visualize, isTest=True)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    state = [0 for _ in range(num_worker)]
    for worker_id, parent_conn in enumerate(parent_conns):
        state[worker_id] = parent_conn.recv()

    while episode < test_episodes:
        predictions_list = agent.Actor.actor_predict(np.reshape(state, [num_worker]+[_ for _ in state[0].shape]))
        actions_list = [np.random.choice(agent.action_space, p=i) for i in predictions_list]

        for worker_id, parent_conn in enumerate(parent_conns):
            parent_conn.send(actions_list[worker_id])

        for worker_id, parent_conn in enumerate(parent_conns):
            next_state, reward, realaction, done, reset, net_worth, episode_orders, date, price = parent_conn.recv()
            state[worker_id] = next_state
            rewards[worker_id].append(reward)
            net_worths[worker_id].append(net_worth)
            actions[worker_id].append(realaction)
            dates[worker_id].append(date)
            prices[worker_id].append(price)
            if reset:
                episode += 1
                #print(episode, net_worth, episode_orders)
                average_net_worth += net_worth
                average_orders += episode_orders
                if net_worth < initial_balance: no_profit_episodes += 1 # calculate episode count where we had negative profit through episode
                print("episode: {:<5} worker: {:<1} net worth: {:<7.2f} average_net_worth: {:<7.2f} orders: {}".format(episode, worker_id, net_worth, average_net_worth/episode, episode_orders))
                if episode == test_episodes: break
            
    print("No profit episodes: {}".format(no_profit_episodes))
    # save test results to test_results.txt file
    with open("test_results.txt", "a+") as results:
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        results.write(f'{current_date}, {name}, test episodes:{test_episodes}')
        results.write(f', net worth:{average_net_worth/(episode+1)}, orders per episode:{average_orders/test_episodes}')
        results.write(f', no profit episodes:{no_profit_episodes}, comment: {comment}\n')
    
    # terminating processes after while loop
    works.append(work)
    dfs =[]
    order_dfs =[]
    for work in works:
        dict = {'Reward':  rewards[work.env_idx],
         'NetWorth': net_worths[work.env_idx], 
         'Action': actions[work.env_idx], 
         'Date': dates[work.env_idx],
         'CurrentPrice': prices[work.env_idx]
         } 
        dfs.append(work.env.df)
        order_dfs.append( pd.DataFrame.from_dict(dict))
        work.terminate()
        print('TERMINATED:', work)
        work.join()  

    print("FINISHED")
    return dfs, order_dfs
