def Random_games(env, train_episodes=50, train_mode=False, training_batch_size=500):
    average_net_worth = 0

    for episode in range(train_episodes):
        state = env.reset(env_steps_size=training_batch_size)

        while True:
            env.render(False)

            action = np.random.randint(3, size=1)[0]

            state, reward, done = env.step(action)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break

    print("average_net_worth:", average_net_worth/train_episodes)


def Single_game(env):
    state = env.reset()
    average_net_worth = 0

    while True:
        env.render()

        action = np.random.randint(3, size=1)[0]

        state, reward, done = env.step(action,True)

        if env.current_step == env.end_step:
            average_net_worth += env.net_worth
            print("net_worth:", env.net_worth)
            break
    print("average_net_worth:", average_net_worth)
    renderer.render_performance_vbt(env)
