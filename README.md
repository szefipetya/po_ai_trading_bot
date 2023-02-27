# Reinforcement Learning Bitcoin Trading Bot
## Functionalities:
- Download historical price action (yahoo finance apis)
- Executing dry trades based on the reinforcement learning model's output.
- Visualizing the backtesting results with chartJS
# More about the model:
- Trainable model with custom amount of indicator inputs
- Currently using 30 indicatiors with redundancy filtering
- Using Actor-Critic model for learing [[link](https://medium.com/intro-to-artificial-intelligence/the-actor-critic-reinforcement-learning-algorithm-c8095a655c14)]
### The model:
![image](https://user-images.githubusercontent.com/63967245/221597811-41645766-67b0-4e7d-80a2-4e7a300d37ad.png)
### The program in action:
![image](https://user-images.githubusercontent.com/63967245/221595847-297d1624-7a38-4599-b3e1-92dd54e93b9d.png)

### Tensorboard for performance monitoring
![image](https://user-images.githubusercontent.com/63967245/221599415-b0c43ce0-2529-4949-81a7-0d32db89a1f4.png)


