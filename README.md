# CS 370: Pirate Intelligent Agent

A deep Q-learning agent that teaches a "pirate" to find the treasure in an 8×8 maze from any open starting cell.

## Files

| File | Description |
|------|-------------|
| `ProjectTwo.ipynb` | Jupyter notebook containing maze setup, neural network, and the Q-learning training loop |
| `TreasureMaze.py` | The environment: maze matrix, movement rules, rewards, and game state (provided) |
| `GameExperience.py` | Experience replay memory used to train the network (provided) |
| `requirements.txt` | Python dependencies |

## What I was given vs. what I wrote

**Provided starter code:**
- `TreasureMaze.py`, which defines the maze, valid moves, rewards/penalties, and win/lose conditions
- `GameExperience.py`, which stores past episodes and generates training batches
- Notebook: the maze matrix, the `show()` visualizer, `play_game()`, `completion_check()`, the `build_model()` network, the `train_step()` gradient update, and the setup portion of `qtrain()` (hyperparameters, environment, target network, and replay object)

**My work:** I implemented the core deep Q-learning loop in `qtrain()`, the section marked `#TODO`/`START_HERE`. For each epoch, my code:

1. Resets the pirate to a randomly chosen free cell and observes the starting state
2. Chooses each move with an **epsilon-greedy** policy: a random valid action (exploration), otherwise the action with the highest predicted Q-value (exploitation)
3. Applies the action, collects the new state and reward, and records wins and losses in `win_history`
4. Stores each `[state, action, reward, next_state, game_over]` transition in experience replay
5. Samples a training batch from memory and runs `train_step()` to update the network, tracking the average loss per epoch
6. Decays epsilon over time, dropping it to the minimum once the win rate passes 90% so the agent relies on what it has learned
7. Stops early only when the win rate reaches 100% **and** `completion_check()` confirms the pirate can reach the treasure from every free cell

The agent reached a 100% win rate and passed the completion check at epoch 431 (which took about 8 minutes of CPU training).

## Reflection

### What do computer scientists do, and why does it matter?

Computer scientists turn messy, real-world problems into precise models a computer can work with, then build and test solutions against those models. In this project, "find the treasure" became a problem solved with a neural network approach. The agent was never told where the treasure was; it discovered a strategy from experience. That same approach drives many real world problems, including recommendation systems which I have worked with previously. It matters because good modeling and careful engineering decide whether those tools actually help people.

### How do I approach a problem as a computer scientist?

Before I write code, I try to fully understand the problem and the existing codebase. For this project that meant reading `TreasureMaze.py` and `GameExperience.py` so I knew what the methods expect/return. From there I broke the algorithm into small steps (following the pseudocode), implemented them one at a time, and checked output as I went until the pirate agent was able to complere the maze.  However, just because a problem seems "solved" doesn't mean it's the best version of the code, or that it hits every edge case. An early version of my project had the pirate solve the maze on some runs but fail to solve it in 1000 epochs on others. Fixing the code so that the pirate completes the maze on each run of the code cell took extra time but it's important to me to have code that behaves predictably and has results that can be reproduced.

### What are my ethical responsibilities to the end user and the organization?

My responsibility is to build systems that are honest about what they can do, work reliably, and respect the people affected by them. For machine learning in particular, that means validating models thoroughly rather than reporting the best metric, and being transparent about limitations. It also means considering the ethics of a problem, and protecting the data a system learns from. A real recommender or navigation system may be trained on people's behavior, and that data deserves care and privacy. 

 
