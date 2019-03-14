# battle-of-the-agents
Battle of the Agents - balancing experience, curiosity &amp; imitation

#Running Instructions:
1. Training: (Change env to Breakout for breakout runs.)
	a. Lunar Lander Student DQN: 
	```
	python curious-agent.py --mode=train --model=student --curiosity_during_expert_phase=False --env=LunarLander
	```

	b. Lunar Lander Expert DQN:
	```
	python curious-agent.py --mode=train --model=expert --curiosity_during_expert_phase=False --env=LunarLander
	```

	c. Lunar Lander Curious Student DQN: 
	```
	python curious-agent.py --mode=train --model=curious_student --curiosity_during_expert_phase=False --env=LunarLander
	```

	d. Lunar Lander Expert DQN:
	```
	python curious-agent.py --mode=train --model=curious_expert --curiosity_during_expert_phase=False --env=LunarLander
	```

2. Testing: Similar to above. Only change mode.

#Installation Requirements
1. Install Prereqs:

MacOS: 

```
brew install cmake openmpi
```

Ubuntu
```
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

2. Create Virtual Env:

Assuming we are using conda

Conda:
```
conda create --name cs234-project python=3.6.7
conda activate cs234-project
conda install matplotlib
```

Normal Virtualenv:

```
sudo pip install virtualenv      # This may already be installed
virtualenv cs234-project --python=python3.6 # Create a virtual environment
source cs234-project/bin/activate         # Activate the virtual environment 
```

3. Install Python dependencies:
```
pip install -r requirements.txt
```
