# Q-Learning Introductory Exercise

This was a weekend project I worked on. The Q-learning algorithm was based on [this tutorial](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/) and the idea to do this was based on [Code Bullet's YouTube video](https://www.youtube.com/watch?v=r428O_CMcpI).

Overall this was a great foray into Q-learning as well as `numpy` and `pygame`.

# Installation

## Pre-requisites
* Python (I am using v3.7.3)
* Pip (Python's package installer)
* `virtualenvwrapper` (or other environment manager -- highly recommended)

On Unix-based systems install the Python dependencies with

```bash
> pip install -r requirements
```

Finally, you can run with

```bash
> python q.py
```

# Limitations
* This was done over a long weekend so there will be many more things that can be done. I'd recommend using creating maps that are square as the animation was coded for square maps.
* Make sure to resize the animation grid squares and images if you are going to use a map with more than 10 rows/columns.