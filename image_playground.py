import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

RESIZE_FACTOR = 2

class ImageProcessor:
    # I think we can play around with this more
    # Possibilities 
    def __init__(self):
        pass

    def transform(self, frame):
        frame = tf.cast(frame, tf.float32) / 255.0
        # gray_scaled = tf.image.rgb_to_grayscale(frame)
        resized = tf.image.resize(frame, [frame.shape[0] // RESIZE_FACTOR, frame.shape[1] // RESIZE_FACTOR], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.squeeze(resized).numpy()
    
im = ImageProcessor()

env = gym.make('CarRacing-v3', render_mode='rgb_array')

obs, _ = env.reset()
obs = im.transform(obs)

for i in range(100):
    obs, _, _, _, _ = env.step(env.action_space.sample())
    obs = im.transform(obs)
    plt.imshow(obs)
    plt.savefig(f'{i}')


env.close()