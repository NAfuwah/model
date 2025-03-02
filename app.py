from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from stable_baselines3 import PPO
import gym
from gym import spaces

app = Flask(__name__)

class BovineRespiratoryDiseaseEnv(gym.Env):
    def __init__(self, data):
        super(BovineRespiratoryDiseaseEnv, self).__init__()
        self.data = data
        self.current_step = 0
        num_features = data.shape[1] - 1
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_features,), dtype=np.float32)
    
    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step, :-1].values.astype(np.float32)
    
    def step(self, action):
        observation = self.data.iloc[self.current_step, :-1].values.astype(np.float32)
        reward = 10 if action == 1 and self.data.iloc[self.current_step]['BRD_Total'] > 0 else -5
        self.current_step += 1
        done = self.current_step >= len(self.data)
        return observation, reward, done, {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    file = request.files['file']
    data = pd.read_excel(file)
    data['BRD_Total'] = data['BRD_Total'].fillna(0).astype(int)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns] / (data[numeric_columns].max() + 1e-8)
    data = data.fillna(0)
    
    env = BovineRespiratoryDiseaseEnv(data)
    model = PPO('MlpPolicy', env, learning_rate=0.0001, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("brd_model.zip")
    return jsonify({'message': 'Model trained and saved successfully!'})

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    data = request.json
    if not data or "features" not in data or len(data["features"]) != 5:
        return jsonify({'error': 'Please provide exactly 5 feature values'}), 400
    
    if not os.path.exists("brd_model.zip"):
        return jsonify({'error': 'Model not found. Train the model first!'}), 400

    model = PPO.load("brd_model.zip")
    features = np.array(data["features"]).reshape(1, -1)
    action, _ = model.predict(features, deterministic=True)

    return jsonify({'prediction': 'Intervene' if action == 1 else 'No intervention needed'})

if __name__ == '__main__':
    app.run(Host='0.0.0.0' port='10000')
