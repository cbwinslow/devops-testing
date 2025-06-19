#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from typing import List, Dict, Tuple, Optional
import asyncio
import aiohttp
import logging
import os
import json
from datetime import datetime
from dataclasses import dataclass
import yaml
from cryptography.fernet import Fernet
import pickle
import zmq
import zmq.asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('federated_learning')

@dataclass
class FederatedModel:
    name: str
    version: str
    architecture: Dict
    weights: np.ndarray
    metrics: Dict
    timestamp: str

class FederatedLearningNode:
    def __init__(self, config_path: str = '~/devops-testing/config/federated_learning.yaml'):
        self.config_path = os.path.expanduser(config_path)
        self.load_config()
        self.setup_encryption()
        self.setup_communication()
        self.initialize_models()

    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = {
                'node': {
                    'id': 'node1',
                    'role': 'worker'
                },
                'network': {
                    'aggregator_address': 'localhost:5555',
                    'pub_port': 5556,
                    'sub_port': 5557
                },
                'training': {
                    'batch_size': 32,
                    'epochs': 10,
                    'min_clients': 3
                },
                'security': {
                    'encryption_enabled': True,
                    'key_rotation_interval': 3600
                }
            }

    def setup_encryption(self):
        """Set up encryption for secure model transfer"""
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def setup_communication(self):
        """Set up ZMQ communication"""
        context = zmq.asyncio.Context()
        
        # Setup publisher
        self.publisher = context.socket(zmq.PUB)
        self.publisher.bind(f"tcp://*:{self.config['network']['pub_port']}")
        
        # Setup subscriber
        self.subscriber = context.socket(zmq.SUB)
        self.subscriber.connect(f"tcp://{self.config['network']['aggregator_address']}")
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    def initialize_models(self):
        """Initialize federated learning models"""
        # Create core model architecture
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='softmax')
        ])
        
        # Define federated types
        self.input_spec = tf.TensorSpec(shape=[None, 10], dtype=tf.float32)
        self.output_spec = tf.TensorSpec(shape=[None, 8], dtype=tf.float32)

    async def participate_in_round(self, round_number: int):
        """Participate in a federated learning round"""
        # Get local data
        local_data = await self.get_local_data()
        
        # Train on local data
        local_model = await self.train_local_model(local_data)
        
        # Send model updates
        await self.send_model_updates(local_model, round_number)
        
        # Receive aggregated model
        aggregated_model = await self.receive_aggregated_model()
        
        # Update local model
        await self.update_local_model(aggregated_model)

    async def get_local_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get local training data"""
        # Implement data collection logic
        return np.random.rand(100, 10), np.random.rand(100, 8)

    async def train_local_model(self, data: Tuple[np.ndarray, np.ndarray]) -> FederatedModel:
        """Train model on local data"""
        X, y = data
        
        # Train model
        history = self.model.fit(
            X, y,
            batch_size=self.config['training']['batch_size'],
            epochs=self.config['training']['epochs'],
            validation_split=0.2
        )
        
        # Create federated model
        model = FederatedModel(
            name=self.config['node']['id'],
            version=str(datetime.now().timestamp()),
            architecture=self.model.get_config(),
            weights=self.model.get_weights(),
            metrics=history.history,
            timestamp=datetime.now().isoformat()
        )
        
        return model

    async def send_model_updates(self, model: FederatedModel, round_number: int):
        """Send model updates to aggregator"""
        # Serialize model
        model_bytes = pickle.dumps(model)
        
        # Encrypt if enabled
        if self.config['security']['encryption_enabled']:
            model_bytes = self.cipher.encrypt(model_bytes)
        
        # Send update
        message = {
            'type': 'model_update',
            'round': round_number,
            'node_id': self.config['node']['id'],
            'model': model_bytes
        }
        
        await self.publisher.send_multipart([
            b'model_update',
            json.dumps(message).encode()
        ])

    async def receive_aggregated_model(self) -> FederatedModel:
        """Receive aggregated model from server"""
        message = await self.subscriber.recv_multipart()
        data = json.loads(message[1].decode())
        
        if data['type'] == 'aggregated_model':
            model_bytes = data['model']
            
            # Decrypt if needed
            if self.config['security']['encryption_enabled']:
                model_bytes = self.cipher.decrypt(model_bytes)
            
            return pickle.loads(model_bytes)
        
        return None

    async def update_local_model(self, aggregated_model: FederatedModel):
        """Update local model with aggregated weights"""
        if aggregated_model:
            self.model.set_weights(aggregated_model.weights)

    async def run(self):
        """Run federated learning node"""
        round_number = 0
        while True:
            try:
                await self.participate_in_round(round_number)
                round_number += 1
                
                # Rotate encryption key if needed
                if (round_number * self.config['training']['epochs'] * 
                    self.config['training']['batch_size']) % self.config['security']['key_rotation_interval'] == 0:
                    self.setup_encryption()
                
            except Exception as e:
                logger.error(f"Error in federated learning round: {e}")
            
            await asyncio.sleep(60)  # Wait between rounds

class FederatedLearningAggregator:
    def __init__(self, config_path: str = '~/devops-testing/config/federated_learning.yaml'):
        self.config_path = os.path.expanduser(config_path)
        self.load_config()
        self.setup_communication()
        self.initialize_state()

    def load_config(self):
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = {
                'aggregation': {
                    'algorithm': 'fedavg',
                    'min_clients': 3,
                    'timeout': 300
                }
            }

    def setup_communication(self):
        """Set up ZMQ communication"""
        context = zmq.asyncio.Context()
        
        # Setup subscriber for updates
        self.subscriber = context.socket(zmq.SUB)
        self.subscriber.bind(f"tcp://*:{self.config['network']['sub_port']}")
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Setup publisher for aggregated models
        self.publisher = context.socket(zmq.PUB)
        self.publisher.bind(f"tcp://*:{self.config['network']['pub_port']}")

    def initialize_state(self):
        """Initialize aggregator state"""
        self.current_round = 0
        self.received_models = {}
        self.model_versions = []

    async def aggregate_models(self, models: List[FederatedModel]) -> FederatedModel:
        """Aggregate received models"""
        if len(models) < self.config['aggregation']['min_clients']:
            return None
        
        # Compute average weights
        weights = [model.weights for model in models]
        avg_weights = np.mean(weights, axis=0)
        
        # Create aggregated model
        return FederatedModel(
            name='aggregated',
            version=str(datetime.now().timestamp()),
            architecture=models[0].architecture,
            weights=avg_weights,
            metrics={},
            timestamp=datetime.now().isoformat()
        )

    async def run(self):
        """Run federated learning aggregator"""
        while True:
            try:
                # Collect model updates
                message = await self.subscriber.recv_multipart()
                data = json.loads(message[1].decode())
                
                if data['type'] == 'model_update':
                    model = pickle.loads(data['model'])
                    self.received_models[data['node_id']] = model
                
                # Check if we have enough models
                if len(self.received_models) >= self.config['aggregation']['min_clients']:
                    # Aggregate models
                    aggregated_model = await self.aggregate_models(list(self.received_models.values()))
                    
                    if aggregated_model:
                        # Send aggregated model
                        message = {
                            'type': 'aggregated_model',
                            'round': self.current_round,
                            'model': pickle.dumps(aggregated_model)
                        }
                        
                        await self.publisher.send_multipart([
                            b'aggregated_model',
                            json.dumps(message).encode()
                        ])
                        
                        self.current_round += 1
                        self.received_models.clear()
                
            except Exception as e:
                logger.error(f"Error in aggregation: {e}")
            
            await asyncio.sleep(1)

def main():
    # Parse arguments to determine role
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['node', 'aggregator'], required=True)
    args = parser.parse_args()
    
    if args.role == 'node':
        node = FederatedLearningNode()
        asyncio.run(node.run())
    else:
        aggregator = FederatedLearningAggregator()
        asyncio.run(aggregator.run())

if __name__ == '__main__':
    main()

