from gym.envs.registration import register

register('fraud_detect-v0',
         entry_point='fraud_detect.envs:FraudDetectEnv')
