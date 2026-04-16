import os
import yaml
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class ConfigLoader:
    def __init__(self, config_path: str = "config/config.yml"):
        self.config_path = config_path
        self._config = self._load_and_interpolate()
    
    def _interpolate_env_vars(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(config_dict, dict):
            return {k: self._interpolate_env_vars(v) for k, v in config_dict.items()}
        elif isinstance(config_dict, str):
            if config_dict.startswith("{{") and config_dict.endswith("}}"):
                env_var = config_dict[2:-2].strip()
                return os.getenv(env_var, "")
            return config_dict
        elif isinstance(config_dict, list):
            return [self._interpolate_env_vars(item) for item in config_dict]
        else:
            return config_dict
    
    def _load_and_interpolate(self) -> Dict[str, Any]:
        with open(self.config_path, 'r', encoding='utf-8') as file:
            raw_config = yaml.safe_load(file)
        return self._interpolate_env_vars(raw_config)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        return self._config.get(model_name, {})

    @property
    def jailbreak_setting(self):
        return self._config.get('jailbreak_setting', {})
    
    @property
    def target_model(self) -> Dict[str, Any]:
        return self.get_model_config('target_model')
    
    @property
    def attacker(self) -> Dict[str, Any]:
        return self.get_model_config('attacker')
    
    @property
    def judger(self) -> Dict[str, Any]:
        return self.get_model_config('judger')
    
    @property
    def optimizer(self) -> Dict[str, Any]:
        return self.get_model_config('optimizer')
    
    @property
    def summary_model(self) -> Dict[str, Any]:
        return self.get_model_config('summary_model')
    
    @property
    def refusal_checker(self) -> Dict[str, Any]:
        return self.get_model_config('refusal_checker')

    # @property
    # def evaluator(self) -> Dict[str, Any]:
    #     return self.get_model_config('evaluator')

    @property
    def judge(self) -> Dict[str, Any]:
        return self.get_model_config('judge')

    @property
    def knowledge_extractor(self) -> Dict[str, Any]:
        return self.get_model_config('knowledge_extractor')

    @property
    def safeguard(self) -> Dict[str, Any]:
        return self.get_model_config('safeguard')

    @property
    def dataset_path(self) -> str:
        return self._config.get('paths').get('dataset')

    @property
    def logs_dir(self) -> str:
        return self._config.get('paths').get('logs_dir')

    @property
    def result_dir(self) -> str:
        return self._config.get('paths').get('result_dir')

    @property
    def successful_example_cache(self) -> str:
        return self._config.get('paths').get('successful_example_cache')

        
    @property
    def out_dir(self) -> str:
        return self._config.get('paths').get('out_dir')

    @property
    def paper_name(self) -> str:
        return self._config.get('context').get('paper_name')
    @property
    def context_json_path(self) -> str:
        return self._config.get('context').get('context_json_path')


if __name__ == '__main__':
    config_loader = ConfigLoader()
    print(config_loader.successful_example_cache)
    print(config_loader.result_dir)
    print(config_loader.logs_dir)
    print(config_loader.dataset_path)