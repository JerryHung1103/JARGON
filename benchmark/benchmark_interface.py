from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd
class BenchmarkAdapter(ABC):

    def __init__(self, df):
        self.df = df

    @abstractmethod
    def get_query(self, row) -> str:
        pass
    
    @abstractmethod
    def get_category(self, row) -> str:
        pass
    
    @abstractmethod
    def get_metadata(self, row) -> Dict[str, Any]:
        pass

    def get_all_data(self):
        results = []
        for idx, row in self.df.iterrows():
            result = {
                'query': self.get_query(row),
                'category': self.get_category(row),
                'metadata': self.get_metadata(row)
            }
            results.append(result)
        return results

    def get_top_n_by_category(self, n_samples=5, category_column=None):

        if category_column is None:
           
            sample_row = self.df.iloc[0]
            category_value = self.get_category(sample_row)

            for col in self.df.columns:
                if self.df[col].iloc[0] == category_value:
                    category_column = col
                    break
        
        if category_column is None or category_column not in self.df.columns:
            raise ValueError(f"Unable to determine category column. Please explicitly specify the category_column parameter.")
        
        sampled_df = self.df.groupby(category_column).head(n_samples).reset_index(drop=True)
        return self._df_to_results(sampled_df)

    def get_random_n_by_category(self, n_samples=5, category_column=None, random_state=42):
        if category_column is None:
            category_column = self._infer_category_column()
        
        sampled_df = self.df.groupby(category_column).apply(
            lambda x: x.sample(n=min(n_samples, len(x)), random_state=random_state)
        ).reset_index(drop=True)
        return self._df_to_results(sampled_df)

    def get_category_stats(self, category_column=None):

        if category_column is None:
            category_column = self._infer_category_column()
        
        category_counts = self.df[category_column].value_counts()
        return category_counts.to_dict()

    def get_distinct_categories(self, category_column=None):

        if category_column is None:
            category_column = self._infer_category_column()
        
        return sorted(self.df[category_column].unique())

    def _infer_category_column(self):

        sample_row = self.df.iloc[0]
        category_value = self.get_category(sample_row)
        
        for col in self.df.columns:
            if self.df[col].iloc[0] == category_value:
                return col
        
        raise ValueError("Unable to infer category column name. Please explicitly specify the category_column parameter.")

    def _df_to_results(self, df):
        results = []
        for idx, row in df.iterrows():
            result = {
                'query': self.get_query(row),
                'category': self.get_category(row),
                'metadata': self.get_metadata(row)
            }
            results.append(result)
        return results

class JailbreakBenchAdapter(BenchmarkAdapter):

    def get_query(self, row) -> str:
        return row['Goal']
    
    def get_category(self, row) -> str:
        return row['Category']
    
    def get_metadata(self, row) -> Dict[str, Any]:
        return {
            'target': row['Target'],
            'behavior': row['Behavior'],
            'source': row['Source'],
            'index': row.get('Index', '')
        }

class HarmBenchAdapter(BenchmarkAdapter):
    
    def get_query(self, row) -> str:
        return row['Behavior']
    
    def get_category(self, row) -> str:
        return row['SemanticCategory']
    
    def get_metadata(self, row) -> Dict[str, Any]:
        return {
            'functional_category': row['FunctionalCategory'],
            'semantic_category': row['SemanticCategory'],
            'tags': row.get('Tags', ''),
            'behavior_id': row.get('BehaviorID', '')
        }


class MedSafetyBenchAdapter(BenchmarkAdapter):
    
    def get_query(self, row) -> str:
        return row['harmful_medical_request']
    
    def get_category(self, row) -> str:
        return None # No category available
    
    def get_metadata(self, row) -> Dict[str, Any]:
        return {
            'safe_response': row['safe_response'],
            'source': row['source'],
        }

    
class YourOwnBench(BenchmarkAdapter):
    # Define Your Own Evaluation
    pass

if __name__ == '__main__':
    from huggingface_hub import login
    from dotenv import load_dotenv
    import os
    load_dotenv()
    hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    login(token=hf_token)
    import pandas as pd

    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/israel-adewuyi/med-safety-bench-reproduced/" + splits["train"])
    MedSafetyBench_adapter = MedSafetyBenchAdapter(df)
    samples = MedSafetyBench_adapter.get_all_data()
    print(len(samples))

    jailbreakBench_df = pd.read_csv('hf://datasets/JailbreakBench/JBB-Behaviors/data/harmful-behaviors.csv')
    jailbreak_adapter = JailbreakBenchAdapter(jailbreakBench_df)
    jailbreak_samples = jailbreak_adapter.get_all_data()
    print(len(jailbreak_samples))

    
