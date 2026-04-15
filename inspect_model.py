import pickle
import pandas as pd
import joblib

def inspect_model(model_path):
    try:
        # HMM models are often saved with joblib or pickle
        try:
            model = joblib.load(model_path)
        except:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        print(f"Model Type: {type(model)}")
        
        if isinstance(model, dict):
            print(f"Dictionary Keys: {list(model.keys())}")
            print(f"Selected Features: {model.get('selected_features')}")
            print(f"State Map: {model.get('state_map')}")
            # Try to find the model object in the dict
            for key, value in model.items():
                print(f"Key '{key}' type: {type(value)}")
        
        # Check for number of components
        if hasattr(model, 'n_components'):
            print(f"Components (States): {model.n_components}")
        
        # Check for features if it's a GaussianHMM or similar
        if hasattr(model, 'n_features'):
             print(f"Expected Number of Features: {model.n_features}")
             
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    inspect_model("mars_golden_model.pkl")
