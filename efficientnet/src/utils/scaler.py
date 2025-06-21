import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class NutrientScaler:
    """
    Utility class to scale nutrient values for training and inference
    """
    def __init__(self, scaler_type='standard', feature_range=(0, 1)):
        """
        Initialize the scaler
        
        Args:
            scaler_type (str): Type of scaler ('standard' or 'minmax')
            feature_range (tuple): Range for MinMaxScaler (only used if scaler_type is 'minmax')
        """
        self.scaler_type = scaler_type
        self.feature_range = feature_range
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=feature_range)
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
            
        self.is_fitted = False
        self.output_names = ['mass', 'calories', 'fat', 'carbs', 'protein']
        
    def fit(self, data):
        """
        Fit the scaler to the data
        
        Args:
            data (np.ndarray): Nutrient data to fit the scaler [N, 5]
        """
        self.scaler.fit(data)
        self.is_fitted = True
        return self
        
    def transform(self, data):
        """
        Transform the data using the fitted scaler
        
        Args:
            data (np.ndarray or torch.Tensor): Nutrient data to transform [N, 5]
            
        Returns:
            Same type as input: Scaled nutrient data
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler is not fitted yet. Call fit() first.")
        
        is_torch_tensor = False
        import_torch = False
        
        try:
            import torch
            import_torch = True
            is_torch_tensor = isinstance(data, torch.Tensor)
        except ImportError:
            pass
        
        if is_torch_tensor:
            data_numpy = data.cpu().numpy()
        else:
            data_numpy = data
            
        scaled_data = self.scaler.transform(data_numpy)
        
        if is_torch_tensor and import_torch:
            return torch.tensor(scaled_data, dtype=data.dtype, device=data.device)
        
        return scaled_data
        
    def inverse_transform(self, scaled_data):
        """
        Inverse transform scaled data back to original scale
        
        Args:
            scaled_data (np.ndarray or torch.Tensor): Scaled nutrient data [N, 5]
            
        Returns:
            Same type as input: Original scale nutrient data
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler is not fitted yet. Call fit() first.")
            
        is_torch_tensor = False
        import_torch = False
        
        try:
            import torch
            import_torch = True
            is_torch_tensor = isinstance(scaled_data, torch.Tensor)
        except ImportError:
            pass
        
        if is_torch_tensor:
            scaled_numpy = scaled_data.cpu().numpy()
        else:
            scaled_numpy = scaled_data
            
        original_data = self.scaler.inverse_transform(scaled_numpy)
        
        if is_torch_tensor and import_torch:
            return torch.tensor(original_data, dtype=scaled_data.dtype, device=scaled_data.device)
        
        return original_data
    
    def fit_transform(self, data):
        """
        Fit the scaler to the data and transform it
        
        Args:
            data (np.ndarray or torch.Tensor): Nutrient data to fit and transform [N, 5]
            
        Returns:
            Same type as input: Scaled nutrient data
        """
        self.fit(data)
        return self.transform(data)
    
    def save(self, file_path):
        """
        Save the fitted scaler to a file
        
        Args:
            file_path (str): Path to save the scaler
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'scaler_type': self.scaler_type,
                'feature_range': self.feature_range,
                'is_fitted': self.is_fitted,
                'output_names': self.output_names
            }, f)
    
    @classmethod
    def load(cls, file_path):
        """
        Load a fitted scaler from a file
        
        Args:
            file_path (str): Path to the saved scaler file
            
        Returns:
            NutrientScaler: Loaded scaler
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        instance = cls(scaler_type=data['scaler_type'], feature_range=data['feature_range'])
        instance.scaler = data['scaler']
        instance.is_fitted = data['is_fitted']
        instance.output_names = data['output_names']
        
        return instance
    
    def get_params(self):
        """
        Get the parameters of the fitted scaler
        
        Returns:
            dict: Scaler parameters
        """
        if self.scaler_type == 'standard':
            return {
                'mean': self.scaler.mean_,
                'scale': self.scaler.scale_
            }
        elif self.scaler_type == 'minmax':
            return {
                'min': self.scaler.min_,
                'scale': self.scaler.scale_
            } 