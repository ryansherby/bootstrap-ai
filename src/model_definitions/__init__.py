import embedding_models as embedding_models
import transformer_models as transformer_models


embedding_model_dict = {model.name: model for model in embedding_models.__dict__.values() if isinstance(model, type) and issubclass(model, embedding_models.BaseModel)}
transformer_model_dict = {model.name: model for model in transformer_models.__dict__.values() if isinstance(model, type) and issubclass(model, transformer_models.BaseModel)}

def init_transformer_model(gpu_number, device, temperature, model_name):
    
    
    return transformer_model_dict[model_name](gpu_number=gpu_number, device=device, temperature=temperature)

def init_embedding_model(gpu_number, device, padding, dimensions, model_name):
    
    return embedding_model_dict[model_name](gpu_number=gpu_number, device=device, padding=padding, dimensions=dimensions)