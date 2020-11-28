import os
import pickle
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
from tensorflow.keras.models import load_model


base_path = Path(os.path.dirname(__file__)) / 'models'

class Predict(ABC):
    @abstractmethod
    def predict():
        pass


class PredictNN(Predict):
    def __init__(self, all_elements, info_path, model_path, **kwargs):
        super().__init__()

        features, domain, x_mean, x_std, y_mean, y_std = \
            pickle.load(open(info_path, "rb"))

        self.info_dict = {
            'domain': domain,
            'features': features,
            'x_mean': x_mean,
            'x_std': x_std,
            'y_mean': y_mean,
            'y_std': y_std,
            'model': load_model(model_path),
            'index': [all_elements.index(el) for el in features],
            'not_index': [
                all_elements.index(el) for el in all_elements if el not in features
            ],
        }

    def get_domain(self):
        return self.info_dict['domain']

    def is_within_domain(self, population_dict):
        atomic_array = population_dict['atomic_array']
        x = atomic_array[:, self.info_dict['not_index']]

        is_within_domain = x.sum(axis=1).astype(bool)

        for n, el in enumerate(self.info_dict['features']):
            x = atomic_array[:,n]
            logic1 = np.less(x, self.info_dict['domain'][el][0])
            logic2 = np.greater(x, self.info_dict['domain'][el][1])
            not_in_domain = np.logical_or(logic1, logic2)
            is_within_domain = np.logical_and(
                is_within_domain, np.logical_not(not_in_domain))

        return is_within_domain

    def predict(self, population_dict):
        atomic_array = population_dict['atomic_array']
        x = atomic_array[:, self.info_dict['index']]

        x_mean = self.info_dict['x_mean']
        x_std = self.info_dict['x_std']
        y_mean = self.info_dict['y_mean']
        y_std = self.info_dict['y_std']
        model = self.info_dict['model']

        y_scaled = model.predict((x - x_mean) / x_std)
        y = (y_scaled * y_std + y_mean).flatten()

        return y


class PredictGlassTransitionNN(PredictNN):
    def __init__(self, all_elements, **kwargs):
        model_path = base_path / 'model_glass_transition_temperature.h5'
        info_path = base_path / 'Tg.p'
        super().__init__(all_elements, info_path, model_path)


class PredictRefractiveIndexNN(PredictNN):
    def __init__(self, all_elements, **kwargs):
        model_path = base_path / 'model_refractive_index.h5'
        info_path = base_path / 'nd.p'
        super().__init__(all_elements, info_path, model_path)


class PredictAbbeNN(PredictNN):
    def __init__(self, all_elements, **kwargs):
        model_path = base_path / 'model_abbe.h5'
        info_path = base_path / 'abbe.p'
        super().__init__(all_elements, info_path, model_path)


class PredictCost(Predict):
    def __init__(self, weight_cost_dict, compound_list, all_elements, **kwargs):
        super().__init__()

        self.all_elements = all_elements
        self.weight_cost = np.array([
            weight_cost_dict.get(comp, np.nan) for comp in compound_list
        ]).reshape(-1, 1)

    def get_domain(self):
        return {el: [0,1] for el in self.all_elements}

    def is_within_domain(self, population_dict):
        return np.ones(len(population_dict['atomic_array'])).astype(bool)

    def predict(self, population_dict):
        cost = (
            population_dict['population_weight'] @ self.weight_cost
        ).flatten()
        return cost
