from ...models import ProbabilisticModelType
from ...models.gpflow import BatchReparametrizationSampler 
from ..interface import AcquisitionFunctionBuilder, AcquisitionFunction
from ...observer import OBJECTIVE, INEQUALITY_CONSTRAINT_PREFIX
from typing import Optional

import tensorflow as tf

class BatchExpectedConstrainedImprovement(AcquisitionFunctionBuilder[ProbabilisticModelType]):
    def __init__(self,
                 sample_size: int, 
                 threshold: float,
                 constraint_builder: AcquisitionFunctionBuilder[ProbabilisticModelType]):
        self._sample_size = sample_size
        self._threshold = threshold
        self._constraint_builder = constraint_builder
        self._constraint_fn: Optional[AcquisitionFunction] = None
        self._expected_improvement_fn: Optional[AcquisitionFunction] = None
        self._constrained_improvement_fn: Optional[AcquisitionFunction] = None
        self._best_valid_observation = None

    def prepare_acquisition_function(self, models, datasets):

        satisfied_mask = tf.constant(value=True, shape=datasets[OBJECTIVE].observations.shape)
        for tag, dataset in datasets.items():
            if tag.startswith(INEQUALITY_CONSTRAINT_PREFIX):
                constraint_vals = dataset.observations
                valid_constraint_vals = constraint_vals <= 0
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)

        if tf.reduce_sum(tf.cast(satisfied_mask, tf.int32)) != 0:
            objective_vals = datasets[OBJECTIVE].observations
            valid_y = tf.boolean_mask(objective_vals, satisfied_mask)
            self._best_valid_observation = tf.math.reduce_min(valid_y)
        print(f"Best Valid Observation: {self._best_valid_observation}")

        objective_model = models[OBJECTIVE]
        objective_dataset = datasets[OBJECTIVE]

        samplers = {
            tag: BatchReparametrizationSampler(
                self._sample_size, model
            )
            for tag, model in models.items()
        }

        self._constraint_fn = self._constraint_builder.prepare_acquisition_function(
            models, datasets=datasets
        )
        pof = self._constraint_fn(objective_dataset.query_points[:, None, ...])
        is_feasible = pof >= 0.5

        if not tf.reduce_any(is_feasible):
            # If no points are feasible, set "best feasible point" to a large value
            # to force the acquisition function to resort to probability of feasibility
            eta = 1e6
        else:
            mean, _ = objective_model.predict(objective_dataset.query_points)
            eta = tf.reduce_min(tf.boolean_mask(mean, is_feasible), axis=0)
        print(f"eta: {eta}")

        def batch_efi(at):
            samples = {
                tag: tf.squeeze(sampler.sample(at), -1)
                for tag, sampler in samplers.items()
            }

            feasible_mask = tf.constant(value=True, shape=samples[OBJECTIVE].shape)  # [N, S, B]
            print(f"Feasible Mask Shape: {feasible_mask.shape} should be [N, S, B]")
            for tag, sample in samples.items():
                if tag.startswith(INEQUALITY_CONSTRAINT_PREFIX):
                    feasible_mask = tf.logical_and(feasible_mask, sample <= self._threshold)
            improvement = tf.where(
                feasible_mask, tf.maximum(eta - samples[OBJECTIVE], 0.0), 0.0
            )  # [N, S, B]
            batch_improvement = tf.reduce_max(improvement, axis=-1)  # [N, S]
            return tf.reduce_mean(
                batch_improvement, axis=-1, keepdims=True
            )  # [N, 1]

        return batch_efi
