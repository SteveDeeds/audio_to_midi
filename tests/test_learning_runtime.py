import unittest

from core.learning_runtime import (
    InferencePriority,
    InferenceRequest,
    LearningDomain,
    LearningRuntimeState,
    RuntimeToken,
    TrainingLabel,
)


def label(key: str) -> TrainingLabel:
    return TrainingLabel(LearningDomain.PITCH, key, True)


def request(request_id: int, priority: InferencePriority,
            model: int = 0) -> InferenceRequest:
    return InferenceRequest(
        request_id,
        priority,
        RuntimeToken(session=1, model=model, edits=0, settings=0),
        0.0,
        4.0,
    )


class TestLearningRuntimeState(unittest.TestCase):
    def test_labels_arriving_during_training_form_next_batch(self) -> None:
        state = LearningRuntimeState()
        state.submit_labels([label("first")])

        first = state.take_training_batch(base_model_version=3)
        self.assertIsNotNone(first)
        self.assertEqual([item.key for item in first.labels], ["first"])

        state.submit_labels([label("second"), label("third")])
        self.assertIsNone(state.take_training_batch(base_model_version=3))

        state.finish_training(published_model_version=4)
        state.request_inference(request(1, InferencePriority.VISIBLE, model=4))
        visible = state.take_inference_request()
        self.assertIsNotNone(visible)
        state.accept_inference(visible)

        second = state.take_training_batch(base_model_version=4)
        self.assertIsNotNone(second)
        self.assertEqual(
            [item.key for item in second.labels], ["second", "third"]
        )

    def test_next_batch_waits_for_published_model_viewport(self) -> None:
        state = LearningRuntimeState()
        state.submit_labels([label("first")])
        self.assertIsNotNone(state.take_training_batch(base_model_version=0))
        state.submit_labels([label("later")])
        state.finish_training(published_model_version=1)

        state.request_inference(request(1, InferencePriority.VISIBLE, model=0))
        stale_visible = state.take_inference_request()
        state.accept_inference(stale_visible)
        self.assertIsNone(state.take_training_batch(base_model_version=1))

        state.request_inference(request(2, InferencePriority.VISIBLE, model=1))
        current_visible = state.take_inference_request()
        state.accept_inference(current_visible)
        self.assertIsNotNone(state.take_training_batch(base_model_version=1))

    def test_inference_priority_and_visible_coalescing(self) -> None:
        state = LearningRuntimeState()
        state.request_inference(request(1, InferencePriority.OFFSCREEN))
        state.request_inference(request(2, InferencePriority.VISIBLE))
        state.request_inference(request(3, InferencePriority.VISIBLE))
        state.request_inference(request(4, InferencePriority.EXPORT))

        visible = state.take_inference_request()
        self.assertEqual(visible.request_id, 3)
        state.accept_inference(visible)
        export = state.take_inference_request()
        self.assertEqual(export.request_id, 4)
        state.accept_inference(export)
        offscreen = state.take_inference_request()
        self.assertEqual(offscreen.request_id, 1)
        state.accept_inference(offscreen)

    def test_export_blocks_queued_training(self) -> None:
        state = LearningRuntimeState()
        state.submit_labels([label("queued")])
        state.request_inference(request(1, InferencePriority.EXPORT))

        self.assertIsNone(state.take_training_batch(base_model_version=0))
        export = state.take_inference_request()
        self.assertEqual(export.priority, InferencePriority.EXPORT)
        self.assertIsNone(state.take_training_batch(base_model_version=0))
        state.accept_inference(export)
        self.assertIsNotNone(state.take_training_batch(base_model_version=0))

    def test_failed_batch_is_restored_before_new_labels(self) -> None:
        state = LearningRuntimeState()
        state.submit_labels([label("failed")])
        batch = state.take_training_batch(base_model_version=0)
        state.submit_labels([label("new")])

        state.fail_training(batch)

        self.assertIsNone(state.take_training_batch(base_model_version=0))
        state.retry_failed_training()
        retry = state.take_training_batch(base_model_version=0)
        self.assertEqual([item.key for item in retry.labels], ["failed", "new"])


if __name__ == "__main__":
    unittest.main()