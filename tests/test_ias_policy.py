"""Environment-owned IAS policy tests (no Isaac stage required)."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

try:
    from env_wrapper import PushEnv
except Exception:  # pragma: no cover
    PushEnv = None


@unittest.skipIf(PushEnv is None, "当前 Python 环境缺少环境依赖")
class IasPolicyTests(unittest.TestCase):
    def _env(self):
        env = PushEnv.__new__(PushEnv)
        env.num_envs = 1
        env.invalid_actions_by_env = [set()]
        env.ias_exhausted = [False]
        return env

    def test_empty_actions_are_masked_until_scene_changes(self):
        env = self._env()
        for action in (0, 4, 7):
            self.assertFalse(env._update_invalid_action_state(0, action, "empty"))
        self.assertEqual(env.get_invalid_actions(0), [0, 4, 7])
        env._update_invalid_action_state(0, 2, "push")
        self.assertEqual(env.get_invalid_actions(0), [])

    def test_eighth_distinct_empty_action_exhausts_episode(self):
        env = self._env()
        exhausted = False
        for action in range(8):
            exhausted = env._update_invalid_action_state(0, action, "empty")
        self.assertTrue(exhausted)
        self.assertTrue(env.ias_exhausted[0])
        self.assertEqual(env.get_invalid_actions(0), list(range(8)))


if __name__ == "__main__":
    unittest.main()
