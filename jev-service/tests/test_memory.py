import unittest

from jev_service.memory import MemoryService


class MemoryTests(unittest.TestCase):
    def test_inferred_fact_is_not_persisted(self) -> None:
        memory = MemoryService()
        proposal = memory.propose_write("c1", "用户很生气", source="inference", confidence=0.99)
        self.assertFalse(proposal.eligible)
        self.assertEqual(memory.retrieve("c1", "生气"), [])

    def test_confirmed_fact_can_be_persisted_and_corrected(self) -> None:
        memory = MemoryService()
        proposal = memory.propose_write("c1", "用户偏好周末见面", source="confirmed", confidence=0.95)
        self.assertTrue(proposal.eligible)
        evidence = memory.retrieve("c1", "周末见面")
        self.assertEqual(len(evidence), 1)
        record_id = evidence[0].id
        record = memory.feedback(record_id, "correct", "用户偏好周六下午见面")
        self.assertTrue(record.confirmed)
        self.assertIn("周六", record.content)

    def test_invalid_memory_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MemoryService().propose_write("c1", "bad", record_type="emotion")
