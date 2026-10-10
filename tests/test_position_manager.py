import unittest
from unittest.mock import MagicMock
from core.mt5.types import MT5Position, OrderResult, RetCode
from core.position.manager import PositionManager, ManualPolicy, PositionState

class TestPositionManager(unittest.TestCase):
    def setUp(self):
        self.mock_adapter = MagicMock()
        self.mock_execution = MagicMock()
        self.manager = PositionManager(
            adapter=self.mock_adapter,
            execution_engine=self.mock_execution,
            bot_magic=888999,
            default_manual_policy=ManualPolicy.OBSERVE_ONLY
        )

    def test_reconcile_new_positions(self):
        # Setup mock positions (one bot, one manual)
        bot_pos = MT5Position(
            ticket=1, symbol="EURUSD", type=0, volume=0.1, price_open=1.1, price_current=1.1,
            time=123, magic=888999
        )
        manual_pos = MT5Position(
            ticket=2, symbol="EURUSD", type=1, volume=0.2, price_open=1.1, price_current=1.1,
            time=123, magic=0
        )
        self.mock_adapter.get_positions.return_value = [bot_pos, manual_pos]
        
        self.manager.reconcile()
        
        # Verify local state
        self.assertIn(1, self.manager._state)
        self.assertIn(2, self.manager._state)
        self.assertEqual(self.manager._state[1].policy, ManualPolicy.FULL_MANAGEMENT)
        self.assertEqual(self.manager._state[2].policy, ManualPolicy.OBSERVE_ONLY)

    def test_reconcile_closed_positions(self):
        # Start with one known position
        self.manager._state[1] = PositionState(
            ticket=1, symbol="EURUSD", direction=1, volume=0.1, magic=888999, policy=ManualPolicy.FULL_MANAGEMENT
        )
        # Broker returns empty list -> position closed
        self.mock_adapter.get_positions.return_value = []
        self.manager.reconcile()
        self.assertNotIn(1, self.manager._state)

    def test_reconcile_volume_change(self):
        self.manager._state[1] = PositionState(
            ticket=1, symbol="EURUSD", direction=1, volume=1.0, magic=888999, policy=ManualPolicy.FULL_MANAGEMENT
        )
        # Broker returns same position with lower volume
        partial_pos = MT5Position(
            ticket=1, symbol="EURUSD", type=0, volume=0.5, price_open=1.1, price_current=1.1,
            time=123, magic=888999
        )
        self.mock_adapter.get_positions.return_value = [partial_pos]
        self.manager.reconcile()
        self.assertEqual(self.manager._state[1].volume, 0.5)

if __name__ == '__main__':
    unittest.main()
