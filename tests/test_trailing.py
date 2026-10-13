import unittest
from core.position.trailing import (
    TrailingConfig,
    calculate_proposed_sl,
    validate_sl_modification,
    normalize_price
)

class TestTrailing(unittest.TestCase):

    def test_normalize_price(self):
        self.assertAlmostEqual(normalize_price(1.234567, 0.00001), 1.23457)
        self.assertAlmostEqual(normalize_price(150.123, 0.01), 150.12)
        
    def test_calculate_breakeven_long(self):
        config = TrailingConfig(breakeven_activation=50.0, breakeven_offset=5.0)
        # Price moved 60 points in our favor (long)
        sl = calculate_proposed_sl(direction=1, price_open=1000.0, price_current=1060.0, current_sl=900.0, config=config)
        self.assertEqual(sl, 1005.0)
        
    def test_calculate_breakeven_short(self):
        config = TrailingConfig(breakeven_activation=50.0, breakeven_offset=5.0)
        # Price moved 60 points in our favor (short)
        sl = calculate_proposed_sl(direction=-1, price_open=1000.0, price_current=940.0, current_sl=1100.0, config=config)
        self.assertEqual(sl, 995.0)

    def test_progressive_breakeven_long(self):
        config = TrailingConfig(breakeven_activation=50.0, breakeven_offset=5.0, profit_locking_ratio=0.5)
        # Step 1: Profit = 50 (trigger reached, extra = 0) -> SL = 1000 + 5 = 1005
        sl1 = calculate_proposed_sl(direction=1, price_open=1000.0, price_current=1050.0, current_sl=900.0, config=config)
        self.assertEqual(sl1, 1005.0)

        # Step 2: Profit = 70 (extra = 20, locks 5 + 10 = 15) -> SL = 1015
        sl2 = calculate_proposed_sl(direction=1, price_open=1000.0, price_current=1070.0, current_sl=sl1, config=config)
        self.assertEqual(sl2, 1015.0)

        # Step 3: Profit = 100 (extra = 50, locks 5 + 25 = 30) -> SL = 1030
        sl3 = calculate_proposed_sl(direction=1, price_open=1000.0, price_current=1100.0, current_sl=sl2, config=config)
        self.assertEqual(sl3, 1030.0)

        # Step 4: Retracement to profit = 80 (SL must NEVER loosen, stays at 1030)
        sl4 = calculate_proposed_sl(direction=1, price_open=1000.0, price_current=1080.0, current_sl=sl3, config=config)
        self.assertEqual(sl4, 1030.0)

    def test_progressive_breakeven_short(self):
        config = TrailingConfig(breakeven_activation=50.0, breakeven_offset=5.0, profit_locking_ratio=0.5)
        # Step 1: Profit = 50 -> SL = 1000 - 5 = 995
        sl1 = calculate_proposed_sl(direction=-1, price_open=1000.0, price_current=950.0, current_sl=1100.0, config=config)
        self.assertEqual(sl1, 995.0)

        # Step 2: Profit = 70 -> extra = 20, locks 5 + 10 = 15 -> SL = 985
        sl2 = calculate_proposed_sl(direction=-1, price_open=1000.0, price_current=930.0, current_sl=sl1, config=config)
        self.assertEqual(sl2, 985.0)

        # Step 3: Profit = 100 -> extra = 50, locks 5 + 25 = 30 -> SL = 970
        sl3 = calculate_proposed_sl(direction=-1, price_open=1000.0, price_current=900.0, current_sl=sl2, config=config)
        self.assertEqual(sl3, 970.0)

        # Step 4: Retracement to 920 (profit 80) -> SL must never widen, stays at 970
        sl4 = calculate_proposed_sl(direction=-1, price_open=1000.0, price_current=920.0, current_sl=sl3, config=config)
        self.assertEqual(sl4, 970.0)

    def test_calculate_fixed_trailing_long(self):
        config = TrailingConfig(mode="fixed", activation_distance=50.0, trailing_distance=20.0)
        sl = calculate_proposed_sl(direction=1, price_open=1000.0, price_current=1060.0, current_sl=1000.0, config=config)
        # Trail distance = 20, so price_current(1060) - 20 = 1040
        self.assertEqual(sl, 1040.0)
        
    def test_profit_locking_trailing(self):
        config = TrailingConfig(mode="profit_locking", profit_locking_ratio=0.5)
        # Profit = 100
        sl = calculate_proposed_sl(direction=1, price_open=1000.0, price_current=1100.0, current_sl=950.0, config=config)
        self.assertEqual(sl, 1050.0)

    def test_never_loosen_sl(self):
        config = TrailingConfig(mode="fixed", activation_distance=10.0, trailing_distance=10.0)
        # We already have an awesome SL at 1045. The calculated trail would be 1050 - 10 = 1040, which is worse.
        sl = calculate_proposed_sl(direction=1, price_open=1000.0, price_current=1050.0, current_sl=1045.0, config=config)
        self.assertEqual(sl, 1045.0) # Should keep current SL

    def test_validate_sl_modification_valid(self):
        # proposed=1010, current=990, stops_level=10, point=1, tick_size=1
        # current_price_for_sl = 1050. Dist from 1050 to 1010 = 40 >= 10*1. Valid.
        res = validate_sl_modification(1, 1010.1, 990.0, 1050.0, 1.0, 1.0, 10, 5)
        self.assertEqual(res, 1010.0)
        
    def test_validate_sl_modification_invalid_widen(self):
        # Long position: trying to move SL from 990 to 980
        res = validate_sl_modification(1, 980.0, 990.0, 1050.0, 1.0, 1.0, 10, 5)
        self.assertIsNone(res)

    def test_validate_sl_modification_freeze_level(self):
        # Distance from 1050 to 1045 is 5.
        # But freeze level is 10. So it should reject.
        res = validate_sl_modification(1, 1045.0, 990.0, 1050.0, 1.0, 1.0, 5, 10)
        self.assertIsNone(res)
        
    def test_validate_sl_modification_insignificant(self):
        # Modifying SL by less than tick size
        res = validate_sl_modification(1, 990.001, 990.0, 1050.0, 0.01, 0.01, 0, 0)
        self.assertIsNone(res)

if __name__ == '__main__':
    unittest.main()
