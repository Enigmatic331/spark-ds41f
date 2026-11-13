import unittest
from visual_region_check import search_field_clipped


class RegionCheck(unittest.TestCase):
    def test_unrelated_table_clipping_does_not_explain_search_cursor(self):
        entries = [{"region":"Transactions","change":"The To column is truncated."},
                   {"region":"Top navigation search bar","change":"A yellow cursor is visible."}]
        self.assertFalse(search_field_clipped({"regressions":entries}))

    def test_reduced_width_in_navigation_region(self):
        entry = {"region":"Top navigation","change":"Search bar width is drastically reduced."}
        self.assertTrue(search_field_clipped({"regressions":[entry]}))

    def test_negative_control(self):
        self.assertFalse(search_field_clipped({"regressions":[]}))

    def test_unrelated_price_clipping_in_same_entry(self):
        entry = {"region":"Header", "change":"The top search bar is shifted left, and the ETH price text is clipped."}
        self.assertFalse(search_field_clipped({"regressions":[entry]}))

    def test_wrong_axis_does_not_identify_seeded_defect(self):
        entry = {"region":"Header Search Bar", "change":"The placeholder text is vertically clipped."}
        self.assertFalse(search_field_clipped({"regressions":[entry]}))


if __name__ == "__main__":
    unittest.main()
