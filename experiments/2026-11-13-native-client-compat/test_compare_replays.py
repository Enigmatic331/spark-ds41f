import unittest
from compare_replays import compare


def waves():
    for condition in ("alone", "four", "four-prefill"):
        for repeat in (1,2):
            ids = [34170] if condition == "alone" else range(34168,34172)
            yield dict(condition=condition, repeat=repeat, model="fixture",
                       injected=None if condition!="four-prefill" else
                       dict(needle_exact=True,http_status=200,done_count=1,error=None,model_exact=True,finish_reason="stop"),
                       rows=[dict(case_id=f"resident-{n}", request_fingerprint=f"prompt-{n}",
                                  content_sha256=f"answer-{n}", count=1948 if n==34170 else 2048,
                                  transport_ok=True, progress_during=True,gaps_during={"max_s":.5},
                                  sequence_exact=n!=34170) for n in ids])


class Compare(unittest.TestCase):
    def test_consistent_but_wrong(self):
        r = compare(list(waves()))
        self.assertTrue(r["same_focal_answer_across_conditions"])
        self.assertFalse(r["full_task_pass"])

    def test_load_dependent_change(self):
        w = list(waves())
        w[-1]["rows"][2]["content_sha256"]="different"
        self.assertFalse(compare(w)["same_focal_answer_across_conditions"])

    def test_nonidentical_resident_input(self):
        w = list(waves())
        w[-1]["rows"][0]["request_fingerprint"]="different"
        with self.assertRaises(ValueError):
            compare(w)

    def test_slow_is_not_an_output_mismatch(self):
        w=list(waves())
        w[-1]["rows"][2]["gaps_during"]["max_s"]=6
        r=compare(w)
        self.assertTrue(r["same_focal_answer_across_conditions"])
        self.assertFalse(r["cells"][-1]["during_prefill_timing_ok"])


if __name__ == "__main__":
    unittest.main()
