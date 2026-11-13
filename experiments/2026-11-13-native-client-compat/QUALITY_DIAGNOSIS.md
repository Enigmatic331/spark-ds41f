# Visual and counting investigation on November 13

The candidate uses Christopher Owen's pinned deployment and its imported vLLM
and TileLang stack. The historical Etherscan prompts, screenshots and original
scorer come from the site's October 18 synthetic corpus. The supplemental
assertion, diagnostic tools and report are AI-assisted site changes. Claude
reviewed the experimental design. No upstream publication is proposed.

## What the full screenshot answers say

Five isolated native requests used the original two 1440×900 screenshots,
the archived prompt, temperature 0.6, top-p 0.95, seed 0 and a fresh cache
salt. Each completed normally and used 1,774 input tokens. All five reported
the displaced transaction table. Four omitted the header search defect. The
fifth reported a yellow cursor there but never described the shortened field.
Some answers placed a white obstruction over the ETH price. Pixel comparison
puts the white cover at the search control in the top strip. The ordinary
greedy full-frame answer also missed the clipped field.

The archived scorer checks its keyword stems across the **whole** JSON.
That fifth answer contains “search” in the cursor entry and “truncated” in the
table entry, so the historical checker gave it full credit. The supplemental
assertion now checks the change associated with the search control in a single
reported regression. The historical score is still recorded. The previously
recorded failed answer was stored only as a hash. Reproductions, rather than
that historical turn's unobserved wording, establish the current failure.

On the installed CPU preprocessing path, all three fixture images yielded
bit-for-bit identical BF16 patch tensors and image-token roles against the
checkpoint reference. Both paths produced a 65×103 vision patch grid, a
22×35 aligner grid and 794 image-span tokens per screenshot. This test ends
at the input to the neural tower. It does not measure hidden states, tower
weights or end-to-end numerical equivalence.

The header-only comparison correctly identified the field narrowing and the
placeholder being cut off. Full screenshots plus three hand-selected crops
also got all three seeded changes. These exploratory crops use known fixture
regions and cannot substitute for an unaided acceptance pass. TensorFold
passed both fresh uncached full-frame controls including the region-scoped
search check. Sample sizes are small. The exact source of variance between
engines has not been established.

Geometric quadrants covering each entire page did not reliably restore
the comparison. One answer associated a shifted search bar with clipped
ETH text, another omitted the search bar, and another described vertical
placeholder clipping. The planted mask removes the right part of the field.
The identical-image quadrant control passed. Explicit high reasoning effort
and two full-page reading-order reminders also missed it. The region
checker now distinguishes the observed cursor, other-clause clipping, and
the explicitly wrong axis; it remains a heuristic secondary to reading the
failed synthetic answer.

## Counting experiment

The hash-checked historical instruction asks for 170962510 through 170964557
inclusive. The deterministic native answer reproduced its historical hash.
After 1,948 correct integers it ended at 170964457. Output ID 1 is the
checkpoint's EOS. Usage included 9,739 output tokens of a 12,288-token limit.
Replaying the exact prompt IDs and all generated IDs except that EOS with
one uncued next-token step returned EOS again. Its log probability was
−0.00961 (99.04%) and the next alternative was comma, −4.69126 (0.92%).
The online SSE pipeline cannot explain the observed termination.

TensorFold's instrumented isolated control also returned actual EOS ID 1.
Its 1,548 integers were consecutive through 170964057, 500 early, using
7,739 completion tokens with the same 12,288-token limit and zero cached
prompt tokens. The log-probability counterfactuals below were measured on
Christopher. The differing stop positions across engines remain unexplained.

| Counterfactual, temperature zero | Next-token evidence |
|---|---|
| Immediately after 170964456 | Comma 99.9962%, EOS 0.0037%. |
| Immediately after 170964457, original target 170964557 | EOS 99.0436%, comma 0.9175%. |
| Same 1,948-number prefix, change only the target to 170964558 | Comma 99.9970%, EOS 0.0029%. Both raw next-token prompts have 9,794 IDs. |
| Original target; supply exactly one comma after 170964457 | Generated 170964458, 170964459, 170964460 in order. |
| Original target; supply the correct continuation through 170964557 | EOS 99.9904%. |
| Shift both endpoints by 26, start a new generation | All 2,048 values through 170964583 were exact. This matches integer count; the output used 8,192 IDs, so token length differs. |

The evidence localizes the failure to the learned stopping decision. The false
endpoint is 100 below the requested one and shares its last two decimal
digits. Changing a single target digit abolishes that false stop with the
same generated prefix. It does not identify a particular attention head or
prove a universal suffix rule. Treat exact long enumeration as a known
quality diagnostic. The historical request and its failure remain recorded.
The checkpoint tokenizer splits the nine-digit numbers into three groups.
The first two are `170` and `964` in every endpoint comparison. The last
groups `057`, `457`, `557` and `558` have different token IDs (33369, 23369,
27278 and 27277). The near-match is not a tokenizer collision.
One exploratory attempt to bias EOS returned HTTP 400. The speculative
configuration rejects non-empty logit bias; the later comma test supplied a
token in the prompt and left the sampler unchanged.

The existing app bindings are sufficient for a qualified in-place cutover.
The enabled Sembang provider and the active Tokios connector were both
verified against the original upstream URL; app registrations and auth were
left alone. An unauthenticated request to each actual ingress returned 401.
The diagnostic was staging-only. No authenticated user request through
either app is claimed on the candidate.
