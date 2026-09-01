# Prompt-Suite Quality Evaluation — Search & Chat (2026-09-01)

Full-prompt sweep of every prompt in `backend/scripts/prompt_suite_prompts.txt`
(100 prompts, 10 sections × 10) against the live backend on the deployment box
(`ubuntu@13.206.46.160`, gunicorn port 8001), followed by independent,
fresh-context LLM judging of the raw outputs.

**Verdict at a glance**

| Suite | Avg score | Distribution (5→1) | Errors | Fallbacks | Avg latency |
|---|---|---|---|---|---|
| Search (`/search`) | **3.55 / 5** | 26, 29, 21, 22, 2 | 0 | n/a | 156 ms |
| Chat (`/api/chat`) | **3.59 / 5** | 30, 27, 24, 10, 9 | 0 (after retry) | 9 / 100 | 2,296 ms |

Both systems are solid on entity lookup, firm-portfolio recall and
sector×investor intersection queries (avg 4+ in those sections) and both break
down on the same axis: **recency intent** ("latest / this week / this month /
recently") and **aggregation intent** ("biggest / top / compare / most active").

---

## 1. Methodology

- **Runner**: `backend/scripts/prompt_suite_eval.py` (branch
  `ops/prompt-suite-eval`, merged to main). Parses the 100-prompt fixture,
  substitutes deterministic real entities for placeholders (Paytm, Zomato,
  Lenskart, Zerodha, Swiggy, PhonePe, Razorpay, Byju's; Blackstone/KKR;
  Peak XV/Accel/Blume; Kunal Shah/Nithin Kamath; fintech/edtech/healthcare;
  quick commerce/UPI; year=2025).
- **Search suite**: `GET /search?q=…&top_k=8`, full result list captured
  (title, date, industry/dealtype, score, summary excerpt), plus `note`,
  `cached`, latency. 100/100 HTTP-ok.
- **Chat suite**: fresh session per prompt, `POST /api/chat/sessions/{id}/messages`,
  full answer + sources + cost + latency captured, session deleted after. 10
  prompts hit transient `503 LLM temporarily unavailable` (clustered in
  consecutive pairs — provider rate-limit bursts, not app bugs); all 10
  succeeded on retry and were merged in. Total billed cost: **₹0.187**
  (~₹0.0019/answer) — negligible against the $5 daily cap.
- **Judging**: two independent fresh-context subagents (one per suite), given
  only the raw result JSON and a 0–5 rubric. Judge rubric: 5 = top results
  directly and specifically satisfy the query intent; 3 = partial/generic;
  1 = essentially off-topic / canned refusal; 0 = error. Chat judge additionally
  checked grounding vs. source titles, staleness framing, fallback usage and
  the dataviz contract.

## 2. Search suite — detailed findings

### 2.1 Section summary (judge)

| Section | Avg | #scores ≤2 | Notable failure pattern |
|---|---|---|---|
| Private Equity | 4.00 | 2 | "Biggest/this year" superlatives |
| Industry-Specific | 4.00 | 1 | Staleness on "latest" (id 68 → 2014 wafer plants) |
| Venture Capital | 3.90 | 3 | Co-occurrence + superlative intents |
| Company & Brand | 3.80 | 1 | Intent-type misses (founders, competitors, business model) |
| Sector Analysis | 3.60 | 1 | "Consumer internet" ambiguity; stale trends |
| Mergers & Acquisitions | 3.50 | 2 | **Relation reversal (id 48)**; superlatives |
| People & Leadership | 3.50 | 3 | Interview/appointment intent; recency |
| Funding & Startup Deals | 3.30 | 3 | Superlatives + recency |
| Search, Analysis & Custom | 3.10 | 3 | Analytical/aggregation asks get single items |
| Trends & Market Intelligence | **2.80** | 5 | "This week/today" recency totally unmet |

### 2.2 Praise-worthy (consistently 5/5)

1. **Firm-portfolio recall** — Blackstone (id 27), KKR (28), Accel (33), Peak XV (32): every result firm-specific.
2. **Named-acquirer M&A** — PhonePe (id 8: Zopper Retail, ZestMoney, GigIndia), Paytm (47): textbook.
3. **Sector×VC intersections** — VCs-in-AI (34), fintech investors (35), D2C (64), climate-tech (69).
4. **Person-name variant handling** — Kamath brothers / Nikhil / Nithin (72); Kunal Shah + CRED (71).
5. **Honest signalling** — `note` flags track real quality; 0 errors, 0 empty pages in 200 requests across both suites.

### 2.3 Ten worst search prompts

| id | Prompt (intent) | Score | Evidence |
|---|---|---|---|
| 82 | "Summarize today's most important VCCircle news" | 1 | 2012–2017 corporate-launch announcements |
| 2 | "Latest news about Banyan Netfaqs Pvt Ltd" | 1 | Entity confusion → Banyan Tree Finance / Banyan Green |
| 48 | "Who acquired Zomato?" | 2 | **Relation reversed**: all 8 results are Zomato-as-buyer |
| 79 | "Show me interviews with Nithin Kamath" | 2 | No interviews; funding news + unrelated interviews |
| 75 | "Which founders raised the most capital recently?" | 2 | No ranking; NEA fund raise / Tyke pre-seed |
| 38 | "Startups backed by both Accel and Peak XV" | 2 | Every result Peak XV-only; Accel never co-featured |
| 81 | "Biggest business stories this week" | 2 | 2024/2020 most-read flashbacks |
| 78 | "Business leaders making headlines this month" | 2 | All 2015 content (Arnab Goswami, NHB CEO) |
| 22 | "Biggest PE investments this year" | 2 | Top-3 all score 0.00 (compensation survey, Oman exit) |
| 42 | "Biggest acquisitions this year" | 2 | Goldman global M&A volume stats ×2 |

### 2.4 Perf / ops

- Latency avg/median 156 ms (min 107, max 227) — very consistent.
- 41/100 flagged by `note` (34 "weakly related", 7 date-fallback variants) — flags align with judge scores.
- 0/100 cache hits (cold eval run) — fine for benchmarking.

## 3. Chat suite — detailed findings

### 3.1 Section summary (judge)

| Section | Avg | #scores ≤2 | Dominant failure pattern |
|---|---|---|---|
| Company & Brand | **4.30** | 0 | — (entity lookup works) |
| Private Equity | 4.30 | 1 | One stale-as-current buyout list (id 25) |
| Venture Capital | 4.30 | 1 | One stale-as-current list (id 37) |
| Mergers & Acquisitions | 3.90 | 2 | Old deals presented undated as "recent" |
| Funding & Startup Deals | 3.60 | 1 | Mixed-vintage deals for recency asks |
| Sector Analysis | 3.70 | 1 | Thin-source answers (id 54: one 2023 article) |
| Industry-Specific | 3.30 | 1 | **False-positive weak-retrieval refusal (id 66)** |
| Search, Analysis & Custom | 2.90 | 3 | 1-source retrieval can't serve "everything/compare/find-all" |
| People & Leadership | 2.90 | 5 | **Refusals on answerable people queries** |
| Trends & Market Intelligence | **2.70** | 4 | Zero-source refusals for "this week/now" |

### 3.2 Correctness & contract findings

- **Hallucinations: 0 confirmed**, 1 suspect (id 4: "Sequoia Capital India" line weakly cited).
- **Dataviz contract: 0 violations** — no ` ```dataviz ` block appeared (correct: no prompt explicitly requested a visual). Contract-adjacent noise: id 69 says "see the table below" with no table; ~42/100 answers end with an unsolicited "want a table?" offer.
- **Grounding is strong**: answers cite `[n]` markers matching source titles; Flashback/year-recap sources produce precise numbers (ids 24, 31, 41, 49, 86).
- **Weak-retrieval gate false positives**: ids 66, 73, 76 refused despite clearly on-topic sources (e.g. id 76 refused while source[0] was literally an ACC CEO-appointment article). 9/100 total fallbacks (6 with sources attached, 3 with zero sources: 78, 81, 100); ≥5 of the 9 look answerable.
- **Stale-data framing**: good answers state data vintage explicitly (ids 9, 12, 14, 56: "as of December 2021", "week ended 22 Nov 2024"); bad answers present 2009–2019 deals in present tense (ids 25, 37, 43, 44, 75, 87).

### 3.3 Ten worst chat prompts

| id | Prompt | Score | Evidence |
|---|---|---|---|
| 78 | "Business leaders making headlines this month" | 1 | 0 sources → canned refusal |
| 81 | "Biggest business stories this week" | 1 | 0 sources → canned refusal |
| 100 | "Developments I may have missed this week" | 1 | 0 sources → canned refusal |
| 73 | "Recent news involving Kunal Shah" | 1 | Refused while all 3 sources are about him (CRED $450mn, Expertrons) |
| 76 | "Latest CEO appointments" | 1 | Refused while source[0] = ACC appoints Harish Badami as CEO |
| 66 | "Deals in logistics & supply chain" | 1 | Refused while sources include TVS Supply Chain $100mn, NewCold |
| 87 | "Emerging startups to watch" | 2 | Presents 2010's Ecologix and 2015's Zapr as today's emerging startups |
| 82 | "Today's most important VCCircle news" | 1 | Retrieval returned 2012/2017 sources → refusal |
| 79 | "Interviews with Nithin Kamath" | 1 | Bare refusal; should have offered his latest news instead |
| 43 | "Companies acquired recently" | 2 | 2009–2019 deals (Nikkos, QEI, FuGen) listed as "recent", undated |

### 3.4 Perf / ops

- Latency avg 2,296 ms / median 2,242 ms (fallback answers ~114 ms).
- Cost: ₹0.1875 total → ₹0.0019/answer (≈ $0.00002/answer at ₹95.6/$); avg 5,682 prompt / 303 completion tokens per turn.
- `n_sources`: mode 8 (55%), then 10 (7%); 23 answers with 1–4 sources including 7 single-source and 3 zero-source. Single-source answers average ~2.9 vs ~4.4 for 7–10-source answers — retrieval depth correlates directly with answer quality.

## 4. Cross-suite signals

- **Recency is the #1 root cause in both suites** — ~13 search prompts and ~13 chat prompts asking latest/this week/recently get 2009–2019 "evergreen" articles even though the corpus demonstrably has 2024–26 coverage (search ids 12/22/32/86 and chat ids 12/22/32/86 prove it).
- **Aggregation/superlatives**: "biggest / top / most active / compare" get isolated news items instead of ranked or synthesized output in both suites.
- **Relation direction**: search returned Zomato-as-buyer for "Who acquired Zomato?" (id 48), but the chat LLM recovered and correctly stated the sources show Zomato as acquirer (chat id 48 = 5/5). The rerank layer, not the LLM, is the weak link for role-reversed queries.
- **Entity disambiguation**: Banyan Netfaqs → Banyan Tree Finance confusion in search (id 2 = 1) is at least honestly handled in chat (id 2 = 3, names the gap).
- **Weak-retrieval gate**: it protects against nonsense but fired false positives in chat on answerable queries — it's the single biggest chat-only defect (6 of 9 lowest chat scores are refusals).

## 5. Consolidated recommendations (ranked by expected impact)

1. **Recency-aware retrieval** — date-decay boost / implicit date filter for "latest / recent / this week / this month / now" prompts, plus a "data as of <date>" caveat in chat. One change lifts both suites (~26 low scores).
2. **Fix the weak-retrieval gate** — recalibrate the threshold (ids 66/73/76 were refused with on-topic sources attached); when sources exist, have the LLM summarize what was found instead of the canned refusal.
3. **Temporal no-hit handler** — "this week/today/headlines" queries with no semantic hits should fall back to a recency-sorted "latest N articles" stream instead of 0 sources → refusal.
4. **Aggregation intent synthesis** — detect "biggest/top/most/compare" asks and synthesize ranked output across multiple retrieved articles (also fixes search's superlative misses).
5. **Relation-direction awareness** — parse acquirer/target roles for "who acquired X" queries (search-side; chat already recovers).
6. **Multi-entity intersection** — "backed by both A and B" / "X vs Y" need intersection logic (id 38 search = 2, id 93 chat = 2).
7. **Staleness guardrail in CHAT_PROMPT** — require vintage disclosure when sources are >12 months old; forbid present-tense framing of old deals.
8. **Output hygiene** — drop dangling "see the table below" (id 69), trim unsolicited "want a table?" offers (42% of answers), normalize citation style (`[1, 5]` vs `[1][5]`).
9. **Content-type awareness** — "interviews / appointments / founders / competitors / business model" are intent modifiers; currently they degrade to generic company news (search ids 5, 6, 10, 79; chat 79).

## 6. Artifacts

| File | Contents |
|---|---|
| `docs/evals/prompt_suite_20260901T073756Z_1041239_search.json` | Raw search results: 100 prompts × top-8 results w/ scores, dates, facets, latency, notes |
| `docs/evals/prompt_suite_20260901_chat_final_100.json` | Raw chat results: 100 full answers + sources + tokens + cost + latency (incl. 10 retried) |
| `backend/scripts/prompt_suite_eval.py` | Runner (merged to main via `ops/prompt-suite-eval`) |
| `backend/scripts/prompt_suite_prompts.txt` | The 100-prompt fixture |

Runner output logs: `/tmp/prompt_suite_search.log`, `/tmp/prompt_suite_chat_retry.log` on the box (ephemeral).

## Appendix A — Search per-prompt scores

### Company & Brand (avg 3.80)
| id | score | reason |
|---|---|---|
| 1 | 5 | All top-3 Paytm-specific (GMV, anchor round, IPO) |
| 2 | 1 | Entity confusion: returned Banyan Tree Finance/Banyan Green, not Banyan Netfaqs |
| 3 | 4 | 2/3 Paytm funding; "Refrens raises from Paytm founder" off-intent |
| 4 | 5 | All Zomato investor/financing news (Glade Brook, Alibaba, exit) |
| 5 | 3 | Lenskart-specific but funding news; founders (Bansal) never surfaced |
| 6 | 4 | All Zerodha, but merchant banking/founder fund ≠ business model |
| 7 | 4 | GMV growth + loss-narrowing on-topic; 2016 funding piece stale |
| 8 | 5 | Exact: Zopper Retail, ZestMoney, GigIndia — all PhonePe acquisitions |
| 9 | 4 | All Razorpay but valuation/fundraising, not financial performance |
| 10 | 3 | All Byju's news, zero competitor content — intent missed |

### Funding & Startup Deals (avg 3.30)
| id | score | reason |
|---|---|---|
| 11 | 4 | Real funding deals (Bigspoon, $10bn report) but 2022-24 "latest" |
| 12 | 3 | Weekly digest format matches, but dates 2019/2023/2024 ≠ "this week" |
| 13 | 2 | $10bn aggregate + 2012 VC fund; "$9.5mn Skeps" below $10M bar |
| 14 | 2 | Weekly/Gulf digests and Series A; no biggest-rounds ranking |
| 15 | 3 | Strong Q2-2025 Series A analysis; other two marginal (MENA, trends) |
| 16 | 4 | All genuine seed rounds but 2015-2020 dates ≠ "recently" |
| 17 | 3 | Biz2Credit Series B great; Moglix is Series C; ReshaMandi only "looking" |
| 18 | 5 | All fintech funding: OfBusiness, GetVantage, Aye Finance |
| 19 | 5 | All AI investments: Elevation voiceover, Zendesk fund, Niki.ai |
| 20 | 2 | One Blackstone deal + Goldman M&A + gold ETFs ≠ active investors |

### Private Equity (avg 4.00)
| id | score | reason |
|---|---|---|
| 21 | 4 | ChrysCapital deal on-point; survey/market-state pieces marginal |
| 22 | 2 | Compensation survey, Oman exit, weekly digest — all score 0.00 |
| 23 | 5 | KKR medical devices, MS PE urology, hospital-buyout flashback |
| 24 | 3 | $27bn 2025 exits great; charts + 2021 recap marginal/stale |
| 25 | 2 | Icahn-Dell 2013, LBO essay, 2015 Crompton — no latest buyouts |
| 26 | 5 | All PE sector-allocation: financial services/IT/realty |
| 27 | 5 | All Blackstone investments (Financial Tech, Sona BLW, infra) |
| 28 | 5 | All KKR portfolio-firm stories ($500mn tag, infra exit) |
| 29 | 5 | Lighthouse, Multiples consumer fund, consumer PE recap |
| 30 | 4 | 2/3 compare Blackstone-KKR; KKR-only impact fund result |

### Venture Capital (avg 3.90)
| id | score | reason |
|---|---|---|
| 31 | 5 | All most-active-VC rankings (2018/2019 flashbacks, pecking order) |
| 32 | 5 | Neo Group round (Jul 2026) + portfolio stories, all Peak XV |
| 33 | 5 | All Accel startup backing (cloud analytics, deep-tech) |
| 34 | 5 | All VC-in-AI: Niki.ai, AI funding report, Accel enterprise AI |
| 35 | 5 | All fintech backers: YC/SaveIN, Omidyar/Kaleidofin, Shivalik |
| 36 | 2 | Funding-sinks trend, awards list, single Rusk round — no top rounds |
| 37 | 3 | All VC-in-SaaS but 2014-2017 ≠ "recently" |
| 38 | 2 | All Peak XV-only results; Accel co-backing never shown |
| 39 | 5 | Bessemer India fund, Fundamental debut, Gulf VC funds |
| 40 | 2 | 2015 Sequoia fundraise + deals digests; VC fundraising intent missed |

### Mergers & Acquisitions (avg 3.50)
| id | score | reason |
|---|---|---|
| 41 | 4 | All M&A-India but 2016-2024 for "latest" ask |
| 42 | 2 | Goldman global volume stats; real Indian M&A buried at 0.00 |
| 43 | 3 | All acquisitions but 2009-2019 ≠ "recently" |
| 44 | 3 | On-topic tech M&A but 2010, plus duplicate M&M/Tech Mahindra pair |
| 45 | 4 | Cross-border intent present: Motherson, Aegis, WNS |
| 46 | 4 | CredFlow, MonkeyBox acquisitions + trend piece; "recently" stale |
| 47 | 5 | All Paytm acquisitions: Balance, Raheja QBE, Nightstay |
| 48 | 2 | Relation reversed: 8× "Zomato acquires X" for "who acquired Zomato" |
| 49 | 3 | M&A recaps but no sector-comparison structure |
| 50 | 5 | All M&A-trend pieces: buoyancy, record 2022, sector flashback |

### Sector Analysis (avg 3.60)
| id | score | reason |
|---|---|---|
| 51 | 4 | Two fintech-trend pieces (2018) + PayU news; "latest" stale |
| 52 | 4 | All ecosystem commentary but 2011-2021 |
| 53 | 4 | Healthcare summit + 2025 dealmaking outlook + gap-bridging |
| 54 | 3 | Tata EV plans, EV policy are EV items but 2022-23, scores ≤0.39 |
| 55 | 4 | Speech AI deeptech, AI-IT impact — real AI trends despite note flag |
| 56 | 4 | IVCA-EY 14-sectors + consumer-investment pieces |
| 57 | 3 | Edtech consolidation/Veranda on-topic but scores ≤0.05, stale |
| 58 | 4 | Real-estate PE inflow/outlook pieces answer via investment lens |
| 59 | 2 | "Consumer internet" ask returned generic consumer-sector pieces |
| 60 | 4 | Sectoral deal-flow interview + banking/healthcare deal flashbacks |

### Industry-Specific Questions (avg 4.00)
| id | score | reason |
|---|---|---|
| 61 | 4 | All SaaS-investment but anchored in 2014 |
| 62 | 4 | Green-energy digest + Brookfield-Leap Green; Masdar non-India |
| 63 | 3 | Tesla India entity on-point; others are policy pieces |
| 64 | 5 | All D2C investments: D'chica, Mother Sparsh, Masqa |
| 65 | 5 | Faarms, Omnivore fund, Cropin — agritech funding core |
| 66 | 4 | NewCold warehousing + TVS Supply Chain; event promo ranked #1 |
| 67 | 4 | All healthcare funding (IThrive, Poshtick, Xcode) but stale |
| 68 | 2 | 2014 wafer-plant approval + podcast; "latest" development absent |
| 69 | 5 | All climate-tech investment (GrowX, ranking report, 2022 woes) |
| 70 | 4 | TVS Capital investor view + Zepto/TPG news; adoption stat marginal |

### People & Leadership (avg 3.50)
| id | score | reason |
|---|---|---|
| 71 | 5 | All Kunal Shah (WhatsApp-CRED, Thrasio-style, Carl Pei) |
| 72 | 5 | Kamath investments: InCred, GreenLine, AssetPlus |
| 73 | 3 | Kunal Shah named in all 3 but 2019-2022 ≠ "recent" |
| 74 | 5 | All most-active-investor lists (angels 2015, VCs 2025) |
| 75 | 2 | Tyke pre-seed + NEA fund raise + digest; founders ranking absent |
| 76 | 3 | All CEO appointments but 2009-2024 ≠ "latest" |
| 77 | 4 | Snapdeal head joining Lenskart perfect; 2 funding pieces |
| 78 | 2 | All 2015 (Arnab Goswami, NHB CEO) for "this month" ask |
| 79 | 2 | No Kamath interviews; funding news + Modi/Tata interviews |
| 80 | 4 | DAM churn + Pantheon/Bain people moves on-point; HSBC marginal |

### Trends & Market Intelligence (avg 2.80)
| id | score | reason |
|---|---|---|
| 81 | 2 | "This week" ask got 2024/2020 most-read flashbacks |
| 82 | 1 | "Today's news" → 2012-2017 VCCircle corporate launch announcements |
| 83 | 3 | Trend-watch pieces exist but all 2018, "right now" missed |
| 84 | 2 | No YoY comparison; one Q2 funding story + MENA digests |
| 85 | 5 | All sector-interest: distressed, bulk-drug, financial services |
| 86 | 3 | 2025 flashbacks relevant but retrospective, not forward themes |
| 87 | 2 | 2010 Startup Watch + entrepreneur tips; no emerging-startup list |
| 88 | 2 | Hiring piece + edtech event + vague 2009 essay — no challenges |
| 89 | 5 | Flipkart IPO, MobiKwik refiling, SMC Global — exact IPO-prep |
| 90 | 3 | Exit pieces relevant but retrospective; "expected" exits absent |

### Search, Analysis & Custom Queries (avg 3.10)
| id | score | reason |
|---|---|---|
| 91 | 4 | Top-3 mostly Zerodha (FY24 results, merchant banking) |
| 92 | 4 | All Swiggy events (IPO, valuation) but scores ~0.01, unsorted |
| 93 | 3 | No comparative piece; Paytm-only and PhonePe-only items interleaved |
| 94 | 2 | Event promo + single licence news; no top-5 fintech stories |
| 95 | 4 | All quick-commerce: profitability, antitrust, Flipkart/Amazon |
| 96 | 4 | All UPI: record quarter, RBI-Paytm continuity, CheQ |
| 97 | 2 | "Last 12 months" → 2019 evergreen + one 2022 piece |
| 98 | 2 | Only TVS Capital piece on-topic; 2013 panel + awards filler |
| 99 | 3 | Two quick-commerce pieces on-point; e-commerce rules marginal |
| 100 | 3 | Correct digest format but May 2026 ≈ 4 months stale |

## Appendix B — Chat per-prompt scores

Flags: **F**=fallback (llm_used=false) · **S**=stale (pre-2023 data for a recency prompt) · **H**=hallucination-suspect · **D**=dataviz violation (none in run)

### Company & Brand — avg 4.30
| id | score | flags | reason |
|---|---|---|---|
| 1 | 5 | | Paytm profile: GMV, IPO anchor, funding, payments bank all match sources |
| 2 | 3 | | Honest "no info on Banyan Netfaqs"; correctly distinguishes Banyan Tree Finance |
| 3 | 5 | | Funding history 2014→2021 anchor round, all source-matched |
| 4 | 4 | H | Investors grounded, but "Sequoia Capital India" claim weakly cited ([6][8] off-topic) |
| 5 | 4 | | Founders named with explicit 2008-vs-2010 conflict disclosure; Neha Bansal unverified |
| 6 | 5 | | Business model (brokerage, Coin, APIs) matches sources incl. 2026 article |
| 7 | 4 | | 2022 + Q4-FY26 data points honestly frame 3-year arc; thin middle |
| 8 | 5 | | Zopper Retail, ZestMoney, GigIndia, Indus OS all in sources; notes Flipkart deal |
| 9 | 3 | S | "Latest" financials = Dec 2021 data; honestly dated but stale |
| 10 | 5 | | Vedantu/Unacademy/PW/upGrad directly from FY22 comparison source |

### Funding & Startup Deals — avg 3.60
| id | score | flags | reason |
|---|---|---|---|
| 11 | 4 | | Leads with 2025 Gupshup; mixes 2022–24 deals without dates |
| 12 | 4 | | Explicitly frames "week ended 22 Nov 2024"; Zepto/HealthKart grounded |
| 13 | 3 | S | 2010–2022 mix for "recently"; honestly flags Skeps below threshold |
| 14 | 4 | | Nov-2024 framing; only 2 deals but grounded and direct |
| 15 | 4 | | Good 2025 Series A stats; trailing "***" artifact; Zension is MENA not India |
| 16 | 2 | S | 2015/2017 seed deals presented as "recent"; no date framing |
| 17 | 3 | | "Largest Series B" from only 3 thin sources; can't support superlative |
| 18 | 4 | | Cars24/Razorpay/Oxyzo grounded; mixed 2019–23 vintage |
| 19 | 5 | | H2O.ai, Observe.AI, Arya.ai all source-matched |
| 20 | 3 | | Single-source answer generalizes "most active investors" from one Blackstone piece |

### Private Equity — avg 4.30
| id | score | flags | reason |
|---|---|---|---|
| 21 | 4 | | ChrysCapital X close + 2024 recap deals; "latest" leans on year-old recap |
| 22 | 5 | | 2026 deals (Neysa $600M, Novartis $159M, Wellbeing $175M) match sources |
| 23 | 5 | | Exhaustive, well-cited PE-in-healthcare mapping |
| 24 | 5 | | 2025 exits ($27.5bn, Temasek/Schneider, KKR/JB) straight from Flashback source |
| 25 | 2 | S | 2012–2019 buyouts (Aman, Fortis) framed as "recent activity" |
| 26 | 5 | | 2025 FS/IT vs 2024 realty/healthcare shift, source-matched |
| 27 | 5 | | Blackstone realty/PE/credit inventory well grounded |
| 28 | 5 | | KKR infra/education/logistics holdings match 2025 sources |
| 29 | 4 | | Multiples/Lighthouse consumer grounding; somewhat list-y |
| 30 | 3 | | Solid compare structure but 2008–2011 AUM data, dated framing |

### Venture Capital — avg 4.30
| id | score | flags | reason |
|---|---|---|---|
| 31 | 5 | | 2025 Flashback: Accel #1, Peak XV, Blume — precise |
| 32 | 5 | | July-2026 Neo Group, Primer, Exaforce — current and cited |
| 33 | 5 | | Accel portfolio incl. Atoms 3.0 cohort, sector-organized |
| 34 | 4 | | Investors grounded; 2015–2024 vintage mix |
| 35 | 4 | | SaveIN/Kaleidofin/Zolve grounded; slightly list-y |
| 36 | 5 | | 2026 rounds + honest "no deal crossed $100M" caveat |
| 37 | 2 | S | 2013–2017 SaaS deals presented as "recently" |
| 38 | 5 | | Accel∩Peak XV intersection (Primer, HomeLane) with careful sourcing |
| 39 | 5 | | Transition VC (Jul 2026), Bessemer, Fundamental — current |
| 40 | 3 | | Leads with 2024 fund closes for "latest"; no date framing |

### Mergers & Acquisitions — avg 3.90
| id | score | flags | reason |
|---|---|---|---|
| 41 | 5 | | 2025 top M&As (Schneider/Temasek, MUFG/Shriram) grounded |
| 42 | 5 | | 2026 mega-deals incl. India deals; all as-reported by sources |
| 43 | 2 | S | 2009–2019 acquisitions framed as "recently" |
| 44 | 2 | S | 2009–2016 tech deals for a present-tense ask; undated in text |
| 45 | 3 | | Trend-heavy, few named deals for "cross-border acquisitions" |
| 46 | 4 | | Leads 2023 deals; marks older ones "historical" |
| 47 | 5 | | Paytm acquisitions incl. terminated Raheja QBE nuance |
| 48 | 5 | | Correctly infers sources show Zomato as acquirer, not acquired |
| 49 | 5 | | Best structured sector comparison; 2024/25 Flashback grounded |
| 50 | 3 | S | 2022–23 trends presented as current |

### Sector Analysis — avg 3.70
| id | score | flags | reason |
|---|---|---|---|
| 51 | 3 | S | "Latest fintech trends" = 2018–22 material |
| 52 | 3 | | Generic "third largest ecosystem" from mixed 2011–24 sources |
| 53 | 4 | | Dealmaking-focused healthcare view, decently grounded |
| 54 | 2 | S | Entire answer from one 2023 residential-chargers article |
| 55 | 5 | | Sharp AI trends from two fresh 2026 sources |
| 56 | 3 | S | Answers "most investment" entirely from 2021 report (though dated) |
| 57 | 4 | | Veranda/Next Education lead; 2023–24 freshness |
| 58 | 4 | | Asset-class-by-asset-class realty view, grounded |
| 59 | 4 | | Outlook 2025 premiumization/quick-commerce grounding |
| 60 | 5 | | BFSI/healthcare deal activity from 2024–25 recaps |

### Industry-Specific Questions — avg 3.30
| id | score | flags | reason |
|---|---|---|---|
| 61 | 3 | | "Top SaaS investments" from 2014 Rewind-era data |
| 62 | 4 | | TPG/Siemens, Actis/Stride from Mar-2025 Deals Digest |
| 63 | 3 | S | EV leaders from 2020–21 sources, present tense |
| 64 | 4 | | ChrysCap/Lenskart 2023, D2C Insider 2024 grounded |
| 65 | 4 | | Ergos/Cropin detail; 2023 freshness ceiling |
| 66 | 1 | F | Refused despite clearly on-topic sources (NewCold, TVS Supply Chain) — false-positive weak gate |
| 67 | 3 | S | Dates every deal honestly, but 2013–22 material for "recently" |
| 68 | 3 | | Vedanta-Foxconn stall grounded; 2023 vintage for "latest" |
| 69 | 4 | | Good 2022 climate data; dangling "see the table below" with no table |
| 70 | 4 | | Swiggy/Zomato-Blinkit/Zepto grounded; mild vintage mix |

### People & Leadership — avg 2.90
| id | score | flags | reason |
|---|---|---|---|
| 71 | 5 | | Rich Kunal Shah bio incl. June-2026 WhatsApp move |
| 72 | 5 | | Zerodha/Rainmatter/InCred/AssetPlus all source-matched |
| 73 | 1 | F | Refused although all 3 sources are literally about Kunal Shah |
| 74 | 5 | | VC/PE/angel breakdown from 2025 Flashback |
| 75 | 2 | S | 2021–22 raises as "most capital recently"; superlative unsupported |
| 76 | 1 | F | Refused; sources include an actual CEO-appointment article |
| 77 | 3 | S | One 2017 hire for "recently joined" — thin but honest |
| 78 | 1 | F | Zero sources; canned "No sufficiently relevant articles" |
| 79 | 1 | F | Refused; no interview content exists — but could have summarized available news |
| 80 | 5 | | Named 2025–26 moves across Pantheon/Bain/DAM — current and precise |

### Trends & Market Intelligence — avg 2.70
| id | score | flags | reason |
|---|---|---|---|
| 81 | 1 | F | Zero sources; canned refusal for "this week" |
| 82 | 1 | F | "Today's news" → 2012/2017 sources → refusal |
| 83 | 1 | F | "Right now" trends → 2017–18 sources → refusal |
| 84 | 4 | | India 2025-vs-2024 comparison with numbers; MENA digression |
| 85 | 3 | S | 2019–21 sectors framed as "currently" |
| 86 | 5 | | Ten Flashback-2025 sources → best-in-run thematic synthesis |
| 87 | 2 | S | "Emerging startups to watch" = Ecologix (2010) and Zapr (2015) |
| 88 | 3 | | Generic challenges from grab-bag 2009–25 sources |
| 89 | 4 | S | Zetwerk/Jio current; 2020 Flipkart claim in present tense |
| 90 | 3 | S | Recaps 2021 exits; doesn't answer what's "expected" |

### Search, Analysis & Custom Queries — avg 2.90
| id | score | flags | reason |
|---|---|---|---|
| 91 | 3 | | FY24 Zerodha summary solid, but "everything reported" = 1 article |
| 92 | 3 | | Timeline = founding + IPO dates from one source; thin |
| 93 | 2 | | Paytm-vs-PhonePe comparison impossible from one 2017 article |
| 94 | 4 | | Five fintech stories constructed coherently from sources |
| 95 | 2 | S | "Find all recent articles" → summarizes one Feb-2023 piece |
| 96 | 3 | S | UPI insights solid but June-2022 vintage |
| 97 | 3 | S | "Last 12 months" answered with 2022–23 state |
| 98 | 4 | | Investor-skepticism view well grounded in Srinivasan piece |
| 99 | 4 | | Balanced pre-investment briefing; 2025 data + regulatory caveats |
| 100 | 1 | F | Zero sources; canned refusal for "this week" |
