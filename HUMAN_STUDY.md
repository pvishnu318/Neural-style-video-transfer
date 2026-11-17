# Human rating study (Google Form) - design

Status: **clips prepared, form NOT created, nobody invited, no ratings collected.** Creating the form and inviting people is your action (outward-facing).

## What is compared
Four methods per video+style (all on TEST videos, with the 3 showcase styles; never used for selection):
B0 (pretrained AdaIN) | B1 lt=150 | B1 lt=500 | B0 + flow blend 0.3.
(control ~= B0 on every metric, so it is left out to keep the form short; B2 is left out because it is not distinguishable from B1.)

## Material (already made)
`results/human_study/S1_A.mp4 ... S9_D.mp4` = 9 sets x 4 clips, 854x480, 24 fps, H.264, no audio. Letters A-D are shuffled per set (fixed seed 2026).
`results/human_study/KEY_DO_NOT_SHARE.csv` maps set+letter -> method. **Do not open it before collecting answers and do not share it with raters.**
Recommended: use 6 of the 9 sets (about 24 clips) so the form takes ~10 minutes; pick them before looking at any result.

## Form layout
Intro: "You will see short stylised videos of the same clip and style image. Watch each clip fully. No clip is 'the right answer'."
Show the style image once per set. For each of the 6 sets, on one page: 4 clips (A-D) and for **each clip three 1-10 sliders** (1 = very bad, 10 = excellent):
1. *Style*: how well does it look like the style image?
2. *Content*: can you clearly recognise the scene/objects?
3. *Smoothness*: how free is it from flicker/jitter between frames? (10 = perfectly steady)
Plus one optional overall question per set: "Which clip do you like best overall?" (A/B/C/D).
Add one attention check: one extra clip that is the **same video repeated twice** (e.g. S1_A shown again as "E") - answers that differ by >3 points are flagged.
Collect: age group (optional), "have you watched style-transfer videos before?" (yes/no). No names or emails (use an anonymous link).

## Hosting
Google Forms cannot play local files: upload the mp4s to Google Drive (link sharing "anyone with the link") or YouTube (unlisted) and embed/link them.
Keep file names anonymous (S1_A.mp4 etc.) - they are.

## Analysis (decide now, so it cannot be tuned to the outcome)
* Target >= 20 raters (more is better; say honestly how many you got).
* Per clip: mean of each criterion; per method: mean over sets and raters, 95% bootstrap CI **over raters** (rater = independent unit).
* Compare methods with paired differences per rater (B1 lt150 vs B0, lt500 vs B0, lt500 vs blend0.3, ...). Report effect sizes with CIs; avoid claiming significance from tiny N.
* Report "best overall" as vote shares.
* Expected from the numbers (a hypothesis, not a result): smoothness B1-lt500 > B1-lt150 > B0; style B0 >= lt150 > lt500; blend 0.3 smooth but visibly worse style.
  If raters disagree with the metrics, that is a finding to report, not something to hide.
* Do not include the metric numbers in the form.
